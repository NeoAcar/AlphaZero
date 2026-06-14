"""
Batched MCTS with virtual losses.

Same algorithm as mcts.MCTS, but parallelises leaf evaluation: instead of
one NN forward per simulation, the search collects `batch_size` leaves in
parallel using virtual losses to discourage threads from converging on the
same path, then runs ONE NN forward on the whole batch. Trades a small
change in exploration behaviour for a 5-10x speedup on GPU at batch=8-16
(GPU is severely underutilised at batch=1).

Interface is drop-in compatible with mcts.MCTS:

    mcts = BatchedMCTS(args, model)
    probs = mcts.search(board, move_counter)   # np.ndarray (action_space,)

`args` extension over mcts.MCTS:
    batch_size      (int, default 8)  number of sims per NN forward pass

Behaviour relative to sequential mcts.MCTS:
    - Limit as sims -> infinity: identical.
    - Finite sims: slightly different exploration. Visit distributions can
      differ because some sims see only virtual losses where sequential
      would see fully-updated Q values. Net Elo effect is roughly neutral;
      the 5-10x wall-clock speedup is the real win.
    - Determinism: with fixed seed AND fixed batch_size, deterministic.
      Different batch_size -> potentially different move choices.

Notes:
    - Terminal nodes don't need NN evaluation; they're processed inline
      in the selection phase.
    - Re-entry into the same unexpanded leaf within one batch is allowed
      (wastes one slot of compute but is correct).
    - All proven_value / discount / force-win-on-proof logic from mcts.py
      carries over unchanged via shared Node class.
"""
import logging
import math
import time
from typing import Callable, Optional

import chess
import numpy as np
import torch

from . import utils as f
from .mcts import MCTSBase, Node, _amp_ctx
from .nn import value_scalar_and_wdl

_log = logging.getLogger(__name__)


def evaluate_leaves(model, leaves: list[Node], args: dict) -> dict[int, float]:
    """Run ONE batched NN forward over `leaves` (distinct unexpanded Nodes),
    cache their priors / value / WDL on each leaf (if not already set), and
    return {id(leaf): collapsed_value}.

    Tree-independent: single-game BatchedMCTS and the cross-game
    MultiGameSearcher both route their evaluation phase through this so a batch
    can span many games' leaves in one forward. Single GPU→CPU sync."""
    if not leaves:
        return {}
    use_history = args.get("input_planes") == 119
    inputs = torch.stack([
        f.prepare_input(
            leaf.state, leaf.move_counter,
            history=(leaf.history if use_history else None),
            rep_count=max(leaf.rep_count, 1),
        )
        for leaf in leaves
    ]).to(args["device"])
    if inputs.device.type == "cuda" and args.get("channels_last", True):
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode(), _amp_ctx(inputs.device):
        value_t, policy_t = model(inputs)

    masks_np = np.stack([
        (leaf.legal_mask_np if leaf.legal_mask_np is not None
         else f.legal_mask(leaf.state))
        for leaf in leaves
    ])
    masks_t = torch.from_numpy(masks_np).to(policy_t.device, non_blocking=True)

    # One softmax for both the collapsed scalar and the cached WDL probs.
    values_t, wdl_t = value_scalar_and_wdl(
        value_t, mode=args.get("value_scalar", "expected")
    )
    values_t = values_t.reshape(-1, 1).float()
    masked_logits = policy_t.float().masked_fill(~masks_t, float("-inf"))
    policies_t = torch.softmax(masked_logits, dim=1)

    # WDL probs ride along in the same single GPU→CPU sync. .float() guards
    # fp16 autocast outputs from leaking into numpy storage.
    if wdl_t is not None:
        combined = torch.cat([values_t, wdl_t.float(), policies_t], dim=1).cpu().numpy()
        values = combined[:, 0]
        wdl_probs = combined[:, 1:4]
        policies = combined[:, 4:]
    else:
        combined = torch.cat([values_t, policies_t], dim=1).cpu().numpy()
        values = combined[:, 0]
        wdl_probs = None
        policies = combined[:, 1:]

    leaf_value: dict[int, float] = {}
    for i, leaf in enumerate(leaves):
        val = float(values[i])
        if leaf.raw_policy is None:
            leaf.raw_policy = policies[i]
            leaf.policy = policies[i].copy()
            leaf.n_legal = int(masks_np[i].sum())
            leaf.legal_mask_np = masks_np[i]
            leaf.raw_nn_value = val
            if wdl_probs is not None:
                leaf.raw_nn_wdl = (
                    float(wdl_probs[i, 0]), float(wdl_probs[i, 1]), float(wdl_probs[i, 2])
                )
        leaf_value[id(leaf)] = val
    return leaf_value


class BatchedMCTS(MCTSBase):
    """Batched MCTS. Shares all state/tree-reuse/proof/finalisation machinery
    with sequential MCTS via MCTSBase; only leaf evaluation is batched."""

    def __init__(self, args: dict, model) -> None:
        super().__init__(args, model)
        self.model.eval()  # defensive: no train-mode BN/dropout during batched eval
        self.batch_size = int(args.get("batch_size", 8))

    # ---------- selection with virtual loss ----------

    @staticmethod
    def _select_action_with_vloss(node: Node) -> tuple[int, float]:
        """PUCT with μ-FPU and Virtual Mean (Cazenave 2021 "Batch MCTS").

        Two refinements over the plain "unscored virtual visit" scheme:
          * μ-FPU: legal moves with no real or in-flight visits are NOT scored
            as Q=0 (which is pessimistic for the side to move). Instead, they
            get FPU = mean Q of explored children at this node (`mu_fpu`).
            Cazenave §3.3: "set [FPU] to the average mean of the node".
          * Virtual Mean: each child's effective Q in UCB blends real and
            virtual statistics:
                Q_eff = (Q + virtual_Q) / (N + virtual_loss)
            Cazenave §4.2: "the Virtual Mean ... adds the mean of the move to
            the sum of its evaluations in order to have more realistic
            statistics for the next descent."

        Returns:
            (action, mu_used) where `mu_used` is the Q value treated as this
            move's evaluation for purposes of UCB. The caller MUST add
            `mu_used` to the chosen child's `virtual_Q` (and 1 to its
            `virtual_loss`) so subsequent sims in the same batch see the
            Virtual-Mean-updated statistics, and revert both on undo.
        """
        c_base: float = node.args["c_base"]
        c_init: float = node.args["c_init"]
        t: float = node.args["t"]
        # PUCT invariant: N(parent) = sum_a N(parent, a). With virtual loss
        # inflating child denominators, the parent numerator must inflate too
        # so the exploration term scales correctly with in-flight visits.
        parent_N_eff: int = node.N + node.virtual_loss
        c_puct: float = math.log((1 + parent_N_eff + c_base) / c_base) + c_init
        sqrt_parent_N: float = math.sqrt(max(parent_N_eff, 1))

        assert node.policy is not None
        legal: np.ndarray = np.nonzero(node.policy)[0]
        priors: np.ndarray = node.policy[legal]

        # μ-FPU baseline: visit-weighted mean Q over children with at least one
        # real visit. Σ Q / Σ N treats every observation equally rather than
        # every child equally, so a 1-visit outlier can't yank the FPU around.
        # Excludes virtual-only contributions (we want a stable signal here,
        # not one that drifts with the in-flight batch).
        explored_Q_sum: float = 0.0
        explored_N_sum: int = 0
        for c in node.children.values():
            if c.N > 0:
                explored_Q_sum += c.Q
                explored_N_sum += c.N
        mu_fpu: float = (explored_Q_sum / explored_N_sum) if explored_N_sum > 0 else 0.0

        # FPU-reduction (Leela / KataGo): subtract cFPU·sqrt(P_explored) so
        # unvisited children are more pessimistic when little policy mass has
        # been explored yet. Disabled at root if Dirichlet noise is driving
        # exploration -- otherwise the reduction would fight the noise.
        c_fpu: float = float(node.args.get("c_fpu", 0.0))
        if node.parent is None and float(node.args.get("dirichlet_epsilon", 0.0)) > 0:
            c_fpu = 0.0
        if c_fpu > 0.0 and node.children:
            p_explored: float = 0.0
            for a, c in node.children.items():
                if c.N > 0:
                    p_explored += float(node.policy[a])
            mu_fpu -= c_fpu * math.sqrt(p_explored)

        # Per-action effective Q: combine real + virtual when there are any
        # visits, else fall back to mu_fpu.
        child_Q: np.ndarray = np.full(len(legal), mu_fpu, dtype=np.float64)
        child_N_started: np.ndarray = np.zeros(len(legal), dtype=np.int64)
        for i, a in enumerate(legal):
            child = node.children.get(int(a))
            if child is None:
                continue
            total_n: int = child.N + child.virtual_loss
            child_N_started[i] = total_n
            if total_n > 0:
                child_Q[i] = (child.Q + child.virtual_Q) / total_n

        ucb: np.ndarray = (
            child_Q + c_puct * (priors ** (1.0 / t)) * sqrt_parent_N / (1 + child_N_started)
        )
        best_local: int = int(np.argmax(ucb))
        chosen_action: int = int(legal[best_local])
        mu_used: float = float(child_Q[best_local])
        return chosen_action, mu_used

    # _try_prove is inherited from MCTSBase (identical proof propagation).

    # ---------- one batch of sims ----------

    def _select_one_leaf(
        self, root: Node
    ) -> tuple[list[tuple[Node, float]], Node, str, float | None]:
        """Walk from root to a leaf, applying virtual loss + Virtual Mean to
        each node on the way down.

        Returns (path, leaf, status, terminal_value-or-None) where `path` is
        a list of `(node, mu_used)` tuples. `mu_used` is the Q value that was
        added to that node's `virtual_Q` when the descent picked it. The root
        has `mu_used=0.0` because it is never "selected" by a parent.

        The caller is responsible for reverting BOTH `virtual_loss` (decrement
        by 1) AND `virtual_Q` (subtract `mu_used`) on every node in the path
        whether the sim is backpropagated or skipped.
        """
        path: list[tuple[Node, float]] = [(root, 0.0)]
        node: Node = root
        # Virtual loss on root inflates only its own N denominator for any
        # peer batched sims that bottom-out at root (none in practice, but
        # keeps the bookkeeping symmetric with the rest of the path).
        node.virtual_loss += 1
        trunc = int(self.args.get("truncation_halfmoves", 1000))
        while True:
            # Single game_result call. value is from current player's POV:
            # -1 mated, 0 drawn / truncated.
            term_value, is_term = f.game_result(
                node.state, node.move_counter, trunc, node.rep_count
            )
            if is_term:
                node.proven_value = int(term_value)
                return path, node, "terminal", float(term_value)
            if not node.is_expanded():
                return path, node, "needs_eval", None
            action, mu_used = self._select_action_with_vloss(node)
            if action in node.children:
                node = node.children[action]
            else:
                node = node.materialize_child(action)
                node.rep_count = self._compute_rep_count(node)
            node.virtual_loss += 1
            node.virtual_Q += mu_used
            path.append((node, mu_used))

    # The batch loop is split into three phases so a cross-game coordinator
    # (MultiGameSearcher) can run the *selection* and *backprop* phases per tree
    # while coalescing the *evaluation* phase (the only GPU step) across many
    # games into a single NN forward. Single-game search calls all three here.

    def _collect_batch(self, root: Node, batch_target: int) -> list:
        """Phase 1: selection. Returns `in_flight`, a list of
        (path, leaf, status, value-or-None). Applies virtual loss along each
        descent; the caller must eventually backprop every entry to revert it."""
        in_flight = []
        for _ in range(batch_target):
            in_flight.append(self._select_one_leaf(root))
        return in_flight

    @staticmethod
    def _unique_needs_eval(in_flight) -> list[Node]:
        """Distinct unexpanded leaves in `in_flight` (deduped by identity) that
        need an NN forward. Duplicates still backprop (see _backprop_batch); we
        only dedup the *evaluation* to save GPU compute."""
        seen: set[int] = set()
        leaves: list[Node] = []
        for _, leaf, status, _ in in_flight:
            if status == "needs_eval" and id(leaf) not in seen:
                seen.add(id(leaf))
                leaves.append(leaf)
        return leaves

    def _backprop_batch(self, in_flight, leaf_value: dict) -> int:
        """Phase 3: backprop every sim + revert its virtual stats + propagate
        proofs. `leaf_value` maps id(leaf)->value for evaluated leaves. EVERY
        needs_eval sim (including dedup duplicates) backprops the shared value,
        so visit counts are not undercounted; double-counting the identical
        value is safe (Q/N average and sign alternation hold)."""
        # Hand the evaluated value to every needs_eval sim (unique + duplicates).
        for sim_idx, (path, leaf, status, _) in enumerate(in_flight):
            if status == "needs_eval":
                v = leaf_value.get(id(leaf))
                if v is not None:
                    in_flight[sim_idx] = (path, leaf, "evaluated", v)

        gamma: float = self.args.get("discount", 1.0)
        effective: int = 0
        for sim_idx, (path, leaf, _status, value) in enumerate(in_flight):
            if value is None:
                # No value available (shouldn't normally happen): revert virtual
                # stats only, don't backprop.
                for n, mu_used in path:
                    if n.virtual_loss > 0:
                        n.virtual_loss -= 1
                        n.virtual_Q -= mu_used
                continue
            sign: float = -1.0
            for n, mu_used in reversed(path):
                n.N += 1
                n.Q += sign * value
                if n.virtual_loss > 0:
                    n.virtual_loss -= 1
                    n.virtual_Q -= mu_used
                sign = -sign * gamma
            self._visited_depths.add(leaf.depth)
            for n, _mu in reversed(path):
                self._try_prove(n)
            effective += 1
        return effective

    def _simulate_batch(self, root: Node, sims_remaining: int) -> int:
        """Run up to `min(batch_size, sims_remaining)` parallel sims through the
        three phases. Returns the number of sims backpropagated."""
        batch_target: int = min(self.batch_size, sims_remaining)
        in_flight = self._collect_batch(root, batch_target)
        leaves = self._unique_needs_eval(in_flight)
        leaf_value = evaluate_leaves(self.model, leaves, self.args)
        return self._backprop_batch(in_flight, leaf_value)

    # ---------- top-level search ----------

    def search(
        self,
        state: chess.Board,
        move_counter: int,
        info_callback: Optional[Callable[["BatchedMCTS", int, float, int], None]] = None,
        info_interval_s: float = 0.2,
    ) -> np.ndarray:
        root_state = state.copy()
        if not self.args.get("tree_reuse", True):
            self.root = None          # reuse disabled -> always search a fresh tree
        if self.root is not None:
            self.update_root(state, move_counter)

        if self.root is None:
            self.root = Node(
                self.args, state, move_counter,
                history=self._ext_history,
            )
            self.root.rep_count = self._compute_rep_count(self.root)
            self.root.expand_lazy(self.model)
            min_depth = 0
        else:
            min_depth = self.root.depth
            # Reused subtree: refresh rep_counts across the whole subtree against
            # the current external counter (stored values are relative to a prior
            # root). Fixes 3-fold/terminal detection deep in the reused tree.
            old_root_rep = self.root.rep_count
            self._refresh_rep_counts(self.root)
            if self.args.get("input_planes") == 119 and self.root.rep_count != old_root_rep:
                self.root.expand_lazy(self.model)

        self._apply_root_dirichlet()

        self._visited_depths.clear()
        base = int(self.args["num_simulation"])
        es = bool(self.args.get("early_stop", False))
        # Floor defaults to one batch so the eval/PV isn't from a near-empty search.
        min_sims = max(int(self.args.get("early_stop_min_sims", self.batch_size)), 1)
        max_borrow = int(self.args.get("max_borrow", 0))
        # Budget = base + borrowed-from-bank (capped). Easy positions early-stop
        # and refund to the bank; hard ones spend it -- allocation is emergent.
        target = base + (min(self.sim_bank, max_borrow) if es else 0)
        completed = 0
        t_start = time.monotonic()
        t_last_report = t_start
        while completed < target:
            completed += self._simulate_batch(self.root, target - completed)
            # Update last_max_depth incrementally so live info_callback can read it.
            if self._visited_depths:
                self.last_max_depth = max(self._visited_depths) - min_depth
            if es and completed >= min_sims and self._early_stop_decided(completed, target):
                break
            if info_callback is not None:
                now = time.monotonic()
                if now - t_last_report >= info_interval_s:
                    try:
                        info_callback(self, completed, now - t_start, self.last_max_depth)
                    except Exception:
                        # Never let monitoring break the search.
                        _log.exception("info_callback failed (mid-search); continuing")
                    t_last_report = now
        if es:
            self.sim_bank = max(0, self.sim_bank + base - completed)
        self.last_max_depth = (
            max(self._visited_depths) - min_depth if self._visited_depths else 0
        )
        if info_callback is not None:
            # Final tick so the GUI gets the last state before bestmove is sent.
            try:
                info_callback(self, completed, time.monotonic() - t_start, self.last_max_depth)
            except Exception:
                _log.exception("info_callback failed (final); continuing")

        return self._finalize_action_probs()


class MultiGameSearcher:
    """Run the search step for several concurrent games, coalescing ALL games'
    leaf evaluations into one NN forward per simulation round.

    A single game's batch (batch_size leaves) underfills the GPU; running G
    games together puts ~G*batch_size leaves in each forward, which is what
    actually saturates the GPU and gives the self-play throughput win. Each
    game keeps its OWN independent BatchedMCTS tree and its own per-move sim
    budget (PCR) / Dirichlet eps; only the NN forward is shared, so per-game
    results are identical to searching each game on its own.

    Usage (one move across all active games):
        searcher = MultiGameSearcher(model)
        pis = searcher.search_all(engines, states, move_counters)
    Set each engine's args["num_simulation"] / ["dirichlet_epsilon"] before the
    call (e.g. for PCR). Pass only the currently-active (non-terminal) games.
    """

    def __init__(self, model) -> None:
        self.model = model

    def search_all(self, engines: list, states: list, move_counters: list) -> list:
        if not engines:
            return []
        args0 = engines[0].args

        # 1. Root setup. Reuse the prior tree where possible, build+expand fresh
        #    roots otherwise, batching all fresh root evaluations into one forward.
        roots_to_eval: list[Node] = []
        for eng, st, mc in zip(engines, states, move_counters):
            if eng.root is not None:
                eng.update_root(st, mc)
            if eng.root is None:
                eng.root = Node(eng.args, st, mc, history=eng._ext_history)
                eng.root.rep_count = eng._compute_rep_count(eng.root)
            if eng.root.raw_policy is None:
                roots_to_eval.append(eng.root)
        evaluate_leaves(self.model, roots_to_eval, args0)
        for eng in engines:
            eng._apply_root_dirichlet()
            eng._visited_depths.clear()

        # 2. Simulation rounds. Each round: every active game selects a batch
        #    (CPU), all their leaves are evaluated in ONE forward, then each game
        #    backprops its own batch. Per-engine sim budgets may differ (PCR).
        totals = [int(eng.args["num_simulation"]) for eng in engines]
        completed = [0] * len(engines)
        while any(completed[i] < totals[i] for i in range(len(engines))):
            in_flights: list = [None] * len(engines)
            all_leaves: list[Node] = []
            for i, eng in enumerate(engines):
                rem = totals[i] - completed[i]
                if rem <= 0:
                    continue
                bt = min(eng.batch_size, rem)
                infl = eng._collect_batch(eng.root, bt)
                in_flights[i] = infl
                all_leaves.extend(eng._unique_needs_eval(infl))
            leaf_value = evaluate_leaves(self.model, all_leaves, args0)
            for i, eng in enumerate(engines):
                if in_flights[i] is not None:
                    completed[i] += eng._backprop_batch(in_flights[i], leaf_value)

        # 3. Finalise each game's move distribution.
        return [eng._finalize_action_probs() for eng in engines]
