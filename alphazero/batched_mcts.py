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
import math
import time
from typing import Callable, Optional

import chess
import numpy as np
import torch

from . import utils as f
from .mcts import Node
from .nn import value_to_scalar


class BatchedMCTS:
    def __init__(self, args: dict, model) -> None:
        self.args = args
        self.model = model.to(args["device"])
        self.model.eval()  # defensive: ensure no train-mode BN/dropout during batched eval
        self.root: Node | None = None
        self.batch_size = int(args.get("batch_size", 8))
        self.last_max_depth = 0
        self.last_was_proven_mate: bool = False
        # Per-search set of leaf depths; cleared at the start of `search`.
        self._visited_depths: set[int] = set()

    def apply_action(self, action: int) -> None:
        """O(1) walk by known action -- see mcts.MCTS.apply_action."""
        if self.root is None:
            return
        child = self.root.children.get(action)
        if child is None:
            self.root = None
            return
        self.root = child
        self.root.parent = None

    def update_root(self, state: chess.Board, move_counter: int | None = None) -> None:
        """Walk to the descendant whose position matches `state` (within two
        levels) and make it the new root. Reset if no match.

        Depth 1: self-play (search called every half-move). Depth 2:
        match.py / uci.py / lichess-bot, where search runs only on our turn
        and the new state is a grandchild of the previous root (our move +
        opp's reply). Without the depth-2 walk, the entire tree was being
        discarded every move in those contexts.

        Idempotent: if root already matches `state`, no-op (allows callers
        to pre-walk via `apply_action`).

        See mcts.py:MCTS.update_root for full rationale.
        """
        if self.root is None:
            return
        # Already at the right state.
        if self.root.state == state and (
            move_counter is None or self.root.move_counter == move_counter
        ):
            return
        # Depth 1: direct child (self-play half-move transition).
        for child in self.root.children.values():
            if child.state == state and (
                move_counter is None or child.move_counter == move_counter
            ):
                self.root = child
                self.root.parent = None
                return
        # Depth 2: grandchild (bot's move + opp's reply between searches).
        for child in self.root.children.values():
            for grandchild in child.children.values():
                if grandchild.state == state and (
                    move_counter is None or grandchild.move_counter == move_counter
                ):
                    self.root = grandchild
                    self.root.parent = None
                    return
        self.root = None

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

        # μ-FPU: visit-weighted mean Q over children with at least one real
        # visit. Σ Q / Σ N treats every observation equally rather than every
        # child equally, so a 1-visit outlier can't yank the FPU around.
        # Excludes virtual-only contributions (we want a stable signal here,
        # not one that drifts with the in-flight batch).
        explored_Q_sum: float = 0.0
        explored_N_sum: int = 0
        for c in node.children.values():
            if c.N > 0:
                explored_Q_sum += c.Q
                explored_N_sum += c.N
        mu_fpu: float = (explored_Q_sum / explored_N_sum) if explored_N_sum > 0 else 0.0

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

    # ---------- proof propagation (mirrors mcts.MCTS._try_prove) ----------

    @staticmethod
    def _try_prove(node: Node) -> None:
        """Proven value from THIS node's player-to-move perspective.
        Perspective-free, no dependency on tree root.
        OR-rule: any child whose player (opp) loses (proven_value=-1) →
                 this player wins → node.proven_value=+1.
        Otherwise need all legal moves expanded; node.proven_value
                 = -min(child.proven_value)."""
        if node.proven_value is not None:
            return
        if not node.children:
            return
        if any(c.proven_value == -1 for c in node.children.values()):
            node.proven_value = 1
            return
        if node.n_legal is None or len(node.children) < node.n_legal:
            return
        child_provens = [c.proven_value for c in node.children.values()]
        if any(pv is None for pv in child_provens):
            return
        node.proven_value = -min(child_provens)  # type: ignore

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
        while True:
            if node.is_terminal():
                value: float = float(
                    f.game_result(node.state, node.move_counter, 1000)[0]
                )
                # value is from current player's POV: -1 mated, 0 drawn.
                if value == 0.0 or value == -1.0:
                    node.proven_value = int(value)
                return path, node, "terminal", value
            if not node.is_expanded():
                return path, node, "needs_eval", None
            action, mu_used = self._select_action_with_vloss(node)
            if action in node.children:
                node = node.children[action]
            else:
                node = node.materialize_child(action)
            node.virtual_loss += 1
            node.virtual_Q += mu_used
            path.append((node, mu_used))

    def _simulate_batch(self, root: Node, sims_remaining: int) -> int:
        """Run up to `min(batch_size, sims_remaining)` parallel sims.
        Returns the number of sims that actually contributed (duplicates
        whose leaf is already in the batch are skipped to avoid double-
        backprop of the same value)."""
        batch_target: int = min(self.batch_size, sims_remaining)

        # Phase 1: selection. Each entry is (path, leaf, status, value-or-None).
        # `path` is a list of (node, mu_used) tuples produced by _select_one_leaf;
        # mu_used is the Virtual-Mean Q value added to that node's virtual_Q.
        in_flight: list[tuple[list[tuple[Node, float]], Node, str, float | None]] = []
        for _ in range(batch_target):
            in_flight.append(self._select_one_leaf(root))

        # Deduplicate needs_eval leaves: if two sims hit the same unexpanded
        # leaf, only the first one is evaluated/backpropagated. The duplicate
        # has its virtual stats undone and is skipped. This matches sequential
        # MCTS semantics where after expanding a leaf, subsequent sims descend
        # deeper rather than re-evaluating the same position.
        seen_leaves: set[int] = set()
        skip: set[int] = set()
        unique_eval: list[tuple[int, Node]] = []
        for sim_idx, (_, leaf, status, _) in enumerate(in_flight):
            if status == "needs_eval":
                lid: int = id(leaf)
                if lid in seen_leaves:
                    skip.add(sim_idx)
                    continue
                seen_leaves.add(lid)
                unique_eval.append((sim_idx, leaf))

        # Phase 2: batched NN evaluation for unique needs_eval leaves.
        # Single GPU→CPU sync: concat values + policies into one (B, 4673)
        # tensor and pull across PCIe once. Same idea as mcts.py:expand_lazy.
        if unique_eval:
            inputs: torch.Tensor = torch.stack([
                f.prepare_input(leaf.state, leaf.move_counter)
                for _, leaf in unique_eval
            ]).to(self.args["device"])
            with torch.inference_mode():
                value_t, policy_t = self.model(inputs)

            masks_np: np.ndarray = np.stack(
                [f.legal_mask(leaf.state) for _, leaf in unique_eval]
            )
            masks_t: torch.Tensor = torch.from_numpy(masks_np).to(policy_t.device)

            values_t: torch.Tensor = value_to_scalar(
                value_t, mode=self.args.get("value_scalar", "expected")
            ).reshape(-1, 1)                                    # (B, 1) on GPU
            masked_logits: torch.Tensor = policy_t.masked_fill(~masks_t, float("-inf"))
            policies_t: torch.Tensor = torch.softmax(masked_logits, dim=1)  # (B, 4672)

            # ONE sync replaces two separate .cpu() calls.
            combined: np.ndarray = torch.cat(
                [values_t, policies_t], dim=1
            ).cpu().numpy()
            values: np.ndarray = combined[:, 0]
            policies: np.ndarray = combined[:, 1:]

            for (sim_idx, leaf), val, pol, mask_row in zip(
                unique_eval, values, policies, masks_np
            ):
                if leaf.raw_policy is None:
                    leaf.raw_policy = pol
                    leaf.policy = pol.copy()
                    leaf.n_legal = int(mask_row.sum())
                    leaf.raw_nn_value = float(val)
                path, _, _, _ = in_flight[sim_idx]
                in_flight[sim_idx] = (path, leaf, "evaluated", float(val))

        # Phase 3: backprop + virtual-stat revert + proven-value propagation.
        # Every in-flight sim -- whether successfully backpropagated or skipped
        # -- MUST revert both virtual_loss (-1) and virtual_Q (-mu_used) on
        # every (node, mu_used) pair in its path, or the in-flight stats leak
        # into future batches and corrupt selection.
        gamma: float = self.args.get("discount", 1.0)
        effective: int = 0
        for sim_idx, (path, leaf, _status, value) in enumerate(in_flight):
            if sim_idx in skip or value is None:
                # Duplicate (or unexpected): undo virtual stats, don't backprop.
                for n, mu_used in path:
                    if n.virtual_loss > 0:
                        n.virtual_loss -= 1
                        n.virtual_Q -= mu_used
                continue
            # Successful sim: real backprop + virtual revert in the same walk.
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

    # ---------- top-level search ----------

    def search(
        self,
        state: chess.Board,
        move_counter: int,
        info_callback: Optional[Callable[["BatchedMCTS", int, float, int], None]] = None,
        info_interval_s: float = 0.2,
    ) -> np.ndarray:
        root_state = state.copy()
        if self.root is not None:
            self.update_root(state, move_counter)

        if self.root is None:
            self.root = Node(self.args, state, move_counter)
            self.root.expand_lazy(self.model)
            min_depth = 0
        else:
            min_depth = self.root.depth

        # Dirichlet noise over LEGAL moves only (sampling over the full 4672
        # action space would waste ~99% of the noise mass on illegal indices).
        eps = self.args["dirichlet_epsilon"]
        assert self.root.raw_policy is not None
        if eps > 0:
            legal_idx = np.nonzero(self.root.raw_policy)[0]
            if len(legal_idx) > 0:
                noise = np.random.dirichlet([self.args["dirichlet_alpha"]] * len(legal_idx))
                mixed = self.root.raw_policy.copy()
                mixed[legal_idx] = (1 - eps) * mixed[legal_idx] + eps * noise
                total = mixed.sum()
                self.root.policy = mixed / total if total > 0 else mixed
            else:
                self.root.policy = self.root.raw_policy.copy()
        else:
            self.root.policy = self.root.raw_policy.copy()

        self._visited_depths.clear()
        total_sims = int(self.args["num_simulation"])
        completed = 0
        t_start = time.monotonic()
        t_last_report = t_start
        while completed < total_sims:
            completed += self._simulate_batch(self.root, total_sims - completed)
            # Update last_max_depth incrementally so live info_callback can read it.
            if self._visited_depths:
                self.last_max_depth = max(self._visited_depths) - min_depth
            if info_callback is not None:
                now = time.monotonic()
                if now - t_last_report >= info_interval_s:
                    try:
                        info_callback(self, completed, now - t_start, self.last_max_depth)
                    except Exception:
                        # Never let monitoring break the search.
                        pass
                    t_last_report = now
        self.last_max_depth = (
            max(self._visited_depths) - min_depth if self._visited_depths else 0
        )
        if info_callback is not None:
            # Final tick so the GUI gets the last state before bestmove is sent.
            try:
                info_callback(self, completed, time.monotonic() - t_start, self.last_max_depth)
            except Exception:
                pass

        # Force-mate: bot wins by playing into a child whose player (opp) is
        # proven losing -- child.proven_value == -1 in the perspective-free
        # convention. child.Q is still from PARENT's (= bot's) POV; pick the
        # child with the highest Q/N (most-confident win, shortest mate under
        # depth discount).
        action_probs = np.zeros(self.args["action_space"])
        proven_winners = [
            (action, child) for action, child in self.root.children.items()
            if child.proven_value == -1 and child.N > 0
        ]
        if proven_winners:
            best_action, _ = max(proven_winners, key=lambda ac: ac[1].Q / ac[1].N)
            action_probs[best_action] = 1.0
            self.last_was_proven_mate = True
            return action_probs

        # Otherwise, visit-count distribution.
        self.last_was_proven_mate = False
        for action, child in self.root.children.items():
            action_probs[action] = child.N
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        return action_probs
