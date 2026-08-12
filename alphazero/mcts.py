import contextlib
import logging
import math
import time
from typing import Callable, Optional

import chess
import numpy as np
import torch

from . import utils as f
from .nn import ResNet, value_scalar_and_wdl

_log = logging.getLogger(__name__)


def _amp_ctx(device: torch.device):
    """fp16 autocast on CUDA, no-op elsewhere. Used to wrap inference forwards."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class Node:
    """
    Lazy-expanded MCTS node.

    A node is "expanded" once `self.policy` is set. Children are materialised
    on demand: priors live for every legal action, but Node objects only
    exist for actions that were actually selected.

    `Q` is stored from the PARENT's perspective. Maximising `child.Q/child.N`
    in PUCT selects the move that's best for the parent's side to move.
    """

    def __init__(self, args, state, move_counter, depth=0,
                 parent=None, action=None, prior=None, history=None):
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.depth = depth
        self.move_counter = move_counter
        # Up to 7 prior canonical boards (chronological, oldest first). Used
        # only when input_planes=119 -- expand_lazy threads it into
        # board_to_matrix to fill the historical frames. Empty list = clean
        # start (zero-padded historical frames).
        self.history: list = list(history) if history else []

        # `raw_policy` = NN output masked to legal moves; immutable after expand.
        # `policy` = `raw_policy` plus any Dirichlet noise (added only when this
        # node is the root). Selection always reads from `policy`.
        self.raw_policy: np.ndarray | None = None
        self.policy: np.ndarray | None = None
        self.children: dict[int, "Node"] = {}
        self.Q = 0.0
        self.N = 0

        # Solver: proven value from THIS node's player-to-move perspective.
        # Perspective-free (no dependency on tree root), so it stays correct
        # under update_root tree reuse.
        # None = unproven. +1 = this player provably wins.
        # -1 = this player provably loses. 0 = provably drawn.
        # Set on terminal nodes immediately, propagated up by _try_prove().
        self.proven_value: int | None = None
        # Cached after expand_lazy: number of legal moves at this position
        # (used by the AND-node "all expanded" check during proof propagation).
        self.n_legal: int | None = None
        # Raw NN value head output for this position (scalar in [-1, +1] after
        # value_to_scalar collapse). Cached on first expansion so dashboards /
        # diagnostics can compare the NN's pre-search verdict against the
        # search-refined Q without spending an extra forward.
        self.raw_nn_value: float | None = None
        # Raw NN WDL distribution (P(W), P(D), P(L)) for WDL heads, None for
        # scalar-tanh heads. Same single GPU→CPU sync as raw_nn_value above.
        # Lets the dashboard compute variance for the win-prob band without
        # a second forward pass.
        self.raw_nn_wdl: tuple[float, float, float] | None = None
        # Optional auxiliary prediction from ChessFormerWDL. The training
        # target is raw remaining plies; this is telemetry only and is never
        # consulted by selection, backup, early-stop, or time management.
        self.raw_nn_moves_left: float | None = None
        self.raw_nn_moves_left_alpha: float | None = None
        # Cached legal-move mask (bool (4672,)) computed once in expand_lazy.
        # The board state is immutable, so the mask never needs invalidation;
        # lets batched dedup / repeat visits reuse it without recomputing.
        self.legal_mask_np: np.ndarray | None = None
        # Virtual loss counter for batched MCTS (number of in-flight sims
        # that have passed through this node). Always 0 in sequential MCTS.
        self.virtual_loss: int = 0
        # Virtual Q for batched MCTS (Cazenave 2021 "Virtual Mean"): cumulative
        # mu_used values from in-flight sims that selected this node. Used
        # alongside `virtual_loss` so the in-flight sims contribute a realistic
        # value estimate (not just an inflated visit count) when computing
        # combined Q during selection. Always 0.0 in sequential MCTS.
        self.virtual_Q: float = 0.0

        # Transposition key for 3-fold repetition tracking. `state.mirror()`
        # internally calls `copy(stack=False)` which wipes _transpositions, so
        # we can't ask the board itself -- we maintain the count externally.
        # `rep_count` = how many times this position has appeared in
        # (real-game history before MCTS root) + (tree path from root to here,
        # including this node). Filled in by MCTS after construction.
        self.tk = state._transposition_key()
        self.rep_count: int = 0

    def is_expanded(self) -> bool:
        return self.policy is not None

    def select_action(self) -> int:
        """PUCT with μ-FPU (Cazenave 2021 "Batch MCTS", §3.3, visit-weighted):
        unvisited children inherit FPU = Σ(child.Q) / Σ(child.N) over explored
        children, instead of the pessimistic Q=0 default.

        Confirmed +Elo over FPU=0 in an internal A/B (mcts_seresnet_mufpu vs
        mcts_seresnet, 4-0). LOS 93.75%; lower-bound 95% CI ≈ +30 Elo.
        """
        c_base: float = self.args["c_base"]
        c_init: float = self.args["c_init"]
        c_factor: float = self.args.get("c_factor", 1.0)
        t: float = self.args["t"]
        c_puct: float = c_init + c_factor * math.log((1 + self.N + c_base) / c_base)
        sqrt_N: float = math.sqrt(max(self.N, 1))

        legal: np.ndarray = np.nonzero(self.policy)[0]
        priors: np.ndarray = self.policy[legal]

        # Visit-weighted μ-FPU baseline: Σ Q / Σ N over explored children. More
        # stable than mean-of-means; a 1-visit outlier can't yank it around.
        explored_Q_sum: float = 0.0
        explored_N_sum: int = 0
        for c in self.children.values():
            if c.N > 0:
                explored_Q_sum += c.Q
                explored_N_sum += c.N
        mu_fpu: float = (explored_Q_sum / explored_N_sum) if explored_N_sum > 0 else 0.0

        # FPU-reduction (Leela / KataGo): subtract cFPU·sqrt(P_explored) from
        # the baseline so unvisited children are penalized when little policy
        # mass has been explored. Disabled at root if Dirichlet noise is
        # driving exploration -- otherwise the reduction would fight the noise.
        c_fpu: float = float(self.args.get("c_fpu", 0.0))
        if self.parent is None and float(self.args.get("dirichlet_epsilon", 0.0)) > 0:
            c_fpu = 0.0
        if c_fpu > 0.0 and self.children:
            p_explored: float = 0.0
            for a, c in self.children.items():
                if c.N > 0:
                    p_explored += float(self.policy[a])
            mu_fpu -= c_fpu * math.sqrt(p_explored)

        child_Q: np.ndarray = np.full(len(legal), mu_fpu, dtype=np.float64)
        child_N: np.ndarray = np.zeros(len(legal), dtype=np.int64)
        for i, a in enumerate(legal):
            child = self.children.get(int(a))
            if child is not None and child.N > 0:
                child_Q[i] = child.Q / child.N
                child_N[i] = child.N

        ucb: np.ndarray = (
            child_Q + c_puct * (priors ** (1.0 / t)) * sqrt_N / (1 + child_N)
        )
        best_local: int = int(np.argmax(ucb))
        return int(legal[best_local])

    def materialize_child(self, action: int) -> "Node":
        new_state = self.state.copy()
        move = f.alphazero_to_move(action, self.state)
        new_state.push_uci(move)
        new_state.apply_mirror()
        # Child's history = parent's recent history + parent's own state
        # (chronological, oldest first). Keep at most the last 7 entries --
        # 119-plane representation only uses 7 historical frames.
        new_history = self.history[-6:] + [self.state] if self.history else [self.state]
        child = Node(
            self.args, new_state, self.move_counter + 1,
            depth=self.depth + 1, parent=self, action=action,
            prior=float(self.policy[action]),
            history=new_history,
        )
        self.children[action] = child
        return child

    @torch.inference_mode()
    def expand_lazy(self, model: ResNet) -> float:
        """Evaluate this node with the network. Store masked priors. Return value.

        Single GPU→CPU sync: we concat value scalar + policy probs into one
        (4673,) tensor and pull it across the PCIe boundary in one go. Each
        `.cpu()` was previously ~5ms of host-stall waiting on the forward;
        halving the count was ~16% of wall-clock in batched-MCTS profile.
        n_legal comes from the numpy mask (no extra sync).
        """
        model.eval()
        # input_planes=119 -> pass history for 8-frame representation; otherwise
        # legacy 19-plane (current board only). Also fill the frame-0 repetition
        # planes from this node's rep_count (already maintained by MCTS).
        hist = self.history if self.args.get("input_planes") == 119 else None
        inputs = f.prepare_input(
            self.state, self.move_counter, history=hist,
            rep_count=max(self.rep_count, 1),
        ).unsqueeze(0).to(self.args["device"])
        if inputs.device.type == "cuda" and self.args.get("channels_last", True):
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with _amp_ctx(inputs.device):
            if self.args.get("moves_left_aux", False):
                value_t, policy_t, aux_t = model(inputs, return_aux=True)
            else:
                value_t, policy_t = model(inputs)
                aux_t = None

        legal_mask_np = f.legal_mask(self.state)
        self.n_legal = int(legal_mask_np.sum())
        self.legal_mask_np = legal_mask_np
        mask = torch.from_numpy(legal_mask_np).to(policy_t.device, non_blocking=True)

        # One softmax for both the collapsed scalar and the cached WDL probs.
        value_scalar, wdl = value_scalar_and_wdl(
            value_t, mode=self.args.get("value_scalar", "expected")
        )
        value_scalar = value_scalar.flatten().float()       # (1,) fp32 on GPU
        masked_logits = policy_t.squeeze(0).float().masked_fill(~mask, float("-inf"))
        policy_probs = torch.softmax(masked_logits, dim=0)   # (4672,) on GPU

        # Single GPU→CPU sync: concat value scalar (+ optional WDL probs) with
        # the policy probs and pull across the PCIe boundary once. .float()
        # guards against fp16 autocast outputs leaking into numpy storage.
        aux_parts = []
        if aux_t is not None:
            aux_parts = [
                aux_t["moves_left_mu"].flatten().float(),
                aux_t["moves_left_alpha"].flatten().float(),
            ]
        if wdl is not None:
            combined = torch.cat(
                [value_scalar, wdl.flatten().float(), *aux_parts, policy_probs]
            ).cpu().numpy()
            value = float(combined[0])
            self.raw_nn_wdl = (
                float(combined[1]), float(combined[2]), float(combined[3])
            )
            policy_offset = 4
            if aux_t is not None:
                self.raw_nn_moves_left = float(combined[4])
                self.raw_nn_moves_left_alpha = float(combined[5])
                policy_offset = 6
            self.raw_policy = combined[policy_offset:]
        else:
            combined = torch.cat([value_scalar, *aux_parts, policy_probs]).cpu().numpy()
            value = float(combined[0])
            policy_offset = 1
            if aux_t is not None:
                self.raw_nn_moves_left = float(combined[1])
                self.raw_nn_moves_left_alpha = float(combined[2])
                policy_offset = 3
            self.raw_policy = combined[policy_offset:]
        self.policy = self.raw_policy.copy()
        self.raw_nn_value = value
        return value


class MCTSBase:
    """Shared state + tree-reuse machinery for sequential and batched MCTS.

    Holds everything independent of HOW leaves are evaluated: external
    repetition / history injection, O(1) and state-based tree reuse, root
    Dirichlet noise, proof propagation, and final action-prob extraction.
    Subclasses add the per-engine `_simulate*` and `search` loop.
    """

    def __init__(self, args: dict, model) -> None:
        self.args = args
        self.model = model.to(args["device"])
        self.root: Node | None = None
        self.last_was_proven_mate: bool = False
        self.last_max_depth: int = 0
        # Per-search set of leaf depths reached; cleared at the start of `search`.
        self._visited_depths: set[int] = set()
        # Repetition tracking: external history counter (real-game positions
        # before MCTS root). Caller updates via set_rep_counter() before each
        # search. Keyed by transposition_key (tuple). Includes the root
        # position itself (so ext_rep[root.tk] >= 1 always).
        self._ext_rep: dict = {}
        # Board history (chronological list of prior canonical boards). Set
        # by callers via set_history() so a fresh root knows its 8-frame
        # history context for the 119-plane representation.
        self._ext_history: list = []
        # Early-stop sim bank: sims saved on "decided" positions, lent to harder
        # ones later in the game. Only used when args["early_stop"] is on (play
        # only -- never in self-play, where the full visit distribution is the
        # policy target). Reset per game by the caller (uci.cmd_ucinewgame).
        self.sim_bank: int = 0

    def set_rep_counter(self, counter) -> None:
        """Inject the game-level Counter[transposition_key] before search.
        Stored as a plain dict; we only read it. Callers should reset it on
        new games and bump it after each played move."""
        self._ext_rep = dict(counter) if counter else {}

    def set_history(self, history) -> None:
        """Inject the game-level history (chronological list of prior canonical
        boards, EXCLUDING the current root state) before each search. Used to
        seed `Node.history` when a fresh root has to be built. Tree-reused
        roots ignore this -- their history is already correctly inherited
        from their previous-life parents. Only meaningful when input_planes
        is 119; harmless when 19."""
        self._ext_history = list(history) if history else []

    def _compute_rep_count(self, node: "Node") -> int:
        """Total repetition count for `node`: real-game history (ext_rep) +
        tree path from root to node (excluding root, since root is already
        in ext_rep)."""
        tk = node.tk
        extra = 0
        cur = node
        # Walk up, counting matches of `tk` among non-root ancestors AND self.
        # Stop one short of root (cur.parent is None means cur is root).
        while cur.parent is not None:
            if cur.tk == tk:
                extra += 1
            cur = cur.parent
        return self._ext_rep.get(tk, 0) + extra

    def _refresh_rep_counts(self, root: "Node") -> None:
        """Recompute rep_count for the ENTIRE reused subtree against the current
        external counter. Stored rep_counts were computed relative to a PREVIOUS
        root + a stale ext_rep, so after a reroot they'd misfire the 3-fold /
        terminal check deep in the tree. One O(subtree) iterative DFS, once per
        search; equivalent to calling _compute_rep_count on every node but linear
        instead of O(N*depth). `path[tk]` = count of that key among non-root
        nodes on the current path (matching _compute_rep_count's convention)."""
        path: dict = {}
        stack = [(root, True, False)]   # (node, is_root, is_exit_marker)
        while stack:
            node, is_root, exiting = stack.pop()
            if exiting:
                path[node.tk] -= 1
                continue
            if not is_root:
                path[node.tk] = path.get(node.tk, 0) + 1
                node.rep_count = self._ext_rep.get(node.tk, 0) + path[node.tk]
                stack.append((node, False, True))   # exit marker to decrement on the way out
            else:
                node.rep_count = self._ext_rep.get(node.tk, 0)
            for c in node.children.values():
                stack.append((c, False, False))

    def apply_action(self, action: int) -> None:
        """O(1) tree walk by known action: set the action's child as new root,
        drop the rest. Use this from callers that *know* the action played
        (selfplay sampling its move, uci.py pushing a UCI move). Much faster
        than `update_root(state)` which has to scan + state-compare descendants.

        If `action` was never materialised in the tree (PUCT didn't visit it),
        we have no subtree to inherit -- reset to None so the next `search()`
        builds fresh. Same end behaviour as a `update_root` miss.
        """
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

        Depth-1 case: matches when `search()` is called every half-move (selfplay).
        Depth-2 case: matches when `search()` is called only on our turn after the
        opponent has replied (match.py / uci.py / lichess-bot). Without the
        depth-2 walk, the tree was discarded every move in those contexts --
        losing the hundreds of visits accumulated for likely opponent replies.

        If `move_counter` is given, also require it to match -- prevents reusing
        a subtree with stale counter that would feed the wrong move-counter
        plane into board_to_matrix.

        Idempotent: if root already matches `state`, no-op. This lets callers
        pre-walk via `apply_action` and still call `update_root` defensively
        without losing the pre-walked subtree.
        """
        if self.root is None:
            return
        # Already at the right state (e.g. caller called apply_action).
        if self.root.state == state and (
            move_counter is None or self.root.move_counter == move_counter
        ):
            return
        # Depth 1: opp's move directly produces this state (self-play loop).
        for child in self.root.children.values():
            if child.state == state and (
                move_counter is None or child.move_counter == move_counter
            ):
                self.root = child
                self.root.parent = None
                return
        # Depth 2: bot's move (last search picked one) + opp's reply produced
        # this state. The bot already searched some opp replies via PUCT --
        # if opp picked one of them, that grandchild has accumulated visits.
        for child in self.root.children.values():
            for grandchild in child.children.values():
                if grandchild.state == state and (
                    move_counter is None or grandchild.move_counter == move_counter
                ):
                    self.root = grandchild
                    self.root.parent = None
                    return
        self.root = None

    @staticmethod
    def _try_prove(node: Node) -> None:
        """If `node`'s status can be inferred from its expanded children, set
        node.proven_value (from THIS node's player-to-move perspective).

        OR-rule: any child where opp proves to lose (proven_value=-1) means
                 this player can force a win → node.proven_value = +1.
        Otherwise need all legal moves expanded and proven:
            node.proven_value = -min(child.proven_value over all children).
        """
        if node.proven_value is not None:
            return
        if not node.children:
            return
        # OR-rule first (cheap): any child where opp is proven losing.
        if any(c.proven_value == -1 for c in node.children.values()):
            node.proven_value = 1
            return
        # For -1 or 0 proofs we need all legal moves expanded and proven.
        if node.n_legal is None or len(node.children) < node.n_legal:
            return
        child_provens = [c.proven_value for c in node.children.values()]
        if any(pv is None for pv in child_provens):
            return
        node.proven_value = -min(child_provens)  # type: ignore

    def _apply_root_dirichlet(self) -> None:
        """Mix Dirichlet noise into the root's *raw* (un-noised) policy each
        call so tree reuse doesn't compound noise. Noise is sampled over LEGAL
        moves only -- sampling the full 4672 space wastes ~99% of the mass on
        illegal indices (already softmax(-inf)=0), cutting effective noise 100x.
        """
        eps = self.args["dirichlet_epsilon"]
        assert self.root is not None and self.root.raw_policy is not None
        if eps > 0:
            legal_idx = np.nonzero(self.root.raw_policy)[0]
            if len(legal_idx) > 0:
                noise = np.random.dirichlet([self.args["dirichlet_alpha"]] * len(legal_idx))
                mixed = self.root.raw_policy.copy()
                mixed[legal_idx] = (1 - eps) * mixed[legal_idx] + eps * noise
                total = mixed.sum()
                # Fall back to the (already-normalised) raw policy rather than
                # leaving an unnormalised distribution if total underflows to 0.
                self.root.policy = mixed / total if total > 0 else self.root.raw_policy.copy()
            else:
                self.root.policy = self.root.raw_policy.copy()
        else:
            # Reset to raw in case this node previously served as a noised root.
            self.root.policy = self.root.raw_policy.copy()

    def _early_stop_decided(self, completed: int, target: int) -> bool:
        """True if the played move (most-visited child) can no longer change
        within the remaining budget. The runner-up is the only move that could
        overtake #1; if even pouring ALL remaining sims into it can't catch up,
        the argmax is locked and we can stop. Also stops once the root is proven
        (the forced result is fixed). Exact -> never changes the chosen move."""
        root = self.root
        if root is None:
            return False
        if root.proven_value is not None:
            return True
        n1 = n2 = 0
        for c in root.children.values():
            if c.N >= n1:
                n2 = n1
                n1 = c.N
            elif c.N > n2:
                n2 = c.N
        return (n1 - n2) > (target - completed)

    @staticmethod
    def _mate_plies(node: Node) -> int | None:
        """Plies to checkmate from `node` along the proven line, or None if not
        a proven win/loss. The proven-mate subtree is fully materialised (that's
        how the proof was established), so this recursion stays in the tree and
        is bounded by the (short) mate depth.

        proven_value is from node's player-to-move POV:
          +1 win  -> deliver via the FASTEST mating child (opp proven losing),
          -1 loss -> opponent delays the LONGEST,
          terminal checkmate (proven -1, no children) -> 0 plies."""
        pv = node.proven_value
        if pv is None or pv == 0:
            return None
        if not node.children:
            return 0 if pv == -1 else None
        if pv == 1:
            ds = [MCTSBase._mate_plies(c) for c in node.children.values()
                  if c.proven_value == -1]
            ds = [d for d in ds if d is not None]
            return 1 + min(ds) if ds else None
        ds = [MCTSBase._mate_plies(c) for c in node.children.values()]
        ds = [d for d in ds if d is not None]
        return 1 + max(ds) if ds else None

    def _finalize_action_probs(self, verbose: bool = False) -> np.ndarray:
        """Extract the move distribution after search, honouring the solver:

        1. WIN: if any root child has proven_value == -1 (opp loses there),
           force-play the FASTEST mate -- the child with the shortest distance to
           checkmate (tie-break: highest Q/N). With discount==1 every forced win
           has Q ~= +1, so ranking by Q alone can't tell M1 from M9; rank by
           proof depth instead.
        2. AVOID MATE: otherwise, never play a move that is a PROVEN LOSS
           (child.proven_value == +1) while any non-losing move exists.
        3. LOST: if every move is a proven loss, force-play the LONGEST defence
           (deepest forced mate against us).
        Else: the normalised visit-count distribution over the eligible moves."""
        action_probs = np.zeros(self.args["action_space"])
        children = self.root.children

        # 1. Fastest proven mate.
        winners = [(a, c) for a, c in children.items()
                   if c.proven_value == -1 and c.N > 0]
        if winners:
            def _win_key(ac):
                d = self._mate_plies(ac[1])
                return (d if d is not None else 1 << 30, -(ac[1].Q / ac[1].N))
            best_action, best_child = min(winners, key=_win_key)
            if verbose:
                d = self._mate_plies(best_child)
                mtxt = f"M{(d + 1) // 2 + 1}" if d is not None else "M?"
                print(f"Proven win: force-playing fastest mate ({mtxt}) "
                      f"among {len(winners)} winning move(s)")
            action_probs[best_action] = 1.0
            self.last_was_proven_mate = True
            return action_probs

        self.last_was_proven_mate = False

        # 2/3. Avoid proven losses; if all moves lose, delay the longest.
        losing = {a for a, c in children.items() if c.proven_value == 1}
        non_losing = [a for a in children if a not in losing]
        if losing and not non_losing:
            best_action = max(children.items(),
                              key=lambda ac: (self._mate_plies(ac[1]) or 0))[0]
            if verbose:
                print(f"Proven loss in all {len(losing)} move(s); playing longest defence")
            action_probs[best_action] = 1.0
            return action_probs

        for a, c in children.items():
            if a not in losing:
                action_probs[a] = c.N
        total = action_probs.sum()
        if total <= 0:
            # Degenerate (e.g. eligible moves had 0 visits): fall back to raw
            # visit counts over all children so we always return a valid move.
            action_probs[:] = 0.0
            for a, c in children.items():
                action_probs[a] = c.N
            total = action_probs.sum()
        if total > 0:
            action_probs /= total
        return action_probs


class MCTS(MCTSBase):
    def _simulate(self, root: Node) -> None:
        trunc = int(self.args.get("truncation_halfmoves", 1000))
        path = [root]
        node = root
        while True:
            # Single game_result call (was previously is_terminal() + a second
            # game_result for the value). value is from node's player-to-move
            # perspective: -1 = current player mated, 0 = drawn / truncated.
            term_value, is_term = f.game_result(
                node.state, node.move_counter, trunc, node.rep_count
            )
            if is_term:
                value = float(term_value)
                node.proven_value = int(value)
                break
            if not node.is_expanded():
                value = node.expand_lazy(self.model)
                break
            action = node.select_action()
            if action in node.children:
                node = node.children[action]
            else:
                node = node.materialize_child(action)
                node.rep_count = self._compute_rep_count(node)
            path.append(node)

        self._visited_depths.add(node.depth)

        # Backprop. `value` is in leaf's perspective. We want each node's Q
        # in its PARENT's perspective, so flip sign as we attach to leaf and
        # alternate going up. Optional discount γ (args["discount"], default 1.0)
        # shrinks the leaf's contribution one extra factor per ply -- biases
        # the search toward shorter wins / longer losses.
        gamma = self.args.get("discount", 1.0)
        sign = -1.0
        for n in reversed(path):
            n.N += 1
            n.Q += sign * value
            sign = -sign * gamma

        # Solver bubble-up: try to mark nodes as proven based on their
        # expanded children. Cheap; no NN involvement.
        for n in reversed(path):
            self._try_prove(n)

    def search(
        self,
        state: chess.Board,
        move_counter: int,
        info_callback: Optional[Callable[["MCTS", int, float, int], None]] = None,
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
            # Reused subtree: rep_counts were computed in a PRIOR search's context
            # (old root + stale ext_rep). Refresh the WHOLE subtree so 3-fold /
            # terminal detection is correct everywhere, not just at the root.
            old_root_rep = self.root.rep_count
            self._refresh_rep_counts(self.root)
            # If the root's repetition planes changed (119-plane only), the cached
            # NN eval is stale -> refresh it.
            if self.args.get("input_planes") == 119 and self.root.rep_count != old_root_rep:
                self.root.expand_lazy(self.model)

        self._apply_root_dirichlet()

        self._visited_depths.clear()
        base = int(self.args["num_simulation"])
        es = bool(self.args.get("early_stop", False))
        min_sims = max(int(self.args.get("early_stop_min_sims", 1)), 1)
        max_borrow = int(self.args.get("max_borrow", 0))
        # Budget = base sims + whatever we can borrow from the bank (capped by
        # max_borrow for clock safety). Easy positions early-stop well short and
        # refund the unused sims to the bank; hard ones keep searching and spend
        # it -- the allocation is emergent, no "is this hard?" check needed.
        target = base + (min(self.sim_bank, max_borrow) if es else 0)
        completed = 0
        t_start = time.monotonic()
        t_last_report = t_start
        while completed < target:
            self._simulate(self.root)
            completed += 1
            # Forced mate FOR US proven -> stop instantly and play it. On for
            # play (uci/match), OFF in self-play via args["mate_stop"]=False
            # (self-play needs the full visit distribution as its policy target).
            # Independent of EarlyStop. Proven LOSS keeps searching -- only our
            # own mates trigger the instant play.
            if self.args.get("mate_stop", True) and self.root.proven_value == 1:
                break
            if es and completed >= min_sims and self._early_stop_decided(completed, target):
                break
            if info_callback is not None:
                now = time.monotonic()
                if now - t_last_report >= info_interval_s:
                    if self._visited_depths:
                        self.last_max_depth = max(self._visited_depths) - min_depth
                    try:
                        info_callback(self, completed, now - t_start, self.last_max_depth)
                    except Exception:
                        _log.exception("info_callback failed (mid-search); continuing")
                    t_last_report = now
        if es:
            # Bank unused sims (completed < base) or repay borrowed ones
            # (completed > base). Clamp at 0.
            self.sim_bank = max(0, self.sim_bank + base - completed)
        self.last_max_depth = (
            max(self._visited_depths) - min_depth if self._visited_depths else 0
        )
        if info_callback is not None:
            try:
                info_callback(self, completed, time.monotonic() - t_start, self.last_max_depth)
            except Exception:
                _log.exception("info_callback failed (final); continuing")

        return self._finalize_action_probs(verbose=True)
