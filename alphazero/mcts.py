import math
import time
from typing import Callable, Optional

import chess
import numpy as np
import torch

from . import utils as f
from .nn import ResNet, value_to_scalar


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
                 parent=None, action=None, prior=None):
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.depth = depth
        self.move_counter = move_counter

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

    def is_terminal(self) -> bool:
        return f.game_result(self.state, self.move_counter, 1000, self.rep_count)[1]

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
        t: float = self.args["t"]
        c_puct: float = math.log((1 + self.N + c_base) / c_base) + c_init
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
        child = Node(
            self.args, new_state, self.move_counter + 1,
            depth=self.depth + 1, parent=self, action=action,
            prior=float(self.policy[action]),
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
        inputs = f.prepare_input(self.state, self.move_counter).unsqueeze(0).to(self.args["device"])
        value_t, policy_t = model(inputs)

        legal_mask_np = f.legal_mask(self.state)
        self.n_legal = int(legal_mask_np.sum())
        mask = torch.from_numpy(legal_mask_np).to(policy_t.device)

        value_scalar = value_to_scalar(
            value_t, mode=self.args.get("value_scalar", "expected")
        ).flatten()                                         # (1,) on GPU
        masked_logits = policy_t.squeeze(0).masked_fill(~mask, float("-inf"))
        policy_probs = torch.softmax(masked_logits, dim=0)  # (4672,) on GPU

        combined = torch.cat([value_scalar, policy_probs]).cpu().numpy()
        value = float(combined[0])
        self.raw_policy = combined[1:]
        self.policy = self.raw_policy.copy()
        self.raw_nn_value = value
        return value


class MCTS:
    def __init__(self, args: dict, model: ResNet) -> None:
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

    def set_rep_counter(self, counter) -> None:
        """Inject the game-level Counter[transposition_key] before search.
        Stored as a plain dict; we only read it. Callers should reset it on
        new games and bump it after each played move."""
        self._ext_rep = dict(counter) if counter else {}

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
        a subtree with stale counter that would feed the wrong move_counter/300
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

    def _simulate(self, root: Node) -> None:
        path = [root]
        node = root
        while True:
            if node.is_terminal():
                # value is already from node's player-to-move perspective:
                # -1 = current player is mated, 0 = drawn (stalemate / etc.)
                value = float(f.game_result(node.state, node.move_counter, 1000, node.rep_count)[0])
                if value == 0.0 or value == -1.0:
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

    def _try_prove(self, node: Node) -> None:
        """If `node`'s status can be inferred from its expanded children,
        set node.proven_value.

        Convention: proven_value is from THIS node's player-to-move perspective.
        node's player picks a move, then opp moves at the resulting child.
        node's value from playing move m = -child_m.proven_value
        (child_m.proven_value is from opp's perspective).

        OR-rule: any child where opp proves to lose (proven_value=-1) means
                 this player can force a win → node.proven_value = +1.
        Otherwise need all legal moves expanded to claim 0 or -1:
            node.proven_value = -min(child.proven_value over all children)
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

        # All expanded and proven. value = max(-child.proven_value) = -min(child.proven_value)
        node.proven_value = -min(child_provens)

    def search(
        self,
        state: chess.Board,
        move_counter: int,
        info_callback: Optional[Callable[["MCTS", int, float, int], None]] = None,
        info_interval_s: float = 0.2,
    ) -> np.ndarray:
        root_state = state.copy()
        if self.root is not None:
            self.update_root(state, move_counter)

        if self.root is None:
            self.root = Node(self.args, state, move_counter)
            self.root.rep_count = self._compute_rep_count(self.root)
            self.root.expand_lazy(self.model)
            min_depth = 0
        else:
            min_depth = self.root.depth

        # Dirichlet noise at the root, mixed into the *raw* (un-noised) policy
        # each call so tree reuse doesn't compound noise. Critically, we sample
        # noise over LEGAL moves only -- sampling over the full 4672 action
        # space wastes ~99% of the noise mass on illegal indices (which are
        # already softmax(-inf) = 0), reducing effective noise by 100x.
        eps = self.args["dirichlet_epsilon"]
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
            # Reset policy to raw in case this node previously served as root
            # with noise applied.
            self.root.policy = self.root.raw_policy.copy()

        self._visited_depths.clear()
        total_sims = int(self.args["num_simulation"])
        t_start = time.monotonic()
        t_last_report = t_start
        for completed in range(1, total_sims + 1):
            self._simulate(self.root)
            if info_callback is not None:
                now = time.monotonic()
                if now - t_last_report >= info_interval_s:
                    if self._visited_depths:
                        self.last_max_depth = max(self._visited_depths) - min_depth
                    try:
                        info_callback(self, completed, now - t_start, self.last_max_depth)
                    except Exception:
                        pass
                    t_last_report = now
        self.last_max_depth = (
            max(self._visited_depths) - min_depth if self._visited_depths else 0
        )
        if info_callback is not None:
            try:
                info_callback(self, total_sims, time.monotonic() - t_start, self.last_max_depth)
            except Exception:
                pass

        # If the solver has proven that bot wins, play it.
        # Bot wins by playing into a child whose player (opp) is proven losing,
        # i.e., child.proven_value == -1 (perspective: child's player = opp).
        # child.Q is stored from PARENT's (= bot's) POV unchanged; among proven
        # winners pick the highest Q/N (most confident win, shortest mate under
        # depth discount).
        action_probs = np.zeros(self.args["action_space"])
        proven_winners = [
            (action, child) for action, child in self.root.children.items()
            if child.proven_value == -1 and child.N > 0
        ]
        if proven_winners:
            print(f"Proven win found among {len(proven_winners)} children; picking best Q/N")
            best_action, _ = max(proven_winners, key=lambda ac: ac[1].Q / ac[1].N)
            action_probs[best_action] = 1.0
            self.last_was_proven_mate = True
            return action_probs

        self.last_was_proven_mate = False
        for action, child in self.root.children.items():
            action_probs[action] = child.N
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        return action_probs
