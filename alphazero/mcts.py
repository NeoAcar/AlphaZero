import math

import chess
import numpy as np
import torch

from . import utils as f
from .nn import ResNet


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
        # Virtual loss counter for batched MCTS (number of in-flight sims
        # that have passed through this node). Always 0 in sequential MCTS.
        self.virtual_loss: int = 0

    def is_terminal(self) -> bool:
        return f.game_result(self.state, self.move_counter, 1000)[1]

    def is_expanded(self) -> bool:
        return self.policy is not None

    def select_action(self) -> int:
        """PUCT over all legal actions, treating unvisited as N=0, Q=0 (FPU=0)."""
        c_base = self.args["c_base"]
        c_init = self.args["c_init"]
        t = self.args["t"]
        c_puct = math.log((1 + self.N + c_base) / c_base) + c_init
        sqrt_N = math.sqrt(max(self.N, 1))

        legal = np.nonzero(self.policy)[0]
        priors = self.policy[legal]

        child_Q = np.zeros(len(legal), dtype=np.float64)
        child_N = np.zeros(len(legal), dtype=np.int64)
        for i, a in enumerate(legal):
            child = self.children.get(int(a))
            if child is not None and child.N > 0:
                child_Q[i] = child.Q / child.N
                child_N[i] = child.N

        ucb = child_Q + c_puct * (priors ** (1.0 / t)) * sqrt_N / (1 + child_N)
        best_local = int(np.argmax(ucb))
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

    @torch.no_grad()
    def expand_lazy(self, model: ResNet) -> float:
        """Evaluate this node with the network. Store masked priors. Return value."""
        model.eval()
        inputs = f.prepare_input(self.state, self.move_counter).unsqueeze(0).to(self.args["device"])
        value_t, policy_t = model(inputs)
        value = float(value_t.cpu().item())
        mask = torch.from_numpy(f.legal_mask(self.state)).to(policy_t.device)
        masked_logits = policy_t.squeeze(0).masked_fill(~mask, float("-inf"))
        self.raw_policy = torch.softmax(masked_logits, dim=0).cpu().numpy()
        self.policy = self.raw_policy.copy()
        self.n_legal = int(mask.sum().item())
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

    def update_root(self, state: chess.Board, move_counter: int | None = None) -> None:
        """Walk to the child whose position matches `state`. Reset if no match.
        If `move_counter` is given, also require the child's stored move_counter
        to match -- prevents reusing a subtree with stale move_counter, which
        would feed the wrong move_counter / 500 plane into board_to_matrix."""
        if self.root is None:
            return
        for child in self.root.children.values():
            if child.state == state and (
                move_counter is None or child.move_counter == move_counter
            ):
                self.root = child
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
                value = float(f.game_result(node.state, node.move_counter, 1000)[0])
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

    def search(self, state: chess.Board, move_counter: int) -> np.ndarray:
        root_state = state.copy()
        if self.root is not None:
            self.update_root(state, move_counter)

        if self.root is None:
            self.root = Node(self.args, state, move_counter)
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
        for _ in range(self.args["num_simulation"]):
            self._simulate(self.root)
        self.last_max_depth = (
            max(self._visited_depths) - min_depth if self._visited_depths else 0
        )

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
