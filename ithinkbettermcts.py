import math
import optimized_functions as gf
import numpy as np
import torch
import chess
from resnet import ResNet

global max_depth
max_depth = set()


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

        self.policy: np.ndarray | None = None
        self.children: dict[int, "Node"] = {}
        self.Q = 0.0
        self.N = 0

    def is_terminal(self) -> bool:
        return gf.game_result(self.state, self.move_counter, 1000)[1]

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
        move = gf.alphazero_to_move(action, self.state)
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
        inputs = gf.prepare_input(self.state, self.move_counter).unsqueeze(0).to(self.args["device"])
        value_t, policy_t = model(inputs)
        value = float(value_t.cpu().item())
        policy = torch.softmax(policy_t.squeeze(0), dim=0).cpu().numpy()
        self.policy = gf.valid_policy(policy, self.state)
        return value


class MCTS:
    def __init__(self, args: dict, model: ResNet) -> None:
        self.args = args
        self.model = model.to(args["device"])
        self.root: Node | None = None

    def update_root(self, state: chess.Board) -> None:
        """Walk to the child whose position matches `state`. Reset if no match."""
        if self.root is None:
            return
        for child in self.root.children.values():
            if child.state == state:
                self.root = child
                self.root.parent = None
                return
        self.root = None

    def _simulate(self, root: Node) -> None:
        path = [root]
        node = root
        while True:
            if node.is_terminal():
                # value from node's player-to-move perspective (-1 = mated)
                value = float(gf.game_result(node.state, node.move_counter, 1000)[0])
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

        max_depth.add(node.depth)

        # Backprop. `value` is in leaf's perspective. We want each node's Q
        # in its PARENT's perspective, so flip sign as we attach to leaf and
        # alternate going up.
        sign = -1.0
        for n in reversed(path):
            n.N += 1
            n.Q += sign * value
            sign = -sign

    def search(self, state: chess.Board, move_counter: int) -> np.ndarray:
        root_state = state.copy()
        if self.root is not None:
            self.update_root(state)

        if self.root is None:
            self.root = Node(self.args, state, move_counter)
            self.root.expand_lazy(self.model)
            min_depth = 0
        else:
            min_depth = self.root.depth

        eps = self.args["dirichlet_epsilon"]
        if eps > 0:
            noise = np.random.dirichlet([self.args["dirichlet_alpha"]] * self.args["action_space"])
            mixed = (1 - eps) * self.root.policy + eps * noise
            self.root.policy = gf.valid_policy(mixed, root_state)

        max_depth.clear()
        for _ in range(self.args["num_simulation"]):
            self._simulate(self.root)
        if max_depth:
            print("max_depth", max(max_depth) - min_depth)

        action_probs = np.zeros(self.args["action_space"])
        for action, child in self.root.children.items():
            action_probs[action] = child.N
        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        return action_probs
