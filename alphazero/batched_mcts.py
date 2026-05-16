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

    def update_root(self, state: chess.Board, move_counter: int | None = None) -> None:
        """Walk to the child whose position matches `state`. Reset if no match.
        Optional move_counter check prevents reusing subtrees with stale counters."""
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

    # ---------- selection with virtual loss ----------

    @staticmethod
    def _select_action_with_vloss(node: Node) -> int:
        """PUCT with LC0-style 'unscored virtual visit' in-flight accounting.

        Differs from classic AlphaZero virtual loss (which treated in-flight
        sims as fake -1 outcomes, distorting Q during search): we leave Q
        untouched and only inflate the U-term denominator via N + vloss.
        This still deters duplicate paths (via the 1/(1+N+vloss) factor)
        without biasing the value estimate.

        Parent side uses real N (matches sequential PUCT)."""
        c_base = node.args["c_base"]
        c_init = node.args["c_init"]
        t = node.args["t"]
        c_puct = math.log((1 + node.N + c_base) / c_base) + c_init
        sqrt_parent_N = math.sqrt(max(node.N, 1))

        assert node.policy is not None
        legal = np.nonzero(node.policy)[0]
        priors = node.policy[legal]

        child_Q = np.zeros(len(legal), dtype=np.float64)
        # `child_N_started` = real visits + in-flight (vloss). Used in U denom only.
        child_N_started = np.zeros(len(legal), dtype=np.int64)
        for i, a in enumerate(legal):
            child = node.children.get(int(a))
            if child is None:
                continue
            if child.N > 0:
                child_Q[i] = child.Q / child.N  # unbiased: only real visits
            child_N_started[i] = child.N + child.virtual_loss

        ucb = child_Q + c_puct * (priors ** (1.0 / t)) * sqrt_parent_N / (1 + child_N_started)
        best_local = int(np.argmax(ucb))
        return int(legal[best_local])

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

    def _select_one_leaf(self, root: Node) -> tuple[list[Node], Node, str, float | None]:
        """Walk from root to a leaf, applying virtual loss to each node on
        the way down. Returns (path, leaf, status, terminal_value-or-None)."""
        path = [root]
        node = root
        # Apply virtual loss to root (also counts as in-flight through root).
        node.virtual_loss += 1
        while True:
            if node.is_terminal():
                value = float(f.game_result(node.state, node.move_counter, 1000)[0])
                # value is from current player's POV: -1 mated, 0 drawn.
                if value == 0.0 or value == -1.0:
                    node.proven_value = int(value)
                return path, node, "terminal", value
            if not node.is_expanded():
                return path, node, "needs_eval", None
            action = self._select_action_with_vloss(node)
            if action in node.children:
                node = node.children[action]
            else:
                node = node.materialize_child(action)
            node.virtual_loss += 1
            path.append(node)

    def _simulate_batch(self, root: Node, sims_remaining: int) -> int:
        """Run up to `min(batch_size, sims_remaining)` parallel sims.
        Returns the number of sims that actually contributed (duplicates
        whose leaf is already in the batch are skipped to avoid double-
        backprop of the same value)."""
        batch_target = min(self.batch_size, sims_remaining)

        # Phase 1: selection. Each entry is (path, leaf, status, value-or-None).
        in_flight: list[tuple[list[Node], Node, str, float | None]] = []
        for _ in range(batch_target):
            in_flight.append(self._select_one_leaf(root))

        # Deduplicate needs_eval leaves: if two sims hit the same unexpanded
        # leaf, only the first one is evaluated/backpropagated. The duplicate
        # has its virtual losses undone and is skipped. This matches sequential
        # MCTS semantics where after expanding a leaf, subsequent sims descend
        # deeper rather than re-evaluating the same position.
        seen_leaves: set[int] = set()
        skip: set[int] = set()
        unique_eval: list[tuple[int, Node]] = []
        for sim_idx, (path, leaf, status, _) in enumerate(in_flight):
            if status == "needs_eval":
                lid = id(leaf)
                if lid in seen_leaves:
                    skip.add(sim_idx)
                    continue
                seen_leaves.add(lid)
                unique_eval.append((sim_idx, leaf))

        # Phase 2: batched NN evaluation for unique needs_eval leaves.
        if unique_eval:
            inputs = torch.stack([
                f.prepare_input(leaf.state, leaf.move_counter)
                for _, leaf in unique_eval
            ]).to(self.args["device"])
            with torch.no_grad():
                value_t, policy_t = self.model(inputs)
            values = value_to_scalar(
                value_t, mode=self.args.get("value_scalar", "expected")
            ).cpu().numpy().flatten()
            masks_np = np.stack([f.legal_mask(leaf.state) for _, leaf in unique_eval])
            masks_t = torch.from_numpy(masks_np).to(policy_t.device)
            masked_logits = policy_t.masked_fill(~masks_t, float("-inf"))
            policies = torch.softmax(masked_logits, dim=1).cpu().numpy()

            for (sim_idx, leaf), val, pol, mask_row in zip(unique_eval, values, policies, masks_np):
                if leaf.raw_policy is None:
                    leaf.raw_policy = pol
                    leaf.policy = pol.copy()
                    leaf.n_legal = int(mask_row.sum())
                path, _, _, _ = in_flight[sim_idx]
                in_flight[sim_idx] = (path, leaf, "evaluated", float(val))

        # Phase 3: backprop + virtual loss removal + proven-value propagation.
        gamma = self.args.get("discount", 1.0)
        effective = 0
        for sim_idx, (path, leaf, status, value) in enumerate(in_flight):
            if sim_idx in skip or value is None:
                # Duplicate (or unexpected): undo virtual losses, don't backprop.
                for n in path:
                    if n.virtual_loss > 0:
                        n.virtual_loss -= 1
                continue
            sign = -1.0
            for n in reversed(path):
                n.N += 1
                n.Q += sign * value
                if n.virtual_loss > 0:
                    n.virtual_loss -= 1
                sign = -sign * gamma
            self._visited_depths.add(leaf.depth)
            for n in reversed(path):
                self._try_prove(n)
            effective += 1

        return effective

    # ---------- top-level search ----------

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
        while completed < total_sims:
            completed += self._simulate_batch(self.root, total_sims - completed)
        self.last_max_depth = (
            max(self._visited_depths) - min_depth if self._visited_depths else 0
        )

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
