"""Player abstractions for match.py.

A Player decides a move given the real board (game-perspective) and the
mirror-canonical state (always white-to-move) that the MCTS code uses.
It returns the UCI string in both coordinate systems so the game loop
can advance both boards consistently.

Player types
------------
random        no NN, no eval -- uniform random over legal moves
piece_value   no NN -- classical piece value sum, picks best
value_only    NN value head only (one-ply lookahead), no policy / MCTS
policy_only   NN policy head only (argmax over legal moves), no MCTS
mcts          full MCTS + policy + value
stockfish     external UCI engine

Each is constructed from a JSON config; see configs/ for templates.
"""
from __future__ import annotations

import json
import random
from typing import Protocol

import chess
import chess.engine
import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from . import utils as f
from .mcts import MCTS
from .nn import ResNet, SEResNet, SEResNetWDL, detect_in_channels, value_to_scalar


_ARCH_CLASSES = {
    "resnet":      ResNet,
    "seresnet":    SEResNet,
    "seresnetwdl": SEResNetWDL,
}


def build_model(cfg: dict):
    """Instantiate the NN architecture named in cfg['architecture'] (default
    'resnet'). Auto-detects in_channels from the checkpoint (legacy 19 vs
    history 119) so old SFT checkpoints still load."""
    name = cfg.get("architecture", "resnet").lower()
    if name not in _ARCH_CLASSES:
        raise ValueError(
            f"unknown architecture: {name!r}; expected 'resnet', 'seresnet', or 'seresnetwdl'"
        )
    in_ch = 19   # legacy default; overridden if a checkpoint is given
    ckpt_path = cfg.get("checkpoint")
    if ckpt_path:
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            in_ch = detect_in_channels(state)
        except Exception:
            pass
    cfg["_in_channels"] = in_ch                  # stash so MctsPlayer can read
    return _ARCH_CLASSES[name](in_channels=in_ch)


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
INF = 1e9

DEFAULT_MCTS_ARGS = {
    "num_simulation": 200,
    "truncation": 200,
    "c_base": 19652,
    "c_init": 1.25,
    "c_fpu": 0.2,  # Leela/KataGo FPU-reduction; 0.0 falls back to plain μ-FPU
    "dirichlet_epsilon": 0.0,
    "dirichlet_alpha": 0.3,
    "memory_size": 1000,
    "action_space": 4672,
    "t": 1,
}


class Player(Protocol):
    name: str

    def select_move(self, real_board: chess.Board, mirrored_state: chess.Board,
                    move_counter: int) -> tuple[str, str]:
        """Return (real_uci, mirrored_uci) for the chosen move."""
        ...

    def reset(self) -> None:
        """Clear per-game state. Called at the start of each game."""
        ...

    def close(self) -> None:
        """Release resources (subprocesses, GPU memory). Called at end of match."""
        ...


def _to_mirrored(real_uci: str, real_board_turn: chess.Color) -> str:
    """real_board.turn tells us who's MOVING; if black, we mirror."""
    if real_board_turn == chess.WHITE:
        return real_uci
    return f.mirror_move(real_uci)


def _to_real(mirrored_uci: str, real_board_turn: chess.Color) -> str:
    if real_board_turn == chess.WHITE:
        return mirrored_uci
    return f.mirror_move(mirrored_uci)


class RandomPlayer:
    name = "random"

    def __init__(self, cfg: dict):
        seed = cfg.get("seed")
        self.rng = random.Random(seed)

    def select_move(self, real_board, mirrored_state, move_counter):
        legal = list(real_board.legal_moves)
        move = self.rng.choice(legal)
        real_uci = move.uci()
        return real_uci, _to_mirrored(real_uci, real_board.turn)

    def reset(self): pass
    def close(self): pass


class PieceValuePlayer:
    """Pick the move that maximises (own_material - opp_material), with mate as inf."""
    name = "piece_value"

    def __init__(self, cfg: dict):
        pass

    @staticmethod
    def _material(board: chess.Board, color: chess.Color) -> float:
        s = 0.0
        for piece_type, value in PIECE_VALUES.items():
            s += value * len(board.pieces(piece_type, color))
        return s

    def _score_after_move(self, board: chess.Board, mover: chess.Color) -> float:
        # board state is post-move; check terminal first.
        if board.is_checkmate():
            # side to move (= opponent of mover) is mated -> mover won
            return INF
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        return self._material(board, mover) - self._material(board, not mover)

    def select_move(self, real_board, mirrored_state, move_counter):
        mover = real_board.turn
        best_move = None
        best_score = -INF - 1
        for move in real_board.legal_moves:
            real_board.push(move)
            score = self._score_after_move(real_board, mover)
            real_board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        real_uci = best_move.uci()
        return real_uci, _to_mirrored(real_uci, real_board.turn)

    def reset(self): pass
    def close(self): pass


class ValueOnlyPlayer:
    """Use the NN value head to score each legal move's resulting position.
    No policy, no MCTS -- one-ply lookahead by value only."""
    name = "value_only"

    def __init__(self, cfg: dict):
        if "checkpoint" not in cfg:
            raise ValueError("value_only config needs 'checkpoint'")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg).to(self.device)
        self.in_channels = int(cfg.get("_in_channels", 19))
        state = torch.load(cfg["checkpoint"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass
        # For game diversity in matches, the first `temperature_moves` plies are
        # sampled from softmax(value/T) over the legal-move values. After that,
        # pure argmax. Defaults: temperature_moves=0 -> always argmax.
        self.temperature_moves = int(cfg.get("temperature_moves", 0))
        self.temperature = float(cfg.get("temperature", 1.0))
        self._rng = np.random.default_rng(cfg.get("sampling_seed"))
        # WDL-only knob: how to collapse (P(W), P(D), P(L)) -> scalar.
        # "expected" = P(W) - P(L) (default). "win_only" = P(W) (ignores draw rate).
        self.value_scalar = cfg.get("value_scalar", "expected")

    @torch.inference_mode()
    def _batch_values(self, mirrored_states: list[chess.Board], move_counter: int) -> np.ndarray:
        # ValueOnlyPlayer doesn't track per-game history (stateless across
        # moves), so 119-plane models get zero-padded historical frames. Less
        # strong than MCTS path but works.
        hist = [] if self.in_channels == 119 else None
        inputs = torch.stack(
            [f.prepare_input(s, move_counter, history=hist) for s in mirrored_states]
        ).to(self.device)
        values, _ = self.model(inputs)
        return value_to_scalar(values, mode=self.value_scalar).cpu().numpy().flatten()

    def select_move(self, real_board, mirrored_state, move_counter):
        mover_was_white = real_board.turn == chess.WHITE
        legal = list(real_board.legal_moves)
        # Build the mirror-canonical post-move state for each legal move.
        post_states = []
        for move in legal:
            real_uci = move.uci()
            mir_uci = real_uci if mover_was_white else f.mirror_move(real_uci)
            post = mirrored_state.copy()
            post.push_uci(mir_uci)
            post.apply_mirror()
            post_states.append(post)

        # NN outputs each state's value from its player-to-move perspective,
        # which is the opponent of the mover. Negate to get the mover's value.
        opp_values = self._batch_values(post_states, move_counter + 1)
        mover_values = -opp_values

        # Override with +inf for moves that deliver checkmate.
        for i, ps in enumerate(post_states):
            if ps.is_checkmate():
                mover_values[i] = INF

        # If a mate-in-1 is available, take it deterministically -- temperature
        # shouldn't ever risk drawing/losing in front of a forced mate.
        if self.temperature > 0 and move_counter < self.temperature_moves \
                and not np.isinf(mover_values).any():
            scaled = mover_values / max(self.temperature, 1e-6)
            scaled = scaled - scaled.max()         # numerical stability
            probs = np.exp(scaled)
            probs = probs / probs.sum()
            best_idx = int(self._rng.choice(len(legal), p=probs))
        else:
            best_idx = int(np.argmax(mover_values))
        best_move = legal[best_idx]
        real_uci = best_move.uci()
        return real_uci, _to_mirrored(real_uci, real_board.turn)

    def reset(self): pass
    def close(self): pass


class PolicyOnlyPlayer:
    """Use the NN policy head only -- one forward pass per move, argmax over
    legal-move probabilities. No lookahead, no MCTS."""
    name = "policy_only"

    def __init__(self, cfg: dict):
        if "checkpoint" not in cfg:
            raise ValueError("policy_only config needs 'checkpoint'")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg).to(self.device)
        self.in_channels = int(cfg.get("_in_channels", 19))
        state = torch.load(cfg["checkpoint"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        try:
            self.model = torch.compile(self.model)
            with torch.inference_mode():
                _ = self.model(torch.zeros(1, self.in_channels, 8, 8, device=self.device))
        except Exception:
            pass
        # Temperature controls policy sampling for the first temperature_moves plies;
        # after that, pure argmax. Same semantics as ValueOnlyPlayer / MctsPlayer.
        self.temperature_moves = int(cfg.get("temperature_moves", 0))
        self.temperature = float(cfg.get("temperature", 1.0))
        self._rng = np.random.default_rng(cfg.get("sampling_seed"))

    @torch.inference_mode()
    def select_move(self, real_board, mirrored_state, move_counter):
        hist = [] if self.in_channels == 119 else None
        inputs = f.prepare_input(
            mirrored_state, move_counter, history=hist,
        ).unsqueeze(0).to(self.device)
        _value, policy_logits = self.model(inputs)
        mask = torch.from_numpy(f.legal_mask(mirrored_state)).to(self.device)
        masked_logits = policy_logits.squeeze(0).masked_fill(~mask, float("-inf"))
        policy = torch.softmax(masked_logits, dim=0).cpu().numpy()

        if self.temperature > 0 and move_counter < self.temperature_moves:
            scaled = np.where(policy > 0, policy ** (1.0 / self.temperature), 0.0)
            total = scaled.sum()
            if total > 0:
                action = int(self._rng.choice(len(scaled), p=scaled / total))
            else:
                action = int(np.argmax(policy))
        else:
            action = int(np.argmax(policy))
        mir_uci = f.alphazero_to_move(action, mirrored_state)
        return _to_real(mir_uci, real_board.turn), mir_uci

    def reset(self): pass
    def close(self): pass


class MctsPlayer:
    """MCTS-driven player.

    For game diversity in matches, the first `temperature_moves` plies of
    each game are sampled from the MCTS visit-count distribution with the
    given `temperature` (visits ** (1/T), renormalised). After that, the
    move is chosen by argmax for strongest play.

    Defaults: temperature_moves=0 (always argmax). Set temperature_moves=20
    + temperature=1.0 for AlphaZero-style opening sampling.
    """
    name = "mcts"

    def __init__(self, cfg: dict):
        if "checkpoint" not in cfg:
            raise ValueError("mcts config needs 'checkpoint'")
        args = dict(DEFAULT_MCTS_ARGS)
        args.update(cfg)
        args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(cfg).to(args["device"])
        in_ch = int(cfg.get("_in_channels", 19))
        args["input_planes"] = in_ch                # MCTS routes board_to_matrix accordingly
        state = torch.load(args["checkpoint"], map_location=args["device"], weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        # Optional optimizations. compile is on by default; batched is opt-in.
        use_compile = bool(cfg.get("compile", True))
        use_batched = bool(cfg.get("batched", False))
        batch_size = int(cfg.get("batch_size", 8))

        if use_compile:
            try:
                self.model = torch.compile(self.model)
                warm_bs = batch_size if use_batched else 1
                # Warm with a real starting position, not torch.zeros: the JIT
                # trace specializes on input content too (BN/SE paths behave
                # differently on all-zero input), so a zeros-warmup leaves the
                # first REAL forward to pay a ~6s re-trace cost on move 1.
                real_one = f.prepare_input(
                    chess.Board(), 0,
                    history=([] if in_ch == 119 else None),
                ).unsqueeze(0).to(args["device"])
                with torch.inference_mode():
                    if warm_bs > 1:
                        _ = self.model(real_one.expand(warm_bs, -1, -1, -1).contiguous())
                    _ = self.model(real_one)
            except Exception:
                pass

        if use_batched:
            from .batched_mcts import BatchedMCTS
            args["batch_size"] = batch_size
            self.mcts = BatchedMCTS(args, self.model)
        else:
            self.mcts = MCTS(args, self.model)
        self.args = args

        # MCTS pipeline pre-warm. The model warm-up above covers the NN graph,
        # but the search loop has extra lazy CUDA/CPU paths (masked_fill+softmax
        # with -inf, torch.cat + .cpu().numpy(), legal_mask numpy interop) that
        # only fire on the FIRST search call -- adding ~6s of unaccounted time
        # to move 1 otherwise. A 4-sim throwaway absorbs that cost into startup.
        try:
            orig_sims = args["num_simulation"]
            args["num_simulation"] = 4
            self.mcts.search(chess.Board(), 0)
            args["num_simulation"] = orig_sims
            self.mcts.root = None
        except Exception:
            pass

        self.temperature_moves = int(cfg.get("temperature_moves", 0))
        self.temperature = float(cfg.get("temperature", 1.0))
        seed = cfg.get("sampling_seed")
        self._rng = np.random.default_rng(seed)

    def _sample_action(self, probs: np.ndarray) -> int:
        if self.temperature <= 0:
            return int(np.argmax(probs))
        scaled = np.where(probs > 0, probs ** (1.0 / self.temperature), 0.0)
        total = scaled.sum()
        if total <= 0:
            return int(np.argmax(probs))
        scaled = scaled / total
        return int(self._rng.choice(len(scaled), p=scaled))

    def select_move(self, real_board, mirrored_state, move_counter):
        probs = self.mcts.search(mirrored_state, move_counter)
        depth = getattr(self.mcts, "last_max_depth", 0)
        name = getattr(self, "display_name", self.name)
        print(f"  [{name}] max_depth {depth}", flush=True)
        if move_counter < self.temperature_moves:
            action = self._sample_action(probs)
        else:
            action = int(probs.argmax())
        mir_uci = f.alphazero_to_move(action, mirrored_state)
        return _to_real(mir_uci, real_board.turn), mir_uci

    def reset(self):
        self.mcts.root = None

    def close(self): pass


class StockfishPlayer:
    name = "stockfish"

    def __init__(self, cfg: dict):
        binary = cfg.get("binary", "stockfish")
        self.engine = chess.engine.SimpleEngine.popen_uci(binary)
        uci_options = {}
        if cfg.get("skill_level") is not None:
            uci_options["Skill Level"] = int(cfg["skill_level"])
        if cfg.get("limit_strength") and cfg.get("elo") is not None:
            uci_options["UCI_LimitStrength"] = True
            uci_options["UCI_Elo"] = int(cfg["elo"])
        if uci_options:
            self.engine.configure(uci_options)

        if cfg.get("depth") is not None:
            self.limit = chess.engine.Limit(depth=int(cfg["depth"]))
        elif cfg.get("time_ms") is not None:
            self.limit = chess.engine.Limit(time=float(cfg["time_ms"]) / 1000.0)
        else:
            self.limit = chess.engine.Limit(depth=10)

    def select_move(self, real_board, mirrored_state, move_counter):
        result = self.engine.play(real_board, self.limit)
        real_uci = result.move.uci()
        return real_uci, _to_mirrored(real_uci, real_board.turn)

    def reset(self): pass

    def close(self):
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass


PLAYER_TYPES = {
    "random": RandomPlayer,
    "piece_value": PieceValuePlayer,
    "value_only": ValueOnlyPlayer,
    "policy_only": PolicyOnlyPlayer,
    "mcts": MctsPlayer,
    "stockfish": StockfishPlayer,
}


def load_player(config_path: str) -> Player:
    with open(config_path) as fh:
        cfg = json.load(fh)
    ptype = cfg.get("type")
    if ptype not in PLAYER_TYPES:
        raise ValueError(f"{config_path}: 'type' must be one of {sorted(PLAYER_TYPES)}, got {ptype!r}")
    return PLAYER_TYPES[ptype](cfg)
