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
from .nn import ResNet, SEResNet, SEResNetWDL, value_to_scalar


def build_model(cfg: dict):
    """Instantiate the NN architecture named in cfg['architecture'] (default 'resnet')."""
    name = cfg.get("architecture", "resnet").lower()
    if name == "resnet":
        return ResNet()
    if name == "seresnet":
        return SEResNet()
    if name == "seresnetwdl":
        return SEResNetWDL()
    raise ValueError(
        f"unknown architecture: {name!r}; expected 'resnet', 'seresnet', or 'seresnetwdl'"
    )


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

    @torch.no_grad()
    def _batch_values(self, mirrored_states: list[chess.Board], move_counter: int) -> np.ndarray:
        inputs = torch.stack(
            [f.prepare_input(s, move_counter) for s in mirrored_states]
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
    legal-move probabilities. No lookahead, no MCTS.

    Two backends:
      backend="torch" (default) -- FP16 + channels_last + torch.compile
        (mode='max-autotune') over model.forward_policy.
      backend="onnx" -- loads an ONNX file exported by export_onnx.py and
        runs it through ONNX Runtime's CUDAExecutionProvider. Typically
        ~15-30% faster than the torch backend at batch=1.

    Sampling/argmax math is identical between backends.
    """
    name = "policy_only"

    def __init__(self, cfg: dict):
        if "checkpoint" not in cfg and cfg.get("backend", "torch") != "onnx":
            raise ValueError("policy_only config needs 'checkpoint'")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = cfg.get("backend", "torch").lower()

        if self.backend == "onnx":
            self._init_onnx(cfg)
        else:
            self._init_torch(cfg)

        # Temperature controls policy sampling for the first temperature_moves plies;
        # after that, pure argmax. Same semantics as ValueOnlyPlayer / MctsPlayer.
        self.temperature_moves = int(cfg.get("temperature_moves", 0))
        self.temperature = float(cfg.get("temperature", 1.0))
        self._rng = np.random.default_rng(cfg.get("sampling_seed"))

    def _init_torch(self, cfg: dict) -> None:
        self.model = build_model(cfg).to(self.device)
        state = torch.load(cfg["checkpoint"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        self._use_half = self.device.type == "cuda"
        if self._use_half:
            self.model = self.model.half().to(memory_format=torch.channels_last)
        try:
            self._policy_fn = torch.compile(self.model.forward_policy, mode="max-autotune")
            warm = torch.zeros(
                1, 19, 8, 8, device=self.device,
                dtype=torch.float16 if self._use_half else torch.float32,
            )
            if self._use_half:
                warm = warm.to(memory_format=torch.channels_last)
            with torch.inference_mode():
                _ = self._policy_fn(warm)
        except Exception:
            self._policy_fn = self.model.forward_policy

    def _init_onnx(self, cfg: dict) -> None:
        from .onnx_io import OnnxPolicyRunner
        onnx_path = cfg.get("onnx_path")
        if not onnx_path:
            raise ValueError("policy_only backend='onnx' needs 'onnx_path' in config")
        fp16 = bool(cfg.get("onnx_fp16", True))
        self._ort = OnnxPolicyRunner(
            onnx_path,
            device="cuda" if self.device.type == "cuda" else "cpu",
            fp16=fp16,
        )

    def _policy_logits(self, mirrored_state, move_counter) -> np.ndarray:
        """Run the forward and return a (4672,) numpy logits array."""
        if self.backend == "onnx":
            planes = f.board_to_matrix(mirrored_state, move_counter)  # (19,8,8) float32
            return self._ort(planes)
        with torch.inference_mode():
            inputs = f.prepare_input(mirrored_state, move_counter).unsqueeze(0).to(self.device)
            if self._use_half:
                inputs = inputs.half().to(memory_format=torch.channels_last)
            logits = self._policy_fn(inputs).squeeze(0)
            return logits.float().cpu().numpy()

    def select_move(self, real_board, mirrored_state, move_counter):
        policy_logits = self._policy_logits(mirrored_state, move_counter)
        # Gather only the legal-action logits (~30 entries) instead of softmaxing
        # over the full 4672-wide vector.
        legal_idx = np.fromiter(
            (f.move_obj_to_alphazero(m) for m in mirrored_state.legal_moves),
            dtype=np.int64,
        )
        legal_logits = policy_logits[legal_idx]

        if self.temperature > 0 and move_counter < self.temperature_moves:
            shifted = legal_logits - legal_logits.max()
            legal_probs = np.exp(shifted, dtype=np.float64)
            legal_probs /= legal_probs.sum()
            scaled = legal_probs ** (1.0 / self.temperature)
            total = scaled.sum()
            if total > 0:
                pos = int(self._rng.choice(len(scaled), p=scaled / total))
            else:
                pos = int(np.argmax(legal_probs))
        else:
            pos = int(np.argmax(legal_logits))
        action = int(legal_idx[pos])
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
                with torch.no_grad():
                    _ = self.model(torch.zeros(warm_bs, 19, 8, 8, device=args["device"]))
                    if use_batched and warm_bs != 1:
                        _ = self.model(torch.zeros(1, 19, 8, 8, device=args["device"]))
            except Exception:
                pass

        if use_batched:
            from .batched_mcts import BatchedMCTS
            args["batch_size"] = batch_size
            self.mcts = BatchedMCTS(args, self.model)
        else:
            self.mcts = MCTS(args, self.model)
        self.args = args

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
