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
mcts          full MCTS + policy + value (ithinkbettermcts)
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

import optimized_functions as f
from ithinkbettermcts import MCTS
from resnet import ResNet


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
INF = 1e9


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
        self.model = ResNet().to(self.device)
        state = torch.load(cfg["checkpoint"], map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def _batch_values(self, mirrored_states: list[chess.Board], move_counter: int) -> np.ndarray:
        inputs = torch.stack(
            [f.prepare_input(s, move_counter) for s in mirrored_states]
        ).to(self.device)
        values, _ = self.model(inputs)
        return values.cpu().numpy().flatten()

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

        best_idx = int(np.argmax(mover_values))
        best_move = legal[best_idx]
        real_uci = best_move.uci()
        return real_uci, _to_mirrored(real_uci, real_board.turn)

    def reset(self): pass
    def close(self): pass


class MctsPlayer:
    name = "mcts"

    def __init__(self, cfg: dict):
        from match import DEFAULT_MCTS_ARGS  # reuse the same defaults
        if "checkpoint" not in cfg:
            raise ValueError("mcts config needs 'checkpoint'")
        args = dict(DEFAULT_MCTS_ARGS)
        args.update(cfg)
        args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResNet().to(args["device"])
        state = torch.load(args["checkpoint"], map_location=args["device"], weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self.mcts = MCTS(args, self.model)
        self.args = args

    def select_move(self, real_board, mirrored_state, move_counter):
        probs = self.mcts.search(mirrored_state, move_counter)
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
