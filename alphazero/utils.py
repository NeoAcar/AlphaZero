import math

import chess
import numpy as np
import torch


def board_to_matrix(board: chess.Board, move_counter: int,
                    history: list | None = None,
                    rep_count: int = 1) -> np.ndarray:
    """Input planes.

    `history`:
      - None → legacy 19-plane representation (current board only).
      - list of chess.Board → 119-plane AlphaZero representation. The list
        should be chronological (oldest first), holding the prior canonical
        boards (NOT including the current `board`). Up to 7 entries are
        consumed; earlier ones are ignored. Missing slots are zero-padded.

    `rep_count`: how many times the CURRENT position has appeared in the
    real-game history so far (including this occurrence). Used to fill the
    frame-0 repetition planes per AZ S1: plane 12 lit if rep_count >= 2
    (seen at least once before), plane 13 lit if rep_count >= 3 (3-fold
    draw imminent). For historical frames we leave the rep planes zero --
    tracking per-frame historical rep counts would require a parallel
    history of rep_counts which we don't maintain. Frame-0 carries ~80% of
    the signal anyway.

    The 119-plane layout matches AlphaZero (Silver et al. 2018 chess Table S1):
      - 8 time-step frames × 14 planes (12 pieces + 2 repetition flags).
      - Frame 0 is the current board. Frame -1 is the previous ply, etc.
      - Historical frames are rotated to the CURRENT player's perspective.
        Since each `mirror_state` ply alternates canonical frame, odd-offset
        historical frames need a vertical flip + P1/P2 swap to align.
      - 7 constant planes follow: real side-to-move, move counter, 4 castling
        flags, no-progress counter.
    """
    if history is None:
        return _board_to_matrix_19(board, move_counter)
    return _board_to_matrix_119(board, move_counter, history, rep_count=rep_count)


def _board_to_matrix_19(board: chess.Board, move_counter: int) -> np.ndarray:
    matrix = np.zeros((19, 8, 8), dtype=np.float32)
    for color in [True, False]:
        piece_offset = 0 if color else 6
        for piece_type in range(1, 7):  # pawn=1, knight=2, ..., king=6
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                piece_index = piece_offset + (piece_type - 1)
                matrix[piece_index, row, col] = 1

    # "Colour" plane (AZ S1): real side-to-move (player-to-move's actual color
    # in the un-mirrored game), NOT the canonical board.turn (always True here).
    matrix[12, :, :] = 1.0 if (move_counter % 2 == 0) else 0.0
    # Keep the /300 scale that 19-plane checkpoints were trained on, but CLAMP to
    # [0,1]. Pre-clamp this was move_counter/300 unclamped at inference while the
    # training data was clipped to 1.0 at uint8 quantization -> a train/inference
    # mismatch (and uint8 overflow) for games past ply 300. Clamping fixes that
    # WITHOUT changing the scale, so it stays backward-compatible with existing
    # 19-plane models. (The 119-plane path uses /500 to match 119-plane models;
    # the two representations are separate families, hence the different scales.)
    matrix[13, :, :] = min(move_counter / 300.0, 1.0)
    matrix[14, :, :] = board.has_kingside_castling_rights(True)
    matrix[15, :, :] = board.has_queenside_castling_rights(True)
    matrix[16, :, :] = board.has_kingside_castling_rights(False)
    matrix[17, :, :] = board.has_queenside_castling_rights(False)
    # Clamp: halfmove_clock can exceed 100 under the 75-move rule, which would
    # push the plane >1 (and overflow uint8 at quantization). Keep it in [0,1].
    matrix[18, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    return matrix


def _write_pieces_into(board: chess.Board, dest_frame: np.ndarray) -> None:
    """Fill 12 piece planes (own [0:6] + opp [6:12]) for a canonical board.
    In the canonical frame, the player-to-move's pieces are board.WHITE."""
    for piece_type in range(1, 7):
        for sq in board.pieces(piece_type, True):   # own (canonical WHITE)
            r, c = divmod(sq, 8)
            dest_frame[piece_type - 1, r, c] = 1.0
        for sq in board.pieces(piece_type, False):  # opp (canonical BLACK)
            r, c = divmod(sq, 8)
            dest_frame[6 + piece_type - 1, r, c] = 1.0


def _flip_frame_to_current_view(frame: np.ndarray) -> np.ndarray:
    """A canonical frame that is in the OPPOSITE perspective from the current
    player (i.e., an odd-offset historical frame). Bring it to the current
    player's view: vertical rank flip + swap own/opp piece planes. Repetition
    planes flip ranks too (rep is a per-position fact, no color swap needed)."""
    out = np.empty_like(frame)
    out[0:6, :, :] = np.flip(frame[6:12, :, :], axis=1)
    out[6:12, :, :] = np.flip(frame[0:6, :, :], axis=1)
    out[12:14, :, :] = np.flip(frame[12:14, :, :], axis=1)
    return out


def _board_to_matrix_119(board: chess.Board, move_counter: int,
                         history: list, rep_count: int = 1) -> np.ndarray:
    planes = np.zeros((119, 8, 8), dtype=np.float32)

    # Frame 0: current board (already in current-player canonical view).
    _write_pieces_into(board, planes[0:14, :, :])
    # Frame-0 repetition flags. AZ Table S1: plane 12 = position has been
    # seen at least once before (rep_count >= 2); plane 13 = at least twice
    # before (rep_count >= 3, draw imminent). Historical frames stay zero.
    if rep_count >= 2:
        planes[12, :, :] = 1.0
    if rep_count >= 3:
        planes[13, :, :] = 1.0

    # Frames -1 through -7: historical boards (chronological list, oldest first).
    # Iterate most-recent past first so offset_idx=0 → 1 ply ago, etc.
    recent = history[-7:]                 # at most 7 prior boards
    for offset_idx, hist_board in enumerate(reversed(recent)):
        offset = offset_idx + 1            # 1..7
        start = offset * 14                # planes [14:28] for offset=1, ...
        frame = np.zeros((14, 8, 8), dtype=np.float32)
        _write_pieces_into(hist_board, frame)
        if offset % 2 == 1:
            # Odd offsets are in opposite canonical frame from current.
            frame = _flip_frame_to_current_view(frame)
        planes[start:start + 14, :, :] = frame

    # 7 constant planes [112:119].
    planes[112, :, :] = 1.0 if (move_counter % 2 == 0) else 0.0  # real color
    planes[113, :, :] = min(move_counter / 500.0, 1.0)            # total moves
    planes[114, :, :] = float(board.has_kingside_castling_rights(True))
    planes[115, :, :] = float(board.has_queenside_castling_rights(True))
    planes[116, :, :] = float(board.has_kingside_castling_rights(False))
    planes[117, :, :] = float(board.has_queenside_castling_rights(False))
    planes[118, :, :] = min(board.halfmove_clock / 100.0, 1.0)  # clamp (75-move rule)
    return planes


def move_to_alphazero(move: str) -> int:
    start_file = ord(move[0]) - 97
    start_rank = int(move[1]) - 1
    end_file = ord(move[2]) - 97
    end_rank = int(move[3]) - 1
    start_idx = start_file + start_rank * 8

    file_diff = end_file - start_file
    rank_diff = end_rank - start_rank

    # Promotion moves
    if len(move) == 5 and move[4] != 'q':
        promotion_map = {'n': 0, 'b': 1, 'r': 2}
        move_type_index = 64 + promotion_map[move[4]] * 3 + (file_diff + 1)
    else:
        if file_diff == 0:  # Vertical moves
            move_type_index = 14 + rank_diff - 1 if rank_diff > 0 else 21 + abs(rank_diff) - 1
        elif rank_diff == 0:  # Horizontal moves
            move_type_index = file_diff - 1 if file_diff > 0 else 7 + abs(file_diff) - 1
        elif abs(file_diff) == abs(rank_diff):  # Diagonal moves
            if file_diff > 0 and rank_diff > 0:
                move_type_index = 28 + rank_diff - 1  # North-east
            elif file_diff < 0 and rank_diff > 0:
                move_type_index = 35 + rank_diff - 1  # North-west
            elif file_diff > 0 and rank_diff < 0:
                move_type_index = 42 + abs(rank_diff) - 1  # South-east
            else:
                move_type_index = 49 + abs(rank_diff) - 1  # South-west

        else:  # Knight moves
            move_type_index = 56 + (file_diff == 2) * 0 + (file_diff == 1) * 1 + (file_diff == -1) * 2 + (file_diff == -2) * 3
            if rank_diff < 0:
                move_type_index += 4

    return move_type_index * 64 + start_idx


def moves_to_alphazero(moves: list[chess.Move]) -> list[int]:
    return [move_to_alphazero(move.uci()) for move in moves]


def alphazero_to_move(action: int, board: chess.Board | None = None) -> str:
    start_idx = action % 64
    move_type_index = action // 64
    start_file = start_idx % 8
    start_rank = start_idx // 8
    start_square = chr(start_file + 97) + str(start_rank + 1)

    # Underpromotions (knight / bishop / rook) -- queen promotions fall
    # through to the sliding-move branch below by AlphaZero convention.
    if move_type_index >= 64:
        promotion_map = {0: 'n', 1: 'b', 2: 'r'}
        promotion_type_index = (move_type_index - 64) // 3
        promotion_piece = promotion_map[promotion_type_index]
        file_diff = (move_type_index - 64) % 3 - 1
        end_file = start_file + file_diff
        end_rank = start_rank + (1 if start_rank == 6 else -1)
        end_square = chr(end_file + 97) + str(end_rank + 1)
        return start_square + end_square + promotion_piece

    # Regular moves
    if move_type_index < 56:
        if move_type_index < 14:
            file_diff = (move_type_index % 7 + 1) * (1 if move_type_index < 7 else -1)
            rank_diff = 0
        elif move_type_index < 28:
            rank_diff = (move_type_index % 7 + 1) * (1 if move_type_index < 21 else -1)
            file_diff = 0
        else:
            diff = move_type_index % 7 + 1
            file_diff = diff * (1 if move_type_index < 35 or 42 <= move_type_index < 49 else -1)
            rank_diff = diff * (1 if 28 <= move_type_index < 42 else -1)
    elif 56 <= move_type_index < 64:
        knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
        file_diff, rank_diff = knight_moves[move_type_index - 56]

    end_file = start_file + file_diff
    end_rank = start_rank + rank_diff
    end_square = chr(end_file + 97) + str(end_rank + 1)
    uci = start_square + end_square

    # Queen-promotion disambiguation: if a pawn slides to the last rank,
    # python-chess requires an explicit promotion piece in UCI.
    if board is not None and end_rank == 7:
        piece = board.piece_at(chess.square(start_file, start_rank))
        if piece is not None and piece.piece_type == chess.PAWN:
            uci += 'q'

    return uci


def game_result(board: chess.Board, move_counter: int, truncation: int,
                rep_count: int = 1) -> tuple[int, bool]:
    """`rep_count` is how many times THIS position has occurred in the real
    game so far (including the current occurrence). Caller maintains a
    Counter[transposition_key] across the game and looks up the current
    board's key, because `chess.Board.mirror()` does not preserve the
    repetition history (`copy(stack=False)` internally), so we can't ask
    the board itself."""
    if board.is_checkmate():
        return -1, True
    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
            or rep_count >= 3
            or move_counter >= truncation):
        return 0, True
    return 0, False


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean array of shape (4672,) — True at indices of legal moves at `board`."""
    encoded = moves_to_alphazero(list(board.legal_moves))
    mask = np.zeros(4672, dtype=bool)
    mask[encoded] = True
    return mask


def valid_policy(policy: np.ndarray, board: chess.Board) -> np.ndarray:
    """Zero the illegal indices of a (4672,) policy and renormalise to sum 1.

    Always apply this before sampling from a raw policy-head output. Returns a
    new float array. If the policy has no mass on legal moves (degenerate /
    fully-masked), falls back to a uniform distribution over legal moves."""
    mask = legal_mask(board)
    masked = np.where(mask, policy, 0.0).astype(np.float32)
    total = masked.sum()
    if total > 0:
        return masked / total
    legal = mask.astype(np.float32)
    n = legal.sum()
    return legal / n if n > 0 else legal


def prepare_input(board: chess.Board, move_counter: int,
                  history: list | None = None,
                  rep_count: int = 1) -> torch.Tensor:
    """Float-tensor wrapper around board_to_matrix. Pass `history` for 119
    planes (omit for legacy 19); `rep_count` populates the current-frame
    repetition flags when running 119-plane."""
    matrix = board_to_matrix(board, move_counter, history=history, rep_count=rep_count)
    # board_to_matrix already returns a fresh, contiguous float32 array, so
    # from_numpy (zero-copy) is safe and avoids torch.tensor's extra copy.
    return torch.from_numpy(matrix)


def mirror_move(move: str) -> str:
    if move is None:
        return None
    return f"{move[0]}{9 - int(move[1])}{move[2]}{9 - int(move[3])}{move[4:]}"


def centipawn_to_prob(cp: float) -> float:
    return 0.64017665102 * math.atan(0.89513781885 * cp)
