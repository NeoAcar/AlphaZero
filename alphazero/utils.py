import math

import chess
import numpy as np
import torch


# Per-square bit masks for vectorised bitboard -> plane conversion. Square `i`
# in python-chess corresponds to bit `i` of any bitboard, with rank = i // 8 and
# file = i % 8. Flat index `i` of the resulting (8, 8) reshape lands at
# (rank=i//8, file=i%8), matching the existing matrix[plane, row, col] layout.
_BIT_MASKS = (np.uint64(1) << np.arange(64, dtype=np.uint64))  # shape (64,)


def board_to_matrix(board: chess.Board, move_counter: int) -> np.ndarray:
    # Pull all 12 piece bitboards at once (white pawns..kings, then black).
    bbs = np.fromiter(
        (board.pieces_mask(pt, color)
         for color in (True, False)
         for pt in range(1, 7)),
        dtype=np.uint64,
        count=12,
    )
    # Broadcast bit-AND: (12, 1) & (1, 64) -> (12, 64). Boolean -> float32.
    pieces = ((bbs[:, None] & _BIT_MASKS[None, :]) > np.uint64(0)).astype(np.float32)

    matrix = np.empty((19, 8, 8), dtype=np.float32)
    matrix[:12] = pieces.reshape(12, 8, 8)
    # "Colour" plane (AZ S1): real side-to-move (player-to-move's actual color
    # in the un-mirrored game), NOT the canonical board.turn (always True here).
    matrix[12].fill(1.0 if (move_counter % 2 == 0) else 0.0)
    # Move-counter normalisations: keep values in [0, 1] before save_shard's
    # clamp+uint8 quantisation. halfmove_clock can reach 99 before the 50-move
    # rule forces a draw; total move counter is capped at ~300 plies which
    # matches our self-play truncation and covers the bulk of real games.
    matrix[13].fill(move_counter / 300)
    matrix[14].fill(float(board.has_kingside_castling_rights(True)))
    matrix[15].fill(float(board.has_queenside_castling_rights(True)))
    matrix[16].fill(float(board.has_kingside_castling_rights(False)))
    matrix[17].fill(float(board.has_queenside_castling_rights(False)))
    matrix[18].fill(board.halfmove_clock / 100)
    return matrix


# python-chess piece-type constants -> AZ underpromotion slot.
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def move_obj_to_alphazero(move: chess.Move) -> int:
    """Encode a chess.Move directly to the 4672 AlphaZero action index.

    Equivalent to move_to_alphazero(move.uci()) but skips the string
    round-trip (~30-100us per move at typical legal-move counts).
    """
    fs = move.from_square
    ts = move.to_square
    start_file = fs & 7
    start_rank = fs >> 3
    end_file = ts & 7
    end_rank = ts >> 3
    file_diff = end_file - start_file
    rank_diff = end_rank - start_rank

    promo = move.promotion
    if promo is not None and promo != chess.QUEEN:
        move_type_index = 64 + _UNDERPROMO[promo] * 3 + (file_diff + 1)
    elif file_diff == 0:                                      # vertical
        move_type_index = 14 + rank_diff - 1 if rank_diff > 0 else 21 + (-rank_diff) - 1
    elif rank_diff == 0:                                      # horizontal
        move_type_index = file_diff - 1 if file_diff > 0 else 7 + (-file_diff) - 1
    elif abs(file_diff) == abs(rank_diff):                    # diagonal
        if file_diff > 0 and rank_diff > 0:
            move_type_index = 28 + rank_diff - 1              # NE
        elif file_diff < 0 and rank_diff > 0:
            move_type_index = 35 + rank_diff - 1              # NW
        elif file_diff > 0 and rank_diff < 0:
            move_type_index = 42 + (-rank_diff) - 1           # SE
        else:
            move_type_index = 49 + (-rank_diff) - 1           # SW
    else:                                                     # knight
        move_type_index = 56 + (file_diff == 2) * 0 + (file_diff == 1) * 1 \
                             + (file_diff == -1) * 2 + (file_diff == -2) * 3
        if rank_diff < 0:
            move_type_index += 4
    return move_type_index * 64 + fs


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
    return [move_obj_to_alphazero(move) for move in moves]


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


def game_result(board: chess.Board, move_counter: int, truncation: int) -> tuple[int, bool]:
    if board.is_checkmate():
        return -1, True
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or move_counter >= truncation:
        return 0, True
    return 0, False


def legal_mask(board: chess.Board) -> np.ndarray:
    """Boolean array of shape (4672,) — True at indices of legal moves at `board`."""
    encoded = moves_to_alphazero(list(board.legal_moves))
    mask = np.zeros(4672, dtype=bool)
    mask[encoded] = True
    return mask


def prepare_input(board: chess.Board, move_counter: int) -> torch.Tensor:
    matrix = board_to_matrix(board, move_counter)
    X_tensor = torch.tensor(matrix, dtype=torch.float32)
    # shape = (19, 8, 8)
    return X_tensor


def mirror_move(move: str) -> str:
    if move is None:
        return None
    return f"{move[0]}{9 - int(move[1])}{move[2]}{9 - int(move[3])}{move[4:]}"


def centipawn_to_prob(cp: float) -> float:
    return 0.64017665102 * math.atan(0.89513781885 * cp)
