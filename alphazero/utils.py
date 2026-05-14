import math

import chess
import numpy as np
import torch


def board_to_matrix(board: chess.Board, move_counter: int) -> np.ndarray:
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
    # Move-counter normalisations: keep values in [0, 1] before save_shard's
    # clamp+uint8 quantisation. halfmove_clock can reach 99 before the 50-move
    # rule forces a draw; total move counter is capped at ~300 plies which
    # matches our self-play truncation and covers the bulk of real games.
    matrix[13, :, :] = move_counter / 300
    matrix[14, :, :] = board.has_kingside_castling_rights(True)
    matrix[15, :, :] = board.has_queenside_castling_rights(True)
    matrix[16, :, :] = board.has_kingside_castling_rights(False)
    matrix[17, :, :] = board.has_queenside_castling_rights(False)
    matrix[18, :, :] = board.halfmove_clock / 100
    return matrix


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
