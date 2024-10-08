import chess
import numpy as np
from chess import pgn
import torch
from tqdm import tqdm # type: ignore
import chess.engine
import math

def board_to_matrix(board: chess.Board, move_counter:int) -> np.ndarray:
    matrix = np.zeros((19, 8, 8), dtype=np.int32)
    for color in [True, False]:
        piece_offset = 0 if color else 6
        for piece_type in range(1, 7):  # pawn=1, knight=2, ..., king=6
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                piece_index = piece_offset + (piece_type - 1)
                matrix[piece_index, row, col] = 1

    matrix[12, :, :] = 1 if board.turn else 0
    matrix[13, :, :] = move_counter / 500
    matrix[14, :, :] = board.has_kingside_castling_rights(True)
    matrix[15, :, :] = board.has_queenside_castling_rights(True)
    matrix[16, :, :] = board.has_kingside_castling_rights(False)
    matrix[17, :, :] = board.has_queenside_castling_rights(False)
    matrix[18, :, :] = board.halfmove_clock / 50
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

def alphazero_to_move(action: int) -> str:
    start_idx = action % 64
    move_type_index = action // 64
    start_file = start_idx % 8
    start_rank = start_idx // 8
    start_square = chr(start_file + 97) + str(start_rank + 1)

    # Promotion moves
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
        knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        file_diff, rank_diff = knight_moves[move_type_index - 56]

    

    end_file = start_file + file_diff
    end_rank = start_rank + rank_diff
    end_square = chr(end_file + 97) + str(end_rank + 1)

    return start_square + end_square

def game_result(board: chess.Board, move_counter: int, truncation:int) -> tuple[int, bool]:
    if board.is_checkmate():
        return -1, True
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or  move_counter >= truncation:
        return 0, True
    return 0, False

def parallel_valid_policy(policies: np.ndarray, boards: list[chess.Board]) -> np.ndarray:
    valid_moves = [list(board.legal_moves) for board in boards]
    encoded_valid_moves = [moves_to_alphazero(moves) for moves in valid_moves]
    mask = np.zeros(shape=(len(boards), 4672))
    for i, moves in enumerate(encoded_valid_moves):
        mask[i, moves] = 1
    valid_policy = mask * policies
    row_sums = np.sum(valid_policy, axis=1, keepdims=True, dtype=np.float32)
    valid_policy /=  np.where(row_sums != 0, row_sums, 1)
    return valid_policy

def valid_policy(policy: np.ndarray, board: chess.Board) -> np.ndarray:
    valid_moves = list(board.legal_moves)
    encoded_valid_moves = moves_to_alphazero(valid_moves)
    mask = np.zeros(4672)
    mask[encoded_valid_moves] = 1
    valid_policy = mask * policy
    valid_policy /= (np.sum(valid_policy) if np.sum(valid_policy) != 0 else 1)
    return valid_policy
     
def prepare_input(board: chess.Board, move_counter: int) -> torch.Tensor:
    matrix = board_to_matrix(board, move_counter)
    X_tensor = torch.tensor(matrix, dtype=torch.float32)
    # shape = (19, 8, 8)
    return X_tensor


def board_value(board: chess.Board) -> int:
    piece_values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 10, "k": 1000}
    value = 0
    if board.is_checkmate():
        return -10000
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0
    for square, piece in board.piece_map().items():
        value += piece_values[piece.symbol().lower()] if piece.color else -piece_values[piece.symbol().lower()]
    return value

def mirror_move(move: str) -> str:
    if move is None:
        return None
    return f"{move[0]}{9 - int(move[1])}{move[2]}{9 - int(move[3])}"
    
def create_nn_input(games: list[chess.pgn.Game]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = []
    y = []
    z = []
    for game in tqdm(games):
        board = chess.Board()
        result = float(eval(game.headers["Result"].split("-")[0])) * 2 - 1
        for move_counter, move in enumerate(game.mainline_moves()):
            move = move.uci()
            X.append(board_to_matrix(board, move_counter))
            if move_counter % 2 == 1:
                move = mirror_move(move)
            y.append(move_to_alphazero(move))
            z.append([result * (-1 if move_counter % 2 == 1 else 1)])
            board.push_uci(move)
            board = board.mirror()
        
    b = torch.tensor(np.array(X), dtype=torch.float32)
    m = torch.tensor(np.array(y), dtype=torch.long)
    r = torch.tensor(np.array(z), dtype=torch.float32)
    X = []
    y = []
    z = []
    return b,m,r

def load_pgn(file_path: str, max_games: int) -> list[chess.pgn.Game]:
    games = []
    with open(file_path, 'r') as pgn_file:
        for i in tqdm(range(max_games)):
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def stockfish(board: chess.Board, engine: chess.engine.SimpleEngine) -> float:
    info = engine.analyse(board, chess.engine.Limit(depth=0))
    return (info["score"].white().score(mate_score=100000)) / 100

def analyze_boards(boards: list[chess.Board]) -> list[float]:   
    results = []
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        for board in boards:
            try:
                result = stockfish(board, engine)
                results.append(result)
            except chess.engine.EngineError as e:
                print(f"Error analyzing board {board.fen()}: {e}")
                continue
    return results
def centipawn_to_prob(cp: float) -> float:
    return 0.64017665102 * math.atan(0.89513781885*cp)

def malstockfish(board: chess.Board) -> float:
    
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=0))
        return (info["score"].white().score(mate_score=100000))/100