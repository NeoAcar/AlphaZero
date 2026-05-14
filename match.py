"""
Player-vs-player match runner.

Each player is a JSON config with a "type" field plus type-specific options.
See players.py and configs/ for the available types and templates.

Usage:
    uv run python match.py --player1 configs/new_mcts.json --player2 configs/random.json --games 20

Players alternate colours (player1 plays white in games 0, 2, ...).
"""
import argparse
import json
import time

import chess
import chess.pgn
import numpy as np
import torch

from alphazero import utils as f
from alphazero.players import load_player


def termination_reason(board: chess.Board, move_counter: int, truncation: int) -> str:
    """Identify why the game ended. Truncation takes priority because the
    real_board may not actually be terminal under chess rules."""
    if move_counter >= truncation:
        return "truncation"
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_fivefold_repetition():
        return "5-fold_repetition"
    if board.is_seventyfive_moves():
        return "75-move_rule"
    if board.can_claim_threefold_repetition():
        return "3-fold_repetition"
    if board.can_claim_fifty_moves():
        return "50-move_rule"
    return "unknown"


def play_game(p1, p2, p1_color: chess.Color, truncation: int) -> tuple[int, chess.Board, str]:
    """Returns (result_for_p1, final_board, termination_reason).
    result: +1 win, -1 loss, 0 draw."""
    p1.reset()
    p2.reset()

    real_board = chess.Board()
    mirrored_state = chess.Board()
    move_counter = 0

    while not f.game_result(mirrored_state, move_counter, truncation)[1]:
        active = p1 if real_board.turn == p1_color else p2
        real_uci, mir_uci = active.select_move(real_board, mirrored_state, move_counter)
        real_board.push_uci(real_uci)
        mirrored_state.push_uci(mir_uci)
        mirrored_state = mirrored_state.mirror()
        move_counter += 1

    reason = termination_reason(real_board, move_counter, truncation)
    if real_board.is_checkmate():
        loser_color = real_board.turn
        return (+1 if loser_color != p1_color else -1), real_board, reason
    return 0, real_board, reason


def board_to_pgn(board: chess.Board, p1_color: chess.Color, result_for_p1: int,
                 p1_name: str, p2_name: str) -> chess.pgn.Game:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Player match"
    game.headers["White"] = p1_name if p1_color == chess.WHITE else p2_name
    game.headers["Black"] = p2_name if p1_color == chess.WHITE else p1_name
    if result_for_p1 == 0:
        game.headers["Result"] = "1/2-1/2"
    else:
        p1_won = result_for_p1 > 0
        white_won = (p1_won and p1_color == chess.WHITE) or (not p1_won and p1_color == chess.BLACK)
        game.headers["Result"] = "1-0" if white_won else "0-1"
    return game


def elo_diff_from_score(score: float, n_games: int) -> float | None:
    if n_games == 0:
        return None
    wr = score / n_games
    if wr <= 0 or wr >= 1:
        return None
    return -400.0 * float(np.log10(1.0 / wr - 1.0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--player1", required=True, help="JSON config for player 1")
    p.add_argument("--player2", required=True, help="JSON config for player 2")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--truncation", type=int, default=200)
    p.add_argument("--output", help="optional multi-game PGN output file")
    p.add_argument("--p1-name", default="p1")
    p.add_argument("--p2-name", default="p2")
    cli = p.parse_args()

    if torch.cuda.is_available():
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")

    print(f"Loading {cli.p1_name} from {cli.player1}")
    p1 = load_player(cli.player1)
    p1.display_name = cli.p1_name
    print(f"  type: {p1.name}")
    print(f"Loading {cli.p2_name} from {cli.player2}")
    p2 = load_player(cli.player2)
    p2.display_name = cli.p2_name
    print(f"  type: {p2.name}")

    print(f"\nPlaying {cli.games} games (truncation={cli.truncation}). "
          f"{cli.p1_name} plays white in odd-numbered games (1, 3, 5, ...).\n")

    wins = draws = losses = 0
    pgns = []
    try:
        for i in range(cli.games):
            p1_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_str = "white" if p1_color == chess.WHITE else "black"

            t0 = time.time()
            result, board, reason = play_game(p1, p2, p1_color, cli.truncation)
            dt = time.time() - t0

            if result > 0:
                wins += 1; tag = f"{cli.p1_name} WIN"
            elif result < 0:
                losses += 1; tag = f"{cli.p1_name} LOSS"
            else:
                draws += 1; tag = "DRAW"

            plies = len(board.move_stack)
            print(f"Game {i+1:>3}/{cli.games}: {cli.p1_name} as {color_str:>5} -> "
                  f"{tag:<13} {plies:>3} plies ({dt:5.1f}s) | {reason:<22} | "
                  f"running: {wins}W {draws}D {losses}L")
            print(f"  final FEN: {board.fen()}")

            if cli.output:
                pgns.append(board_to_pgn(board, p1_color, result, cli.p1_name, cli.p2_name))
    finally:
        p1.close()
        p2.close()

    score = wins + 0.5 * draws
    print(f"\n=== Final ===")
    print(f"{cli.p1_name}: {wins} W / {draws} D / {losses} L")
    print(f"{cli.p1_name} score: {score:.1f}/{cli.games} ({100 * score / cli.games:.1f}%)")
    elo = elo_diff_from_score(score, cli.games)
    if elo is not None:
        print(f"Elo diff ({cli.p1_name} - {cli.p2_name}): {elo:+.0f}  "
              f"(~{cli.games} games -- noisy below 30)")
    else:
        print("Elo diff: undefined (perfect or null score)")

    if cli.output:
        with open(cli.output, "w") as fh:
            for g in pgns:
                fh.write(str(g) + "\n\n")
        print(f"PGNs written to {cli.output}")


if __name__ == "__main__":
    main()
