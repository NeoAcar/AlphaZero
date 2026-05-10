"""
Bot-vs-Stockfish match runner.

Usage:
    uv run python stockfish_match.py \\
        --bot configs/new_model.json \\
        --stockfish configs/stockfish.json \\
        --games 20

Requires the `stockfish` binary on PATH (Ubuntu: `sudo apt install stockfish`).

Stockfish config keys (all optional):
    binary:           path to engine (default: "stockfish")
    skill_level:      0..20  (0 ~= 1300 Elo, 20 = full strength)
    elo:              integer between 1320 and 3190
    limit_strength:   true to honour `elo` (UCI_LimitStrength)
    depth:            search depth (e.g. 1 = ~1500 Elo)
    time_ms:          per-move time budget in milliseconds
"""
import argparse
import json
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

import optimized_functions as f
from ithinkbettermcts import MCTS
from resnet import ResNet


DEFAULT_BOT_ARGS = {
    "num_simulation": 200,
    "truncation": 200,
    "c_base": 19652,
    "c_init": 1.25,
    "dirichlet_epsilon": 0.0,
    "dirichlet_alpha": 0.03,
    "memory_size": 1000,
    "action_space": 4672,
    "top_actions": 5,
    "t": 1,
}

DEFAULT_SF_CFG = {
    "binary": "stockfish",
    "skill_level": None,
    "elo": None,
    "limit_strength": False,
    "depth": None,
    "time_ms": None,
}


def load_bot(config_path: str) -> tuple[MCTS, dict]:
    with open(config_path) as fh:
        cfg = json.load(fh)
    if "checkpoint" not in cfg:
        raise ValueError(f"{config_path} must specify 'checkpoint'")
    args = dict(DEFAULT_BOT_ARGS)
    args.update(cfg)
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet().to(args["device"])
    state = torch.load(args["checkpoint"], map_location=args["device"], weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return MCTS(args, model), args


def start_stockfish(config_path: str) -> tuple[chess.engine.SimpleEngine, dict]:
    with open(config_path) as fh:
        cfg = json.load(fh)
    sf = dict(DEFAULT_SF_CFG)
    sf.update(cfg)

    engine = chess.engine.SimpleEngine.popen_uci(sf["binary"])

    uci_options = {}
    if sf["skill_level"] is not None:
        uci_options["Skill Level"] = int(sf["skill_level"])
    if sf["limit_strength"] and sf["elo"] is not None:
        uci_options["UCI_LimitStrength"] = True
        uci_options["UCI_Elo"] = int(sf["elo"])
    if uci_options:
        engine.configure(uci_options)

    return engine, sf


def stockfish_limit(sf_cfg: dict) -> chess.engine.Limit:
    if sf_cfg["depth"] is not None:
        return chess.engine.Limit(depth=int(sf_cfg["depth"]))
    if sf_cfg["time_ms"] is not None:
        return chess.engine.Limit(time=float(sf_cfg["time_ms"]) / 1000.0)
    # Default: small fixed depth so games don't take forever.
    return chess.engine.Limit(depth=10)


def play_game(bot_mcts: MCTS, engine: chess.engine.SimpleEngine, sf_cfg: dict,
              bot_color: chess.Color, truncation: int
              ) -> tuple[int, chess.Board]:
    """Returns (result_for_bot, final_board). result: +1 win, -1 loss, 0 draw."""
    bot_mcts.root = None
    real_board = chess.Board()
    mirrored_state = chess.Board()
    move_counter = 0
    limit = stockfish_limit(sf_cfg)

    while not f.game_result(mirrored_state, move_counter, truncation)[1]:
        mover_was_white = real_board.turn == chess.WHITE

        if real_board.turn == bot_color:
            probs = bot_mcts.search(mirrored_state, move_counter)
            action = int(probs.argmax())
            uci_mirrored = f.alphazero_to_move(action, mirrored_state)
            real_uci = uci_mirrored if mover_was_white else f.mirror_move(uci_mirrored)
            real_board.push_uci(real_uci)
            mirrored_state.push_uci(uci_mirrored)
        else:
            result = engine.play(real_board, limit)
            move = result.move
            real_uci = move.uci()
            real_board.push(move)
            # Convert SF's real-coord move into mirrored-coord for the bot's tree.
            uci_mirrored = real_uci if mover_was_white else f.mirror_move(real_uci)
            mirrored_state.push_uci(uci_mirrored)

        mirrored_state = mirrored_state.mirror()
        move_counter += 1

    if real_board.is_checkmate():
        loser_color = real_board.turn
        return (+1 if loser_color != bot_color else -1), real_board
    return 0, real_board


def board_to_pgn(board: chess.Board, bot_color: chess.Color, result_for_bot: int,
                 bot_name: str, sf_name: str) -> chess.pgn.Game:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Bot vs Stockfish"
    game.headers["White"] = bot_name if bot_color == chess.WHITE else sf_name
    game.headers["Black"] = sf_name if bot_color == chess.WHITE else bot_name
    if result_for_bot == 0:
        game.headers["Result"] = "1/2-1/2"
    else:
        bot_won = result_for_bot > 0
        white_won = (bot_won and bot_color == chess.WHITE) or (not bot_won and bot_color == chess.BLACK)
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
    p.add_argument("--bot", required=True, help="JSON config for the bot")
    p.add_argument("--stockfish", required=True, help="JSON config for Stockfish")
    p.add_argument("--games", type=int, default=20, help="number of games (default 20)")
    p.add_argument("--truncation", type=int, default=200, help="max plies per game (default 200)")
    p.add_argument("--output", help="optional PGN output file")
    p.add_argument("--bot-name", default="bot", help="bot name in PGN headers")
    p.add_argument("--sf-name", default="stockfish", help="stockfish name in PGN headers")
    cli = p.parse_args()

    print(f"Loading {cli.bot_name} from {cli.bot}")
    bot_mcts, bot_args = load_bot(cli.bot)
    print(f"  checkpoint: {bot_args['checkpoint']}, sims: {bot_args['num_simulation']}, "
          f"top_actions: {bot_args['top_actions']}")

    print(f"Starting {cli.sf_name} from {cli.stockfish}")
    engine, sf_cfg = start_stockfish(cli.stockfish)
    print(f"  binary: {sf_cfg['binary']}")
    print(f"  skill_level: {sf_cfg['skill_level']}, "
          f"elo: {sf_cfg['elo']} (limit_strength={sf_cfg['limit_strength']}), "
          f"depth: {sf_cfg['depth']}, time_ms: {sf_cfg['time_ms']}")

    print(f"\nPlaying {cli.games} games (truncation={cli.truncation}). "
          f"{cli.bot_name} plays white in even-indexed games.\n")

    wins = draws = losses = 0
    pgns = []
    try:
        for i in range(cli.games):
            bot_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_str = "white" if bot_color == chess.WHITE else "black"

            t0 = time.time()
            result, board = play_game(bot_mcts, engine, sf_cfg, bot_color, cli.truncation)
            dt = time.time() - t0

            if result > 0:
                wins += 1; tag = f"{cli.bot_name} WIN"
            elif result < 0:
                losses += 1; tag = f"{cli.bot_name} LOSS"
            else:
                draws += 1; tag = "DRAW"

            print(f"Game {i+1:>3}/{cli.games}: {cli.bot_name} as {color_str:>5} -> "
                  f"{tag:<13} ({dt:5.1f}s) | running: {wins}W {draws}D {losses}L")

            if cli.output:
                pgns.append(board_to_pgn(board, bot_color, result, cli.bot_name, cli.sf_name))
    finally:
        engine.quit()

    score = wins + 0.5 * draws
    print(f"\n=== Final ===")
    print(f"{cli.bot_name}: {wins} W / {draws} D / {losses} L")
    print(f"{cli.bot_name} score: {score:.1f}/{cli.games} ({100 * score / cli.games:.1f}%)")
    elo = elo_diff_from_score(score, cli.games)
    if elo is not None:
        print(f"Elo diff ({cli.bot_name} - {cli.sf_name}): {elo:+.0f}  "
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
