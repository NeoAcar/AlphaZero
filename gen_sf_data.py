"""
Generate sharded supervised dataset: boards + policy targets + Stockfish evals.

Walks each game in mirror-canonical frame (same loop structure as
optimized_functions.create_nn_input) and calls Stockfish on each position to
produce an eval aligned 1:1 with the boards.

Output layout (one directory, sharded files; with multiple workers each
worker writes into its own subdir):

    --output-dir/
      worker_0000/shard_0000.pt   (when --workers > 1)
      worker_0000/shard_0001.pt
      worker_0001/shard_0000.pt
      ...

      shard_0000.pt               (when --workers == 1)
      shard_0001.pt
      ...

Each shard is a dict with self-contained:
    boards (N,19,8,8) float32
    moves  (N,)       long
    evals  (N,1)      float32 in [-1, 1]
    positions_per_game (G,) int64
    meta (dict)

Streaming design: only ONE shard is held in memory at a time during
generation, so RAM is bounded regardless of total dataset size.

Multi-worker mode: --workers N spawns N independent Python+Stockfish
subprocess pairs, each handling a contiguous slice of games. This is the
right way to use CPU when --depth is low (Stockfish's internal threading
is useless at depth=0; you parallelize at the *game* level instead).

The eval is taken from the player-to-move's perspective and mapped from
centipawns to [-1, 1] via --formula:

    lichess  (DEFAULT)   v = 2 / (1 + exp(-0.00368208 * cp)) - 1
                         Empirical Lichess WDL fit.

    arctan               v = 0.64017665102 * atan(0.89513781885 * cp/100)
                         Older hand-tuned approximation.

Usage:
    uv run python gen_sf_data.py \
        --pgn data/games.pgn \
        --depth 0 --shard-games 5000 \
        --workers 8 \
        --output-dir data/sf_shards

Resume after a crash:
    Each shard is written atomically. Just rerun the same command --
    the script detects existing shards per worker and skips past them.
"""
import argparse
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from tqdm import tqdm  # type: ignore

import optimized_functions as f


def lichess_sigmoid(cp: float) -> float:
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0


def arctan_sigmoid(cp: float) -> float:
    return f.centipawn_to_prob(cp / 100.0)


FORMULAS = {
    "lichess": lichess_sigmoid,
    "arctan": arctan_sigmoid,
}


def safe_eval(engine, board, limit, engine_path, configure_kwargs, formula):
    try:
        info = engine.analyse(board, limit)
        cp = info["score"].pov(board.turn).score(mate_score=100000)
        return formula(cp), engine, False
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, BrokenPipeError):
        try:
            engine.quit()
        except Exception:
            pass
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        if configure_kwargs:
            try:
                engine.configure(configure_kwargs)
            except Exception:
                pass
        return 0.0, engine, True


def save_shard(output_dir, shard_idx, boards, moves, evals, positions_per_game, meta):
    boards_t = torch.from_numpy(np.stack(boards))
    moves_t = torch.tensor(moves, dtype=torch.long)
    evals_t = torch.tensor(evals, dtype=torch.float32).reshape(-1, 1)
    path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
    tmp = path + ".tmp"
    torch.save({
        "boards": boards_t,
        "moves": moves_t,
        "evals": evals_t,
        "positions_per_game": np.asarray(positions_per_game, dtype=np.int64),
        "meta": meta,
    }, tmp)
    os.replace(tmp, path)


def count_pgn_games(pgn_path: str) -> int:
    """Quickly count games by scanning for the [Event ...] header line."""
    count = 0
    with open(pgn_path, "rb") as fh:
        for line in fh:
            if line.startswith(b"[Event "):
                count += 1
    return count


def fast_skip(fh, n: int) -> int:
    """Skip n games. Returns how many were actually skipped (may be less on EOF)."""
    skipped = 0
    for _ in range(n):
        if not chess.pgn.skip_game(fh):
            break
        skipped += 1
    return skipped


def process_chunk(cfg: dict) -> dict:
    """Worker entry point. cfg is a plain dict for picklability."""
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Auto-resume per worker: pick up after any existing shards.
    existing = sorted(Path(output_dir).glob("shard_*.pt"))
    if existing and cfg["skip_games"] >= 0:
        last_idx = int(existing[-1].stem.split("_")[1])
        shard_idx = last_idx + 1
        cfg["skip_games"] += shard_idx * cfg["shard_games"]
        cfg["max_games"] = max(0, cfg["max_games"] - shard_idx * cfg["shard_games"])
        resumed = shard_idx
    else:
        shard_idx = 0
        resumed = 0

    configure_kwargs = {"Threads": cfg["threads"], "Hash": cfg["hash_mb"]}
    formula = FORMULAS[cfg["formula"]]
    engine = chess.engine.SimpleEngine.popen_uci(cfg["engine"])
    try:
        engine.configure(configure_kwargs)
    except chess.engine.EngineError:
        pass
    limit = chess.engine.Limit(depth=cfg["depth"])

    shard_boards: list[np.ndarray] = []
    shard_moves: list[int] = []
    shard_evals: list[float] = []
    shard_ppg: list[int] = []
    shard_games_done = 0

    total_games = 0
    total_positions = 0
    total_failures = 0
    games_skipped_for_bad_header = 0
    t0 = time.time()

    shard_meta = {
        "depth": cfg["depth"],
        "formula": cfg["formula"],
        "engine": cfg["engine"],
        "threads": cfg["threads"],
        "shard_games": cfg["shard_games"],
        "worker_id": cfg.get("worker_id", 0),
    }

    log_prefix = f"[worker {cfg.get('worker_id', 0):>2}]" if cfg.get("multi_worker") else ""

    def log(msg: str) -> None:
        print(f"{log_prefix} {msg}" if log_prefix else msg, flush=True)

    if resumed > 0:
        log(f"resume: skipping {resumed * cfg['shard_games']} already-completed games, "
            f"continuing as shard_{shard_idx:04d}")

    with open(cfg["pgn"]) as fh:
        actually_skipped = fast_skip(fh, cfg["skip_games"])
        if actually_skipped < cfg["skip_games"]:
            log(f"PGN ended after {actually_skipped} games; can't reach skip_games={cfg['skip_games']}.")
            try:
                engine.quit()
            except Exception:
                pass
            return {"games": 0, "positions": 0, "shards": 0}

        use_tqdm = cfg.get("use_tqdm", True)
        pbar = tqdm(desc=f"worker {cfg.get('worker_id', 0)}",
                    unit="game", total=cfg["max_games"]) if use_tqdm else None
        last_print = time.time()

        while total_games < cfg["max_games"]:
            game = chess.pgn.read_game(fh)
            if game is None:
                break

            try:
                result_str = game.headers["Result"]
                _ = float(eval(result_str.split("-")[0])) * 2 - 1  # noqa: S307
            except (KeyError, SyntaxError, ValueError, ZeroDivisionError, NameError):
                games_skipped_for_bad_header += 1
                continue

            real_board = chess.Board()
            mirror_board = chess.Board()
            game_positions = 0

            for move_counter, move in enumerate(game.mainline_moves()):
                move_str = move.uci()
                val, engine, failed = safe_eval(
                    engine, real_board, limit, cfg["engine"], configure_kwargs, formula
                )
                if failed:
                    total_failures += 1

                shard_boards.append(f.board_to_matrix(mirror_board, move_counter))
                mover_was_white = (move_counter % 2 == 0)
                mir_move_str = move_str if mover_was_white else f.mirror_move(move_str)
                shard_moves.append(f.move_to_alphazero(mir_move_str))
                shard_evals.append(val)

                real_board.push(move)
                mirror_board.push_uci(mir_move_str)
                mirror_board = mirror_board.mirror()
                game_positions += 1
                total_positions += 1

            shard_ppg.append(game_positions)
            shard_games_done += 1
            total_games += 1

            if pbar is not None:
                pbar.update(1)
                elapsed = max(time.time() - t0, 1e-6)
                pbar.set_postfix(shard=shard_idx, in_shard=shard_games_done,
                                 pos=total_positions, fail=total_failures,
                                 rate=f"{total_positions / elapsed:.0f}/s")
            else:
                # Multi-worker: periodic plain prints (every 30s).
                if time.time() - last_print > 30.0:
                    elapsed = max(time.time() - t0, 1e-6)
                    log(f"games {total_games}/{cfg['max_games']}, "
                        f"positions {total_positions}, "
                        f"rate {total_positions/elapsed:.0f}/s, "
                        f"shard {shard_idx} ({shard_games_done}/{cfg['shard_games']})")
                    last_print = time.time()

            if shard_games_done >= cfg["shard_games"]:
                save_shard(output_dir, shard_idx, shard_boards, shard_moves,
                           shard_evals, shard_ppg, shard_meta)
                shard_boards, shard_moves, shard_evals, shard_ppg = [], [], [], []
                shard_games_done = 0
                shard_idx += 1

        if pbar is not None:
            pbar.close()

    if shard_games_done > 0:
        save_shard(output_dir, shard_idx, shard_boards, shard_moves,
                   shard_evals, shard_ppg, shard_meta)
        shard_idx += 1

    try:
        engine.quit()
    except Exception:
        pass

    log(f"done in {time.time()-t0:.0f}s: {total_games} games, "
        f"{total_positions} positions, {total_failures} fails, "
        f"{games_skipped_for_bad_header} bad headers, {shard_idx} shards")

    return {"games": total_games, "positions": total_positions,
            "shards": shard_idx, "failures": total_failures}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pgn", required=True)
    p.add_argument("--max-games", type=int, default=None,
                   help="max games to process across all workers (default: count PGN)")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--engine", default="stockfish")
    p.add_argument("--threads", type=int, default=None,
                   help="Stockfish UCI Threads option per worker. Default: 1 if --workers>1 else 4.")
    p.add_argument("--hash-mb", type=int, default=128)
    p.add_argument("--skip-games", type=int, default=0)
    p.add_argument("--shard-games", type=int, default=5000)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes (default: 1)")
    p.add_argument("--formula", choices=list(FORMULAS), default="lichess")
    p.add_argument("--output-dir", required=True)
    cli = p.parse_args()

    if cli.threads is None:
        cli.threads = 1 if cli.workers > 1 else 4

    if cli.max_games is None:
        print(f"Counting games in {cli.pgn} ...")
        t0 = time.time()
        cli.max_games = count_pgn_games(cli.pgn)
        cli.max_games = max(0, cli.max_games - cli.skip_games)
        print(f"  found {cli.max_games} games after --skip-games={cli.skip_games} "
              f"(scan took {time.time()-t0:.1f}s)")

    os.makedirs(cli.output_dir, exist_ok=True)

    if cli.workers <= 1:
        # Single-worker: write shards directly into --output-dir (no subdir).
        cfg = {
            "pgn": cli.pgn, "engine": cli.engine, "depth": cli.depth,
            "threads": cli.threads, "hash_mb": cli.hash_mb,
            "skip_games": cli.skip_games, "max_games": cli.max_games,
            "shard_games": cli.shard_games, "formula": cli.formula,
            "output_dir": cli.output_dir, "worker_id": 0,
            "multi_worker": False, "use_tqdm": True,
        }
        process_chunk(cfg)
        return

    # Multi-worker: each gets a contiguous chunk of games, writes to its own subdir.
    chunk = cli.max_games // cli.workers
    remainder = cli.max_games - chunk * cli.workers
    worker_cfgs = []
    cursor = cli.skip_games
    for w in range(cli.workers):
        n = chunk + (1 if w < remainder else 0)
        worker_cfgs.append({
            "pgn": cli.pgn, "engine": cli.engine, "depth": cli.depth,
            "threads": cli.threads, "hash_mb": cli.hash_mb,
            "skip_games": cursor, "max_games": n,
            "shard_games": cli.shard_games, "formula": cli.formula,
            "output_dir": os.path.join(cli.output_dir, f"worker_{w:04d}"),
            "worker_id": w,
            "multi_worker": True, "use_tqdm": False,
        })
        cursor += n

    print(f"Spawning {cli.workers} workers, "
          f"~{chunk} games each, threads/worker={cli.threads}")
    print(f"Output dirs: {cli.output_dir}/worker_0000/ ... worker_{cli.workers-1:04d}/")

    t0 = time.time()
    with mp.get_context("spawn").Pool(cli.workers) as pool:
        results = pool.map(process_chunk, worker_cfgs)

    elapsed = time.time() - t0
    total_games = sum(r["games"] for r in results)
    total_positions = sum(r["positions"] for r in results)
    total_shards = sum(r["shards"] for r in results)
    total_failures = sum(r["failures"] for r in results)
    print(f"\nAll workers done in {elapsed:.0f}s")
    print(f"  games: {total_games}, positions: {total_positions}, "
          f"shards: {total_shards}, failures: {total_failures}")
    print(f"  effective throughput: {total_positions/elapsed:.0f} positions/sec")


if __name__ == "__main__":
    main()
