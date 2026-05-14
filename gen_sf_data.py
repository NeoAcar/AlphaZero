"""
Generate sharded supervised dataset: boards + policy targets + Stockfish evals.

Walks each game in mirror-canonical frame and calls Stockfish on each
position to produce an eval aligned 1:1 with the boards.

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
    evals  (N,1)      float32 in [-1, 1]   # scalar value target
    wdls   (N,3)      float32              # (P(W), P(D), P(L)) from Stockfish
    legal_masks_packed (N, 584) uint8      # bit-packed 4672-wide bool mask
    positions_per_game (G,) int64
    meta (dict)

`wdls` is the raw per-position win/draw/loss probability triple from
Stockfish's UCI_ShowWDL (SF 14+). It's always stored so the same shards
can later train a 3-output WDL value head without regenerating data --
even when the current run uses a single scalar tanh head. When `--formula
wdl`, `evals` = wdls[:, 0] - wdls[:, 2] (the WDL-derived scalar). When
`--formula lichess` or `arctan`, `evals` comes from cp+sigmoid; `wdls`
is still stored for future use.

Streaming design: only ONE shard is held in memory at a time during
generation, so RAM is bounded regardless of total dataset size.

Multi-worker mode: --workers N spawns N independent Python+Stockfish
subprocess pairs, each handling a contiguous slice of games. This is the
right way to use CPU when --depth is low (Stockfish's internal threading
is useless at depth=0; you parallelize at the *game* level instead).

The scalar value target is taken from the player-to-move's perspective
and produced via --formula:

    wdl      (DEFAULT)   v = P(W) - P(L)   from Stockfish's UCI_ShowWDL.
                         Position-aware (Stockfish's own win-rate model);
                         strictly better calibration than cp+sigmoid.
                         Requires Stockfish 14+ with UCI_ShowWDL support.

    lichess              v = 2 / (1 + exp(-0.00368208 * cp)) - 1
                         Empirical Lichess WDL fit. Position-agnostic.

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

from alphazero import utils as f


def lichess_sigmoid(cp: float) -> float:
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0


def arctan_sigmoid(cp: float) -> float:
    return f.centipawn_to_prob(cp / 100.0)


CP_FORMULAS = {
    "lichess": lichess_sigmoid,
    "arctan": arctan_sigmoid,
}
FORMULA_CHOICES = ["wdl", *CP_FORMULAS.keys()]
NAN_WDL = (float("nan"), float("nan"), float("nan"))


def normalize_wdl(wdl_obj) -> tuple[float, float, float]:
    """Stockfish Wdl (ints summing to ~1000) → (P(W), P(D), P(L)) in [0, 1]."""
    total = wdl_obj.wins + wdl_obj.draws + wdl_obj.losses
    if total <= 0:
        return (0.0, 1.0, 0.0)
    return (wdl_obj.wins / total, wdl_obj.draws / total, wdl_obj.losses / total)


def safe_eval(engine, board, limit, engine_path, configure_kwargs, formula_name):
    """Returns (value, wdl_tuple, engine, failed). wdl_tuple is (w, d, l) ∈ [0,1]."""
    try:
        info = engine.analyse(board, limit)
        wdl_raw = info.get("wdl")
        wdl = normalize_wdl(wdl_raw.pov(board.turn)) if wdl_raw is not None else NAN_WDL
        if formula_name == "wdl":
            if wdl_raw is None:
                raise RuntimeError(
                    "--formula wdl requires Stockfish with UCI_ShowWDL (SF 14+)"
                )
            value = wdl[0] - wdl[2]
        else:
            cp = info["score"].pov(board.turn).score(mate_score=100000)
            value = CP_FORMULAS[formula_name](cp)
        return value, wdl, engine, False
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
        return 0.0, (0.0, 1.0, 0.0), engine, True


def save_shard(output_dir, shard_idx, boards, moves, evals, wdls, legal_masks,
               positions_per_game, meta):
    # boards and legal_masks arrive pre-quantized/packed (uint8) -- the workers
    # do this at append-time to keep per-shard memory low. See process_chunk.
    boards_t = torch.from_numpy(np.stack(boards))                    # (N, 19, 8, 8) uint8
    moves_t = torch.tensor(moves, dtype=torch.long)
    evals_t = torch.tensor(evals, dtype=torch.float32).reshape(-1, 1)
    wdls_t = torch.tensor(wdls, dtype=torch.float32)                  # (N, 3)
    masks_packed = torch.from_numpy(np.stack(legal_masks))            # (N, 584) uint8
    shard_meta = dict(meta)
    shard_meta.setdefault("boards_dtype", "uint8")
    shard_meta.setdefault("boards_scale", 255)
    shard_meta.setdefault("action_space", 4672)
    path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
    tmp = path + ".tmp"
    torch.save({
        "boards": boards_t,
        "moves": moves_t,
        "evals": evals_t,
        "wdls": wdls_t,
        "legal_masks_packed": masks_packed,
        "positions_per_game": np.asarray(positions_per_game, dtype=np.int64),
        "meta": shard_meta,
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
    formula_name = cfg["formula"]
    engine = chess.engine.SimpleEngine.popen_uci(cfg["engine"])
    try:
        engine.configure(configure_kwargs)
    except chess.engine.EngineError:
        pass
    # Enable WDL output. Required when --formula wdl; harmless (just makes the
    # field available) otherwise. Add to configure_kwargs so engine restarts
    # re-enable it.
    try:
        engine.configure({"UCI_ShowWDL": True})
        configure_kwargs["UCI_ShowWDL"] = True
    except chess.engine.EngineError:
        if formula_name == "wdl":
            print(f"[worker {cfg.get('worker_id', 0)}] FATAL: Stockfish at "
                  f"{cfg['engine']} doesn't support UCI_ShowWDL; can't use "
                  f"--formula wdl. Use a newer Stockfish (14+) or pick "
                  f"--formula lichess.", flush=True)
            try:
                engine.quit()
            except Exception:
                pass
            return {"games": 0, "positions": 0, "shards": 0, "failures": 0}
    limit = chess.engine.Limit(depth=cfg["depth"])

    shard_boards: list[np.ndarray] = []
    shard_moves: list[int] = []
    shard_evals: list[float] = []
    shard_wdls: list[tuple[float, float, float]] = []
    shard_masks: list[np.ndarray] = []
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
                val, wdl, engine, failed = safe_eval(
                    engine, real_board, limit, cfg["engine"], configure_kwargs,
                    formula_name,
                )
                if failed:
                    total_failures += 1

                # Quantize to uint8 and bit-pack the mask immediately, so per-shard
                # accumulation memory stays ~5x smaller (uint8 boards + packed
                # masks instead of float32 + bool arrays in Python lists).
                b = f.board_to_matrix(mirror_board, move_counter)
                shard_boards.append((np.clip(b, 0.0, 1.0) * 255.0).round().astype(np.uint8))
                shard_masks.append(np.packbits(f.legal_mask(mirror_board)))
                mover_was_white = (move_counter % 2 == 0)
                mir_move_str = move_str if mover_was_white else f.mirror_move(move_str)
                shard_moves.append(f.move_to_alphazero(mir_move_str))
                shard_evals.append(val)
                shard_wdls.append(wdl)

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
                           shard_evals, shard_wdls, shard_masks, shard_ppg,
                           shard_meta)
                shard_boards, shard_moves, shard_evals, shard_wdls = [], [], [], []
                shard_masks, shard_ppg = [], []
                shard_games_done = 0
                shard_idx += 1

        if pbar is not None:
            pbar.close()

    if shard_games_done > 0:
        save_shard(output_dir, shard_idx, shard_boards, shard_moves,
                   shard_evals, shard_wdls, shard_masks, shard_ppg, shard_meta)
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
    p.add_argument("--engine", default=os.path.expanduser("~/bin/stockfish18"))
    p.add_argument("--threads", type=int, default=None,
                   help="Stockfish UCI Threads option per worker. Default: 1 if --workers>1 else 4.")
    p.add_argument("--hash-mb", type=int, default=128)
    p.add_argument("--skip-games", type=int, default=0)
    p.add_argument("--shard-games", type=int, default=5000)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes (default: 1)")
    p.add_argument("--formula", choices=FORMULA_CHOICES, default="wdl")
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
