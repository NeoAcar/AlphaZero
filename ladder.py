"""Checkpoint ladder. Plays a new checkpoint vs N prior versions, records
per-opponent win/draw/loss and aggregate Elo. The output JSON is the input
to analytics.py for the Elo curve.

A "ladder" call is the natural cadence: after every training run, run this
once to see whether the new version is actually stronger than older ones.

Each opponent uses the same player type as the new model (mcts), same sims,
same temperature, etc. -- so the only varying axis is the model weights.

Usage:
    uv run python ladder.py \\
        --new models/v03.pth \\
        --opponents models/v02.pth models/v01.pth models/v00_seed.pth \\
        --games-per-opponent 20 \\
        --output logs/matches/v03_ladder.json
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def make_player_config(checkpoint: str, sims: int, temperature: float,
                       temperature_moves: int, dirichlet_eps: float,
                       architecture: str | None) -> dict:
    cfg = {
        "type":                "mcts",
        "checkpoint":          checkpoint,
        "num_simulation":      sims,
        "discount":            1,
        "c_init":              1.25,
        "c_base":              19652,
        "c_fpu":               0.2,
        "dirichlet_epsilon":   dirichlet_eps,
        "dirichlet_alpha":     0.3,
        "t":                   1,
        "temperature_moves":   temperature_moves,
        "temperature":         temperature,
        "compile":             True,
        "batched":             True,
        "batch_size":          8,
    }
    if architecture:
        cfg["architecture"] = architecture
    return cfg


def write_tmp_config(cfg: dict, tmp_dir: Path, name: str) -> Path:
    path = tmp_dir / f"_ladder_{name}.json"
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def parse_match_output(stdout: str, p1_name: str, p2_name: str) -> dict:
    """Pull the W/D/L counts + per-side avg/max depth out of match.py's stream."""
    wins = draws = losses = None
    p1_depths: list[int] = []
    p2_depths: list[int] = []
    for line in stdout.splitlines():
        s = line.strip()
        # Example: "  [model_last] max_depth 8"  → per-move depth tick
        if "max_depth" in s and s.startswith("["):
            try:
                name = s.split("]")[0].lstrip("[").strip()
                depth = int(s.split("max_depth")[1].strip().split()[0])
                if name == p1_name:
                    p1_depths.append(depth)
                elif name == p2_name:
                    p2_depths.append(depth)
            except Exception:
                pass
        # Example: "p1: 14 W / 3 D / 3 L"
        if " W / " in s and " D / " in s and " L" in s:
            try:
                parts = s.split(":")[-1].split("/")
                wins = int(parts[0].strip().split()[0])
                draws = int(parts[1].strip().split()[0])
                losses = int(parts[2].strip().split()[0])
            except Exception:
                pass

    def _stats(ds: list[int]) -> dict:
        if not ds:
            return {"n": 0, "avg": None, "max": None}
        return {"n": len(ds), "avg": sum(ds) / len(ds), "max": max(ds)}

    return {
        "wins": wins, "draws": draws, "losses": losses,
        "p1_depth": _stats(p1_depths),
        "p2_depth": _stats(p2_depths),
    }


def elo_from_score(score: float, games: int) -> float | None:
    if games <= 0:
        return None
    wr = score / games
    if wr <= 0 or wr >= 1:
        return None
    return -400.0 * math.log10(1.0 / wr - 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, help="path to the new checkpoint under test")
    p.add_argument("--opponents", nargs="+", required=True,
                   help="checkpoint paths to play against")
    p.add_argument("--games-per-opponent", type=int, default=20)
    p.add_argument("--sims", type=int, default=400, help="MCTS sims per move (both sides)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--temperature-moves", type=int, default=10)
    p.add_argument("--dirichlet-eps", type=float, default=0.0,
                   help="0 -> deterministic / strength test (recommended for ladder)")
    p.add_argument("--architecture", default=None,
                   help="override; otherwise read from each checkpoint's meta")
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--match-py", default="match.py", help="path to match.py")
    p.add_argument("--keep-tmp", action="store_true",
                   help="do not delete the temporary player configs")
    cli = p.parse_args()

    out_path = Path(cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent
    new_name = Path(cli.new).stem

    new_cfg_path = write_tmp_config(
        make_player_config(cli.new, cli.sims, cli.temperature, cli.temperature_moves,
                           cli.dirichlet_eps, cli.architecture),
        tmp_dir, f"new_{new_name}",
    )

    matches = []
    total_w = total_d = total_l = 0
    t_session = time.time()
    pgn_dir = out_path.parent

    interrupted = False
    for opp in cli.opponents:
        if interrupted:
            break
        opp_name = Path(opp).stem
        opp_cfg = write_tmp_config(
            make_player_config(opp, cli.sims, cli.temperature, cli.temperature_moves,
                               cli.dirichlet_eps, cli.architecture),
            tmp_dir, f"opp_{opp_name}",
        )
        pgn_path = pgn_dir / f"{new_name}_vs_{opp_name}.pgn"
        cmd = [
            sys.executable, cli.match_py,
            "--player1", str(new_cfg_path),
            "--player2", str(opp_cfg),
            "--games", str(cli.games_per_opponent),
            "--p1-name", new_name,
            "--p2-name", opp_name,
            "--output", str(pgn_path),
        ]
        print(f"\n=== {new_name} vs {opp_name} : {cli.games_per_opponent} games ===")
        t0 = time.time()
        # Stream match.py's stdout line-by-line so we see per-game progress
        # live, while also buffering the full output for parse_match_output.
        # On Ctrl-C, forward SIGINT to match.py (so it can save partial PGN)
        # then collect whatever it streamed, parse it, and break.
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            stdout_lines: list[str] = []
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    stdout_lines.append(line)
                rc = proc.wait()
                if rc != 0:
                    print(f"match.py exited with code {rc}")
                    continue
            except KeyboardInterrupt:
                print("\n[Ctrl-C] forwarding to match.py and saving partial results...")
                proc.send_signal(signal.SIGINT)
                # Drain remaining output so match.py can finish its cleanup.
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    stdout_lines.append(line)
                proc.wait()
                interrupted = True
            stdout_text = "".join(stdout_lines)
        except Exception as e:
            print(f"match.py failed: {e}")
            continue
        dt = time.time() - t0
        result = parse_match_output(stdout_text, new_name, opp_name)
        if result["wins"] is None:
            print("could not parse match output (already streamed above)")
            continue
        w, d, l = result["wins"], result["draws"], result["losses"]
        n = w + d + l
        score = w + 0.5 * d
        elo = elo_from_score(score, n)
        print(f"  result: {w}W / {d}D / {l}L  → score {score:.1f}/{n}  "
              f"Elo {('%.0f' % elo) if elo is not None else 'inf'}  ({dt:.0f}s)")
        d1, d2 = result["p1_depth"], result["p2_depth"]
        if d1["avg"] is not None and d2["avg"] is not None:
            print(f"  avg max_depth: {new_name} {d1['avg']:.1f} "
                  f"(max {d1['max']}) vs {opp_name} {d2['avg']:.1f} "
                  f"(max {d2['max']})")
        matches.append({
            "opponent":          opp,
            "opponent_name":     opp_name,
            "games":             n,
            "wins":              w,
            "draws":             d,
            "losses":            l,
            "score":             score,
            "elo_diff":          elo,
            "duration_s":        round(dt, 1),
            "pgn_path":          str(pgn_path),
            "new_depth_avg":     d1["avg"],
            "new_depth_max":     d1["max"],
            "opp_depth_avg":     d2["avg"],
            "opp_depth_max":     d2["max"],
        })
        total_w += w; total_d += d; total_l += l

    if not cli.keep_tmp:
        for f in tmp_dir.glob("_ladder_*.json"):
            try: f.unlink()
            except OSError: pass

    n_total = total_w + total_d + total_l
    total_score = total_w + 0.5 * total_d
    overall_elo = elo_from_score(total_score, n_total)
    summary = {
        "new":               cli.new,
        "new_name":          new_name,
        "opponents":         cli.opponents,
        "games_per_opponent": cli.games_per_opponent,
        "matches":           matches,
        "total_wins":        total_w,
        "total_draws":       total_d,
        "total_losses":      total_l,
        "overall_score":     total_score,
        "overall_elo_diff":  overall_elo,
        "duration_s":        round(time.time() - t_session, 1),
    }
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nLadder summary → {out_path}")
    print(f"  total: {total_w}W / {total_d}D / {total_l}L  "
          f"Elo {('%.0f' % overall_elo) if overall_elo is not None else 'inf'}")


if __name__ == "__main__":
    main()
