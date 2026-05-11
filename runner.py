"""
AlphaZero self-play loop runner.

For each iteration:
  1. Generate N self-play games with the current `best` checkpoint
  2. Train a fresh copy on (recent self-play + optional supervised data)
  3. Evaluate the trained model vs current `best` over K games
  4. If trained model wins >= win_threshold, promote it to the new `best`

Layout:
    workdir/
      best.pth                       current strongest model
      iter_001/
        selfplay.pt                  iteration 1's self-play data
        ckpt/                        training output (model_last, model_best_*)
        match.log                    eval match output
      iter_002/...

Each step shells out to the existing scripts (selfplay.py, train.py,
match.py) so they stay independently testable. Logs from each step
go to stdout and into per-iteration files.

Usage:
    uv run python runner.py \\
        --workdir runs/r1 \\
        --initial-checkpoint models/model_best.pth \\
        --iters 5 --selfplay-games 50 --sims 200 \\
        --supervised-config train_config.json \\
        --eval-games 10

`--supervised-config` is optional. If provided, train.py also pulls in
supervised data each iteration (data_mix=both). Without it, data_mix
defaults to self_play.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], log_path: str | None = None) -> int:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}")
    if log_path:
        with open(log_path, "w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd)
    return proc.returncode


def latest_selfplay_files(workdir: Path, keep: int) -> list[str]:
    """Return up to `keep` most-recent iter_*/selfplay.pt paths."""
    iters = sorted(workdir.glob("iter_*/selfplay.pt"))
    return [str(p) for p in iters[-keep:]]


def parse_match_result(match_log: str) -> tuple[int, int, int, float]:
    """Extract (wins, draws, losses, win_rate) from a match.py log."""
    with open(match_log) as fh:
        text = fh.read()
    wins = draws = losses = 0
    for line in text.splitlines():
        s = line.strip()
        if "W /" in s and "D /" in s and "L" in s:
            # e.g. "candidate: 7 W / 2 D / 1 L"
            try:
                parts = s.split(":", 1)[1].strip()
                bits = parts.replace(" ", "").split("/")
                wins = int(bits[0].rstrip("W"))
                draws = int(bits[1].rstrip("D"))
                losses = int(bits[2].rstrip("L"))
                break
            except Exception:
                pass
    total = wins + draws + losses
    win_rate = (wins + 0.5 * draws) / total if total else 0.0
    return wins, draws, losses, win_rate


def write_match_config(path: Path, checkpoint: str, sims: int, top_actions: int,
                       temperature_moves: int, temperature: float) -> None:
    cfg = {
        "type": "mcts",
        "checkpoint": checkpoint,
        "num_simulation": sims,
        "top_actions": top_actions,
        "c_init": 1.25,
        "c_base": 19652,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.03,
        "t": 1,
        "temperature_moves": temperature_moves,
        "temperature": temperature,
    }
    path.write_text(json.dumps(cfg, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, help="run directory (will be created)")
    p.add_argument("--initial-checkpoint", required=True, help="starting .pth file")
    p.add_argument("--iters", type=int, default=5)

    # Self-play options
    p.add_argument("--selfplay-games", type=int, default=50)
    p.add_argument("--sims", type=int, default=200, help="MCTS sims per move during self-play")
    p.add_argument("--top-actions", type=int, default=10)
    p.add_argument("--temperature-moves", type=int, default=30)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--selfplay-truncation", type=int, default=300)

    # Training options
    p.add_argument("--epochs", type=int, default=3, help="epochs per iteration's training pass")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--l2-weight", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--supervised-config", default=None,
                   help="optional train_config.json for mixing in supervised data")
    p.add_argument("--keep-selfplay-iters", type=int, default=3,
                   help="how many of the most recent self-play files to use for training")

    # Evaluation gate options
    p.add_argument("--eval-games", type=int, default=10, help="games in the gate match")
    p.add_argument("--eval-sims", type=int, default=200, help="MCTS sims during evaluation match")
    p.add_argument("--win-threshold", type=float, default=0.55,
                   help="min win-rate to promote candidate to new best")
    p.add_argument("--eval-truncation", type=int, default=200)
    cli = p.parse_args()

    workdir = Path(cli.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Establish the current `best` model: copy the initial checkpoint in.
    best_path = workdir / "best.pth"
    if not best_path.exists():
        shutil.copy(cli.initial_checkpoint, best_path)
        print(f"Seeded best = {cli.initial_checkpoint}")
    else:
        print(f"Resuming with existing best = {best_path}")

    python = sys.executable

    for it in range(1, cli.iters + 1):
        iter_dir = workdir / f"iter_{it:03d}"
        iter_dir.mkdir(exist_ok=True)
        print(f"\n========== Iteration {it}/{cli.iters} (workdir: {iter_dir}) ==========")

        # ---- 1. Self-play ----
        selfplay_out = iter_dir / "selfplay.pt"
        if not selfplay_out.exists():
            t0 = time.time()
            rc = run([
                python, "selfplay.py",
                "--checkpoint", str(best_path),
                "--games", str(cli.selfplay_games),
                "--sims", str(cli.sims),
                "--top-actions", str(cli.top_actions),
                "--temperature-moves", str(cli.temperature_moves),
                "--dirichlet-eps", str(cli.dirichlet_eps),
                "--truncation", str(cli.selfplay_truncation),
                "--output", str(selfplay_out),
            ], log_path=str(iter_dir / "selfplay.log"))
            if rc != 0:
                raise RuntimeError(f"selfplay.py failed (rc={rc}); see {iter_dir / 'selfplay.log'}")
            print(f"Self-play done in {time.time() - t0:.0f}s")
        else:
            print(f"(skipping self-play; {selfplay_out} already exists)")

        # ---- 2. Train ----
        ckpt_dir = iter_dir / "ckpt"
        ckpt_dir.mkdir(exist_ok=True)
        log_dir = iter_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        recent_sp = latest_selfplay_files(workdir, cli.keep_selfplay_iters)
        train_cmd = [
            python, "train.py",
            "--epochs", str(cli.epochs),
            "--batch-size", str(cli.batch_size),
            "--learning-rate", str(cli.learning_rate),
            "--l2-weight", str(cli.l2_weight),
            "--label-smoothing", str(cli.label_smoothing),
            "--checkpoint-dir", str(ckpt_dir),
            "--log-dir", str(log_dir),
            "--resume", str(best_path),
        ]
        for sp in recent_sp:
            train_cmd += ["--self-play-data", sp]
        if cli.supervised_config:
            train_cmd += ["--config", cli.supervised_config, "--data-mix", "both"]
        else:
            train_cmd += ["--data-mix", "self_play"]

        t0 = time.time()
        rc = run(train_cmd, log_path=str(iter_dir / "train.log"))
        if rc != 0:
            raise RuntimeError(f"train.py failed (rc={rc}); see {iter_dir / 'train.log'}")
        print(f"Training done in {time.time() - t0:.0f}s")

        # Candidate model: the "last" checkpoint -- with no val available in
        # pure self-play, "best" by-loss is not meaningful, so we use the
        # final-epoch weights and let the evaluation gate decide.
        candidate_path = ckpt_dir / "model_last.pth"
        if not candidate_path.exists():
            raise RuntimeError(f"Training produced no model_last.pth in {ckpt_dir}")

        # ---- 3. Evaluation gate ----
        cand_cfg = iter_dir / "candidate.json"
        best_cfg = iter_dir / "best.json"
        write_match_config(cand_cfg, str(candidate_path), cli.eval_sims, cli.top_actions, 20, 1.0)
        write_match_config(best_cfg, str(best_path), cli.eval_sims, cli.top_actions, 20, 1.0)

        match_log = iter_dir / "match.log"
        t0 = time.time()
        rc = run([
            python, "match.py",
            "--player1", str(cand_cfg),
            "--player2", str(best_cfg),
            "--games", str(cli.eval_games),
            "--truncation", str(cli.eval_truncation),
            "--p1-name", "candidate",
            "--p2-name", "best",
            "--output", str(iter_dir / "match.pgn"),
        ], log_path=str(match_log))
        if rc != 0:
            raise RuntimeError(f"match.py failed (rc={rc}); see {match_log}")
        wins, draws, losses, win_rate = parse_match_result(str(match_log))
        print(f"Match done in {time.time() - t0:.0f}s -- candidate "
              f"{wins}W/{draws}D/{losses}L  win-rate={win_rate:.2%}")

        # ---- 4. Promote? ----
        if win_rate >= cli.win_threshold:
            shutil.copy(candidate_path, best_path)
            print(f"PROMOTED: win_rate {win_rate:.2%} >= {cli.win_threshold:.0%} "
                  f"-> new best = {best_path}")
        else:
            print(f"REJECTED: win_rate {win_rate:.2%} < {cli.win_threshold:.0%} -- "
                  f"keeping previous best")

    print(f"\n=== Done after {cli.iters} iterations ===")
    print(f"Final best: {best_path}")


if __name__ == "__main__":
    main()
