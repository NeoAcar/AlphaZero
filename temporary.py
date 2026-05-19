"""
Scratch benchmark: avg MCTS move time vs num_simulation.
Plays a fresh argmax game at each sim count, times each move, plots mean ± std.
Delete this file (and sim_scaling.png) once you've eyeballed the curve.

Usage:
    uv run python temporary.py
"""
import time

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch

from alphazero import utils as f
from alphazero.mcts import MCTS
from alphazero.nn import ResNet


CHECKPOINT = "models/model_5.pth"
SIM_COUNTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
MOVES_PER_POINT = 10   # timed moves averaged per sim count
WARMUP_MOVES = 2       # untimed moves first (covers torch.compile JIT etc.)
TRUNCATION = 1000


def build_mcts(model, device, num_sims):
    return MCTS({
        "num_simulation": num_sims,
        "truncation": TRUNCATION,
        "c_base": 19652,
        "c_init": 1.25,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.3,
        "memory_size": 1000,
        "action_space": 4672,
        "t": 1,
        "device": device,
    }, model)


def time_moves(model, device, num_sims, n_moves, warmup):
    mcts = build_mcts(model, device, num_sims)
    state = chess.Board()
    move_counter = 0
    times = []
    total = warmup + n_moves
    i = 0
    while i < total:
        if f.game_result(state, move_counter, TRUNCATION)[1]:
            # Game ended; restart with a fresh tree so we keep collecting samples.
            state = chess.Board()
            move_counter = 0
            mcts = build_mcts(model, device, num_sims)
        t0 = time.perf_counter()
        probs = mcts.search(state, move_counter)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
        action = int(np.argmax(probs))
        uci = f.alphazero_to_move(action, state)
        state.push_uci(uci)
        state.apply_mirror()
        move_counter += 1
        i += 1
    return float(np.mean(times)), float(np.std(times))


def main():
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = ResNet().to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    try:
        model = torch.compile(model, mode="reduce-overhead")
        with torch.inference_mode():
            _ = model(torch.zeros(1, 19, 8, 8, device=device))
        print("torch.compile + warm-up done")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    means, stds = [], []
    for n in SIM_COUNTS:
        m, s = time_moves(model, device, n, MOVES_PER_POINT, WARMUP_MOVES)
        print(f"sims={n:>4}: {m*1000:7.1f} +/- {s*1000:.1f} ms/move")
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(SIM_COUNTS, [m * 1000 for m in means],
                yerr=[s * 1000 for s in stds], marker="o", capsize=4)
    ax.set_xlabel("MCTS simulations per move")
    ax.set_ylabel("avg time per move (ms)")
    ax.set_title(f"MCTS compute time vs num_simulation ({device.type})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "sim_scaling.png"
    fig.savefig(out, dpi=120)
    print(f"saved plot to {out}")


if __name__ == "__main__":
    main()
