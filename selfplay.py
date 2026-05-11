"""
Generate self-play games for AlphaZero training.

Each move:
  * MCTS runs with Dirichlet noise at root (for exploration).
  * For the first `temperature_moves` plies, action is sampled from the
    visit-count distribution scaled by `1/temperature`. After that, argmax.

Per position, we record:
  * board_matrix (19, 8, 8) -- the mirror-canonical board representation
  * pi          (4672,)    -- MCTS visit-count distribution (sums to 1)
  * z           scalar     -- the eventual game outcome from THIS player's
                              perspective: +1 if they won, -1 if lost, 0 if draw

Output is a .pt file:
    {"boards": Tensor(N,19,8,8), "pis": Tensor(N,4672), "values": Tensor(N,1)}

Usage:
    uv run python selfplay.py \
        --checkpoint models/model_best.pth \
        --games 100 \
        --sims 200 \
        --output selfplay_data/iter_0.pt
"""
import argparse
import json
import os
import time
from pathlib import Path

import chess
import numpy as np
import torch

import optimized_functions as f
from ithinkbettermcts import MCTS
from resnet import ResNet


DEFAULT_MCTS_ARGS = {
    "c_base": 19652,
    "c_init": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.03,
    "memory_size": 1000,
    "action_space": 4672,
    "top_actions": 10,
    "t": 1,
}


def build_mcts(checkpoint: str, sims: int, top_actions: int,
               dirichlet_eps: float) -> tuple[MCTS, ResNet, dict]:
    args = dict(DEFAULT_MCTS_ARGS)
    args["num_simulation"] = sims
    args["top_actions"] = top_actions
    args["dirichlet_epsilon"] = dirichlet_eps
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args["truncation"] = 300

    model = ResNet().to(args["device"])
    state = torch.load(checkpoint, map_location=args["device"], weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return MCTS(args, model), model, args


def sample_action(pi: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(pi))
    scaled = np.where(pi > 0, pi ** (1.0 / temperature), 0.0)
    total = scaled.sum()
    if total <= 0:
        return int(np.argmax(pi))
    return int(rng.choice(len(scaled), p=scaled / total))


def play_one_game(mcts: MCTS, temperature_moves: int, temperature: float,
                  truncation: int, rng: np.random.Generator
                  ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Run one self-play game. Returns (boards, pis, outcome_from_p0_perspective).

    Per ply we record:
      - board_matrix at that ply
      - pi (MCTS visit distribution)
    Players alternate; we'll compute per-position z based on who was to move.
    """
    mcts.root = None
    mirrored_state = chess.Board()
    move_counter = 0

    boards: list[np.ndarray] = []
    pis: list[np.ndarray] = []

    while not f.game_result(mirrored_state, move_counter, truncation)[1]:
        boards.append(f.board_to_matrix(mirrored_state, move_counter))
        pi = mcts.search(mirrored_state, move_counter)
        pis.append(pi.astype(np.float32))

        if move_counter < temperature_moves:
            action = sample_action(pi, temperature, rng)
        else:
            action = int(np.argmax(pi))

        uci_mirrored = f.alphazero_to_move(action, mirrored_state)
        mirrored_state.push_uci(uci_mirrored)
        mirrored_state = mirrored_state.mirror()
        move_counter += 1

    # game_result()[0] is from the perspective of the player to move in
    # mirrored_state right now (i.e. the next-to-move at the time the loop
    # exited). That's the player whose turn it WOULD have been -- equivalently
    # the parity of move_counter.
    final_value, _ = f.game_result(mirrored_state, move_counter, truncation)
    # final_value: -1 means "the next-to-move lost", 0 draw, +1 not really
    # produced by game_result (it returns -1 for mated, 0 for draw, never +1).
    # So we use parity:
    #   If next-to-move is the same player as ply k -> z[k] = final_value
    #   Otherwise z[k] = -final_value
    return boards, pis, int(final_value), move_counter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to .pth model")
    p.add_argument("--games", type=int, default=100, help="games to generate")
    p.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    p.add_argument("--top-actions", type=int, default=10, help="MCTS expansion width")
    p.add_argument("--dirichlet-eps", type=float, default=0.25, help="exploration noise at root")
    p.add_argument("--temperature-moves", type=int, default=30, help="plies of stochastic sampling")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--truncation", type=int, default=300, help="max plies before draw")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for action sampling")
    p.add_argument("--output", required=True, help="output .pt file")
    cli = p.parse_args()

    if torch.cuda.is_available():
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")

    print(f"Loading {cli.checkpoint}")
    mcts, _, args = build_mcts(cli.checkpoint, cli.sims, cli.top_actions, cli.dirichlet_eps)
    rng = np.random.default_rng(cli.seed)

    all_boards: list[np.ndarray] = []
    all_pis: list[np.ndarray] = []
    all_values: list[float] = []
    stats = {"wins_white": 0, "wins_black": 0, "draws": 0, "truncated": 0, "total_plies": 0}

    print(f"\nGenerating {cli.games} games (sims={cli.sims}, temp_moves={cli.temperature_moves})\n")

    t_total = time.time()
    for g in range(cli.games):
        t0 = time.time()
        boards, pis, final_value, plies = play_one_game(
            mcts, cli.temperature_moves, cli.temperature, cli.truncation, rng
        )
        dt = time.time() - t0

        # Compute z for each ply. final_value is from the perspective of the
        # next-to-move at game end. Plies 0, 2, 4, ... had player A to move;
        # plies 1, 3, 5, ... had player B. The "next-to-move at game end" had
        # the same parity as `plies` would have (i.e. if `plies` plies were
        # made, the next-to-move has parity (plies) % 2). So:
        next_parity = plies % 2
        zs = []
        for k in range(plies):
            if (k % 2) == next_parity:
                zs.append(float(final_value))
            else:
                zs.append(float(-final_value))
        # Note: final_value is from mirror-canonical "player-to-move at end"
        # perspective. Because both bots play in the same mirrored frame,
        # the perspective math is uniform.

        all_boards.extend(boards)
        all_pis.extend(pis)
        all_values.extend(zs)

        if final_value == -1:
            # Player to move at end lost. They had parity `next_parity`.
            # In the mirror-canonical frame, plays 0/2/4 are white-mirror;
            # 1/3/5 are post-mirror i.e. black's real move. But since both
            # sides see "white to move" in their mirror, we just track which
            # ply-parity won.
            if next_parity == 0:
                stats["wins_black"] += 1
            else:
                stats["wins_white"] += 1
        else:
            if plies >= cli.truncation:
                stats["truncated"] += 1
            else:
                stats["draws"] += 1
        stats["total_plies"] += plies

        print(f"Game {g+1:>3}/{cli.games}: {plies:>3} plies, "
              f"final_value={final_value:+d}  ({dt:5.1f}s)")

    print(f"\nTotal: {time.time() - t_total:.1f}s for {cli.games} games "
          f"({stats['total_plies']} positions, "
          f"avg {stats['total_plies'] / cli.games:.1f} plies/game)")
    print(f"  wins(parity-1) {stats['wins_white']}, "
          f"wins(parity-0) {stats['wins_black']}, "
          f"draws {stats['draws']}, truncated {stats['truncated']}")

    out_dir = os.path.dirname(cli.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    boards_t = torch.from_numpy(np.stack(all_boards)).to(torch.float32)
    pis_t = torch.from_numpy(np.stack(all_pis)).to(torch.float32)
    values_t = torch.tensor(all_values, dtype=torch.float32).reshape(-1, 1)

    payload = {
        "boards": boards_t,
        "pis": pis_t,
        "values": values_t,
        "meta": {
            "checkpoint": cli.checkpoint,
            "games": cli.games,
            "sims": cli.sims,
            "temperature_moves": cli.temperature_moves,
            "temperature": cli.temperature,
            "dirichlet_eps": cli.dirichlet_eps,
            "truncation": cli.truncation,
            **stats,
        },
    }
    torch.save(payload, cli.output)
    print(f"Saved {len(all_boards)} positions to {cli.output} "
          f"({os.path.getsize(cli.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
