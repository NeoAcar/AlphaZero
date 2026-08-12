"""Compare value-head legal-move rankings with Stockfish on one PGN game.

For every position on the selected game's main line, all legal child positions
are scored by both neural value heads.  Stockfish evaluates the same root moves
with MultiPV.  Metrics are tie-aware top-1 agreement, pairwise ranking accuracy,
centipawn regret, and the largest relative overvaluations.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from alphazero import utils as f
from alphazero.mcts import _amp_ctx
from alphazero.nn import value_to_scalar
from alphazero.players import _channels_last_input, _load_model


MATE_CP = 100_000


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_game(path: str, game_number: int) -> chess.pgn.Game:
    with open(path, encoding="utf-8") as fh:
        for number in range(1, game_number + 1):
            game = chess.pgn.read_game(fh)
            if game is None:
                raise ValueError(f"{path} contains fewer than {game_number} games")
    if game.errors:
        raise ValueError(f"PGN parse errors: {game.errors}")
    return game


class ValueModel:
    def __init__(self, config_path: str, device: torch.device):
        self.cfg = load_config(config_path)
        self.model = _load_model(self.cfg, device)
        self.device = device
        self.in_channels = int(self.cfg.get("_in_channels", 19))
        self.scalar_mode = self.cfg.get("value_scalar", "expected")

    @torch.inference_mode()
    def mover_values(self, post_states: list[chess.Board], ply: int) -> np.ndarray:
        history = [] if self.in_channels == 119 else None
        batch = torch.stack([
            f.prepare_input(board, ply + 1, history=history) for board in post_states
        ]).to(self.device)
        batch = _channels_last_input(batch)
        with _amp_ctx(batch.device):
            opponent_logits, _ = self.model(batch)
        opponent_values = value_to_scalar(
            opponent_logits.float(), mode=self.scalar_mode
        )
        return -opponent_values.cpu().numpy().reshape(-1)


def canonical_children(real_board: chess.Board, canonical: chess.Board):
    mover_is_white = real_board.turn == chess.WHITE
    moves = list(real_board.legal_moves)
    children = []
    for move in moves:
        uci = move.uci()
        canonical_uci = uci if mover_is_white else f.mirror_move(uci)
        child = canonical.copy(stack=False)
        child.push_uci(canonical_uci)
        child.apply_mirror()
        children.append(child)
    return moves, children


def sf_all_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
) -> dict[str, tuple[int, float]]:
    mover = board.turn
    legal_count = board.legal_moves.count()
    infos = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=legal_count,
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    result: dict[str, tuple[int, float]] = {}
    for info in infos:
        if not info.get("pv") or "score" not in info:
            continue
        move = info["pv"][0].uci()
        score = info["score"].pov(mover)
        cp = score.score(mate_score=MATE_CP)
        # Stockfish's UCI WDL is preferable because it shares the NN's [-1, 1]
        # expected-score scale.  Fall back to a smooth CP conversion if absent.
        if "wdl" in info:
            wdl = info["wdl"].pov(mover)
            expected = (wdl.wins - wdl.losses) / 1000.0
        else:
            expected = float(np.tanh(cp / 400.0))
        result[move] = (int(cp), float(expected))
    return result


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--game", type=int, default=1, help="1-based game number")
    parser.add_argument("--new-config", required=True)
    parser.add_argument("--old-config", required=True)
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--sf-depth", type=int, default=1)
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.sf_depth < 1:
        parser.error("Stockfish depth must be >= 1; depth=0 has no stopping condition")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = load_game(args.pgn, args.game)
    played_moves = list(game.mainline_moves())
    print(f"Device: {device}; game {args.game}: {game.headers.get('White')} - "
          f"{game.headers.get('Black')} ({game.headers.get('Result')}); "
          f"{len(played_moves)} plies")
    print(f"Stockfish depth={args.sf_depth}; loading models...")
    models = {
        "new": ValueModel(args.new_config, device),
        "old": ValueModel(args.old_config, device),
    }

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    if "UCI_ShowWDL" in engine.options:
        engine.configure({"UCI_ShowWDL": True})

    aggregate = {
        name: {
            "positions": 0,
            "top1": 0,
            "pairs_correct": 0,
            "pairs_total": 0,
            "regrets": [],
            "chosen_relative_overvalues": [],
            "worst_children": [],
        }
        for name in models
    }
    board = game.board()
    canonical = chess.Board()
    try:
        for ply, played_move in enumerate(played_moves):
            moves, children = canonical_children(board, canonical)
            sf = sf_all_moves(engine, board, args.sf_depth)
            missing = [move.uci() for move in moves if move.uci() not in sf]
            if missing:
                raise RuntimeError(
                    f"Stockfish omitted {len(missing)} legal moves at ply {ply + 1}: {missing}"
                )
            cp = np.asarray([sf[move.uci()][0] for move in moves], dtype=np.int32)
            sf_ev = np.asarray([sf[move.uci()][1] for move in moves], dtype=np.float64)
            sf_best_cp = int(cp.max())
            sf_best_indices = np.flatnonzero(cp == sf_best_cp)

            for name, model in models.items():
                values = model.mover_values(children, ply)
                chosen = int(np.argmax(values))
                best_sf_for_model = int(sf_best_indices[np.argmax(values[sf_best_indices])])
                regret = sf_best_cp - int(cp[chosen])

                correct = total = 0
                for i, j in itertools.combinations(range(len(moves)), 2):
                    sf_delta = int(cp[i]) - int(cp[j])
                    if sf_delta == 0:
                        continue
                    total += 1
                    model_delta = float(values[i]) - float(values[j])
                    correct += int(model_delta * sf_delta > 0)

                # Relative optimism versus Stockfish's best move. This removes
                # per-position calibration offsets and measures ranking error.
                relative_over = (
                    (values - values[best_sf_for_model])
                    - (sf_ev - sf_ev[best_sf_for_model])
                )
                a = aggregate[name]
                a["positions"] += 1
                a["top1"] += int(cp[chosen] == sf_best_cp)
                a["pairs_correct"] += correct
                a["pairs_total"] += total
                a["regrets"].append(float(regret))
                a["chosen_relative_overvalues"].append(float(relative_over[chosen]))

                for idx in np.argsort(relative_over)[-args.worst:]:
                    a["worst_children"].append({
                        "overvalue": float(relative_over[idx]),
                        "ply": ply + 1,
                        "move_number": f"{board.fullmove_number}{'.' if board.turn else '...'}",
                        "fen": board.fen(),
                        "move": moves[idx].uci(),
                        "played": played_move.uci(),
                        "model_value": float(values[idx]),
                        "sf_expected": float(sf_ev[idx]),
                        "sf_cp": int(cp[idx]),
                        "sf_best_moves": [moves[i].uci() for i in sf_best_indices],
                        "sf_best_cp": sf_best_cp,
                    })

            real_uci = played_move.uci()
            canonical_uci = real_uci if board.turn == chess.WHITE else f.mirror_move(real_uci)
            board.push(played_move)
            canonical.push_uci(canonical_uci)
            canonical = canonical.mirror()
            if (ply + 1) % 10 == 0 or ply + 1 == len(played_moves):
                print(f"  {ply + 1}/{len(played_moves)} plies analysed")
    finally:
        engine.quit()

    output = {"game": dict(game.headers), "sf_depth": args.sf_depth, "models": {}}
    print("\n=== Results ===")
    for name, a in aggregate.items():
        positions = a["positions"]
        regrets = a["regrets"]
        worst = sorted(a["worst_children"], key=lambda row: row["overvalue"], reverse=True)
        # A move appears once per position in the retained list; keep the global top N.
        worst = worst[:args.worst]
        result = {
            "positions": positions,
            "top1_agreement": a["top1"] / positions,
            "pairwise_accuracy": a["pairs_correct"] / a["pairs_total"],
            "pair_count": a["pairs_total"],
            "cp_regret_mean": float(np.mean(regrets)),
            "cp_regret_median": percentile(regrets, 50),
            "cp_regret_p90": percentile(regrets, 90),
            "cp_regret_max": float(np.max(regrets)),
            "chosen_relative_overvalue_mean": float(np.mean(a["chosen_relative_overvalues"])),
            "worst_overvaluations": worst,
        }
        output["models"][name] = result
        print(f"\n{name.upper()}: top-1 {100 * result['top1_agreement']:.2f}% | "
              f"pairwise {100 * result['pairwise_accuracy']:.2f}% "
              f"({result['pair_count']:,} pairs)")
        print(f"  CP regret mean/median/p90/max: {result['cp_regret_mean']:.1f} / "
              f"{result['cp_regret_median']:.1f} / {result['cp_regret_p90']:.1f} / "
              f"{result['cp_regret_max']:.1f}")
        print("  Worst relative overvaluations:")
        for row in worst:
            print(f"    ply {row['ply']:>3} {row['move_number']:<4} {row['move']:<5} "
                  f"over={row['overvalue']:+.3f} model={row['model_value']:+.3f} "
                  f"SF={row['sf_expected']:+.3f}/{row['sf_cp']:+d}cp "
                  f"best={','.join(row['sf_best_moves'])}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nDetailed JSON: {output_path}")


if __name__ == "__main__":
    main()
