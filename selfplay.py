"""Generate self-play games for AlphaZero training.

This is the laptop-side game generator. Each invocation is one *session*
that produces fragments under `data/selfplay/<checkpoint_name>/`. Sessions
are arbitrarily small (10 games, 50 games, 100 games) -- a later Colab
training run merges all fragments for a checkpoint and trains on them.

Implements the following:

* **PCR (Playout Cap Randomization)** -- KataGo-style. With probability
  `high_prob`, run `high_sims` simulations; otherwise `low_sims`. Only
  high-sim positions are used for policy training (the `is_high_sim` flag
  rides with each position). Value training uses all positions. No
  Dirichlet noise on low-sim moves.

* **LC0-style termination** -- resign threshold + playthrough sampling +
  hard ply cap. Some fraction of games disable resign so the threshold can
  be calibrated (false-positive rate tracked in analytics later).

* **PGN export** alongside the `.pt` -- human-readable game records.

* **Sparse pi storage** -- only legal/visited action indices + values are
  stored, padded to MAX_LEGAL. Cuts per-position size ~10x.

* **119-plane history** -- self-play maintains the per-game board history
  and feeds it into MCTS. Auto-detects 19 vs 119 from the checkpoint.

Usage:
    uv run python selfplay.py \\
        --checkpoint models/v00_seed.pth \\
        --games 50 \\
        --output-dir data/selfplay/
"""
import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from tqdm import tqdm  # type: ignore

torch.set_float32_matmul_precision("high")

from alphazero import utils as f
from alphazero.batched_mcts import BatchedMCTS as MCTS
from alphazero.nn import (
    INPUT_PLANES_HISTORY,
    ResNet,
    SEResNet,
    SEResNetWDL,
    detect_in_channels,
    value_to_scalar,
)


ARCHITECTURES = {
    "resnet":      ResNet,
    "seresnet":    SEResNet,
    "seresnetwdl": SEResNetWDL,
}


# Max sparse pi entries kept per position. Visit counts at high_sims=1200 with
# batch=8 reach ~40-50 distinct children in worst case; 64 has slack. If a
# position somehow has more, we keep top-K by visit and renormalize.
MAX_LEGAL = 64


DEFAULT_MCTS_ARGS = {
    "c_base": 19652,
    "c_init": 1.25,
    "c_fpu":  0.2,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha":   0.3,
    "memory_size":  1000,
    "action_space": 4672,
    "t": 1,
    # Self-play runs the FULL visit budget every move -- the visit distribution
    # is the policy target, so no proven-mate instant-stop and no early-stop
    # (early_stop is also off by default; spelled out here for clarity).
    "mate_stop":  False,
    "early_stop": False,
}


# ---------- checkpoint loading + MCTS bootstrap -------------------------------

def load_checkpoint(path: str, device: torch.device,
                    arch_override: str | None) -> tuple[torch.nn.Module, str, int]:
    """Loads a checkpoint, detecting architecture (from meta or override) and
    in_channels (from first-conv weight shape). Returns (model, arch_name,
    in_channels)."""
    state = torch.load(path, map_location=device, weights_only=False)
    in_ch = detect_in_channels(state)
    meta = state.get("meta", {})
    arch_name = (arch_override or meta.get("architecture") or "seresnetwdl").lower()
    if arch_name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {arch_name!r}; choices: {list(ARCHITECTURES)}"
        )
    cls = ARCHITECTURES[arch_name]
    model = cls(in_channels=in_ch).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, arch_name, in_ch


def build_mcts(model: torch.nn.Module, in_channels: int, batch_size: int,
               device: torch.device) -> MCTS:
    args = dict(DEFAULT_MCTS_ARGS)
    args["device"]        = device
    args["batch_size"]    = batch_size
    args["truncation_halfmoves"] = 1000        # we apply our own max_plies cap
    args["input_planes"]  = in_channels
    args["num_simulation"] = 1                  # placeholder; overridden per-move
    return MCTS(args, model)


def checkpoint_sha(path: str) -> str:
    """Short hash of a checkpoint's bytes -- so we can tag every session with
    the exact weights that produced it. SHA1 truncated for readability."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:12]


def checkpoint_dir_name(path: str) -> str:
    """Directory name under data/selfplay/ for this checkpoint. Uses the
    checkpoint's stem (e.g. 'v00_seed') so sessions group by version."""
    return Path(path).stem


# ---------- per-position sparse-pi encoding -----------------------------------

def encode_pi_sparse(pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices: int16[MAX_LEGAL], values: float16[MAX_LEGAL]).
    Padded with index=-1 and value=0 for empty slots. If a position has more
    than MAX_LEGAL nonzero entries (shouldn't happen with our sim caps), we
    keep the top-MAX_LEGAL by visit and renormalize."""
    nz = np.flatnonzero(pi)
    vals = pi[nz]
    if len(nz) > MAX_LEGAL:
        top = np.argpartition(-vals, MAX_LEGAL - 1)[:MAX_LEGAL]
        nz = nz[top]
        vals = vals[top]
        # Re-normalize after dropping low-mass moves.
        s = vals.sum()
        if s > 0:
            vals = vals / s
    idx = np.full(MAX_LEGAL, -1, dtype=np.int16)
    val = np.zeros(MAX_LEGAL, dtype=np.float16)
    k = min(len(nz), MAX_LEGAL)
    idx[:k] = nz[:k].astype(np.int16)
    val[:k] = vals[:k].astype(np.float16)
    return idx, val


# ---------- action sampling ---------------------------------------------------

def sample_action(pi: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(pi))
    scaled = np.where(pi > 0, pi ** (1.0 / temperature), 0.0)
    total = scaled.sum()
    if total <= 0:
        return int(np.argmax(pi))
    return int(rng.choice(len(scaled), p=scaled / total))


# ---------- one self-play game ------------------------------------------------

def root_value_from_pov(mcts: MCTS) -> float:
    """Mean value at the root from the side-to-move's POV, in [-1, +1].
    Used by the resign-threshold check after a high-sim search."""
    root = mcts.root
    if root is None or root.N == 0:
        return float(root.raw_nn_value) if (root and root.raw_nn_value is not None) else 0.0
    # root.Q is accumulated with the leaf-perspective alternating-sign rule, so
    # for the root specifically `root.Q / root.N` is in the "grandparent" frame
    # (which doesn't exist); negating gives the root-player perspective.
    return -root.Q / root.N


def play_one_game(mcts: MCTS, in_channels: int, cli, rng: np.random.Generator,
                  allow_resign: bool):
    """Run one self-play game. Returns:
        positions: list of per-position dicts
        plies:     number of plies played
        final_value: from the perspective of the side-to-move at game end
        reason:    "checkmate" | "draw_rule" | "resign" | "truncation"
        real_moves: list of real-frame UCI moves (for PGN export)
    """
    mcts.root = None
    mirrored_state = chess.Board()
    move_counter = 0
    rep_counter: dict = {mirrored_state._transposition_key(): 1}
    history: list = []                # canonical board history, oldest first
    consecutive_low = 0

    positions: list = []
    real_moves: list = []

    use_history = (in_channels == 119)

    while True:
        # Terminal check (checkmate / stalemate / insufficient / 50-move /
        # 3-fold / hard ply cap).
        rep_now = rep_counter.get(mirrored_state._transposition_key(), 0)
        val, terminal = f.game_result(
            mirrored_state, move_counter, cli.max_plies, rep_now,
        )
        if terminal:
            # Categorise the reason for analytics.
            if mirrored_state.is_checkmate():
                reason = "checkmate"
            elif move_counter >= cli.max_plies:
                reason = "truncation"
            elif rep_now >= 3:
                reason = "3-fold"
            elif mirrored_state.is_fifty_moves():
                reason = "50-move"
            else:
                reason = "draw_rule"
            return positions, move_counter, val, reason, real_moves

        # ---- PCR: pick sim count for this move ------------------------------
        use_high = rng.random() < cli.high_prob
        if use_high:
            mcts.args["num_simulation"]    = cli.high_sims
            mcts.args["dirichlet_epsilon"] = cli.dirichlet_eps
        else:
            mcts.args["num_simulation"]    = cli.low_sims
            mcts.args["dirichlet_epsilon"] = 0.0    # no noise on low-sim

        # ---- search ---------------------------------------------------------
        mcts.set_rep_counter(rep_counter)
        if use_history:
            mcts.set_history(history)
        pi = mcts.search(mirrored_state, move_counter)

        # ---- record position ------------------------------------------------
        # rep_now is the count INCLUDING the current occurrence (always >= 1).
        # AZ's frame-0 rep planes fire at >=2 (seen-before) and >=3 (3-fold).
        board_planes = f.board_to_matrix(
            mirrored_state, move_counter,
            history=(history if use_history else None),
            rep_count=max(rep_now, 1),
        )
        idx, val_arr = encode_pi_sparse(pi)
        positions.append({
            "board":        board_planes,              # float32; cast to uint8 later
            "pi_idx":       idx,
            "pi_val":       val_arr,
            "is_high_sim":  np.uint8(1 if use_high else 0),
        })

        # ---- LC0-style resign check (high-sim only) -------------------------
        if allow_resign and use_high:
            v = root_value_from_pov(mcts)
            if v < cli.resign_threshold:
                consecutive_low += 1
                if consecutive_low >= cli.resign_consecutive:
                    # Side-to-move resigns -> their loss.
                    return positions, move_counter, -1, "resign", real_moves
            else:
                consecutive_low = 0
        else:
            consecutive_low = 0

        # ---- play the move --------------------------------------------------
        if move_counter < cli.temperature_moves:
            action = sample_action(pi, cli.temperature, rng)
        else:
            action = int(np.argmax(pi))

        uci_mirrored = f.alphazero_to_move(action, mirrored_state)
        mover_was_white = (move_counter % 2 == 0)
        uci_real = uci_mirrored if mover_was_white else f.mirror_move(uci_mirrored)
        real_moves.append(uci_real)

        # Maintain canonical history: append the BEFORE-push state.
        if use_history:
            history.append(mirrored_state.copy())
            if len(history) > 7:
                history.pop(0)

        mirrored_state.push_uci(uci_mirrored)
        mirrored_state = mirrored_state.mirror()
        move_counter += 1
        tk = mirrored_state._transposition_key()
        rep_counter[tk] = rep_counter.get(tk, 0) + 1
        # O(1) tree walk for reuse on the next search call.
        mcts.apply_action(action)


def _terminal_reason(board: chess.Board, move_counter: int, max_plies: int,
                     rep_now: int) -> str:
    """Categorise a terminal position for analytics (mirrors play_one_game)."""
    if board.is_checkmate():
        return "checkmate"
    if move_counter >= max_plies:
        return "truncation"
    if rep_now >= 3:
        return "3-fold"
    if board.is_fifty_moves():
        return "50-move"
    return "draw_rule"


def play_games_multigame(engines: list, in_channels: int, cli, rng,
                         allow_resign: list[bool]) -> list:
    """Play len(engines) self-play games concurrently, coalescing every active
    game's MCTS leaf evaluations into a single NN forward per simulation round
    (via MultiGameSearcher). Returns a list of per-game result tuples in the
    same format as play_one_game: (positions, plies, final_value, reason,
    real_moves). Behaviour per game is identical to play_one_game; only the GPU
    batching differs."""
    from alphazero.batched_mcts import MultiGameSearcher
    searcher = MultiGameSearcher(engines[0].model)
    use_history = (in_channels == 119)

    games = []
    for e, ar in zip(engines, allow_resign):
        e.root = None
        b = chess.Board()
        games.append({
            "eng": e, "board": b, "mc": 0,
            "rep": {b._transposition_key(): 1}, "hist": [],
            "consec_low": 0, "positions": [], "real_moves": [],
            "allow_resign": ar, "done": False, "result": None,
        })

    while not all(g["done"] for g in games):
        # 1. Terminal check + PCR / search setup for every still-active game.
        active = []
        for g in games:
            if g["done"]:
                continue
            rep_now = g["rep"].get(g["board"]._transposition_key(), 0)
            val, terminal = f.game_result(g["board"], g["mc"], cli.max_plies, rep_now)
            if terminal:
                reason = _terminal_reason(g["board"], g["mc"], cli.max_plies, rep_now)
                g["done"] = True
                g["result"] = (g["positions"], g["mc"], val, reason, g["real_moves"])
                continue
            use_high = rng.random() < cli.high_prob
            e = g["eng"]
            e.args["num_simulation"]    = cli.high_sims if use_high else cli.low_sims
            e.args["dirichlet_epsilon"] = cli.dirichlet_eps if use_high else 0.0
            e.set_rep_counter(g["rep"])
            if use_history:
                e.set_history(g["hist"])
            g["use_high"] = use_high
            g["rep_now"] = rep_now
            active.append(g)
        if not active:
            break

        # 2. One batched search step across all active games.
        pis = searcher.search_all(
            [g["eng"] for g in active],
            [g["board"] for g in active],
            [g["mc"] for g in active],
        )

        # 3. Per-game: record position, resign check, play the sampled move.
        for g, pi in zip(active, pis):
            e = g["eng"]
            rep_now, use_high = g["rep_now"], g["use_high"]
            board_planes = f.board_to_matrix(
                g["board"], g["mc"],
                history=(g["hist"] if use_history else None),
                rep_count=max(rep_now, 1),
            )
            idx, val_arr = encode_pi_sparse(pi)
            g["positions"].append({
                "board": board_planes,
                "pi_idx": idx, "pi_val": val_arr,
                "is_high_sim": np.uint8(1 if use_high else 0),
            })

            if g["allow_resign"] and use_high:
                v = root_value_from_pov(e)
                if v < cli.resign_threshold:
                    g["consec_low"] += 1
                    if g["consec_low"] >= cli.resign_consecutive:
                        g["done"] = True
                        g["result"] = (g["positions"], g["mc"], -1, "resign", g["real_moves"])
                        continue
                else:
                    g["consec_low"] = 0
            else:
                g["consec_low"] = 0

            if g["mc"] < cli.temperature_moves:
                action = sample_action(pi, cli.temperature, rng)
            else:
                action = int(np.argmax(pi))
            uci_mirrored = f.alphazero_to_move(action, g["board"])
            mover_was_white = (g["mc"] % 2 == 0)
            uci_real = uci_mirrored if mover_was_white else f.mirror_move(uci_mirrored)
            g["real_moves"].append(uci_real)

            if use_history:
                g["hist"].append(g["board"].copy())
                if len(g["hist"]) > 7:
                    g["hist"].pop(0)

            g["board"].push_uci(uci_mirrored)
            g["board"] = g["board"].mirror()
            g["mc"] += 1
            tk = g["board"]._transposition_key()
            g["rep"][tk] = g["rep"].get(tk, 0) + 1
            e.apply_action(action)

    return [g["result"] for g in games]


# ---------- session bookkeeping + serialisation -------------------------------

def build_pgn(real_moves: list[str], headers: dict) -> str:
    """Construct a single-game PGN string from real-frame UCI moves."""
    game = chess.pgn.Game()
    for k, v in headers.items():
        game.headers[k] = str(v)
    board = chess.Board()
    node = game
    for uci in real_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break  # safety
        node = node.add_main_variation(move)
        board.push(move)
    return str(game)


def per_position_zs(plies: int, final_value: int) -> list[float]:
    """Compute z (game outcome) for each ply, from THAT ply's side-to-move POV.
    `final_value` is from the side-to-move at game end."""
    next_parity = plies % 2
    return [
        float(final_value) if (k % 2) == next_parity else float(-final_value)
        for k in range(plies)
    ]


def _save_payload(pt_path: Path, pgn_path: Path, json_path: Path,
                  session_positions: list, game_pgns: list[str],
                  session_stats: dict) -> Path:
    """Stack the session arrays and write the .pt + .pgn + .json side-cars to
    the given paths. Returns the .pt path."""
    # Stack arrays. Boards stored uint8 (4x shrink); pi sparse; values float32.
    N = len(session_positions)
    boards_u8 = np.stack([np.clip(p["board"] * 255, 0, 255).astype(np.uint8)
                          for p in session_positions])
    pi_idx    = np.stack([p["pi_idx"] for p in session_positions])
    pi_val    = np.stack([p["pi_val"] for p in session_positions])
    is_hi     = np.stack([p["is_high_sim"] for p in session_positions])
    values    = np.array(session_stats["values_per_position"], dtype=np.float32).reshape(-1, 1)

    payload = {
        "boards":       torch.from_numpy(boards_u8),           # (N, P, 8, 8) uint8
        "pi_indices":   torch.from_numpy(pi_idx),              # (N, MAX_LEGAL) int16
        "pi_values":    torch.from_numpy(pi_val),              # (N, MAX_LEGAL) float16
        "values":       torch.from_numpy(values),              # (N, 1) float32
        "is_high_sim":  torch.from_numpy(is_hi),               # (N,) uint8
        "meta": session_stats["meta"],
    }
    torch.save(payload, pt_path)

    with open(pgn_path, "w") as fh:
        for pgn in game_pgns:
            fh.write(pgn + "\n\n")

    with open(json_path, "w") as fh:
        json.dump(session_stats["meta"] | {"positions": N}, fh, indent=2)

    return pt_path


def write_session(out_dir: Path, ckpt_path: str, session_positions: list,
                  game_pgns: list[str], session_stats: dict, cli) -> Path:
    """Write a session fragment under out_dir/<checkpoint_dir>/games_*.pt
    (the default fragmented layout). Returns the .pt path."""
    ckpt_dir = out_dir / checkpoint_dir_name(ckpt_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Include microseconds + PID so multiple parallel selfplay.py processes
    # writing into the same checkpoint dir never collide on filenames.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    n_games = int(session_stats["meta"].get("games_completed", 0))
    base = f"games_{ts}_pid{os.getpid()}_n{n_games}"
    return _save_payload(
        ckpt_dir / f"{base}.pt", ckpt_dir / f"{base}.pgn", ckpt_dir / f"{base}.json",
        session_positions, game_pgns, session_stats,
    )


def write_flat(out_path: Path, session_positions: list, game_pgns: list[str],
               session_stats: dict) -> Path:
    """Write the whole session to a single .pt at exactly out_path (plus sibling
    .pgn/.json). This is the flat layout runner.py expects (iter_dir/selfplay.pt);
    selected via --output. Pair with --flush-every 0 so there's one file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _save_payload(
        out_path, out_path.with_suffix(".pgn"), out_path.with_suffix(".json"),
        session_positions, game_pgns, session_stats,
    )


# ---------- CLI + main loop ---------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="model .pth path")
    p.add_argument("--architecture", choices=list(ARCHITECTURES), default=None,
                   help="optional architecture override; otherwise read from checkpoint meta")
    p.add_argument("--games", type=int, default=10, help="games to generate this session")
    p.add_argument("--output-dir", default="data/selfplay",
                   help="root dir; a per-checkpoint subdir is created automatically")
    p.add_argument("--output", default=None,
                   help="flat mode: write the whole session to this single .pt path "
                        "(plus sibling .pgn/.json) instead of the fragmented --output-dir "
                        "layout. Used by runner.py (iter_dir/selfplay.pt). Forces a single "
                        "end-of-session write (overrides --flush-every).")

    # PCR (KataGo defaults for our scale: 1200/300, p=0.25).
    p.add_argument("--high-sims",   type=int,   default=1200)
    p.add_argument("--low-sims",    type=int,   default=300)
    p.add_argument("--high-prob",   type=float, default=0.25)
    p.add_argument("--dirichlet-eps", type=float, default=0.25,
                   help="Dirichlet noise weight at root (only on HIGH-sim moves)")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)

    # LC0-style termination.
    p.add_argument("--resign-threshold", type=float, default=-0.95,
                   help="side-to-move resigns if root value <= this for N plies in a row")
    p.add_argument("--resign-consecutive", type=int, default=4,
                   help="N plies below threshold required to trigger resign")
    p.add_argument("--resign-disabled-fraction", type=float, default=0.10,
                   help="fraction of games where resign is disabled (for threshold calibration)")
    p.add_argument("--max-plies", type=int, default=500,
                   help="hard ply cap as safety net (long tabula-rasa games)")

    # Exploration / temperature.
    p.add_argument("--temperature",       type=float, default=1.0)
    p.add_argument("--temperature-moves", type=int,   default=20,
                   help="plies of stochastic sampling at start (AZ-style)")

    p.add_argument("--batch-size", type=int, default=8, help="BatchedMCTS batch size")
    p.add_argument("--concurrent-games", type=int, default=1,
                   help="play this many games at once, coalescing all games' MCTS "
                        "leaf evals into one NN forward per round (cross-game batching). "
                        ">1 keeps the GPU busy for ~2-5x self-play throughput. Default 1.")
    p.add_argument("--flush-every", type=int, default=5,
                   help="write a fragment to disk after every N completed games. "
                        "Smaller = less work lost on Ctrl-C, more files. 0 disables "
                        "(only writes at session end). Default 5.")
    p.add_argument("--seed", type=int, default=None)
    cli = p.parse_args()

    # Flat mode accumulates everything and writes one file at the end, so
    # disable periodic fragment flushing.
    if cli.output:
        cli.flush_every = 0

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: cpu")

    print(f"Loading {cli.checkpoint}")
    model, arch, in_ch = load_checkpoint(cli.checkpoint, device, cli.architecture)
    print(f"  architecture={arch}  in_channels={in_ch}")
    cli.in_channels = in_ch                # stash for use inside play_one_game

    sha = checkpoint_sha(cli.checkpoint)
    print(f"  ckpt_sha={sha}")

    # Compile + warm-up. Real-data warm-up (not torch.zeros) to dodge the
    # first-real-forward JIT re-trace we hit earlier.
    try:
        model = torch.compile(model)
        real_one = f.prepare_input(
            chess.Board(), 0,
            history=([] if in_ch == 119 else None),
        ).unsqueeze(0).to(device)
        with torch.inference_mode():
            if cli.batch_size > 1:
                _ = model(real_one.expand(cli.batch_size, -1, -1, -1).contiguous())
            _ = model(real_one)
        print("torch.compile + warm-up done")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    mcts = build_mcts(model, in_ch, cli.batch_size, device)
    rng = np.random.default_rng(cli.seed)

    def fresh_stats() -> dict:
        return {
            "games_completed": 0, "positions_total": 0,
            "wins_white": 0, "wins_black": 0, "draws": 0,
            "truncated": 0, "resigned": 0, "checkmated": 0, "rule_draws": 0,
            "resign_disabled_games": 0,
            "high_sim_positions": 0, "low_sim_positions": 0,
        }

    # Pending fragment (everything not yet flushed to disk).
    pending_positions: list = []
    pending_zs: list[float] = []
    pending_pgns: list[str] = []
    pending_stats: dict = fresh_stats()
    pending_t_start = time.time()
    fragments_written: list = []

    # Aggregate counters for the end-of-session summary (across all fragments).
    total_stats = fresh_stats()

    def flush() -> Path | None:
        """Write the current pending buffer as a fragment .pt/.pgn/.json and
        reset. Returns the .pt path, or None if nothing pending."""
        nonlocal pending_positions, pending_zs, pending_pgns, pending_stats, pending_t_start
        if pending_stats["games_completed"] == 0:
            return None
        elapsed = time.time() - pending_t_start
        frag_stats = {
            "values_per_position": pending_zs,
            "meta": {
                "checkpoint":           cli.checkpoint,
                "checkpoint_sha":       sha,
                "architecture":         arch,
                "in_channels":          in_ch,
                "started":              datetime.datetime.fromtimestamp(pending_t_start).isoformat(),
                "duration_s":           round(elapsed, 1),
                "high_sims":            cli.high_sims,
                "low_sims":             cli.low_sims,
                "high_prob":            cli.high_prob,
                "dirichlet_eps":        cli.dirichlet_eps,
                "dirichlet_alpha":      cli.dirichlet_alpha,
                "resign_threshold":     cli.resign_threshold,
                "resign_consecutive":   cli.resign_consecutive,
                "resign_disabled_fraction": cli.resign_disabled_fraction,
                "max_plies":            cli.max_plies,
                "temperature":          cli.temperature,
                "temperature_moves":    cli.temperature_moves,
                "batch_size":           cli.batch_size,
                **pending_stats,
            }
        }
        if cli.output:
            pt_path = write_flat(
                Path(cli.output), pending_positions, pending_pgns, frag_stats,
            )
        else:
            pt_path = write_session(
                Path(cli.output_dir), cli.checkpoint,
                pending_positions, pending_pgns, frag_stats, cli,
            )
        fragments_written.append(pt_path)
        tqdm.write(f"  → flushed {pending_stats['games_completed']} games to {pt_path.name} "
                   f"({pt_path.stat().st_size / 1e6:.1f} MB)")
        # Reset pending state.
        pending_positions = []
        pending_zs = []
        pending_pgns = []
        pending_stats = fresh_stats()
        pending_t_start = time.time()
        return pt_path

    # Engine pool for concurrent self-play (all share the one compiled model).
    n_concurrent = max(1, int(cli.concurrent_games))
    engine_pool = ([mcts] if n_concurrent == 1
                   else [build_mcts(model, in_ch, cli.batch_size, device)
                         for _ in range(n_concurrent)])

    t_session = time.time()
    pbar = tqdm(total=cli.games, desc="self-play", unit="game")

    def record_game(round_idx: int, result, dt: float) -> None:
        """Consume one play_one_game/play_games_multigame result tuple: append
        positions + z-targets, update stats, build the PGN, advance the bar."""
        positions, plies, final_value, reason, real_moves = result
        zs = per_position_zs(plies, final_value)
        pending_positions.extend(positions)
        pending_zs.extend(zs)

        hi_n = sum(1 for p in positions if p["is_high_sim"])
        lo_n = len(positions) - hi_n
        for s in (pending_stats, total_stats):
            s["positions_total"] += len(positions)
            s["games_completed"] += 1
            s["high_sim_positions"] += hi_n
            s["low_sim_positions"]  += lo_n
            if final_value == -1:
                if reason == "checkmate":
                    s["checkmated"] += 1
                elif reason == "resign":
                    s["resigned"] += 1
                if plies % 2 == 0:
                    s["wins_black"] += 1
                else:
                    s["wins_white"] += 1
            elif reason == "truncation":
                s["truncated"] += 1
            else:
                s["rule_draws"] += 1
                s["draws"] += 1

        pgn_headers = {
            "Event":   "Self-play",
            "Site":    "local",
            "Date":    datetime.datetime.now().strftime("%Y.%m.%d"),
            "Round":   str(round_idx + 1),
            "White":   "AZBot",
            "Black":   "AZBot",
            "Result":  ("1-0" if (final_value == -1 and plies % 2 == 1)
                        else "0-1" if (final_value == -1 and plies % 2 == 0)
                        else "1/2-1/2"),
            "Plies":   str(plies),
            "Reason":  reason,
            "Sims":    f"{cli.high_sims}/{cli.low_sims}@p={cli.high_prob}",
            "Ckpt":    Path(cli.checkpoint).name,
        }
        pending_pgns.append(build_pgn(real_moves, pgn_headers))

        pbar.update(1)
        pbar.set_postfix(
            plies=plies, reason=reason,
            cm=total_stats["checkmated"], rs=total_stats["resigned"],
            d=total_stats["draws"], t=total_stats["truncated"],
            frag=pending_stats["games_completed"],
            tdt=f"{time.time()-t_session:.0f}s",
        )
        tqdm.write(f"  game {round_idx+1:>3}/{cli.games}: {plies:>3} plies, "
                   f"{reason:<10} z_next={final_value:+d}  ({dt:5.1f}s)")

    try:
        produced = 0
        while produced < cli.games:
            wave = min(n_concurrent, cli.games - produced)
            allow = []
            for _ in range(wave):
                ar = (rng.random() >= cli.resign_disabled_fraction)
                allow.append(ar)
                if not ar:
                    pending_stats["resign_disabled_games"] += 1
                    total_stats["resign_disabled_games"] += 1
            t0 = time.time()
            if n_concurrent == 1:
                results = [play_one_game(engine_pool[0], in_ch, cli, rng, allow[0])]
            else:
                results = play_games_multigame(
                    engine_pool[:wave], in_ch, cli, rng, allow,
                )
            dt = time.time() - t0
            # Share the wall-clock across the wave for the per-game log line.
            per_game_dt = dt / max(wave, 1)
            for k, res in enumerate(results):
                record_game(produced + k, res, per_game_dt)
            produced += wave

            # Periodic flush -- so Ctrl-C only loses at most --flush-every games.
            if cli.flush_every > 0 and pending_stats["games_completed"] >= cli.flush_every:
                flush()
    except KeyboardInterrupt:
        tqdm.write("\n  Ctrl-C: flushing pending games before exit...")

    # Final flush for anything still pending (also handles the case where
    # cli.flush_every == 0 -- everything sits in pending until now).
    flush()

    total_dt = time.time() - t_session
    n_games = total_stats["games_completed"]
    print(f"\nSession done: {n_games} games in {total_dt:.1f}s "
          f"across {len(fragments_written)} fragment(s) "
          f"({total_stats['positions_total']} positions, "
          f"avg {total_stats['positions_total']/max(n_games,1):.0f} plies/game)")
    print(f"  win/draw split: W{total_stats['wins_white']} B{total_stats['wins_black']} "
          f"D{total_stats['draws']} T{total_stats['truncated']}")
    print(f"  termination: checkmate={total_stats['checkmated']} resign={total_stats['resigned']} "
          f"rule_draw={total_stats['rule_draws']} truncation={total_stats['truncated']}")
    print(f"  PCR: hi={total_stats['high_sim_positions']} lo={total_stats['low_sim_positions']} "
          f"({total_stats['high_sim_positions']/max(total_stats['positions_total'],1):.0%} hi)")
    for ptp in fragments_written:
        print(f"  {ptp}")


if __name__ == "__main__":
    main()
