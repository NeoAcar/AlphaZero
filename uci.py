"""
UCI protocol wrapper for the AlphaZero bot.

Lets any chess GUI (Cute Chess, Arena, Scid vs PC, En Croissant, etc.)
talk to the bot as a normal UCI engine. Stdin commands, stdout
responses, debug logs to stderr.

Configuration via UCI setoption:
    Checkpoint       (string)  path to .pth model       default: models/model_best.pth
    Sims             (spin)    MCTS sims per move       default: 200
    Temperature      (string)  >=0, 0 = argmax          default: 0.0
    TempMoves        (spin)    plies of stochastic play default: 0
    DirichletEps     (string)  root noise weight        default: 0.0
    DirichletAlpha   (string)  Dirichlet concentration  default: 0.3
    CInit            (string)  PUCT exploration         default: 1.745
    CFactor          (string)  PUCT log-scaling factor  default: 3.894

Wire it up to a GUI: most GUIs let you "add an engine" by pointing
at a command. For Cute Chess:
    Command:   uv
    Arguments: run --quiet python uci.py
    Working dir: /home/neo/PythonProjects/AlphaZero

Time controls (wtime/btime/movetime) are currently ignored; the
bot always thinks for the configured number of simulations. Add
time-aware behaviour later if you want clocked games.
"""
import contextlib
import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

import chess
import numpy as np
import torch

from alphazero import utils as f
from alphazero.batched_mcts import BatchedMCTS as MCTS
from alphazero.mcts import _amp_ctx
from alphazero.nn import ResNet, SEResNet, SEResNetWDL, detect_in_channels, value_to_scalar


ARCHITECTURES = {
    "resnet": ResNet,
    "seresnet": SEResNet,
    "seresnetwdl": SEResNetWDL,
}

PLAYER_TYPES = {"mcts", "policy_only", "value_only"}


torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    # TF32 on cuDNN convolutions + autotune for the static inference shapes.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


ENGINE_NAME = "AlphaZeroBot"
ENGINE_AUTHOR = "NeoAcar"


def log(msg: str) -> None:
    """Debug logging to stderr (UCI requires stdout for protocol)."""
    sys.stderr.write(f"[uci] {msg}\n")
    sys.stderr.flush()


def send(msg: str) -> None:
    """Emit a UCI protocol reply (readyok / bestmove / uciok / info ...).

    Writes to the *original* process stdout (`sys.__stdout__`), NOT `sys.stdout`.
    The background ponder thread holds a `contextlib.redirect_stdout(sys.stderr)`
    for the whole inter-move period, and that redirect mutates the process-global
    `sys.stdout`. If protocol replies went through `sys.stdout` they'd be diverted
    to stderr while pondering -- the engine would log `readyok` to the logfile
    instead of answering the host, the server's drain barrier would time out, and
    every subsequent session would be corrupted (bot stops moving). Going straight
    to `sys.__stdout__` makes protocol output immune to any in-effect redirect."""
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


# Alias for send(): protocol output to the true stdout. Kept as a separate
# name for the live info_callback call sites that documented the intent.
send_raw = send


# Dashboard telemetry: POST JSON events to monitor.py. Fail-fast (50ms) so
# the engine is unaffected when the dashboard isn't running.
MONITOR_URL = os.environ.get("UCI_MONITOR_URL", "http://localhost:8765/event")
_MONITOR_TIMEOUT = 0.05


def post_event(payload: dict) -> None:
    if not MONITOR_URL:
        return
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            MONITOR_URL, data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_MONITOR_TIMEOUT) as resp:
            resp.read(1)  # drain so the socket can be reused
    except (urllib.error.URLError, OSError, TimeoutError):
        # Dashboard not running, or transient hiccup -- swallow.
        pass
    except Exception:
        pass


class UciEngine:
    DEFAULT_OPTS = {
        "Type": "mcts",              # one of: mcts | policy_only | value_only
        "Checkpoint": "models/model_best_combined_wdl.pth",
        "Architecture": "seresnetwdl",
        "ValueScalar": "expected",   # WDL collapse mode: "expected" (P(W)-P(L)) or "win_only" (P(W))
        "Sims": 1200,
        "Temperature": 0.6,
        "TempMoves": 6,
        "DirichletEps": 0.0,
        "DirichletAlpha": 0.3,
        "CInit": 1.33,     # LC0 log-scaling defaults (with CFactor + c_base 38739)
        "CFactor": 1,   # coefficient on the log term; 1.0 = pre-LC0 behaviour #3.894
        "CFPU": 0.2,
        # Engine-driven background pondering. NOT the same as the standard UCI
        # `Ponder` option (which controls GUI-driven `go ponder` and lichess-bot
        # disables by default). Rename avoids the conflict.
        "BackgroundPonder": "true",
        "PonderMaxSims": 3600,
        # Early stop: cut a search once the most-visited move can't be overtaken
        # within the remaining sims, and bank the saved sims to spend on harder
        # positions later (up to MaxBorrow extra on any single move). Play-only;
        # never used in self-play. MaxBorrow 0 = early-stop without lending.
        "EarlyStop": "true",
        "MaxBorrow": 24000,
        # Reuse the search tree (incl. the pondered subtree) across moves.
        # false = fresh tree every move (for A/B testing reuse's effect).
        "TreeReuse": "true",
    }
    OPT_TYPES = {
        "Type": ("string", None, None),
        "Checkpoint": ("string", None, None),
        "Architecture": ("string", None, None),
        "ValueScalar": ("string", None, None),
        "Sims": ("spin", 2, 100000),
        "Temperature": ("string", None, None),
        "TempMoves": ("spin", 0, 200),
        "DirichletEps": ("string", None, None),
        "DirichletAlpha": ("string", None, None),
        "CInit": ("string", None, None),
        "CFactor": ("string", None, None),
        "CFPU": ("string", None, None),
        "BackgroundPonder": ("string", None, None),
        "PonderMaxSims": ("spin", 0, 1000000),
        "EarlyStop": ("string", None, None),
        "MaxBorrow": ("spin", 0, 1000000),
        "TreeReuse": ("string", None, None),
    }

    def __init__(self) -> None:
        self.options = dict(self.DEFAULT_OPTS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.loaded_arch: str | None = None
        self.loaded_checkpoint: str | None = None
        # Plane count of the loaded model; set in _ensure_loaded. Initialised so
        # policy/value paths that read it never hit AttributeError before a load.
        self.loaded_in_channels: int = 19
        self.mcts: MCTS | None = None
        self.real_board = chess.Board()
        self.mirror_state = chess.Board()
        self.move_counter = 0
        # Used to detect when a new `position` command is just an extension
        # of the last one (lichess-bot sends the full move list each turn).
        # Extension -> only push the tail; otherwise reset + walk full list.
        self._move_history: list[str] = []
        self._start_fen: str | None = None
        # Repetition tracking: count of each position (by transposition_key)
        # seen so far in the actual game. python-chess's board.mirror() doesn't
        # preserve the move stack / _transpositions, so MCTS can't see 3-fold
        # via the board alone -- we maintain this counter and inject it into
        # MCTS before each search. Keyed on mirror_state's transposition_key.
        self._rep_counter: dict = {}
        # Canonical board history (chronological, oldest first), used to build
        # the 119-plane input. Empty for 19-plane checkpoints. Kept across
        # _push_move calls; cleared in _reset_position. Length capped at 7.
        self._mirror_history: list = []
        self._rng = np.random.default_rng()
        self._quitting = False
        # Dashboard telemetry state.
        self._go_start_t: float | None = None         # monotonic time of last `go`
        self._bot_color: str | None = None            # set on each `go`
        self._last_bot_chosen_uci: str | None = None  # real-coord UCI of bot's pending move
        self._pending_bot_eval: dict | None = None    # eval/duration to attach to next _push_move
        # Background pondering. After every `bestmove` we push our own move
        # locally and start running sims on the after-our-move root. When the
        # next `position` arrives we stop the thread; the existing extension
        # detection + apply_action then walks the tree to opp's actual reply,
        # inheriting the pondered subtree for the upcoming search.
        self._ponder_stop = threading.Event()
        self._ponder_thread: threading.Thread | None = None
        # Serializes every GPU/model invocation. The foreground search and the
        # background ponder loop both run MCTS on the same torch.compile'd model;
        # invoking it from two threads at once corrupts CUDA state (intermittent
        # crash / OOM that bricks the engine for the rest of the session). Both
        # paths must hold this lock around their search() call.
        self._model_lock = threading.Lock()
        # Serializes model (re)loading so a background warm-up thread and the
        # foreground `go` never load at the same time / double-load.
        self._load_lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None

    # ---------- model / mcts plumbing ----------

    def _ensure_loaded(self) -> None:
        """Load (or refresh) the model, holding the load lock so a concurrent
        background warm-up and the foreground go() coordinate safely."""
        with self._load_lock:
            self._load_locked()

    def _safe_ensure_loaded(self) -> None:
        """_ensure_loaded that never raises -- for the background warm-up thread
        (a load failure there must not crash the process; go() will retry)."""
        try:
            self._ensure_loaded()
        except Exception as e:
            log(f"background warm-up failed (will retry on go): {e}")

    def _warm_async(self) -> None:
        """Kick off model load + compile + warm-up in a background daemon thread
        so the first `go` doesn't pay the ~20s JIT trace on the clock. Idempotent:
        no-op if the right model is already loaded or a warm-up is in flight."""
        ckpt = str(self.options["Checkpoint"])
        arch = str(self.options["Architecture"]).lower()
        if (self.model is not None and self.loaded_checkpoint == ckpt
                and self.loaded_arch == arch):
            return
        if self._warm_thread is not None and self._warm_thread.is_alive():
            return
        self._warm_thread = threading.Thread(target=self._safe_ensure_loaded, daemon=True)
        self._warm_thread.start()
        log("background warm-up started")

    def _load_locked(self) -> None:
        ckpt = str(self.options["Checkpoint"])
        arch = str(self.options["Architecture"]).lower()
        if self.model is not None and self.loaded_checkpoint == ckpt and self.loaded_arch == arch:
            self._refresh_mcts()
            return
        if arch not in ARCHITECTURES:
            raise ValueError(f"Architecture must be one of {list(ARCHITECTURES)}, got {arch!r}")
        log(f"Loading checkpoint {ckpt} (architecture: {arch})")
        # Detect input-plane count from the checkpoint's first-conv weight so
        # legacy 19-plane checkpoints (e.g. the SFT WDL run) still load. New
        # tabula-rasa checkpoints will be 119.
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        in_ch = detect_in_channels(state)
        log(f"  input_planes = {in_ch}")
        model = ARCHITECTURES[arch](in_channels=in_ch).to(self.device)
        model.load_state_dict(state["model_state_dict"])
        if self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        model.eval()
        self.loaded_in_channels = in_ch
        try:
            # NOTE: default mode, NOT mode="reduce-overhead". The latter records
            # CUDA graphs bound to the recording thread's default stream, which
            # crashes the moment our background ponder thread (running on its
            # own stream) tries to invoke the same model. Default mode loses
            # ~1.5x kernel-fusion speedup at batch=1 but is thread-safe.
            model = torch.compile(model)
            # Real starting position, not torch.zeros -- see players.py for why.
            real_one = f.prepare_input(
                chess.Board(), 0,
                history=([] if in_ch == 119 else None),
            ).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                real_one = real_one.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), _amp_ctx(self.device):
                _ = model(real_one)
            log("torch.compile + warm-up done")
        except Exception as e:
            log(f"torch.compile skipped: {e}")
        self.model = model
        self.loaded_checkpoint = ckpt
        self.loaded_arch = arch
        self._refresh_mcts()

    def _refresh_mcts(self) -> None:
        args = {
            "num_simulation": int(self.options["Sims"]),
            "truncation": 1000,
            "c_base": 38739,
            "c_init": float(self.options["CInit"]),
            "c_factor": float(self.options["CFactor"]),
            "c_fpu": float(self.options["CFPU"]),
            "dirichlet_epsilon": float(self.options["DirichletEps"]),
            "dirichlet_alpha": float(self.options["DirichletAlpha"]),
            "memory_size": 1000,
            "action_space": 4672,
            "t": 1,
            "device": self.device,
            "value_scalar": str(self.options["ValueScalar"]),
            "batch_size": 8,
            # 19 (legacy) or 119 (8-frame history) -- decided when the
            # checkpoint was loaded. MCTS routes board_to_matrix accordingly.
            "input_planes": getattr(self, "loaded_in_channels", 19),
            # Early-stop + sim-bank (play only; never set by self-play).
            "early_stop": str(self.options["EarlyStop"]).lower() == "true",
            "max_borrow": int(self.options["MaxBorrow"]),
            "tree_reuse": str(self.options["TreeReuse"]).lower() == "true",
        }
        assert self.model is not None
        # Reuse the existing engine when the model is unchanged so the search
        # tree (tree-reuse + the pondered subtree) AND the early-stop sim bank
        # survive across moves -- otherwise both are thrown away every move.
        # update_root() resets safely if the new position isn't in the tree.
        # Only rebuild on first use or after a checkpoint/architecture reload.
        if self.mcts is not None and self.mcts.model is self.model:
            self.mcts.args.update(args)
            return
        self.mcts = MCTS(args, self.model)
        # MCTS pipeline pre-warm on first build. The forward warm-up alone
        # doesn't cover the masked_fill/softmax-with-inf, torch.cat + cpu(),
        # and legal_mask numpy paths that fire only on the first real search.
        # A 4-sim throwaway absorbs ~6s into isready so move 1 isn't stalled.
        try:
            self.mcts.args["num_simulation"] = 4
            self.mcts.search(chess.Board(), 0)
            self.mcts.args["num_simulation"] = int(self.options["Sims"])
            self.mcts.root = None
            self.mcts.sim_bank = 0
            log("mcts pre-warm done")
        except Exception as e:
            log(f"mcts pre-warm skipped: {e}")

    # ---------- position tracking ----------

    def _reset_position(self, start_fen: str | None = None) -> None:
        if start_fen and start_fen != chess.STARTING_FEN:
            self.real_board = chess.Board(start_fen)
            mir = chess.Board(start_fen)
            # If it's black to move at the starting position, mirror once so
            # the bot always sees a "white-to-move" canonical state.
            if not mir.turn:
                mir.apply_mirror()
            self.mirror_state = mir
            self.move_counter = (self.real_board.fullmove_number - 1) * 2 + (
                0 if self.real_board.turn == chess.WHITE else 1
            )
        else:
            self.real_board = chess.Board()
            self.mirror_state = chess.Board()
            self.move_counter = 0
        self._move_history = []
        self._start_fen = start_fen
        self._pending_bot_eval = None
        self._last_bot_chosen_uci = None
        # Start fresh: the starting position has been seen once.
        self._rep_counter = {self.mirror_state._transposition_key(): 1}
        # Reset canonical board history (used by 119-plane input).
        self._mirror_history = []
        if self.mcts is not None:
            self.mcts.root = None
        # Tell dashboard: fresh game / position reset.
        post_event({
            "kind": "state",
            "fen": self.real_board.fen(),
            "lastmove": None,
            "ply": self.move_counter,
            "bot_color": self._bot_color,
        })

    def _push_move(self, uci_real: str) -> None:
        """Apply a move (in real-coord UCI) to both boards and walk the
        MCTS root forward by the equivalent action (O(1)). Tree subtree for
        this move is retained for the next search.
        """
        mover_was_white = self.real_board.turn == chess.WHITE
        self.real_board.push_uci(uci_real)
        mir_uci = uci_real if mover_was_white else f.mirror_move(uci_real)
        # Compute action BEFORE the mirror() — alphazero_to_move expects the
        # board the move was made from.
        action: int | None = None
        if self.mcts is not None and self.mcts.root is not None:
            try:
                action = f.move_to_alphazero(mir_uci)
            except Exception:
                action = None
        # Append the OLD canonical board to history BEFORE push+mirror.
        # Keep at most 7 entries -- 119-plane uses 8 frames total (current + 7).
        # Only maintain history for 119-plane models (it's unused by 19-plane).
        # Default to accumulating when the plane count isn't known yet (model
        # not loaded), so we never lose history a 119-plane model would need.
        if getattr(self, "loaded_in_channels", 119) == 119:
            self._mirror_history.append(self.mirror_state.copy())
            if len(self._mirror_history) > 7:
                self._mirror_history.pop(0)
        self.mirror_state.push_uci(mir_uci)
        self.mirror_state = self.mirror_state.mirror()
        self.move_counter += 1
        self._move_history.append(uci_real)
        # Bump repetition counter for the new position. A 50-move-rule reset
        # (capture or pawn move) makes the position un-repeatable from here
        # on, but since we key on the full transposition_key (occupancy +
        # castling + ep), captured-pieces positions already get distinct keys.
        tk = self.mirror_state._transposition_key()
        self._rep_counter[tk] = self._rep_counter.get(tk, 0) + 1
        if action is not None and self.mcts is not None:
            self.mcts.apply_action(action)
        # Dashboard `move` event. If this push is the bot's own just-chosen
        # move (already emitted live at end of cmd_go, before bestmove was
        # sent), skip emission — otherwise the browser receives bot+opp moves
        # in the same batch and only the final state is visually perceptible.
        if (self._last_bot_chosen_uci == uci_real
                and self._pending_bot_eval is not None):
            self._pending_bot_eval = None
            self._last_bot_chosen_uci = None
            return
        evt: dict = {
            "kind": "move",
            "fen": self.real_board.fen(),
            "lastmove": uci_real,
            "ply": self.move_counter,
            "turn": "white" if self.real_board.turn == chess.WHITE else "black",
            "mover": "opp",
        }
        post_event(evt)

    # ---------- UCI command handlers ----------

    def cmd_uci(self, _args: list[str]) -> None:
        send(f"id name {ENGINE_NAME}")
        send(f"id author {ENGINE_AUTHOR}")
        for name, default in self.DEFAULT_OPTS.items():
            kind, lo, hi = self.OPT_TYPES[name]
            if kind == "spin":
                send(f"option name {name} type spin default {default} min {lo} max {hi}")
            else:
                send(f"option name {name} type string default {default}")
        send("uciok")

    def cmd_isready(self, _args: list[str]) -> None:
        # Don't BLOCK here -- lichess-bot's SimpleEngine.popen_uci caps the full
        # uci/isready handshake at 60s and the torch.compile JIT can exceed it.
        # Answer immediately, then warm the model in the BACKGROUND (off the
        # clock) so the first move usually finds it ready. cmd_go also calls
        # _ensure_loaded as a fallback if warm-up hasn't finished.
        send("readyok")
        self._warm_async()

    def cmd_ucinewgame(self, _args: list[str]) -> None:
        self._stop_pondering()
        self._reset_position()
        # New game: drop the reused tree and the sim bank (both are per-game).
        if self.mcts is not None:
            self.mcts.root = None
            self.mcts.sim_bank = 0
        log("ucinewgame: reset")
        # Warm up now (off the clock); options are final by this point.
        self._warm_async()

    def cmd_setoption(self, args: list[str]) -> None:
        self._stop_pondering()
        # Format: setoption name <NAME> [value <VALUE>]
        try:
            name_idx = args.index("name")
            value_idx = args.index("value") if "value" in args else None
            name = " ".join(args[name_idx + 1: value_idx if value_idx is not None else len(args)])
            value = " ".join(args[value_idx + 1:]) if value_idx is not None else ""
        except ValueError:
            log(f"bad setoption: {args}")
            return
        if name not in self.options:
            log(f"unknown option: {name}")
            return
        kind, *_ = self.OPT_TYPES[name]
        if kind == "spin":
            try:
                self.options[name] = int(value)
            except ValueError:
                log(f"bad int for {name}: {value!r}")
                return
        else:
            if name == "ValueScalar" and value not in ("expected", "win_only"):
                log(f"bad ValueScalar {value!r}; expected 'expected' or 'win_only'")
                return
            self.options[name] = value
        log(f"set {name} = {self.options[name]!r}")
        # If model checkpoint or architecture changed, force reload on next isready/go
        if name in ("Checkpoint", "Architecture"):
            self.model = None
            self.loaded_checkpoint = None
            self.loaded_arch = None

    def cmd_position(self, args: list[str]) -> None:
        # Always stop pondering first -- the new `position` either extends the
        # known move list (opp's reply arrived; tree-reuse will inherit the
        # pondered subtree via apply_action), or it's a fresh setup that
        # invalidates the tree entirely.
        self._stop_pondering()
        moves: list[str] = []
        start_fen: str | None = None
        if not args:
            return
        if args[0] == "startpos":
            start_fen = None
            rest = args[1:]
        elif args[0] == "fen":
            # FEN is six fields; find "moves" token after them
            try:
                moves_idx = args.index("moves")
            except ValueError:
                moves_idx = len(args)
            start_fen = " ".join(args[1:moves_idx])
            rest = args[moves_idx:]
        else:
            log(f"position: unknown subcommand {args[0]}")
            return
        if rest and rest[0] == "moves":
            moves = rest[1:]

        # Lichess-bot sends the full move list each turn. If this `position`
        # is just an extension of the last one (same start_fen, same move
        # prefix), only push the new tail -- the MCTS tree from previous
        # searches carries forward. Otherwise full reset and walk from scratch.
        is_extension = (
            start_fen == self._start_fen
            and len(moves) >= len(self._move_history)
            and moves[:len(self._move_history)] == self._move_history
        )
        if is_extension:
            new_moves = moves[len(self._move_history):]
            for m in new_moves:
                self._push_move(m)
            log(f"position extended: +{len(new_moves)} moves (total {len(moves)}), "
                f"turn={'w' if self.real_board.turn == chess.WHITE else 'b'}, "
                f"counter={self.move_counter}")
        else:
            self._reset_position(start_fen)
            for m in moves:
                self._push_move(m)
            log(f"position reset: {len(moves)} moves applied, "
                f"turn={'w' if self.real_board.turn == chess.WHITE else 'b'}, "
                f"counter={self.move_counter}")

    def cmd_go(self, args: list[str]) -> None:
        # Defensive: kill any leftover ponder before foreground search runs.
        # In normal flow cmd_position will have already stopped it.
        self._stop_pondering()
        # Time controls don't drive search yet (we always use configured Sims),
        # but we parse wtime/btime/winc/binc/movetime to forward to the dashboard.
        clock: dict = {}
        i = 0
        while i < len(args):
            tok = args[i]
            if tok in ("wtime", "btime", "winc", "binc", "movetime") and i + 1 < len(args):
                try:
                    clock[tok] = int(args[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
            i += 1
        self._go_start_t = time.monotonic()
        self._bot_color = "white" if self.real_board.turn == chess.WHITE else "black"
        post_event({
            "kind": "go_start",
            "clock": clock if clock else None,
            "bot_color": self._bot_color,
            "ply": self.move_counter,
        })
        ptype = str(self.options["Type"]).lower()
        if ptype not in PLAYER_TYPES:
            log(f"unknown Type {ptype!r}; falling back to mcts")
            ptype = "mcts"

        # Produce a move. EVERY failure path (model load, search, illegal move)
        # is funnelled into a guaranteed-legal bestmove below -- the engine must
        # NEVER finish cmd_go without emitting exactly one `bestmove`, or the GUI
        # / lichess-bot hangs waiting for it ("bot doesn't play").
        real_uci = None
        try:
            self._ensure_loaded()          # lazy fallback if warm-up hasn't finished
            if self.model is None:
                raise RuntimeError("model failed to load")
            if ptype == "mcts":
                real_uci = self._go_mcts()
            elif ptype == "policy_only":
                real_uci = self._go_policy_only()
            else:  # value_only
                real_uci = self._go_value_only()
        except Exception as e:
            log(f"go failed ({ptype}): {e}\n{traceback.format_exc()}")
            real_uci = None

        real_uci = self._ensure_legal_move(real_uci)

        # Dashboard: emit the bot's move now (post-move FEN, eval, duration)
        # so the browser shows it as a distinct frame rather than getting
        # batched together with the opponent's reply in the next position cmd.
        self._last_bot_chosen_uci = real_uci
        try:
            self._emit_bot_move(real_uci)
        except Exception as e:
            log(f"emit_bot_move failed: {e}")
        send(f"bestmove {real_uci}")
        # Pre-push the bot's own move so the MCTS root advances to "after our
        # move." The next `position` from the GUI will see this move already
        # in _move_history and only push opp's reply on top -- existing
        # extension detection in cmd_position handles that case.
        if ptype == "mcts" and real_uci != "0000":
            try:
                self._push_move(real_uci)
            except Exception as e:
                log(f"local push after bestmove failed: {e}")
            try:
                self._start_pondering()
            except Exception as e:
                log(f"start pondering failed: {e}")

    def _ensure_legal_move(self, real_uci: str | None) -> str:
        """Return `real_uci` if it's legal on the real board; else any legal
        move; else '0000' (UCI null = no legal move / game already over). This
        is the single guarantee that cmd_go always has something legal to send."""
        if real_uci:
            try:
                if chess.Move.from_uci(real_uci) in self.real_board.legal_moves:
                    return real_uci
            except Exception:
                pass
            log(f"chose illegal/unparseable {real_uci!r}; falling back to a legal move")
        try:
            return next(iter(self.real_board.legal_moves)).uci()
        except StopIteration:
            return "0000"

    def _start_pondering(self) -> None:
        """Spawn a background thread that keeps running MCTS sims from the
        current root until interrupted by the next UCI command. Reads
        `BackgroundPonder` (not `Ponder` -- that one is reserved by the UCI
        protocol for GUI-driven `go ponder` and lichess-bot disables it by
        default, which would silently turn our engine-side pondering off too)."""
        bp = str(self.options.get("BackgroundPonder", "true")).lower()
        if bp != "true":
            log(f"ponder: skipped (BackgroundPonder={bp!r})")
            return
        # Pondering only pays off if its grown tree is reused next move; with
        # TreeReuse off the next search discards it, so pondering would just burn
        # GPU on the opponent's clock for nothing. No-op in that case.
        if str(self.options.get("TreeReuse", "true")).lower() != "true":
            log("ponder: skipped (TreeReuse is off -- pondered tree would be discarded)")
            return
        if self.mcts is None or self.mcts.root is None:
            log("ponder: skipped (mcts root is None)")
            return
        # If a previous ponder is somehow still alive, stop it first. If it
        # refuses to die (drain timed out), DON'T stack a second thread on top
        # of it -- two ponder threads both growing the same tree is exactly the
        # accumulation that leads to OOM. The model lock keeps the straggler
        # harmless; it will exit on its next chunk boundary.
        self._stop_pondering()
        if self._ponder_thread is not None and self._ponder_thread.is_alive():
            log("ponder: previous thread still draining; skip start")
            return
        max_sims = int(self.options["PonderMaxSims"])
        if max_sims <= 0:
            log(f"ponder: skipped (PonderMaxSims={max_sims})")
            return
        self._ponder_stop.clear()
        self._ponder_thread = threading.Thread(
            target=self._ponder_loop, args=(max_sims,), daemon=True
        )
        self._ponder_thread.start()
        log(f"ponder: started (max_sims={max_sims}, root.N={self.mcts.root.N})")

    def _ponder_loop(self, max_sims: int) -> None:
        """Run sims in chunks until stop event or max_sims reached. We bypass
        the search() noise/setup overhead by issuing small `search` calls with
        a temporary num_simulation override and restoring it each time."""
        assert self.mcts is not None
        # Run many sims per `search()` call (NOT just batch_size) so the
        # per-call setup overhead (policy copy, depth-set clear, etc.) is
        # amortised. Stop-event responsiveness caps at chunk_sims * forward_ms:
        # at 64 sims/chunk with batch=8 that's ~240ms worst-case lag, fine.
        chunk = max(64, int(self.mcts.args.get("batch_size", 8)) * 8)
        orig_sims = int(self.mcts.args["num_simulation"])
        # CRITICAL: pondering runs on the OPPONENT's clock. It must NOT touch the
        # foreground early-stop sim bank -- otherwise it banks free opp-time sims
        # that the next foreground move then spends on OUR clock (flagging). Run
        # ponder searches with early-stop OFF so the bank is fed/spent by real
        # moves only. (Pondering also genuinely wants to keep searching to fill
        # the opponent's time, not early-stop.)
        orig_es = self.mcts.args.get("early_stop", False)
        self.mcts.args["early_stop"] = False
        done = 0
        t0 = time.monotonic()
        try:
            while not self._ponder_stop.is_set() and done < max_sims:
                self.mcts.args["num_simulation"] = min(chunk, max_sims - done)
                # Serialize with the foreground search: never two threads in the
                # model at once. Re-check the stop flag after acquiring so a
                # stop requested while we were queued exits before another chunk.
                with self._model_lock:
                    if self._ponder_stop.is_set():
                        break
                    with contextlib.redirect_stdout(sys.stderr):
                        self.mcts.search(self.mirror_state, self.move_counter)
                done += chunk
                elapsed = time.monotonic() - t0
                root_N = self.mcts.root.N if self.mcts.root is not None else 0
                # Ponder root = the post-our-move position (opponent to move), so
                # its top children are the bot's PREDICTED opponent replies. Reuse
                # _top_k_children (frame-correct for the current real/mirror state).
                opp_top = []
                try:
                    opp_top = [{"uci": u, "N": n, "q": qv}
                               for u, n, qv in self._top_k_children(self.mcts, k=3)]
                except Exception:
                    pass
                post_event({
                    "kind": "ponder_tick",
                    "status": "running",
                    "sims": done,
                    "elapsed_s": round(elapsed, 3),
                    "nps": int(done / max(elapsed, 0.001)),
                    "root_N": root_N,
                    "top": opp_top,
                })
        except Exception as e:
            log(f"ponder thread crashed: {e}\n{traceback.format_exc()}")
        finally:
            self.mcts.args["num_simulation"] = orig_sims
            self.mcts.args["early_stop"] = orig_es
            log(f"ponder stopped after {done} sims")
            root_N = (
                self.mcts.root.N
                if self.mcts is not None and self.mcts.root is not None
                else 0
            )
            post_event({
                "kind": "ponder_tick",
                "status": "stopped",
                "sims": 0,
                "elapsed_s": round(time.monotonic() - t0, 3),
                "nps": 0,
                "root_N": root_N,
            })

    def _stop_pondering(self) -> None:
        """Signal the ponder thread and wait for it to exit. Idempotent.

        If the thread doesn't exit within the timeout we KEEP its reference
        rather than nulling it -- losing the handle would let _start_pondering
        spawn a second thread on top, and orphaned ponder threads accumulating
        across moves/games is what drives the OOM. The model lock makes a
        straggler harmless (it can't run the model concurrently), and it exits
        on its next chunk boundary once it sees the stop flag."""
        if self._ponder_thread is None:
            return
        self._ponder_stop.set()
        self._ponder_thread.join(timeout=5.0)
        if self._ponder_thread.is_alive():
            log("warning: ponder thread did not exit within 5s; keeping handle "
                "so it drains instead of being orphaned")
            return
        self._ponder_thread = None

    def _emit_bot_move(self, real_uci: str) -> None:
        """Emit a `move` event for the bot's just-chosen move using a copy of
        real_board (the move isn't pushed yet — the GUI will echo it back via
        the next `position` command, where _push_move suppresses re-emission)."""
        post = self.real_board.copy()
        try:
            post.push_uci(real_uci)
        except Exception:
            return
        evt: dict = {
            "kind": "move",
            "fen": post.fen(),
            "lastmove": real_uci,
            "ply": self.move_counter + 1,   # post-move ply count
            "turn": "white" if post.turn == chess.WHITE else "black",
            "mover": "bot",
        }
        if self._pending_bot_eval is not None:
            evt.update({
                "cp": self._pending_bot_eval["cp"],
                "win_prob": self._pending_bot_eval["win_prob"],
                "duration": self._pending_bot_eval["duration"],
            })
            nn_wp = self._pending_bot_eval.get("nn_win_prob")
            if nn_wp is not None:
                evt["nn_win_prob"] = nn_wp
            nn_std = self._pending_bot_eval.get("nn_std")
            if nn_std is not None:
                evt["nn_std"] = nn_std
            mate = self._pending_bot_eval.get("mate")
            if mate is not None:
                evt["mate"] = mate
        post_event(evt)

    # ---------- per-Type search backends ----------

    def _go_mcts(self) -> str:
        assert self.mcts is not None
        self.mcts.set_rep_counter(self._rep_counter)
        self.mcts.set_history(self._mirror_history)
        # Hold the model lock for the whole foreground search. If a previous
        # ponder thread is still draining (e.g. its join timed out), this blocks
        # until it releases instead of hitting the GPU model concurrently.
        with self._model_lock, contextlib.redirect_stdout(sys.stderr):
            probs = self.mcts.search(
                self.mirror_state,
                self.move_counter,
                info_callback=self._live_mcts_info,
                info_interval_s=0.2,
            )
        if getattr(self.mcts, "last_was_proven_mate", False):
            send("info string proven forced mate")
            log("proven forced mate")
        action = self._select_action(probs)
        mir_uci = f.alphazero_to_move(action, self.mirror_state)
        real_uci = mir_uci if self.real_board.turn == chess.WHITE else f.mirror_move(mir_uci)
        self._emit_mcts_info(action, real_uci)
        self._stash_bot_eval(action, real_uci)
        return real_uci

    def _stash_bot_eval(self, action: int, real_uci: str) -> None:
        """Compute and queue bot's eval for the chosen move; the next
        `_push_move` matching this UCI will attach it to the dashboard event."""
        if self.mcts is None or self.mcts.root is None:
            return
        child = self.mcts.root.children.get(action)
        if child is None or child.N == 0:
            return
        q = child.Q / child.N           # bot's POV in [-1, +1]
        win_prob = (q + 1.0) / 2.0
        cp = self._value_to_cp(q)
        duration = (
            time.monotonic() - self._go_start_t if self._go_start_t is not None else None
        )
        self._last_bot_chosen_uci = real_uci
        # Raw NN value for the pre-move root position (bot's POV — root is in
        # mirror-canonical frame so side-to-move = bot). Cached on Node during
        # expand_lazy / batched leaf eval; no extra forward.
        nn_v = self.mcts.root.raw_nn_value
        nn_win_prob = (nn_v + 1.0) / 2.0 if nn_v is not None else None
        # WDL variance band on the win-prob plot. With outcomes scored
        # +1/0/−1, Var(q) = P(W)+P(L) − q²; win_prob = (q+1)/2 is a linear
        # transform so std(win_prob) = std(q)/2. Cheap closed-form, no
        # additional NN call.
        nn_std = None
        wdl = self.mcts.root.raw_nn_wdl
        if wdl is not None:
            pw, _pd, pl = wdl
            q_nn = pw - pl
            var_q = max((pw + pl) - q_nn * q_nn, 0.0)
            nn_std = math.sqrt(var_q) / 2.0
        self._pending_bot_eval = {
            "q": q, "win_prob": win_prob, "cp": cp, "duration": duration,
            "nn_win_prob": nn_win_prob,
            "nn_std": nn_std,
            "mate": self._mate_in(),          # signed mate-in-N, or None
        }

    # ---------- mate-distance from the proof tree ----------

    def _mate_in(self) -> int | None:
        """Signed mate distance in MOVES from the current root, or None.
        +N = bot mates in N, -N = bot is mated in N. Used to show 'M7' on the
        dashboard instead of a saturated centipawn value. Mate-depth walk lives
        in MCTSBase (shared with the fastest-mate move selection)."""
        mcts = self.mcts
        if mcts is None or mcts.root is None or mcts.root.proven_value in (None, 0):
            return None
        plies = mcts._mate_plies(mcts.root)
        if plies is None:
            return None
        moves = (plies + 1) // 2            # ceil(plies / 2)
        return moves if mcts.root.proven_value == 1 else -moves

    # ---------- live monitoring during MCTS search ----------

    def _pv_from_root(self, mcts, max_len: int = 16) -> list[str]:
        """Principal variation as a list of UCI moves in *real-board* frame.
        Descends from root taking the max-visit child at each ply, mirroring
        the board between plies so coordinate conversion stays correct."""
        if mcts.root is None or not mcts.root.children:
            return []
        real = self.real_board.copy()
        mir = self.mirror_state.copy()
        node = mcts.root
        pv: list[str] = []
        for _ in range(max_len):
            if not node.children:
                break
            best = max(node.children.items(), key=lambda kv: kv[1].N)
            action, child = best
            if child.N == 0:
                break
            try:
                mir_uci = f.alphazero_to_move(action, mir)
            except Exception:
                break
            real_uci = mir_uci if real.turn == chess.WHITE else f.mirror_move(mir_uci)
            try:
                real.push_uci(real_uci)
                mir.push_uci(mir_uci)
                mir.apply_mirror()
            except Exception:
                break
            pv.append(real_uci)
            node = child
        return pv

    def _top_k_children(self, mcts, k: int = 3) -> list[tuple[str, int, float]]:
        """Return [(real_uci, N, Q/N), ...] for the k most-visited root children."""
        if mcts.root is None:
            return []
        items = [(a, c) for a, c in mcts.root.children.items() if c.N > 0]
        items.sort(key=lambda ac: ac[1].N, reverse=True)
        out: list[tuple[str, int, float]] = []
        for action, child in items[:k]:
            try:
                mir_uci = f.alphazero_to_move(action, self.mirror_state)
            except Exception:
                continue
            real_uci = mir_uci if self.real_board.turn == chess.WHITE else f.mirror_move(mir_uci)
            out.append((real_uci, child.N, child.Q / child.N))
        return out

    def _live_mcts_info(self, mcts, completed: int, elapsed_s: float, depth: int) -> None:
        """Emit a UCI `info` line + `info string` summary during search, AND
        push a `tick` event to the dashboard. Called from inside `mcts.search`
        while stdout is redirected, so use send_raw() for UCI output."""
        if mcts.root is None or not mcts.root.children:
            return
        best_action, best_child = max(mcts.root.children.items(), key=lambda kv: kv[1].N)
        if best_child.N == 0:
            return
        q = best_child.Q / best_child.N  # bot's POV
        win_prob = (q + 1.0) / 2.0
        nodes = sum(c.N for c in mcts.root.children.values())
        time_ms = max(int(elapsed_s * 1000), 1)
        nps = int(nodes * 1000 / time_ms)
        pv = self._pv_from_root(mcts, max_len=12)
        if not pv:
            return
        cp = self._value_to_cp(q)
        d = max(int(depth), 1)
        send_raw(
            f"info depth {d} seldepth {d} score cp {cp} nodes {nodes} "
            f"nps {nps} time {time_ms} pv {' '.join(pv)}"
        )
        top = self._top_k_children(mcts, k=3)
        if top:
            tops_str = " ".join(
                f"{u}(N={n},Q={qv:+.2f})" for u, n, qv in top
            )
            send_raw(
                f"info string sims={completed} eval={cp/100:+.2f} top: {tops_str}"
            )
        # Dashboard tick. List-of-dicts for top moves so the JS can format them.
        post_event({
            "kind": "tick",
            "sims": completed,
            "depth": d,
            "nodes": nodes,
            "nps": nps,
            "cp": cp,
            "win_prob": win_prob,
            "elapsed_s": round(elapsed_s, 3),
            "pv": pv,
            "top": [{"uci": u, "N": n, "q": qv} for u, n, qv in top],
            "mate": self._mate_in(),    # show 'M7' live if a mate is already proven
        })

    def _emit_mcts_info(self, action: int, real_uci: str) -> None:
        """Final UCI `info` line for the chosen move. The live callback
        already streamed updates during search; this is the closing reading
        the GUI shows after `bestmove` arrives.
        """
        if self.mcts is None or self.mcts.root is None:
            return
        root = self.mcts.root
        child = root.children.get(action)
        if child is None or child.N == 0:
            return
        q = child.Q / child.N        # bot's-POV mean value of chosen line, in [-1, +1]
        nodes = sum(c.N for c in root.children.values())
        depth = max(int(getattr(self.mcts, "last_max_depth", 0)), 1)
        # PV: chosen move + continuation from the chosen subtree by max-visit descent.
        pv: list[str] = [real_uci]
        try:
            after_real = self.real_board.copy()
            after_real.push_uci(real_uci)
            after_mir = self.mirror_state.copy()
            mir_uci = real_uci if self.real_board.turn == chess.WHITE else f.mirror_move(real_uci)
            after_mir.push_uci(mir_uci)
            after_mir.apply_mirror()
            node = child
            for _ in range(11):
                if not node.children:
                    break
                best = max(node.children.items(), key=lambda kv: kv[1].N)
                a, c = best
                if c.N == 0:
                    break
                cont_mir = f.alphazero_to_move(a, after_mir)
                cont_real = cont_mir if after_real.turn == chess.WHITE else f.mirror_move(cont_mir)
                after_real.push_uci(cont_real)
                after_mir.push_uci(cont_mir)
                after_mir.apply_mirror()
                pv.append(cont_real)
                node = c
        except Exception:
            pass
        pv_str = " ".join(pv)
        if getattr(self.mcts, "last_was_proven_mate", False):
            send(f"info depth {depth} score mate 1 nodes {nodes} pv {pv_str}")
            send(f"info string eval=#mate {pv_str}")
        else:
            cp = self._value_to_cp(q)
            send(f"info depth {depth} score cp {cp} nodes {nodes} pv {pv_str}")
            top = self._top_k_children(self.mcts, k=3)
            tops_str = " ".join(f"{u}(N={n},Q={qv:+.2f})" for u, n, qv in top)
            send(f"info string final eval={cp/100:+.2f} depth={depth} nodes={nodes} top: {tops_str}")

    @staticmethod
    def _value_to_cp(v: float) -> int:
        """Inverse of the Lichess sigmoid used in training:
            v = 2 * sigmoid(0.00368208 * cp) - 1
        gives `cp = log((1+v)/(1-v)) / 0.00368208`. Clipped to ±0.9999 to
        avoid log(inf) on near-saturated values.
        """
        v = max(min(float(v), 0.9999), -0.9999)
        return int(round(math.log((1.0 + v) / (1.0 - v)) / 0.00368208))

    @torch.inference_mode()
    def _go_policy_only(self) -> str:
        assert self.model is not None
        hist = self._mirror_history if self.loaded_in_channels == 119 else None
        rep_now = self._rep_counter.get(self.mirror_state._transposition_key(), 1)
        inputs = f.prepare_input(
            self.mirror_state, self.move_counter, history=hist,
            rep_count=max(rep_now, 1),
        ).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with self._model_lock, _amp_ctx(self.device):
            _value, policy_logits = self.model(inputs)
        mask = torch.from_numpy(f.legal_mask(self.mirror_state)).to(self.device, non_blocking=True)
        masked_logits = policy_logits.squeeze(0).float().masked_fill(~mask, float("-inf"))
        probs = torch.softmax(masked_logits, dim=0).cpu().numpy()
        action = self._select_action(probs)
        mir_uci = f.alphazero_to_move(action, self.mirror_state)
        return mir_uci if self.real_board.turn == chess.WHITE else f.mirror_move(mir_uci)

    @torch.inference_mode()
    def _go_value_only(self) -> str:
        assert self.model is not None
        mover_was_white = self.real_board.turn == chess.WHITE
        legal = list(self.real_board.legal_moves)
        post_states = []
        for move in legal:
            real_uci = move.uci()
            mir_uci = real_uci if mover_was_white else f.mirror_move(real_uci)
            post = self.mirror_state.copy()
            post.push_uci(mir_uci)
            post.apply_mirror()
            post_states.append(post)
        # For 119-plane models, each post_state's history is (current history +
        # current mirror_state) -- the candidate-move state is one ply ahead.
        post_hist = (
            (self._mirror_history + [self.mirror_state])[-7:]
            if self.loaded_in_channels == 119 else None
        )
        inputs = torch.stack(
            [f.prepare_input(s, self.move_counter + 1, history=post_hist)
             for s in post_states]
        ).to(self.device)
        if self.device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with self._model_lock, _amp_ctx(self.device):
            values_t, _ = self.model(inputs)
        mode = str(self.options["ValueScalar"])
        # NN values are from post-move state's player-to-move perspective = opponent. Negate.
        opp_values = value_to_scalar(values_t.float(), mode=mode).cpu().numpy().flatten()
        mover_values = -opp_values
        # Mate-in-1 wins outright.
        for i, ps in enumerate(post_states):
            if ps.is_checkmate():
                mover_values[i] = float("inf")
        # Opening-phase sampling: softmax(values / T) for first TempMoves plies.
        temperature = float(self.options["Temperature"])
        temp_moves = int(self.options["TempMoves"])
        if (temperature > 0 and self.move_counter < temp_moves
                and not np.isinf(mover_values).any()):
            scaled = mover_values / max(temperature, 1e-6)
            scaled = scaled - scaled.max()
            probs = np.exp(scaled); probs /= probs.sum()
            best_idx = int(self._rng.choice(len(legal), p=probs))
        else:
            best_idx = int(np.argmax(mover_values))
        return legal[best_idx].uci()

    def _select_action(self, probs: np.ndarray) -> int:
        temperature = float(self.options["Temperature"])
        temp_moves = int(self.options["TempMoves"])
        if temperature > 0 and self.move_counter < temp_moves:
            scaled = np.where(probs > 0, probs ** (1.0 / temperature), 0.0)
            total = scaled.sum()
            if total > 0:
                return int(self._rng.choice(len(scaled), p=scaled / total))
        return int(np.argmax(probs))

    def cmd_stop(self, _args: list[str]) -> None:
        # Foreground search is single-threaded -- nothing to stop there. But
        # if a ponder thread is running in the background, kill it now.
        self._stop_pondering()

    def cmd_quit(self, _args: list[str]) -> None:
        self._stop_pondering()
        self._quitting = True

    # ---------- main loop ----------

    def run(self) -> None:
        log(f"{ENGINE_NAME} starting, device={self.device}")
        handlers = {
            "uci": self.cmd_uci,
            "isready": self.cmd_isready,
            "ucinewgame": self.cmd_ucinewgame,
            "setoption": self.cmd_setoption,
            "position": self.cmd_position,
            "go": self.cmd_go,
            "stop": self.cmd_stop,
            "quit": self.cmd_quit,
        }
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]
            args = tokens[1:]
            log(f"<- {line}")
            handler = handlers.get(cmd)
            if handler is None:
                log(f"ignoring unknown command: {cmd}")
                continue
            try:
                handler(args)
            except Exception as e:
                log(f"handler {cmd} crashed: {e}\n{traceback.format_exc()}")
            if self._quitting:
                break
        log("bye")


def main() -> None:
    UciEngine().run()


if __name__ == "__main__":
    main()
