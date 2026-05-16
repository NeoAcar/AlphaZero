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
    CInit            (string)  PUCT exploration         default: 1.25

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
import sys
import traceback

import chess
import numpy as np
import torch

from alphazero import utils as f
from alphazero.mcts import MCTS
from alphazero.nn import ResNet, SEResNet, SEResNetWDL, value_to_scalar


ARCHITECTURES = {
    "resnet": ResNet,
    "seresnet": SEResNet,
    "seresnetwdl": SEResNetWDL,
}

PLAYER_TYPES = {"mcts", "policy_only", "value_only"}


torch.set_float32_matmul_precision("high")


ENGINE_NAME = "AlphaZeroBot"
ENGINE_AUTHOR = "NeoAcar"


def log(msg: str) -> None:
    """Debug logging to stderr (UCI requires stdout for protocol)."""
    sys.stderr.write(f"[uci] {msg}\n")
    sys.stderr.flush()


def send(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


class UciEngine:
    DEFAULT_OPTS = {
        "Type": "policy_only",              # one of: mcts | policy_only | value_only
        "Backend": "onnx",                  # policy_only-only: "torch" or "onnx"
        "OnnxPath": "models/policy_seresnetwdl.fp16.onnx",  # required if Backend=onnx
        "Checkpoint": "models/model_best_combined_wdl.pth",
        "Architecture": "seresnetwdl",
        "ValueScalar": "expected",   # WDL collapse mode: "expected" (P(W)-P(L)) or "win_only" (P(W))
        "Sims": 1200,
        "Temperature": 1.0,
        "TempMoves": 6,
        "DirichletEps": 0.0,
        "DirichletAlpha": 0.3,
        "CInit": 1.25,
    }
    OPT_TYPES = {
        "Type": ("string", None, None),
        "Backend": ("string", None, None),
        "OnnxPath": ("string", None, None),
        "Checkpoint": ("string", None, None),
        "Architecture": ("string", None, None),
        "ValueScalar": ("string", None, None),
        "Sims": ("spin", 1, 100000),
        "Temperature": ("string", None, None),
        "TempMoves": ("spin", 0, 200),
        "DirichletEps": ("string", None, None),
        "DirichletAlpha": ("string", None, None),
        "CInit": ("string", None, None),
    }

    def __init__(self) -> None:
        self.options = dict(self.DEFAULT_OPTS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.loaded_arch: str | None = None
        self.loaded_checkpoint: str | None = None
        self.loaded_profile: str | None = None
        self._policy_fn = None
        self._policy_fast = False
        self._ort_runner = None              # OnnxPolicyRunner when Backend=onnx
        self._ort_path: str | None = None
        self.mcts: MCTS | None = None
        self.real_board = chess.Board()
        self.mirror_state = chess.Board()
        self.move_counter = 0
        self._rng = np.random.default_rng()
        self._quitting = False

    # ---------- model / mcts plumbing ----------

    def _ensure_loaded(self) -> None:
        ptype = str(self.options["Type"]).lower()
        backend = str(self.options.get("Backend", "torch")).lower()
        onnx_path = str(self.options.get("OnnxPath", ""))

        # ONNX policy backend: load an ORT session, no torch model needed.
        if ptype == "policy_only" and backend == "onnx":
            if self._ort_runner is not None and self._ort_path == onnx_path:
                return
            if not onnx_path:
                raise ValueError("Backend=onnx requires OnnxPath to be set")
            from alphazero.onnx_io import OnnxPolicyRunner
            log(f"Loading ONNX policy graph {onnx_path}")
            self._ort_runner = OnnxPolicyRunner(
                onnx_path,
                device="cuda" if self.device.type == "cuda" else "cpu",
                fp16=True,
            )
            self._ort_path = onnx_path
            self.loaded_profile = "onnx"
            self.model = None
            self.mcts = None
            self.loaded_checkpoint = None
            self.loaded_arch = None
            return

        # Torch backend (any Type): drop any ORT state from a prior switch.
        self._ort_runner = None
        self._ort_path = None

        ckpt = str(self.options["Checkpoint"])
        arch = str(self.options["Architecture"]).lower()
        # policy_only on CUDA gets a fp16 + channels_last + max-autotune model;
        # other Types need fp32 since MCTS/value paths run the full forward.
        profile = "policy_fast" if (ptype == "policy_only" and self.device.type == "cuda") else "default"
        if (self.model is not None and self.loaded_checkpoint == ckpt
                and self.loaded_arch == arch and self.loaded_profile == profile):
            self._refresh_mcts()
            return
        if arch not in ARCHITECTURES:
            raise ValueError(f"Architecture must be one of {list(ARCHITECTURES)}, got {arch!r}")
        log(f"Loading checkpoint {ckpt} (architecture: {arch}, profile: {profile})")
        model = ARCHITECTURES[arch]().to(self.device)
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        self._policy_fn = None
        self._policy_fast = profile == "policy_fast"
        if self._policy_fast:
            model = model.half().to(memory_format=torch.channels_last)
            try:
                self._policy_fn = torch.compile(model.forward_policy, mode="max-autotune")
                warm = torch.zeros(
                    1, 19, 8, 8, device=self.device, dtype=torch.float16,
                ).to(memory_format=torch.channels_last)
                with torch.inference_mode():
                    _ = self._policy_fn(warm)
                log("fp16/channels_last/max-autotune policy_fn warmed")
            except Exception as e:
                log(f"policy fast-path compile skipped: {e}")
                self._policy_fn = model.forward_policy
        else:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                with torch.no_grad():
                    _ = model(torch.zeros(1, 19, 8, 8, device=self.device))
                log("torch.compile + warm-up done")
            except Exception as e:
                log(f"torch.compile skipped: {e}")
        self.model = model
        self.loaded_checkpoint = ckpt
        self.loaded_arch = arch
        self.loaded_profile = profile
        self._refresh_mcts()

    def _refresh_mcts(self) -> None:
        args = {
            "num_simulation": int(self.options["Sims"]),
            "truncation": 1000,
            "c_base": 19652,
            "c_init": float(self.options["CInit"]),
            "dirichlet_epsilon": float(self.options["DirichletEps"]),
            "dirichlet_alpha": float(self.options["DirichletAlpha"]),
            "memory_size": 1000,
            "action_space": 4672,
            "t": 1,
            "device": self.device,
            "value_scalar": str(self.options["ValueScalar"]),
        }
        if self.model is None:
            # ONNX backend: MCTS isn't used for policy_only ONNX path.
            self.mcts = None
            return
        self.mcts = MCTS(args, self.model)

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
        if self.mcts is not None:
            self.mcts.root = None

    def _push_move(self, uci_real: str) -> None:
        """Apply a move (in real-coord UCI) to both boards."""
        mover_was_white = self.real_board.turn == chess.WHITE
        self.real_board.push_uci(uci_real)
        mir_uci = uci_real if mover_was_white else f.mirror_move(uci_real)
        self.mirror_state.push_uci(mir_uci)
        self.mirror_state = self.mirror_state.mirror()
        self.move_counter += 1

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
        self._ensure_loaded()
        send("readyok")

    def cmd_ucinewgame(self, _args: list[str]) -> None:
        self._reset_position()
        log("ucinewgame: reset")

    def cmd_setoption(self, args: list[str]) -> None:
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
            self.options[name] = value
        log(f"set {name} = {self.options[name]!r}")
        # Anything that changes which model/graph we run forces a reload.
        if name in ("Checkpoint", "Architecture", "Type", "Backend", "OnnxPath"):
            self.model = None
            self.loaded_checkpoint = None
            self.loaded_arch = None
            self.loaded_profile = None
            self._policy_fn = None
            self._policy_fast = False
            self._ort_runner = None
            self._ort_path = None

    def cmd_position(self, args: list[str]) -> None:
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
        self._reset_position(start_fen)
        for m in moves:
            self._push_move(m)
        log(f"position set: {len(moves)} moves applied, "
            f"turn={'w' if self.real_board.turn == chess.WHITE else 'b'}, "
            f"counter={self.move_counter}")

    def cmd_go(self, _args: list[str]) -> None:
        # Time controls are ignored for now; we just use configured Sims.
        self._ensure_loaded()
        # Either a torch model OR an ORT runner must be ready by now.
        assert self.model is not None or self._ort_runner is not None
        ptype = str(self.options["Type"]).lower()
        if ptype not in PLAYER_TYPES:
            log(f"unknown Type {ptype!r}; falling back to mcts")
            ptype = "mcts"

        try:
            if ptype == "mcts":
                real_uci = self._go_mcts()
            elif ptype == "policy_only":
                real_uci = self._go_policy_only()
            else:  # value_only
                real_uci = self._go_value_only()
        except Exception as e:
            log(f"search failed ({ptype}): {e}\n{traceback.format_exc()}")
            legal = list(self.real_board.legal_moves)
            real_uci = legal[0].uci() if legal else "0000"

        # Sanity: verify legal on real board; fall back to any legal otherwise.
        try:
            move = chess.Move.from_uci(real_uci)
            if move not in self.real_board.legal_moves:
                log(f"chose illegal {real_uci}; falling back to a legal move")
                real_uci = next(iter(self.real_board.legal_moves)).uci()
        except Exception:
            log(f"chose unparseable {real_uci}; bailing")
            legal = list(self.real_board.legal_moves)
            real_uci = legal[0].uci() if legal else "0000"
        send(f"bestmove {real_uci}")

    # ---------- per-Type search backends ----------

    def _go_mcts(self) -> str:
        assert self.mcts is not None
        with contextlib.redirect_stdout(sys.stderr):
            probs = self.mcts.search(self.mirror_state, self.move_counter)
        if getattr(self.mcts, "last_was_proven_mate", False):
            send("info string proven forced mate")
            log("proven forced mate")
        action = self._select_action(probs)
        mir_uci = f.alphazero_to_move(action, self.mirror_state)
        return mir_uci if self.real_board.turn == chess.WHITE else f.mirror_move(mir_uci)

    def _go_policy_only(self) -> str:
        legal_idx = np.fromiter(
            (f.move_obj_to_alphazero(m) for m in self.mirror_state.legal_moves),
            dtype=np.int64,
        )
        if self._ort_runner is not None:
            planes = f.board_to_matrix(self.mirror_state, self.move_counter)
            policy_logits = self._ort_runner(planes)  # (4672,) numpy
            legal_logits = policy_logits[legal_idx]
        else:
            assert self.model is not None
            with torch.inference_mode():
                inputs = f.prepare_input(self.mirror_state, self.move_counter).unsqueeze(0).to(self.device)
                if self._policy_fast:
                    inputs = inputs.half().to(memory_format=torch.channels_last)
                fn = self._policy_fn if self._policy_fn is not None else self.model.forward_policy
                policy_logits_t = fn(inputs).squeeze(0)
                idx_t = torch.tensor(legal_idx, device=self.device, dtype=torch.long)
                legal_logits = policy_logits_t.index_select(0, idx_t).float().cpu().numpy()

        temperature = float(self.options["Temperature"])
        temp_moves = int(self.options["TempMoves"])
        if temperature > 0 and self.move_counter < temp_moves:
            shifted = legal_logits - legal_logits.max()
            legal_probs = np.exp(shifted, dtype=np.float64)
            legal_probs /= legal_probs.sum()
            scaled = legal_probs ** (1.0 / temperature)
            total = scaled.sum()
            if total > 0:
                pos = int(self._rng.choice(len(scaled), p=scaled / total))
            else:
                pos = int(np.argmax(legal_probs))
        else:
            pos = int(np.argmax(legal_logits))
        action = int(legal_idx[pos])
        mir_uci = f.alphazero_to_move(action, self.mirror_state)
        return mir_uci if self.real_board.turn == chess.WHITE else f.mirror_move(mir_uci)

    @torch.no_grad()
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
        inputs = torch.stack(
            [f.prepare_input(s, self.move_counter + 1) for s in post_states]
        ).to(self.device)
        values_t, _ = self.model(inputs)
        mode = str(self.options["ValueScalar"])
        # NN values are from post-move state's player-to-move perspective = opponent. Negate.
        opp_values = value_to_scalar(values_t, mode=mode).cpu().numpy().flatten()
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
        # Single-threaded MCTS; nothing to stop. The 'bestmove' from the
        # in-flight 'go' (if any) will arrive whenever it finishes.
        pass

    def cmd_quit(self, _args: list[str]) -> None:
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
