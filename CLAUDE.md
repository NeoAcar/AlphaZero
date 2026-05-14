# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A chess engine built on the AlphaZero algorithm (MCTS-guided ResNet) with the
full training pipeline: supervised pre-training on Stockfish-labelled Lichess
games, optional self-play improvement, and a UCI wrapper for human play via
Lichess-bot or any chess GUI.

`RECIPE.md` has a detailed comparison of this project's pipeline against the
original AlphaGo Zero and AlphaZero papers — read it before making
non-trivial changes to the training/self-play setup.

## Conventions that are easy to get wrong

### Mirror-canonical state

The bot **never** sees positions from black's perspective. After every push
to the search/training board, the board is mirrored, so the player to move
is always "white" in the bot's mental frame. Three implications:

- `Node.state` in MCTS is the mirror-canonical state, not the real-game
  state. UCI / match.py / play.py / uci.py maintain *both* a `real_board`
  (for game logic and legal-move enumeration) and a `mirror_state` (fed to
  the NN/MCTS). They differ whenever it's black's real-turn.
- `mirror_move(uci_str)` flips coordinates *and preserves the promotion
  suffix* (`a7a8q` → `a2a1q`).
- After `board.push_uci(...)` on the mirror board, call `apply_mirror()` (or
  `board.mirror()`) to keep canonical form.

### 4672 action space (AlphaZero convention)

Policy output is shape `(4672,)`. Index = `8×8×73`: source square (64) × move
type (73 = 56 queen-slide + 8 knight + 9 underpromotion). Auto-queen
promotion uses the regular slide encoding, not a promotion encoding.

- `move_to_alphazero(uci_str)` / `alphazero_to_move(action_idx, board)` are
  the inverse pair. `alphazero_to_move` needs the board to disambiguate
  pawn pushes-to-rank-8 (auto-queen promotion).
- ~1858 of the 4672 indices are legal in some position; the rest are
  permanently dead encodings. `valid_policy(policy, board)` zeroes illegal
  indices and renormalises. **Always apply `valid_policy` before sampling
  from policy output.**

### Value convention

- NN value head outputs a scalar in `[-1, +1]` via tanh, from the
  side-to-move's perspective at the input position.
- Terminal positions are **hard-assigned** via `game_result()`: -1 if the
  player-to-move is mated, 0 for draw. Never +1 (the winner already moved,
  game ended on opp's loss).
- `Node.Q` is stored from the **parent's** perspective.
- `Node.proven_value` is stored from **this node's player-to-move**
  perspective (perspective-free; doesn't break under tree reuse). +1 means
  this player wins from here.

### proven_value semantics

`Node.proven_value` is set on terminal nodes immediately and propagated
upward via `_try_prove`. At search end, if any root child has
`proven_value == -1` (opp loses there), the bot **force-plays** that move
regardless of visit count. This guarantees the bot never misses a found
forced mate due to visit-count noise.

The proof propagates with `OR + (-min) rules`:
- Any child with `proven_value == -1` → this node = +1
- Otherwise need *all legal moves* expanded; this node = `-min(children)`

`Node.n_legal` (number of legal moves at this position) must equal
`len(node.children)` before propagation can claim a `-1` or `0` proof.

## Architecture map

Library code lives in the `alphazero/` package; entrypoint scripts
(`train.py`, `selfplay.py`, `match.py`, `runner.py`, `play.py`, `uci.py`,
`gen_sf_data.py`) sit at the repo root and import from it.

### Search

- `alphazero/mcts.py` — sequential MCTS. Lazy expansion (one NN forward per
  sim). Tree reuse via `update_root(state, move_counter)`. Default.
- `alphazero/batched_mcts.py` — same algorithm but batches `batch_size`
  leaf evaluations per NN call. Uses LC0-style "unscored virtual visit"
  (in-flight sims inflate U-denominator only, don't poison Q). Deduplicates
  leaves within a batch. ~5-10× faster on GPU but slightly weaker per-sim
  due to virtual-loss exploration spread. Opt-in via `"batched": true` in
  config.
- Both expose identical interface: `search(state, move_counter) →
  action_probs (np.ndarray, shape [4672])`.

### Network

- `alphazero/nn.py` — ResNet body + policy head (4672 outputs) + value
  head (single scalar through tanh). Conv → BN → ReLU residual blocks.

### Player abstractions (`alphazero/players.py`)

Used by `match.py` and `runner.py`. Each player loads from a JSON config in
`configs/`. Available types via `PLAYER_TYPES` dict:

- `random` / `piece_value` — no NN, baselines
- `value_only` — one-ply lookahead by NN value head only
- `policy_only` — NN policy head only, argmax over legal moves. Fast.
- `mcts` — full MCTS + policy + value. Set `"batched": true` to use
  BatchedMCTS. Set `"compile": true` to enable `torch.compile` with
  warm-up.
- `stockfish` — wraps external Stockfish via `chess.engine.SimpleEngine`.

### Data pipeline

**Supervised path** (current primary):

1. `gen_sf_data.py` — walks a PGN in mirror-canonical frame, calls Stockfish
   per position (default depth=0 = static eval, fastest), saves sharded
   outputs as `shard_NNNN.pt` under an output dir. Multi-worker via
   `--workers N`. Each worker writes its own subdir; outputs are uint8
   boards (4× smaller than float32), int64 moves, float32 evals via Lichess
   WDL sigmoid. Auto-resume on rerun.
2. `compress_shards.py` — post-process float32 shards to uint8 (legacy;
   `gen_sf_data.py` now writes uint8 directly).
3. `train.py --shards-dir <dir>` — loads all shards under the dir
   (recursive glob handles single-worker and multi-worker subdir layouts),
   splits at game level using `positions_per_game` metadata. `ChessDataset`
   transparently scales `uint8 / 255 → float` at `__getitem__`.

**Self-play path** (not currently used heavily):

1. `selfplay.py --checkpoint X` — runs MCTS-driven self-play, saves
   `selfplay.pt` + `selfplay_summary.json`.
2. `train.py --self-play-data <pt> [--data-mix self_play|both]` — trains on
   soft policy targets (MCTS visit distributions) + game-outcome value
   targets.
3. `runner.py` — orchestrator: self-play → train → match-gate → promote.
   AGZ-proportional cadence: evaluate every `--eval-every N` iterations,
   cumulative training across iters with rollback to best on rejection.

### Live play

- `uci.py` — UCI protocol wrapper. Spoken by lichess-bot and any chess
  GUI. Configurable via UCI `setoption`: `Sims`, `Checkpoint`,
  `Temperature`, `TempMoves`, `DirichletEps`, `DirichletAlpha`, `CInit`.
- `alphazero_uci.sh` — shell launcher that cd's into project and execs
  `uci.py` with the venv python. Point lichess-bot config at this.
- `play.py` — local terminal UI for human vs bot.

## Common commands

All commands use `uv` (Python package manager). `uv run` auto-syncs.

### Setup

```bash
uv sync                                    # install dependencies
```

### Generate supervised training data

```bash
uv run python gen_sf_data.py \
    --pgn data/games.pgn \
    --depth 0 --shard-games 5000 --workers 6 \
    --output-dir data/sf_shards
```

### Train

```bash
# Sharded supervised data
uv run python train.py \
    --shards-dir data/sf_shards \
    --epochs 10 --batch-size 256 --learning-rate 1e-4 \
    --val-fraction 0.05 \
    --checkpoint-dir checkpoints/run_v1 \
    --log-dir logs/run_v1 \
    --wandb-project alphazero-chess --wandb-name run_v1

# With self-play data mixed in
uv run python train.py --self-play-data runs/r1/iter_001/selfplay.pt \
                       --data-mix self_play [...]
```

`--vals-per-epoch N` runs N validations per epoch, evenly spaced
(default 1). Best-by-metric checkpoints update on each val pass.

### Evaluation match

```bash
uv run python match.py \
    --player1 configs/new_model.json --player2 configs/old_model.json \
    --games 10 --p1-name new --p2-name old
```

Player1 plays white in odd-numbered games. Logs include final FEN +
termination reason per game.

### Self-play loop

```bash
uv run python runner.py \
    --workdir runs/r1 \
    [--initial-checkpoint models/seed.pth | omit to start from random] \
    --iters 50 --eval-every 7 \
    --wandb-project alphazero-chess --wandb-name r1
```

If `--initial-checkpoint` is omitted, `runner.py` creates a fresh randomly-
initialized ResNet at `runs/r1/best.pth`.

### Run the bot as a UCI engine

```bash
./alphazero_uci.sh    # for lichess-bot to spawn
```

Test it directly:
```bash
echo -e "uci\nposition startpos moves e2e4\nisready\ngo\nquit" \
    | ./alphazero_uci.sh
```

## Performance knobs

- **`torch.compile(mode="reduce-overhead")`** — applied in
  `alphazero/players.py` model load paths. ~1.5-3× speedup at batch=1 GPU.
  First forward is slow (JIT trace ~20s), so we warm up immediately after
  compile.
- **`torch.set_float32_matmul_precision("high")`** — enables TF32 on
  Ampere+ GPUs. ~2× matmul speedup, ~1e-3 logit precision loss
  (irrelevant for move selection). Set globally in `alphazero/players.py`.
- **Batched MCTS** (`"batched": true` in config) — 5-10× faster per
  wall-clock but ~weaker per-sim. Best for self-play throughput, not
  per-game strength matches.

## Gotchas

- **`update_root` only walks one tree level.** Across full bot turns
  (bot move + opp move), the new state is a grandchild of the previous
  root, so tree reuse doesn't fire and the tree resets. Reuse fires
  reliably in `selfplay.py` (which calls `search` every half-move).
- **The 4672 policy array has dead indices.** ~2814 of 4672 are
  geometrically impossible moves (e.g., knight jumps off the board from
  edge squares). Cross-entropy gradients naturally push them to large
  negative logits during training; explicit masking is unnecessary.
- **`gen_sf_data.py` outputs are uint8 boards**: `ChessDataset` rescales
  by `/255.0` at `__getitem__` time. If you add a new dataset class that
  reads shards, do the same.
- **Self-play `dirichlet_alpha` is 0.3 (chess value).** Was 0.03 (Go's)
  in old configs — fixed across all configs and defaults.
- **`runs/`, `wandb/`, `data/`, `models/`, `checkpoints/` are gitignored.**
  Don't commit large binaries; the model weights are too big for GitHub.

## Deferred work

Tracked as memory items for future Claude sessions; surface when relevant:

- 19 → 119 input planes (add 8-frame history)
- WDL value head (3-class instead of single tanh scalar; Stockfish has
  `UCI_ShowWDL` for direct targets)
- Puzzle fine-tuning (Lichess puzzle CSV → tactical fine-tune)
- Resign threshold in self-play with calibration loop
- Virtual losses + batched NN eval already implemented in `alphazero/batched_mcts.py`
- 1858-action policy head (LC0-compact, ~5-10% model size win)
