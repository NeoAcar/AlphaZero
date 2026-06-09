# CLAUDE.md
Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---
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

- NN value head: `ResNet` and `SEResNet` emit a single scalar in
  `[-1, +1]` via tanh from the side-to-move's POV; `SEResNetWDL` emits
  3-class W/D/L logits, collapsed to a scalar in `[-1, +1]` by
  `value_to_scalar(..., mode=...)` — `"expected"` = `P(W) − P(L)`
  (default), `"win_only"` = `P(W)`. MCTS Q stores the collapsed scalar.
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
  sim). Tree reuse via `update_root(state, move_counter)` (now walks 2
  levels — bot's move + opp's reply — so reuse fires reliably in
  match.py / uci.py too, not just selfplay). μ-FPU on unvisited children
  (visit-weighted Σ Q / Σ N over explored children instead of pessimistic
  FPU=0; +Elo confirmed via internal A/B). O(1) `apply_action(action)`
  for callers that already know which action was played. Default.
- `alphazero/batched_mcts.py` — same algorithm but batches `batch_size`
  leaf evaluations per NN call. Uses Cazenave 2021 "Batch MCTS" tricks:
  **μ-FPU** (unvisited get mean-of-explored Q, not 0) and **Virtual Mean**
  (`Q_eff = (Q + virtual_Q) / (N + virtual_loss)` blends real + in-flight
  stats for next descent). Deduplicates leaves within a batch.
  ~5-10× faster on GPU. Opt-in via `"batched": true` in config.
- Both expose identical interface: `search(state, move_counter,
  info_callback=None, info_interval_s=0.2) → action_probs (np.ndarray,
  shape [4672])`. `info_callback(self, completed, elapsed_s, max_depth)`
  fires every ~200 ms during search — used by `uci.py` to push live ticks
  to the monitoring dashboard and UCI `info` lines.
- `Node.raw_nn_value` caches the value-head output when the node is first
  expanded, so dashboards / diagnostics can compare NN's pre-search verdict
  against the search-refined Q without spending an extra forward pass.

### Network

- `alphazero/nn.py` — three variants registered in
  `uci.ARCHITECTURES`: `ResNet` (plain residual body), `SEResNet`
  (adds Squeeze-and-Excitation blocks), and `SEResNetWDL` (SE body
  with a 3-class W/D/L value head instead of a tanh scalar). All share
  the 4672-output policy head. Pick via UCI `setoption Architecture
  <name>` or `architecture` in player configs. `value_to_scalar(out,
  mode)` collapses the WDL logits back to a scalar for MCTS.

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
  GUI. Configurable via UCI `setoption`: `Type` (mcts|policy_only|
  value_only), `Architecture` (resnet|seresnet|seresnetwdl),
  `ValueScalar` (expected|win_only — WDL collapse mode), `Checkpoint`,
  `Sims`, `Temperature`, `TempMoves`, `DirichletEps`, `DirichletAlpha`,
  `CInit`.
- `alphazero_uci.sh` — shell launcher that cd's into project and execs
  `uci.py` with the venv python. Point lichess-bot config at this.
- `play.py` — local terminal UI for human vs bot.

### Live monitoring dashboard

- `monitor.py` — standalone Flask server (port 8765) that renders a live
  browser dashboard while the bot plays: animated chessboard (chessboard.js),
  vertical eval bar, MCTS depth / sims / nps / move-time counters, parsed
  game clock, and a win-prob line plot growing ply-by-ply (MCTS Q/N green +
  raw NN value blue, both bot's POV). Independent of the engine — run
  separately and open `http://localhost:8765`.
- `uci.py` POSTs telemetry events to `http://localhost:8765/event` by
  default (override via `UCI_MONITOR_URL`, empty value disables). Fail-fast
  with 50 ms timeout, so the engine is unaffected when the monitor isn't
  running.
- Events fire at: `_reset_position` (state), `cmd_go` (go_start + clock),
  `_live_mcts_info` (tick every ~200 ms during search), `_emit_bot_move`
  (bot's move at end of cmd_go, with eval + duration), `_push_move`
  (opp moves). Bot's own move event is emitted **before** `bestmove` to
  separate visually from the opponent's reply that arrives bundled in the
  next `position` command.

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

### Live monitoring dashboard

```bash
uv run python monitor.py        # starts Flask on http://localhost:8765
# then open http://localhost:8765 in a browser
# and run lichess-bot / a chess GUI as usual
```

Disable telemetry emission from `uci.py`:
```bash
UCI_MONITOR_URL= ./alphazero_uci.sh
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

- **`update_root` walks up to two tree levels.** Depth-1 catches the
  selfplay half-move case; depth-2 catches the match.py / uci.py case
  where between two `search()` calls the state advances by bot's move
  + opp's reply (i.e., the new state is a grandchild). Also idempotent:
  no-op if root already matches `state`, so callers can pre-walk via
  `apply_action(action)` and then call `update_root` defensively.
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
- Puzzle fine-tuning (Lichess puzzle CSV → tactical fine-tune)
- Resign threshold in self-play with calibration loop
- 1858-action policy head (LC0-compact, ~5-10% model size win)
- UCI pondering (background search on opp's clock; monitor.py already has
  the infra to display it)
- Retune PUCT `c_init` for the sharper WDL Q distribution (current 1.25
  likely under-explores; try LC0-style ~1.7 + log scaling)
- Variance band on the win-prob plot from WDL: `Var = P(W) + P(L) − (P(W)
  − P(L))²` is free given the WDL head; would visualize search uncertainty
