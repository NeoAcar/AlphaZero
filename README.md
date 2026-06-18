# AlphaZero Chess

An AlphaZero-style chess engine: an MCTS-guided residual network with the full
training pipeline. The primary path is supervised pre-training on
Stockfish-labelled Lichess games; optional self-play improvement and a UCI
wrapper let the bot play in any chess GUI or on Lichess via lichess-bot.

See **`CLAUDE.md`** for the conventions that are easy to get wrong
(mirror-canonical state, the 4672 action space, value perspective,
proven-value semantics) and **`RECIPE.md`** for how this pipeline compares to
the AlphaGo Zero / AlphaZero papers.

## File structure

Entrypoint scripts sit at the repo root; library code lives in `alphazero/`.

- **`gen_sf_data.py`**: Builds Stockfish-labelled training shards from a PGN (multi-worker, uint8 boards).
- **`train.py`**: Trains the network from sharded supervised data and/or self-play `.pt` files.
- **`selfplay.py`**: Generates self-play games (sequential or cross-game batched).
- **`runner.py`**: Orchestrates the self-play → train → match-gate → promote loop.
- **`match.py`**: Plays two players against each other (any mix of NN/MCTS/Stockfish/baselines).
- **`ladder.py`**: Runs a new model against a ladder of opponents.
- **`play.py`**: Local terminal UI for human vs bot or bot vs bot.
- **`uci.py`**: UCI protocol wrapper so any chess GUI (or lichess-bot) can drive the engine.
- **`engine_server.py` / `engine_client.py`**: Persistent engine daemon + thin client (lichess-bot spawns the client; the daemon keeps the model warm).
- **`monitor.py`**: Standalone Flask dashboard (port 8765) — live board, eval bar, MCTS counters, win-prob plot.
- **`analytics.py`**: Self-play / game-log analysis reports.
- **`alphazero/`**: Library package.
  - **`mcts.py` / `batched_mcts.py`**: Sequential and batched MCTS (tree reuse, μ-FPU, log-scaling PUCT).
  - **`nn.py`**: Network bodies + heads — `ResNet`, `SEResNet` (Squeeze-and-Excitation), and `SEResNetWDL` (3-class W/D/L value head). All share the 4672-output policy head.
  - **`utils.py`**: Board / move / policy encoding helpers.
  - **`dataset.py`**: PyTorch datasets for supervised shards and self-play `.pt` files.
  - **`players.py`**: Player abstractions (random / piece_value / value_only / policy_only / mcts / stockfish) used by `match.py` and `runner.py`.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/). One command installs
everything, including a CUDA-enabled PyTorch build (resolved from the
`pytorch-cu124` index in `pyproject.toml` on Linux/Windows):

```bash
uv sync
```

All commands are run via `uv run` (which auto-syncs).

## Quickstart

```bash
# Generate Stockfish-labelled training data from a PGN
uv run python gen_sf_data.py --pgn data/games.pgn --depth 0 \
    --shard-games 5000 --workers 6 --output-dir data/sf_shards

# Train
uv run python train.py --shards-dir data/sf_shards \
    --epochs 10 --batch-size 256 --checkpoint-dir checkpoints/run_v1

# Evaluate two players against each other
uv run python match.py --player1 configs/new_model.json \
    --player2 configs/old_model.json --games 10

# Run as a UCI engine (lichess-bot points at this launcher)
./alphazero_uci.sh

# Live dashboard, then open http://localhost:8765
uv run python monitor.py
```

See `CLAUDE.md` for the full command reference, performance knobs, and gotchas.

## Model weights and training data

The model weights, Lichess games, and evaluation result files are too large
for GitHub (and are gitignored). Download them here:

[Model weights and results](https://drive.google.com/drive/folders/12PGjUCOllXaKWY-uzP7fr_iY1kj9HhTz?usp=sharing)
