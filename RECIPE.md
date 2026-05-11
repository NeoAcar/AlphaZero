# Self-Play / Self-Learn Recipe

This project vs. the two original DeepMind papers it descends from. "Ours" is
what `runner.py` does today (with the per-iteration scripts `selfplay.py`,
`train.py`, `match.py`).

## At-a-glance comparison

| Aspect | AlphaGo Zero (AGZ, 2017, Go) | AlphaZero (AZ, 2018, Chess) | **Ours (current)** |
|---|---|---|---|
| Game | Go 19×19 | Chess (also Shogi, Go) | Chess |
| Training style | Discrete generations + gate | Continuous, no gate | Discrete generations + gate (AGZ-style) |
| Eval gate to promote? | Yes — every 1000 grad steps; must win > 55% of 400 games | **No** — every checkpoint goes live | Yes — every ~7 iters (~1000 grad steps); must win ≥ 55% of 10 games |
| MCTS sims / move | 1600 | 800 | 200 |
| `c_init`, `c_base` | 1.25, 19652 | 1.25, 19652 | 1.25, 19652 ✓ |
| Dirichlet α | 0.03 (Go branching ~250) | **0.3 for chess** (branching ~35) | 0.3 ✓ |
| Dirichlet ε at root | 0.25 | 0.25 | 0.25 ✓ |
| Temperature schedule | τ=1 for 30 plies, then ~0 | τ=1 for 30 plies, then ~0 | τ=1 for 30 plies, then 0 ✓ |
| Self-play games per cycle | 25,000 | continuous; 44M lifetime games | 50 per iteration |
| Training data buffer | Last 500k games sliding | Last 500k games sliding | Last 3 self-play files (~150 games) |
| Mini-batch size | 2048 | 4096 | 256 |
| Optimizer | SGD + momentum 0.9 | SGD + momentum 0.9 | AdamW |
| Learning rate | 0.01 → 0.001 → 0.0001 (stepped) | 0.2 → 0.02 → 0.002 → 0.0002 (stepped) | 1e-4 fixed |
| Weight decay (L2) | 1e-4 | 1e-4 | 1e-4 ✓ |
| ResNet blocks × filters | 40 × 256 | 20 × 256 (chess) | smaller (`resnet.py`) |
| Policy target | MCTS visit distribution π | MCTS visit distribution π | MCTS π ✓ |
| Value target | Game outcome z ∈ {-1, 0, 1} | Game outcome z | Game outcome z ✓ |
| Terminal positions in training data? | No | No | No ✓ |
| Top-K action cap during search? | None | None | None ✓ (removed) |
| Input planes | 17 (8 history × 2 + colour) | 119 for chess (8-frame history) | 19 (no history) |
| Value head output | Single scalar tanh | Single scalar tanh | Single scalar tanh ✓ |

✓ = matches a paper exactly.

## Pipeline (one iteration of `runner.py`)

1. **Self-play** — `selfplay.py` plays N games using current `best.pth`.
   Records `(board_matrix, mcts_π, game_outcome_z)` per non-terminal position.

2. **Train** — `train.py` trains a copy on the last K self-play files (rolling
   window, default 3). Optionally mixes in supervised Stockfish-eval data.
   3 epochs, AdamW, soft-CE policy + MSE value.

3. **Evaluation gate** — `match.py` plays candidate vs current best for
   10 games, `dirichlet_eps=0.0` so it tests *skill*, not exploration.

4. **Promote or reject** — if candidate win-rate ≥ 0.55, copy
   `model_last.pth` over `best.pth`. Otherwise discard.

## Why our setup differs (and what's worth fixing)

**Hybrid AGZ/AZ pipeline.** The discrete-iteration + eval-gate structure is
from AGZ. AlphaZero ripped the gate out — they trusted continuous training to
self-correct. The gate makes sense for us because we run small iterations
(50 games × 200 sims) where a noisy single iteration *can* produce a worse
model; the gate prevents regressions.

**Scale.** AZ's 44M lifetime games × 800 sims is several orders of magnitude
above what a single GPU can produce. The recipe above is the same shape;
just at a smaller scale.

**Dirichlet α=0.3 (resolved).** The AZ paper scales α with branching factor:
0.03 for Go (~250 moves), 0.15 for shogi, **0.3 for chess**. We previously
had 0.03 (Go's value) inherited from old code; all configs and defaults are
now 0.3.

**Top-K action cap (removed).** AZ has no cap — every legal move stays in the
search. PUCT's prior weighting naturally focuses budget on high-prior moves
without an explicit cutoff. The `top_actions` knob is gone from `mcts.py`.

**Optimizer choice.** Both papers used vanilla SGD + momentum with a stepped
LR schedule. We use AdamW with a flat 1e-4. Adam's adaptivity is convenient
at small scales but tends to find slightly worse minima than tuned SGD on
big training runs. Worth revisiting if scaling up.

**Buffer size.** A 3-file rolling window is tiny compared to AZ's 500k-game
window. Each iteration trains on essentially the most recent generation's
games — risk of catastrophic forgetting if the policy distribution shifts
between iterations. Increasing `--keep-selfplay-iters` is cheap.

**Input planes.** Both AGZ and AZ feed the last 8 board positions as input
history so the network can detect repetition/3-fold draws. Ours uses only
the current position. Already on the deferred list.

## Faithful-to-paper checklist

If the goal is to converge toward AZ rather than AGZ, the smallest set of
changes would be:

1. ~~Set `dirichlet_alpha=0.3`~~ done
2. Drop the eval gate from `runner.py` (every iteration promotes)
3. ~~Remove `top_actions` cap~~ done
4. Increase buffer to many more recent iterations
5. Switch optimizer to SGD + momentum with a stepped LR schedule
6. Add 8-frame history to input planes (deferred)
