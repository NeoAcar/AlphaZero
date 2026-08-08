# Future Ideas and Roadmap

This is the authoritative list of open ideas for the project. It is intentionally
limited to work that is not already implemented. `fable_review.md` remains useful
as historical research and rationale, but its action plan describes an older
version of the repository.

Last reconciled with the codebase: 2026-08-08.

## Current baseline

The following capabilities already exist and should not be proposed as new work:

- Mirror-canonical chess state and the 4,672-action AlphaZero policy encoding.
- Legacy 19-plane inputs and 119-plane eight-position history, auto-detected from
  checkpoint weights.
- ResNet, SE-ResNet, and SE-ResNet-WDL models.
- Sequential and batched MCTS with log-scaled PUCT, mu-FPU, cFPU, Virtual Mean,
  proof propagation, repetition handling, two-ply tree reuse, early stopping, and
  a reusable simulation bank.
- Cross-game leaf batching through `MultiGameSearcher`.
- Playout Cap Randomization (PCR), sparse policy targets, high-simulation policy
  masking, resign playthrough sampling, and resignation calibration metadata.
- Scalar and WDL training, game-level validation splitting, AMP, channels-last
  tensors, CUDA-oriented data loading, and supervised/self-play data mixing.
- Persistent UCI serving, background pondering, model warm-up, live search
  telemetry, and WDL variance display in the monitoring dashboard.

## Priority roadmap

| Priority | Idea | Main goal | First useful experiment |
|---|---|---|---|
| P1 | Tune WDL-aware search | More Elo from the existing network | Sweep `c_init`, `c_factor`, and `c_fpu` at equal NN evaluations |
| P1 | Gumbel AlphaZero root search | Strong policy improvement at low simulation counts | Implement root-only Gumbel top-k plus Sequential Halving behind a flag |
| P1 | Forced playouts and target pruning | Explore broadly without polluting policy targets | Add only to high-simulation PCR positions and compare policy entropy/Elo |
| P1 | Better replay and value targets | Improve sample efficiency and reduce forgetting | Increase the replay window, then test short-horizon MCTS value targets |
| P2 | Transposition-aware MCTS | Avoid repeated NN evaluations in transposing chess lines | Add a search-local cache with strict history/repetition identity checks |
| P2 | Auxiliary network heads | Extract more learning signal from every game | Start with moves-left and soft-policy auxiliary heads |
| P2 | Central inference service | Saturate the GPU across independent self-play actors | Generalize `MultiGameSearcher` into an actor/evaluator queue |
| P2 | Distilled inference model | Increase nodes per second without losing much strength | Distill the strongest model into a narrower SE-ResNet |
| P3 | Compact policy representation | Reduce policy-head compute and checkpoint size | Prototype an LC0-style legal-move mapping with conversion tests |
| P3 | Broader training positions | Improve tactical and positional coverage | Add puzzle and diverse-start-position importers |

## Search ideas

### WDL-aware PUCT and simulation allocation

The WDL head already exposes both expected value and outcome variance. Use that
signal instead of treating every position identically:

- Benchmark the current `c_init`/`c_factor` defaults before changing the formula.
- Test variance-scaled PUCT exploration.
- Test uncertainty-gated simulation budgets: spend fewer simulations on stable
  roots and more on sharp roots.
- Compare every variant at equal NN evaluations and equal wall-clock time.

### Gumbel AlphaZero

Implement Gumbel root action sampling and Sequential Halving as an optional search
mode. Keep ordinary PUCT below the root initially. The important comparisons are
low-simulation strength, self-play policy quality, and training stability—not just
nodes per second.

### Forced playouts with policy-target pruning

Force minimum exploration for plausible root moves during high-simulation PCR
searches, then subtract visits that were forced but never justified by value. This
should be implemented as one feature: forced playouts without pruning can train
the policy to reproduce search bookkeeping rather than move quality.

### Transpositions and tablebases

- Prototype a search-local transposition cache before a persistent DAG.
- A cache key must include every input that changes evaluation: board state,
  halfmove/repetition context, and the relevant 119-plane history. Board layout
  alone is not safe.
- Consider Syzygy probes for small endgames. Proven results can integrate with
  `Node.proven_value` and can also provide exact training labels.

## Training ideas

### Replay and target quality

- Increase the self-play replay window beyond the current small recent-generation
  window and measure catastrophic forgetting.
- Test freshness-weighted or surprise-weighted sampling.
- Mix final game outcome with short-horizon future MCTS values to reduce value
  target variance. Keep final outcomes as an anchor.
- Record calibration metrics for the resign threshold, not only the disabled-
  resign samples needed to calculate them.

### Auxiliary supervision

- Soft-policy auxiliary head trained on a higher-temperature target.
- Moves-left/game-length head.
- Puzzle fine-tuning from Lichess puzzle positions.
- Stockfish MultiPV policy distillation, complementing the existing value/WDL
  supervision.
- EMA or SWA checkpoints as a low-cost strength/stability experiment.

### Smaller and faster networks

- Distill a strong teacher into a narrower or shallower SE-ResNet.
- Evaluate the student by Elo per millisecond, not parameter count alone.
- Only investigate transformer-style bodies if they improve strength per unit of
  inference cost; they are not automatically a latency improvement.

## Systems and serving ideas

- Turn cross-game batching into a centralized evaluator used by multiple
  self-play processes or machines.
- Explore exported inference backends such as TensorRT or ONNX Runtime after the
  model/input interface is stable.
- Profile a native bitboard encoder and move generator if Python-side search
  becomes the throughput bottleneck.
- Add actual UCI clock management. `uci.py` currently reports clock fields to the
  dashboard but still searches by a fixed simulation budget.
- Add distributed actors only after session metadata, checkpoint identity, and
  interrupted-write recovery are reliable across processes.

## Experimental backlog

These are plausible but lower-confidence or higher-complexity ideas. They should
not displace the priority roadmap without supporting measurements:

- MCTS as regularized policy optimization.
- Two-network leaf cascade: a small triage network plus a strong network for
  high-importance leaves.
- Prioritized self-play replay.
- Async actor/learner training with intentionally stale actor weights.
- Attention-based network bodies.
- EfficientZero/MuZero-style consistency objectives.

## Evaluation rules

Use the same discipline for every roadmap experiment:

1. Change one algorithmic variable at a time and keep a configuration for both
   control and candidate.
2. Report NN evaluations, wall-clock time, nodes per second, maximum depth, and
   game result—not only win rate.
3. Disable Dirichlet noise for strength matches and alternate colors.
4. Run compatibility checks for both 19-plane and 119-plane checkpoints unless
   the change explicitly retires the legacy format.
5. Do not promote a promising small match as established strength; retain raw
   PGNs and rerun with more games.

## Completed-history sources

- `fable_review.md` contains the original broad review, literature notes, and
  implementation recommendations. Many of its engineering items are now done.
- `RECIPE.md` compares this project with AlphaGo Zero and AlphaZero and explains
  the current gated training philosophy.
- Git history is the source of truth for when completed roadmap items landed.
