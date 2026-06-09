# Fable Review — AlphaZero Chess Engine

_Multi-agent code review. Method: 9 subsystem reviewers read their files in full and
cross-checked against the CLAUDE.md invariants; a skeptic adversarially tried to refute
every bug-class finding (killing false positives); a synthesis pass deduped and
prioritized. 45 findings survived verification._

## Headline

The **engine play path is sound.** The sequential MCTS core (`alphazero/mcts.py`) was
verified correct against *every* invariant — mirror-canonical state, parent-vs-node Q
perspective, terminal hard-assignment, the `proven_value` solver, and 2-level tree reuse.
No sign / perspective / encoding bugs in search.

The damage is concentrated in **training & self-play tooling**, plus the long-standing
**BatchedMCTS weakness, which now has a concrete root cause.**

Cleared false alarm: there is **no** eval/board perspective mismatch in `gen_sf_data.py` —
it correctly uses `.pov(board.turn)` matching the canonical side-to-move.

---

## 🔴 Critical / High — silent corruption or fully broken

### 1. Self-play loop is dead end-to-end (`runner.py` ↔ `selfplay.py`) — *critical*
The uncommitted `selfplay.py` rewrite (+733 lines) broke its contract with `runner.py` on
two coupled axes:
- **Args:** runner passes `--sims --truncation --output` (runner.py ~198-206); refactored
  selfplay only knows `--high-sims/--low-sims --max-plies --output-dir` (selfplay.py
  ~376-411). argparse rejects with "unrecognized arguments" before any work runs.
- **Output layout:** new selfplay writes nested `out_dir/<ckpt>/games_*.pt` fragments;
  runner globs flat `iter_*/selfplay.pt` in `latest_selfplay_files()` and finds zero files.

Every self-play iteration fails immediately. **Fix both as one atomic change** or runner
still finds nothing. Smoke-test one full iteration end-to-end.

### 2. BatchedMCTS dedup skips backprop → undercounts visits — *high*
`batched_mcts.py:302-318, 366-398`. **Likely root cause of the known "BatchedMCTS plays
weaker" bug.** When ≥2 sims in a batch hit the same unexpanded leaf, only the first is
evaluated/backpropagated; duplicates have `virtual_loss` reverted but never receive the NN
value (`in_flight` stays None, Phase 3 skips on `value is None`). A leaf hit by *k* sims
gets `N += 1` instead of `N += k` — visit counts systematically undercounted, biasing
exploration. Double-backprop of the shared value is provably safe (Q/N average unchanged,
sign alternation preserved). **Fix:** propagate the single evaluated value to all duplicate
`in_flight` entries on the same leaf, drop them from `skip`, let every sim backprop. Re-run
the batched-vs-sequential A/B. (Comment at :302-306 claiming it "matches sequential MCTS"
is wrong and masks the bug.)

### 3. WDL one-hot target drops z=±0.5 to all-zeros — *high*
`train.py:522-529`. Strict inequalities `W=(z>0.5), D=(-0.5<z<0.5), L=(z<-0.5)` mean
`z==±0.5` produces an all-zero target → ~zero value-head gradient. Self-play z∈{-1,0,+1}
is safe, but **supervised WDL evals legitimately land on ±0.5**, silently zeroing value
learning for that slice. **Fix:** boundary-inclusive bucketing (`W: z>=0.5, L: z<=-0.5, D
otherwise`) or build soft WDL targets from the eval distribution directly; assert rows
sum to 1.

### 4. uint8 quantization clips 19-plane move-counter — *medium*
`gen_sf_data.py:309-310`, `utils.py:56`. 19-plane sets plane 13 = `move_counter/300`
*unclamped*; quantization does `clip(0,1)*255`. Every game ≥ ply 300 maps to 255 → 1.0,
so moves 300/450/600 are indistinguishable. (119-plane path already uses
`min(move_counter/500, 1.0)`.) **Fix:** clamp `move_counter/500` inside
`_board_to_matrix_19`, regenerate affected 19-plane shards.

---

## 🟡 Medium — correctness-adjacent / quality

- **`match.py` never calls `set_history()`** (`match.py:47-72`) — 119-plane models in eval
  matches start every root with empty/zero-padded history, degrading search vs
  selfplay/uci. Mirror the selfplay history-tracking.
- **`uci.loaded_in_channels` not initialized in `__init__`** (`uci.py:151-195`) — read
  directly at :903/:933 with no fallback; any path reaching those before load →
  `AttributeError`. Init to 19 (or `getattr` fallback like `_refresh_mcts`).
- **`valid_policy()` promised by CLAUDE.md but doesn't exist** — only `legal_mask` exists;
  masking inlined in MCTS. Add the helper (mask + zero + renorm with zero guard); route
  callers through it so diagnostics/selfplay don't grow divergent ad-hoc masking.
- **Silent draw substitution on eval failure** (`gen_sf_data.py:116,138`) — engine failure
  → `(0,1,0)` draw, missing WDL → NaN, no audit trail. If failures correlate with position
  type that's systematic label bias. Count + report substitutions in shard metadata.
- **`info_callback` exceptions swallowed bare** (`mcts.py:460-471`) — keep search
  crash-proof but `logging.exception(...)` instead of `pass`.

---

## 🟢 Low / Cleanup

- WDL softmax computed twice in `expand_lazy` (`mcts.py:192-207`).
- `game_result` called twice per terminal node (`mcts.py:89,338`).
- Dead guard `if value==0 or value==-1` (`mcts.py:339-340`) — always true for terminals.
- Truncation limit hardcoded `1000` in two spots (`mcts.py:89,338`) — parameterize.
- `train.py` accuracy denominator inits to `1` not `0` (`:540,655`) — ~0.1% low; metric-only.
- Validation label recomputes `sorted(...).index()` per trigger (`train.py:705`).
- Dataset hardcodes `/255` without reading `boards_scale` meta; label-smoothing lacks
  `n_legal==0` guard; `unpackbits` hardcodes K with no length assert (`dataset.py`).
- 19 vs 119 move-counter scales diverge (300 vs 500) (`utils.py:56,118`) — unify.
- `match.py` docstrings disagree on color-alternation indexing (:10 vs :131); `nn.py:240`
  stray trailing newline.
- BatchedMCTS dedup is identity-based, misses transpositions (optional batch-local
  transposition table).
- uci keeps `_mirror_history` even for 19-plane (dead work); `ValueScalar` not validated
  at setoption.

**Verified-correct (no action):** game-level train/val split (no leakage), self-play
resign value perspective, torch.compile warm-up using a real board, monitor board-flip
orientation & WDL variance bands, ponder ticks correctly omitting eval data, analytics
division guards.

---

## Recommended action plan (ordered, dependency-aware)

| # | Step | Effort | Why this order |
|---|------|--------|----------------|
| 1 | Fix `runner.py`↔`selfplay.py` (flags + output discovery) as one change; smoke-test one iteration | M | Self-play 100% non-functional; unblocks the pipeline |
| 2 | Fix BatchedMCTS dedup backprop; re-run batched-vs-sequential A/B | M | Resolves the known weaker-play bug; independent of #1 |
| 3 | Boundary-inclusive WDL targets + sum-to-1 assertion | S | Pure correctness, tiny blast radius |
| 4 | Unify move-counter on /500+clamp in `_board_to_matrix_19`; regenerate 19-plane shards | S | Must precede new 19-plane data gen / retraining |
| 5 | Plumb history into `match.py` + add `valid_policy()` helper | M | 119-plane eval parity; after data/training fixes |
| 6 | Robustness batch: init `loaded_in_channels`, validate `ValueScalar`, log `info_callback`, zero-guards, assertions, `total=0`, precompute val triggers | M | Independent low-risk hardening |
| 7 | Pure cleanups: dedup softmax / single `game_result` / dead guard / docstrings / trailing newline / data-quality counters / optional transposition table | M | Non-behavioral; last to avoid churn |

---
---

# Performance & Structure — Deep Dive

_Method: 6 specialists read the current code (so nothing already-done is re-suggested) across
NN inference, NN training, MCTS, board encoding, CUDA/implementation, and structure; a
synthesis pass ranked everything by impact/effort._

## Where the time actually goes
- **Inference:** fp32 conv/BN kernels at batch=1. TF32 is on for matmul but **NOT cuDNN**,
  and `torch.compile` runs **default mode despite CLAUDE.md claiming `reduce-overhead`** —
  so the dominant Conv2d/BN kernels run full fp32 and uncompiled-for-latency.
- **Training:** GPU idles on a single-threaded, un-pinned DataLoader that also does per-item
  uint8→float rescale on CPU.

## The 5 biggest impact/effort levers
1. Add `mode="reduce-overhead"` to all 3 `torch.compile` calls in `players.py` (192, 274, 346) → **1.5-3× batch=1 inference**. 1 line ×3, already try/except-guarded.
2. `torch.backends.cudnn.allow_tf32 = True` in `players.py:30` + `uci.py:57` → **~15-30% forward** on conv/BN. 1 line ×2.
3. DataLoader `num_workers`+`pin_memory`+`persistent_workers` + `non_blocking=True` transfers in `train.py` → **~30-50% training throughput**.
4. fp16 `autocast` on inference forwards (`mcts.py:186`, `batched_mcts.py:333`, `uci.py` policy/value) → **15-25% per-forward latency**.
5. AMP (`autocast`+`GradScaler`) in the training loop (`train.py:675-680`) → **~30-50% training + ~2× memory headroom**.

## Quick wins (S effort)

| Win | File | Gain | Risk |
|-----|------|------|------|
| `mode="reduce-overhead"` on compile | `players.py:192,274,346` | 1.5-3× batch=1 | very low (guarded). Keep `uci.py` default — ponder thread / CUDA-graph stream safety |
| cuDNN TF32 (`allow_tf32=True`) | `players.py:30`, `uci.py:57` | 15-30% fwd | negligible |
| `cudnn.benchmark=True` (static shapes) | `players.py:30`, `uci.py` startup | 5-10% batch=1 | very low |
| DataLoader workers + pin + persistent | `train.py:589,594` | 30-50% training | low (~50MB/worker) |
| `non_blocking=True` H2D in training | `train.py:669-672,542` | 10-20% (with pin) | low |
| fp16 autocast on inference | `mcts.py:186`, `batched_mcts.py:333`, `uci.py:910,944` | 15-25%/fwd | low — A/B move selection |
| `torch.from_numpy` not `torch.tensor` | `utils.py:252` | ~19% tensor conv | none |
| Drop redundant `np.clip` (output ∈[0,1]) | `gen_sf_data.py:307` | ~10% quantize | none |
| `non_blocking` on MCTS mask transfer | `mcts.py:190`, `batched_mcts.py:338` | 1-3%/leaf | very low |
| Cache `legal_mask` on Node post-expansion | `mcts.py`, `batched_mcts.py:336` | 5-20% node overhead | low (state immutable) |

## High-impact, bigger (M/L effort)
- **AMP in training** (`train.py`) — 30-50% + memory headroom; verify loss curve. *M*
- **channels_last** for conv paths (inference + training) — 10-20% Conv2d on Ampere+; needs cuDNN 8.1+, verify BN numerics. *M*
- **Batch-rescale uint8→float on GPU** (drop `/255` in `dataset.py`, do it post-`.to(device)`) — 4× smaller PCIe transfer, 15-25% data path. *M*
- **Preallocate/reuse (pinned) board-encoding buffers** in MCTS — 10-15% hot loop. *M*
- **Multi-game batched self-play** (cross-game leaf batching) — **2-5× self-play throughput**, but L effort + high accounting risk; **fix the BatchedMCTS dedup bug first**. *L*

## Structure cleanups (all low-effort, safe)
- Move `2104.05336v1.pdf` + `BatchMCTSFinal.pdf` → `papers/` (untracked, no code refs). *safe*
- Move `temporary.py` (MCTS-scaling bench, self-marked "delete after eyeballing") + `bootstrap.py` (one-time seed util) → `scripts/`; also decide `ladder.py`/`analytics.py`. *safe*
- Fix the CLAUDE.md `torch.compile` mode claim to match reality (or fix the code — see quick win). *safe*
- Add `configs/README.md` documenting the 19 JSON configs (fields per player type, shared hyperparams). *safe (docs)*
- **Optional, needs care:** ~180 LOC duplicated between `mcts.py`/`batched_mcts.py` (`set_rep_counter`, `set_history`, `_compute_rep_count`, `apply_action`, `update_root`) → extract a small `mcts_common.py` / base mixin; keep the two `_simulate` impls separate. Behavior-preserving — diff + run a match after. *M*

## Already implemented (do NOT redo)
- TF32 **matmul** (`set_float32_matmul_precision('high')`, players.py:30, uci.py:57) — but not cuDNN.
- `torch.compile` applied on all paths — but **without** `reduce-overhead` (uci.py default mode is intentional for ponder safety).
- All inference under `@torch.inference_mode()`.
- Single combined GPU→CPU sync in `expand_lazy` (value+policy+optional WDL in one pull).
- uint8 board storage + `/255` rescale at `__getitem__` (4× smaller).
- `torch.compile` warm-up with a **real** start position at target batch (Mcts player).
- Tree reuse via 2-level `update_root`; idempotent.
- Dirichlet over legal moves only; `alpha=0.3` (chess).
- Batched MCTS leaf dedup via `id()` set (O(1)).
- `board_to_matrix` already uses python-chess C-extension `SquareSet` iteration — further `np.unpackbits` vectorization deemed not beneficial; only per-Node `legal_mask` caching is worthwhile.
- μ-FPU + Virtual Mean (Cazenave 2021) in batched; μ-FPU in sequential (+Elo confirmed).

## Recommended sequence
1. Compile `reduce-overhead` (players.py) — *S*
2. cuDNN TF32 + benchmark — *S*
3. DataLoader workers/pin + non_blocking — *S*
4. fp16 inference autocast + mask non_blocking — *S*
5. `from_numpy` + drop clip — *S*
6. Cache `legal_mask` on Node — *S*
7. Structure cleanups (PDFs, scripts/, configs/README, fix CLAUDE.md claim) — *S*
8. Training AMP — *M*
9. uint8→GPU rescale + channels_last — *M*
10. Reuse pinned board-encoding buffers — *M*
11. Extract shared MCTS state methods — *M*
12. Multi-game batched self-play (after the dedup bug fix) — *L*

---
---

# Ideas / Research Directions — Algorithmic Speed (beyond implementation)

_The coding-level wins above buy ~2-4× (kernels). The ideas here attack **strength-per-compute**
— Elo per simulation, Elo per self-play game — where the literature has found 10-50×.
Halving the sims needed for equal strength **is** a 2× speedup. These stack multiplicatively
with the kernel wins (different layers of the stack). ★ marks high leverage._

> Two notions of "speed": (1) throughput/latency (nodes/sec, games/sec — systems/GPU);
> (2) sample/compute efficiency (Elo per sim, Elo per game). The biggest published wins are (2).

## 🔍 MCTS / Search lens
- **★★ Gumbel AlphaZero** (Danihelka et al., ICLR 2022 Spotlight) — Sequential Halving + Gumbel-top-k
  sampling *without replacement* at the root gives a **guaranteed** policy improvement even when not
  all root actions are visited (vanilla AZ's visit-count target fails here), and plays strongly with
  as few as **~2 sims**. _(Verified — but note: the stronger "low-sim Gumbel beats high-sim AZ at
  equal training time" claim was **refuted** in fact-checking; the safe claim is the improvement
  guarantee + few-sim strength, not a head-to-head win.)_ Pairs with the existing high/low-sims split.
- **★ Playout Cap Randomization** (KataGo, Wu 2019) — most moves get few sims; a random ~25% get
  the full budget and produce the policy target. The principled way to schedule the
  `--high-sims/--low-sims` plumbing already present.
- **★ Forced playouts + policy target pruning** (KataGo) — force early visits to each child
  (exploration), then prune them from the policy target so it stays clean.
- **★ Transposition-aware search (DAG-MCTS)** — chess transposes constantly; a transposition table
  (batch-local first) saves NN evals. Gate TT hits on matching repetition/50-move state.
- **Syzygy tablebase probes in-tree** — perfect values for ≤7-piece endgames; prunes endgame
  subtrees (speed) AND upgrades training labels (quality). Slots next to `proven_value`.
- **MCTS as regularized policy optimization** (Grill et al. 2020) — act/target with the regularized
  policy (softmax over Q anchored on the prior) instead of raw visit counts; better at low sims.
- **Dynamic cpuct + log-scaling** (LC0) — already deferred (`project_puct_tuning_for_wdl`).

## 🧠 Deep Learning / network lens
- **★★ Auxiliary prediction targets** (KataGo, Wu 2019) — share the backbone, make each game teach
  more. _(Verified, with an important transfer caveat:_ KataGo's headline ~50× is the **whole bundle**
  vs ELF, and its two biggest single factors — ownership 1.65× and game-specific features 1.55× — are
  **Go-specific and do NOT transfer to chess**.) The **chess-transferable** auxiliary wins: the
  **soft-policy head** (predict policy^(1/T), T=4, 8× weight; KataGo v1.12.0 "greatly improves policy
  learning speed"), **Global Pooling** (1.60×), aux policy targets (1.30×), and a **moves-left/
  game-length head** (LC0) as the chess analog of the Go score head.
- **★ Knowledge distillation to a small fast net** — train strong/slow, distill to narrow/shallow
  student → most of the strength at a fraction of eval cost = more nps everywhere. Often a bigger
  inference win than any kernel optimization.
- **★ Value-target variance reduction (TD/n-step bootstrapping)** — AlphaZero trains value on the
  final outcome (high variance); KataGo/MuZero mix in the MCTS root value / n-step returns → faster
  value convergence, fewer games. Cheap loss change.
- **1858-compact policy head** (deferred) — smaller output, less memory, faster head.
- **Attention/transformer bodies** (LC0 BT-series) — strength-per-param, but slower per-eval; only
  if optimizing strength, not latency. For pure speed, distilled-narrow beats this.
- **SWA / EMA of weights** — near-free Elo + smoother convergence.

## ⚡ GPU / systems lens
- **★★ Centralized batched evaluator + many search actors** — canonical AlphaZero design; the real
  fix for batch=1 GPU starvation. N actors push leaves to a queue, one server coalesces into big
  batches and fans results back (~30%→~90% GPU util). `engine_server.py` is halfway there.
- **★ TensorRT / ONNX-Runtime / `torch.export` AOT** — beat `torch.compile` for static shapes,
  esp. with int8/fp16 calibration. LC0 ships TensorRT/cuDNN/oneDNN backends.
- **★ Native bitboard board encoder in C/Cython** (deferred) — python-chess CPU encode/move-gen
  often caps games/sec before the GPU does; a multi-× self-play win.
- **Async actor-learner with stale weights (IMPALA-style)** — don't block actors on the newest net;
  keep CPUs+GPUs saturated continuously.
- **Multi-GPU / distributed self-play actors** — the embarrassingly-parallel part.

## 🔁 Self-play / pipeline lens
- **★ Resign threshold + calibration** (deferred) — ~half the moves are in decided positions; resign
  with a disabled-resign calibration set to bound false resigns. Pairs with the WDL value.
- **★ Prioritized / surprise-weighted replay** — sample high-loss positions more.
- **★ Diverse start positions** — seed games from opening book / SFT positions / puzzle FENs / random
  midgames; more state-space coverage per game. (Folds in the puzzle-FT deferred item.)
- **EfficientZero** (Ye et al. 2021) — self-supervised consistency loss + value-prefix; ~500× data
  efficiency on Atari (MuZero-framed; the regularization idea transfers).
- **Replay window / freshness weighting** — weight recent games; train near current strength.

## 💡 Original / synthesis ideas (tailored)
1. **Uncertainty-gated sim budget** — use the free WDL variance (`project_wdl_variance_signal`) to
   allocate sims: few on low-variance roots, many on sharp ones. A continuous version of playout-cap
   randomization, nearly free since variance is already computed.
2. **Two-net leaf cascade** — tiny triage net for most (visit-once) leaves; promote only high-visit /
   high-variance nodes to the big net. Cuts effective big-net evals.
3. **Variance-aware cpuct** — fold WDL variance into exploration (explore where the *outcome* is
   uncertain, not just where N is low). A PUCT upgrade the value head uniquely enables.
4. **Stockfish-as-policy-teacher** — already distill SF value; also distill SF multipv move
   preferences into the policy head during SFT → stronger priors → fewer sims needed (compounds).

## Top 5 by leverage
| # | Idea | Lens | Why |
|---|------|------|-----|
| 1 | Gumbel AlphaZero | MCTS | guaranteed improvement + strong at very few sims |
| 2 | Aux targets: soft-policy head + moves-left + global pooling | DL | chess-transferable subset of KataGo's bundle |
| 3 | Centralized batched evaluator + actors | GPU | Fixes batch=1 starvation, ~3× GPU util |
| 4 | Resign + playout-cap randomization | Self-play | Stop computing decided positions |
| 5 | Distillation to a small net | DL/GPU | Most strength at fraction of eval cost |

_Papers referenced (knowledge cutoff Jan 2026; verify latest): AlphaZero (Silver 2018), MuZero
(Schrittwieser 2020), Gumbel MuZero/AlphaZero (Danihelka 2022), KataGo (Wu 2019), MCTS as
Regularized Policy Optimization (Grill 2020), EfficientZero (Ye 2021), Leela Chess Zero
(SE-ResNet, moves-left head, attention BT-series, WDL, cpuct log-scaling)._

---

## ✅ Deep-Research Verification & 2024-25 Update

_103-agent fan-out web search + 3-vote adversarial fact-check (24/25 claims confirmed, 1 killed).
All sources primary. This section supersedes any conflicting framing above._

### Verified citations
- **Gumbel AlphaZero/MuZero** — Danihelka, Guez, Schrittwieser, Silver, **ICLR 2022 Spotlight**
  (openreview `bERaNdoegnO`). Guarantees policy improvement when not all root actions are visited;
  strong at **~2 sims**. Corroborated by **MiniZero** (arXiv:2310.11305): "significant performance
  with as low as only two simulations." **Refuted (0-3):** that low-sim Gumbel beats n=200 AZ at
  equal training time — do NOT claim a head-to-head win.
- **KataGo** — Wu, "Accelerating Self-Play Learning in Go," **arXiv:1902.10565** (AAAI-20 RLG
  workshop) + `docs/KataGoMethods.md`. Verified per-technique ablation factors (Table 2):

  | Technique | Factor | Transfers to chess? |
  |-----------|--------|---------------------|
  | Global Pooling | 1.60× | ✅ |
  | Aux Ownership + Score | 1.65× | ❌ Go-specific |
  | Game-specific features | 1.55× | ❌ Go-specific |
  | Playout Cap Randomization | 1.37× | ✅ |
  | Auxiliary Policy Targets | 1.30× | ✅ |
  | Forced Playouts + Policy Target Pruning | 1.25× | ✅ |

  Product ≈ **9.1×** (ablation); **~50×** end-to-end vs ELF OpenGo (1.4 vs 74 GPU-years). The 50×
  is cross-engine and bundle-wide — not attributable to any single transferable factor.
- **Playout Cap Randomization** (verified params): full search prob `p=0.25`, `(N,n)=(600,100)`
  annealing to `(1000,200)`; record **only** full-search turns for policy targets; disable
  Dirichlet on fast searches.
- **Forced Playouts + Policy Target Pruning** (verified): `n_forced(c) = (k·P(c)·ΣN(c'))^½`, `k=2`;
  set PUCT=∞ below threshold; subtract forced playouts from the recorded target unless the move
  proved good. Decouples the policy target from MCTS/Dirichlet dynamics.
- **Soft-policy auxiliary head** (KataGo v1.12.0, Mar 2023): predict `policy^(1/T)`, T=4, 8× weight —
  "greatly improves the speed of learning of the policy." ✅ transfers.
- **Short-term TD-bootstrapped value targets**: predict exp-averaged future MCTS values over ~6/16/50
  turn horizons → trains slightly faster, lower value-loss. ✅ transfers (author hedges "slightly").
- **Dynamic variance-scaled cPUCT + uncertainty-weighted playouts** (KataGo v1.9.0): **~75 Elo**
  (~50 Elo if the prior release's cPUCT is optimally tuned). ✅ directly reinforces the deferred
  `project_puct_tuning_for_wdl` item. _(Self-reported A/B, high credibility, not third-party.)_

### New in 2024-25
- **Rapfi** (Jin, Duan, Hang, **arXiv:2503.13178**, Mar 2025, Gomoku): distills a CNN into a compact
  pattern-codebook net; **matches ResNet accuracy at orders-of-magnitude less compute**, beats the
  AlphaZero-based Katagomo on CPU (won GomoCup 2024). **Lesson: distillation-to-small-net works.**
  Caveat: the pattern-codebook + incremental-update trick is Gomoku-specific (exploits local line
  patterns + alpha-beta), not chess-MCTS-transferable; ICLR submission was withdrawn (arXiv +
  competition results stand).

### Not independently verified by this pass (treat as plausible, established, but unconfirmed here)
transposition/DAG-MCTS, regularized policy optimization (Grill 2020), 1858-compact head sizing,
attention/transformer bodies (LC0 BT-series), SWA/EMA, TensorRT/ONNX/int8-fp16 export, centralized
batched evaluator / IMPALA-style async, prioritized replay, diverse start positions, EfficientZero,
resign thresholds. (These fell outside the verified-claim budget — not contradicted, just unchecked.)

### Verified priority for THIS engine
1. **Gumbel root selection** (Gumbel-top-k + Sequential Halving) — best strength-per-sim upgrade.
2. **Playout Cap Randomization** — principled schedule for the existing high/low-sims split.
3. **Forced Playouts + Policy Target Pruning** — cleaner policy targets.
4. **Soft-policy aux head + short-term TD value targets** — faster, lower-variance learning.
5. **Dynamic variance-scaled cPUCT** — folds into the WDL-cPUCT retune already on the list.
6. **Distillation to a small net** — once a strong teacher exists, for fast deployment.

_Open question flagged by the research: KataGo's factors are Go-derived — a small A/B on this chess
engine (start with PCR + Forced Playouts) is needed to confirm the real chess-transfer magnitude._


