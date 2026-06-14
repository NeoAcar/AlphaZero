# Player configs

Each JSON file here describes one **player** for `match.py` / `runner.py` /
`ladder.py`. Loaded by `alphazero/players.py:load_player`. The only required
field is `type`; everything else is type-specific with sensible defaults.

```bash
uv run python match.py --player1 configs/mcts_seresnetwdl.json \
                       --player2 configs/policy_only_seresnet.json --games 20
```

## `type` → player class (see `players.py:PLAYER_TYPES`)

| type | needs NN | what it does |
|------|----------|--------------|
| `random` | no | uniform random legal move (baseline) |
| `piece_value` | no | greedy material-count one-ply (baseline) |
| `policy_only` | yes | one NN forward, argmax/sample over legal policy |
| `value_only` | yes | one-ply lookahead by the value head only |
| `mcts` | yes | full MCTS + policy + value (the real engine) |
| `stockfish` | external | wraps a Stockfish binary |

## Common NN fields (`policy_only`, `value_only`, `mcts`)

| field | meaning | typical |
|-------|---------|---------|
| `architecture` | `resnet` \| `seresnet` \| `seresnetwdl` (must match the checkpoint) | — |
| `checkpoint` | path to the `.pth` weights | — |
| `value_scalar` | WDL collapse: `expected` (P(W)−P(L)) or `win_only` (P(W)). WDL nets only | `expected` |
| `temperature` | softmax temperature for move sampling | `1.0` |
| `temperature_moves` | plies of stochastic sampling before switching to argmax | `0`–`15` |
| `sampling_seed` | RNG seed for reproducible sampling | unset |

Input-plane count (19 vs 119) is **auto-detected** from the checkpoint's first
conv; you don't set it here.

## `mcts`-only fields

| field | meaning | typical |
|-------|---------|---------|
| `num_simulation` | sims per move | `100`–`800` |
| `c_init`, `c_base` | PUCT exploration constants | `1.25`, `19652` |
| `c_fpu` | FPU-reduction coefficient (μ-FPU) | `0.0`–`0.2` |
| `dirichlet_epsilon`, `dirichlet_alpha` | root noise weight / concentration (0.3 = chess) | `0.0`, `0.3` |
| `t` | policy-prior temperature in PUCT | `1` |
| `discount` | per-ply value discount γ (shorter wins / longer losses) | `1` |
| `batched` | use `BatchedMCTS` (throughput, not strength) | `false` |
| `batch_size` | leaves per NN forward when `batched` | `8` |
| `compile` | `torch.compile(mode="reduce-overhead")` the model | `true` |
| `channels_last` | channels_last memory format (CUDA) | `true` |

> Strength matches should keep `batched: false`. Batched MCTS is a throughput
> mode; see CLAUDE.md.

## `stockfish` fields

`binary` (path), then one strength knob: `elo` + `limit_strength: true`, or
`skill_level`. Optional search limit: `depth` or `time_ms` (else a default).

## Tips
- Naming convention: `<type>_<architecture>[_variant].json`.
- The fields are a flat superset; unknown keys for a given type are ignored, so
  copying a template and trimming is safe.
