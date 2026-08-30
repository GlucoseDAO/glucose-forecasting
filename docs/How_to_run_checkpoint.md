# How to run a checkpoint

Prefer the platform CLI for all custom PyTorch families (GluMind, GluMind-Uni, SugarOne, SugarJepa).

## Quick demos (bundled weights + CSV)

```bash
# SugarOne + pump-shaped demo CSV
uv run glucose evaluate \
  --run-dir fixtures/checkpoints/sugar_one_1.0 \
  --model-type sugar_one \
  --data fixtures/demo_data/demo_sugar_one_ready.csv \
  --test-split "" \
  --batch-size 256 \
  --no-plot

# GluMind + wearable-shaped demo CSV
uv run glucose evaluate \
  --run-dir fixtures/checkpoints/glumind_1.0 \
  --model-type glumind \
  --data fixtures/demo_data/demo_glumind_ready.csv \
  --test-split "" \
  --batch-size 4096 \
  --no-plot
```

Both `fixtures/checkpoints/*` folders include **`scalers.json`**. You do not need `--train-data` for these smokes.

**Subject P1 notes**

- No `Recommended Split` column → always pass `--test-split ""`.
- Using GluMind CSV with SugarOne → add `--zero-cov`, or switch to `demo_sugar_one_ready.csv`.
- Numbers are a sanity check on personal type-1 data, not a headline benchmark.

## Useful `glucose evaluate` flags

| Flag | When to use it |
|------|----------------|
| `--registry-dir` | Pick lowest `val_mae` from `_analysis_registry.csv` |
| `--checkpoint` | Explicit `.pt` (still need `--run-dir` for architecture meta) |
| `--zero-cov` / `--include-cov` / `--exclude-cov` | Covariate ablation |
| `--train-data` | Only when `scalers.json` is missing (legacy runs) |
| Omit `--data` | Read precomputed `*_metrics_overall.csv` from the run dir |
| Repeated `--run-dir` + `--out` | Multi-model comparison report |

Full table: [CLI_REFERENCE.md](CLI_REFERENCE.md).

## Download GluMind weights from Hugging Face

```bash
uv run download-glumind-hf --help
```

Then point `glucose evaluate --run-dir` at the downloaded folder.

## In-domain SugarOne check (needs private CSV)

With `data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv` available:

```bash
uv run glucose evaluate \
  --run-dir fixtures/checkpoints/sugar_one_1.0 \
  --model-type sugar_one \
  --data data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv \
  --batch-size 256 \
  --no-plot
```

Expect roughly **~12.4 MAE** on the bundled SugarOne checkpoint (see [GLUMIND_VS_SUGARONE_COMPARISON.md](GLUMIND_VS_SUGARONE_COMPARISON.md)).
