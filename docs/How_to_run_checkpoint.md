# How to run a checkpoint

Prefer the platform CLI for all custom PyTorch families (GluMind, GluMind-Uni, SugarOne, SugarJepa).

## Evaluate (`glucose evaluate`)

```bash
uv run glucose evaluate --help

uv run glucose evaluate \
  --run-dir test_model_glumind \
  --model-type glumind \
  --data test_data/livia_glumind_ready.csv \
  --test-split "" \
  --batch-size 4096 \
  --no-plot
```

Useful flags:

- `--registry-dir` — pick lowest `val_mae` from `_analysis_registry.csv`
- `--zero-cov` / `--include-cov` / `--exclude-cov` — covariate ablation
- `--train-data` — legacy scaler fit when `scalers.json` is missing
- Omit `--data` to read precomputed `*_metrics_overall.csv` from the run dir

Multi-run comparison uses repeated `--run-dir` (or YAML `models[]`) and writes under `--out`.

## Download GluMind weights from Hugging Face

```bash
uv run download-glumind-hf --help
```

Then point `glucose evaluate --run-dir` at the downloaded folder.
