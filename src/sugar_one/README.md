# Evaluation

PyTorch checkpoint evaluation lives in ``common.evaluation.checkpoint_eval``
and is invoked by **`uv run glucose evaluate`**.

Sliding-window datasets live under ``common.data`` (one module per class).
CSV loading lives in ``common.data.loading``. Shared column names in
``common.data.columns``.

## Quick start

```bash
uv run glucose evaluate --help

uv run glucose evaluate \
  --run-dir fixtures/checkpoints/glumind_1.0 \
  --model-type glumind \
  --data fixtures/demo_data/demo_glumind_ready.csv \
  --test-split "" \
  --batch-size 4096 \
  --no-plot
```

Full flags: `docs/CLI_REFERENCE.md`.
