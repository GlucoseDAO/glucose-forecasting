# How to fetch a public checkpoint and run evaluation locally.

## Context (data and metrics)

GluMind was developed and tested on an “AI ready” dataset. That dataset cannot be shipped or redistributed with this repository because of licensing. To still show that the published weights run end-to-end, we evaluate on **Livia** data prepared in the same GluMind CSV shape.

Livia has **type 1 diabetes**, which is a harder setting for glucose forecasting than typical cohorts. **Numbers on Livia are a sanity check, not a claim about general model performance**; do not treat them as headline benchmark scores.

---

## Evaluate (`evaluate-glumind`)

You need:

- **`test_model`** — a folder that contains a compatible run (at minimum `config.json` / `tuning_meta.json` and `best_model.pt`, as produced by a training run or by the download step below).
- **`test_data`** — a GluMind-ready CSV (for example Livia’s export).

From the **repository root**, after `uv sync`, usualy `uv sync` could be omitted for briefness:

```bash
uv run evaluate-glumind --run-dir test_model --test-csv test_data/livia_glumind_ready.csv
```

`--run-dir` is the model directory; `--test-csv` is your test CSV path (adjust names to match your machine).

---

## Download from Hugging Face (`download-glumind-hf`)

Use this **only when `test_model` is empty** (or you intentionally want a clean folder). It fills `test_model` from the Hub; if the folder already has files, you risk a confusing mix of old and new artifacts.

From the **repository root**, after `uv sync`:

```bash
uv run download-glumind-hf --repo-id GlucoseDao/glumind-global-h12 --output-dir test_model
```

Then evaluate as above, for example:

```bash
uv run evaluate-glumind --run-dir test_model --test-csv test_data/livia_glumind_ready.csv
```

(Replace the CSV path with your actual file under `test_data`.)
