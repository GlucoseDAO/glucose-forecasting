# Data pipeline and local layout

This repository trains and evaluates forecasting models. It does **not** produce the ML-ready CSVs from raw CGM exports. That work lives in the companion preprocessing repo:

**[AnonymousOrg/glucose_data_processing](https://anonymous.4open.science/r/glucose-data-processing)**

Use that repo to download/convert public datasets, resample to 5-minute grids, interpolate small gaps, and emit ML-ready CSVs. Then copy (or symlink) those outputs into this repo under `data/input/` before training.

## End-to-end flow

```text
raw CGM / pump exports
        │
        ▼
glucose_data_processing  (glucose-process / configs)
        │
        ▼
ML-ready CSVs  (e.g. ai_ready_processed_dataset.csv, loop ML-ready)
        │
        ├─ optional joins in this repo (temp_scripts/loop_ai_ready/)
        │     → loop_ai_ready_joined2.csv
        │
        ▼
glucose-forecasting
  train-* / glucose neuralforecast / glucose evaluate / glucose release
```

## Local data directory

Licensed datasets (AI-READI, Loop, joined benchmarks) are **not** redistributed with this git repo.

```text
data/
  input/                         # preferred place for local ML-ready CSVs
    actual/                      # AI-READI-style wearable tables
      with_complex_steps_processing/
    loop_and_ai_ready/           # joined Loop + AI-READI benchmarks
    personalization/             # personal / holdout CSVs
  output/
    runs/                        # default training / eval / NF / personalization runs
    marked_runs/                 # curated historical runs + RUNS_ANALYSIS.md
  processed/                     # optional intermediate artifacts
  cache/                         # optional caches
```

| Path | Role |
|------|------|
| `data/input/` | Preferred place for local ML-ready CSVs (gitignored except `.gitignore`). |
| `data/output/runs/` | Default root for generated training and evaluation artifacts (`common.paths.DEFAULT_RUNS_ROOT`). |
| `data/output/marked_runs/` | Curated / annotated historical runs (reference, not a write target). |
| `fixtures/demo_data/` | Small demo CSVs shipped in-repo for smoke tests (Subject P1). |
| `fixtures/checkpoints/` | Bundled reviewer checkpoints (`glumind_1.0`, `sugar_one_1.0`, `sugar_jepa_dev`). |

Do **not** use top-level `runs/` or `marked_runs/` as destinations.

Historical docs and some older metadata may still mention flat folders such as:

- `data/loop_and_ai_ready/` — same logical role as `data/input/loop_and_ai_ready/`
- `data/actual/with_complex_steps_processing/` — same as under `data/input/actual/…`
- `test_data/` — now `fixtures/demo_data/`
- `test_model_glumind/` — now `fixtures/checkpoints/glumind_1.0/`
- `test_model_sugar_one/` — now `fixtures/checkpoints/sugar_one_1.0/`
- `sugar_jepa_dev/` — now `fixtures/checkpoints/sugar_jepa_dev/`

Place files under `data/input/` and pass `--csv data/input/<…>/<file>.csv` (or `--data` / `--train-data` for evaluate).

`glucose evaluate` / training helpers also **resolve by basename**: if checkpoint metadata still points at an absolute Windows path or `data/loop_and_ai_ready/foo.csv`, a file named `foo.csv` under `data/input/` (or known subfolders) is used automatically via `common.registry.resolve_csv_path`.

## Important datasets used by this project

| Logical name | Typical filename | Schema / covariates | Produced by |
|--------------|------------------|---------------------|-------------|
| AI-READI processed | `ai_ready_processed_dataset.csv` | glucose, HR, steps (+ split/group columns) | `glucose_data_processing` with `glucose_config_ai_ready.yaml` |
| AI-READI + Type1 | `ai_ready_plus_type1_*.csv` | wearable schema + small T1DM supplement | preprocessing + project-specific combine |
| Loop ML-ready | loop export CSV | glucose, basal, bolus, carbs | `glucose_data_processing` with `glucose_config_loop.yaml` |
| Joined benchmark | `loop_ai_ready_joined2.csv` (~12M rows) | loop-style columns; ~50% loop T1DM / ~50% ai_ready | `temp_scripts/loop_ai_ready/build_loop_ai_ready_joined2.py` |
| Joined dev subset | `loop_ai_ready_joined2_dev.csv` | same schema, ~1/N rows | same builder (`--dev-output-csv`) |

Typical local paths:

- `data/input/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv`
- `data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv`

Column expectations for training/eval are summarized in the root [README.md](../README.md#expected-dataset-columns) and [CLI_REFERENCE.md](CLI_REFERENCE.md).

## Building the joined Loop + AI-READI benchmark

After you have ML-ready Loop and AI-READI CSVs from the preprocessing repo:

```bash
uv run python temp_scripts/loop_ai_ready/build_loop_ai_ready_joined2.py --help
```

Related helpers in `temp_scripts/loop_ai_ready/`:

- `join_loop_ai_ready.py` — earlier join variant
- `export_joined_to_loop_csv.py` — export joined parquet → loop-style CSV
- `sample_joined_csv_dev.py` — smaller dev sample

These are one-off / reusable prep utilities (not part of the installable product surface). Prefer keeping new scratch scripts under `temp_scripts/` as well.

## Demo data (no private exports needed)

| File | Use |
|------|-----|
| `fixtures/demo_data/demo_glumind_ready.csv` | GluMind-shaped personal CGM (~140k rows); no `Recommended Split` |
| `fixtures/demo_data/demo_sugar_one_ready.csv` | Same subject with pump covariates for SugarOne |
| `fixtures/checkpoints/glumind_1.0/` | Bundled GluMind checkpoint + `scalers.json` |
| `fixtures/checkpoints/sugar_one_1.0/` | Bundled SugarOne checkpoint + `scalers.json` |
| `fixtures/checkpoints/sugar_jepa_dev/` | Bundled SugarJEPA dev checkpoint |

See [How_to_run_checkpoint.md](How_to_run_checkpoint.md) for reviewer smoke-test commands.

## Licensing note

AI-READI and some Loop/JAEB exports require registration and cannot be redistributed. Only share derived artifacts that your data agreements allow. Public demo CSVs under `fixtures/demo_data/` are intentionally small personal/demo traces for CI and reviewers.
