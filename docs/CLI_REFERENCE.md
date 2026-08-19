# CLI reference

Prefer built-in help for exact flags and defaults:

| How to run | Help |
|------------|------|
| Console entry (`pyproject.toml` `[project.scripts]`) | `uv run <command> --help` |
| Module under `src/` | `uv run python src/.../script.py --help` |

Argparse CLIs use `--snake_case`. Typer apps use `--kebab-case`.

Worked examples: root [README.md](../README.md). Data layout: [DATA.md](DATA.md).

---

## Platform CLI — `glucose`

```bash
uv run glucose --help
uv run glucose info
```

| Command | Role |
|---------|------|
| `info` | Package version, default runs root, evaluate config path |
| `evaluate` | Unified eval/compare for custom PyTorch + NF run dirs |
| `neuralforecast` | NF holdout train / evaluate / summarize |
| `release` | Pack / check / publish / pull inference bundles (format 1.0) |

There is **no** `glucose train` for custom PyTorch. Train with experiment CLIs (`train-glumind`, `train_sugar_one`, …).

### `glucose evaluate`

Central path for GluMind / GluMind-Uni / SugarOne / SugarJepa (and comparison against NF run dirs). Engine: `common.evaluation.checkpoint_eval` via `common.evaluation.runner`. Window datasets and CSV helpers: `common.data`.

```bash
uv run glucose evaluate --help
uv run glucose evaluate                    # defaults from src/glucose_evaluate.yaml
uv run glucose evaluate \
  --run-dir fixtures/checkpoints/glumind_1.0 --model-type glumind \
  --data fixtures/livia_data/livia_glumind_ready.csv --test-split "" \
  --batch-size 4096 --no-plot
```

| Option | Meaning |
|--------|---------|
| `--run-dir` | Leaf run dir **or** container of runs (repeatable). Containers expand to best-by-val-MAE per family. Default: `models[]` in YAML. |
| `--registry-dir` | Pick lowest `val_mae` from `_analysis_registry.csv` (single model). |
| `--checkpoint` | Explicit `.pt` weights (with `--run-dir` for meta). |
| `--data` | Eval CSV (default from YAML). Omit to use precomputed metrics when present. |
| `--train-data` | Legacy scaler fit CSV when `scalers.json` is missing. |
| `--label` | Label per `--run-dir` (repeatable). |
| `--out` | Comparison report directory (default `data/output/compare`). |
| `--config` | YAML defaults (`src/glucose_evaluate.yaml`). |
| `--model-type` | `auto` \| `glumind` \| `sugar_one` \| `glumind_uni` \| `sugar_jepa`. |
| `--test-split` | `Recommended Split` filter (default `test`; empty string disables). |
| `--batch-size`, `--device` | Inference controls (`device`: `auto` \| `cuda` \| `mps` \| `cpu`). |
| `--zero-cov` / `--include-cov` / `--exclude-cov` | Covariate ablation. |
| `--refit-scalers` / `--allow-fit-on-eval` | Scaler fallback controls. |
| `--covariates` | Inspect CSV covariate columns and exit. |
| `--output-json` | Write metrics JSON for a single-run eval. |
| `--log-interval` | Progress log cadence (seconds). |
| `--plot` / `--no-plot` | Comparison charts under `--out`. |

### `glucose neuralforecast`

SugarOne-compatible geometry by default: **input 128 / horizon 12 / stride 1**.

```bash
uv run glucose neuralforecast train --list-models
uv run glucose neuralforecast train --data <csv> --models NHITS --global-model --device auto
uv run glucose neuralforecast evaluate --run-dir <nf_run> --data <csv>
uv run glucose neuralforecast summarize-holdout --run-dir <a> --run-dir <b>
```

Legacy tuner (kept until parity): `uv run python src/nf_baselines/tune_nf_baselines_by_group.py -h`.

### `glucose release`

```bash
uv run glucose release pack <run_dir> --out <bundle_dir> [--release-id ID]
uv run glucose release check <bundle_dir>
uv run glucose release publish <bundle_dir> --repo ORG/NAME [--private]
uv run glucose release pull --repo ORG/NAME --out <dir> [--revision main]
```

---

## Experiment / console entry points

| Command | Module | Notes |
|---------|--------|------|
| `train-glumind` | `src/glumind/train_glumind.py` | argparse |
| `download-glumind-hf` | `src/glumind/download_from_huggingface.py` | Typer |
| `tune-sugar-one` | `src/sugar_one/tune_sugar_one.py` | Typer + TOML |
| `personal-prepare` | `src/personalization/prepare.py` | Chronological personal CSVs |
| `personal-finetune` | `src/personalization/finetune.py` | One SugarOne personal fine-tune |
| `personal-tune` | `src/personalization/tune.py` | TOML LR grid + leaderboard |
| `personal-sweep-days` | `src/personalization/sweep_data_size.py` | Train-days vs MAE |
| `personal-plot` | `src/personalization/plots.py` | Data-size charts |
| `personal-sweep-lr` | `src/personalization/sweep_holdout_lr.py` | Holdout LR transfer |
| `personal-sweep-lwf` | `src/personalization/sweep_lwf.py` | Independent LwF |
| `personal-study` | `src/personalization/study.py` | Cohort curves + report |

Direct (no console script):

```bash
uv run python src/sugar_one/train_sugar_one.py --help
uv run python src/glumind_uni/train_uniglumind.py train --help
uv run python src/sugar_jepa/train_sugar_jepa.py --help
uv run python src/nf_baselines/tune_nf_baselines_by_group.py -h
uv run python src/glumind/eval_gluformer_val_test_masked.py -h
uv run python src/glumind/upload_to_huggingface.py --help
```

### Personalization (`personal-*`)

SugarOne only (glucose + insulin + carbs). Defaults: Livia fixture + `fixtures/checkpoints/sugar_one_1.0`, train window stride **6**. Full writeup: [PERSONALIZATION.md](PERSONALIZATION.md).

```bash
uv run personal-prepare livia
uv run personal-tune --dry-run
uv run personal-finetune --help
```

The fixture CSV has an empty `Recommended Split`; `personal-prepare livia` assigns chronological train/val/test. Then `personal-tune` / `personal-finetune` read `data/input/personalization/prepared/livia_chronological.csv`.

### `train-glumind`

`uv run train-glumind --help`

| Option | Meaning |
|--------|---------|
| `--csv` | Path to processed dataset CSV (required). |
| `--unique_id` | `sequence_id` or `user_id`: which column defines a series. |
| `--max_train_series` / `--max_eval_series` | Cap series; `0` = all. |
| `--drop_interpolated` | Drop rows with `Event Type == Interpolated`. |
| `--study_groups` | Comma-separated `Study Group` filter; empty = all. |
| `--split_scheme` | `classic` or `trainval_test_as_val`. |
| `--mode` | `global`, `per_group`, `cohort_wise`, or `continual`. |
| `--horizon` / `--input_steps` | Forecast / history length in steps. |
| `--d_model`, `--n_heads`, `--n_blocks`, `--ff_units`, `--dropout` | Architecture. |
| `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--patience` | Training. |
| `--precision` | `fp32`, `bf16`, or `fp16`. |
| `--compile_mode` | `none`, `default`, `reduce-overhead`, `max-autotune`. |
| `--num_workers` | DataLoader workers; `-1` auto (GPU: up to 8). |
| `--resume_from` | Path to full `checkpoint.pt`. |
| `--lwf_lambda` | Learning-without-forgetting weight in `continual` mode. |
| `--device` | `cpu`, `mps`, or `cuda`. |
| `--out_dir` | Base output directory for runs. |

Known dead flags on this CLI: `--mask_interpolated_targets`, `--save_predictions` (parsed, not wired).

### `tune-sugar-one`

`uv run tune-sugar-one --help`

| Option | Meaning |
|--------|---------|
| `--config`, `-c` | TOML config (default `src/sugar_one/tune_sugar_one_full.toml`). |
| `--device` | `cuda`, `cpu`, or `mps`. |
| `--seed` | Override `.random_seed` from the config. |

Shipped configs: `tune_sugar_one_full.toml`, `tune_sugar_one_dev.toml`.

### SugarOne / GluMind-Uni / SugarJEPA train scripts

Typer apps with the same training modes as GluMind (`global`, `per_group`, `cohort_wise`, `continual`). Defaults:

| Script | Default `--out-dir` | Covariates |
|--------|---------------------|------------|
| `src/sugar_one/train_sugar_one.py` | `data/output/runs/sugar_one` | basal, bolus, carbs |
| `src/glumind_uni/train_uniglumind.py` | `data/output/runs/glumind_uni` | glucose only |
| `src/sugar_jepa/train_sugar_jepa.py` | under `data/output/runs/` | SugarOne + JEPA |

### `download-glumind-hf`

`uv run download-glumind-hf --help` — `--repo-id`, `--output-dir`, `--token`, `--revision`.

### GluFormer (external HF baseline)

`uv run python src/glumind/eval_gluformer_val_test_masked.py -h` — not part of `glucose evaluate`.

---

## Expected dataset columns

Produce CSVs with [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing), then place under `data/input/` (see [DATA.md](DATA.md)).

**GluMind / AI-READI-style:**  
`sequence_id`, `User ID`, `Timestamp (YYYY-MM-DDThh:mm:ss)`, `Recommended Split`, `Study Group`, `Event Type`, `Glucose Value (mg/dL)`, `Heart Rate`, `Step Count`

**Loop / SugarOne:**  
`Glucose (mg/dL)` or `Glucose Value (mg/dL)`, `Basal Rate (U/h)`, `Bolus Insulin (U)`, `Carbohydrates (g)`

---

## Checkpoints and run artifacts

Typical outputs under `data/output/runs/<run_name>/`:

```text
val_metrics_overall.csv
val_metrics_by_study_group.csv
test_metrics_overall.csv
test_metrics_by_study_group.csv
tuning_meta.json / config.json
scalers.json
best_model.pt / last_model.pt
checkpoints/
```

Load architecture without the training script:

```python
import torch
from sugar_one.sugar_one_model import SugarOneModel

model = SugarOneModel(
    n_time_steps=128, n_features=4, d_model=32, n_heads=8,
    ff_units=128, n_blocks=5, prediction_horizon=12, dropout=0.1
)
state = torch.load("path/to/best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

---

## Quick smoke

```bash
uv run glucose --help
uv run glucose info
uv run glucose evaluate --run-dir fixtures/checkpoints/glumind_1.0 --model-type glumind \
  --data fixtures/livia_data/livia_glumind_ready.csv --test-split "" --batch-size 4096 --no-plot
uv run pytest -q
```

## Notes

- In `trainval_test_as_val` mode, held-out test metrics are intentionally disabled.
- Demo Livia CSVs often lack `Recommended Split` — always pass `--test-split ""` for them.
- Prefer `scalers.json` from the run dir over `--refit-scalers` when comparing to training-domain metrics.
