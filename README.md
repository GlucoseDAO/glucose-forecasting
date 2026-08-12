# GluMind Glucose Forecasting Project

This repository contains training, tuning, and comparison workflows for blood glucose forecasting on AI-READI-style datasets.

The project currently includes:
- `GluMind` (our architecture) training pipeline — glucose, heart rate, steps.
- `SugarOne` — insulin/carb covariate variant for loop-style CGM + pump data.
- NeuralForecast baselines (`NHITS`, `TFT`, `NBEATSx`, …) via `glucose neuralforecast` (sugarone-compatible holdout) plus legacy tuner.
- `GluFormer` evaluation script.
- Unified evaluation via `glucose evaluate` for GluMind / GluMind-Uni / SugarOne / SugarJepa checkpoints.
- Platform CLI `glucose` (`info`, `evaluate`, `neuralforecast`, `release`) wrapping shared evaluation under `src/common/evaluation/`, NF under `src/nf_baselines/`, and inference bundles under `src/common/release/`.
- Run analysis artifacts and cross-model comparison reports.

**Docs:** [CLI reference](docs/CLI_REFERENCE.md) · [Migration notes](docs/MIGRATION.md) · [Data layout](docs/DATA.md)

## Quick start

```bash
uv sync
uv run glucose --help
uv run glucose info

# Demo eval (bundled GluMind checkpoint + CSV)
uv run glucose evaluate --run-dir test_model_glumind --model-type glumind \
  --data test_data/livia_glumind_ready.csv --test-split "" --batch-size 4096 --no-plot

# Default multi-model compare (YAML: demos + best NF runs under nf_holdout)
uv run glucose evaluate

uv run pytest -q
```

## Project Scope

- Forecast horizon: default `12` steps (`60` minutes at `5min` frequency).
- Main modalities used by GluMind: glucose, heart rate, steps.
- Main modalities used by SugarOne: glucose, basal rate, bolus insulin, carbohydrates.
- Main split mode `classic`: use train/val/test as provided.
- Main split mode `trainval_test_as_val`: merge train+val for training, use test as validation (no held-out test output).

## Repository Structure

- `src/cli.py`: top-level `glucose` Typer app (`info`, `evaluate`, `neuralforecast`, `release`).
- `src/glucose_evaluate.yaml`: default models/dataset/out/plot settings for `glucose evaluate`.
- `src/common/release/`: inference release format 1.0 + Hub publish/pull (`glucose release`).
- `src/glumind/train_glumind.py`: GluMind training/tuning entrypoint (also exposed as `train-glumind`).
- `src/glumind/glumind_model.py`: model architecture module (checkpoint-friendly).
- `src/glumind/download_from_huggingface.py`, `upload_to_huggingface.py`: Hub download/upload.
- `src/glumind_uni/train_uniglumind.py`: univariate GluMind variant (glucose-only).
- `src/sugar_one/train_sugar_one.py`: SugarOne training entrypoint (insulin/carb covariates).
- `src/sugar_one/tune_sugar_one.py`: random-search hyperparameter tuner (`tune-sugar-one`).
- `src/sugar_one/sugar_one_model.py`: SugarOne architecture module.
- `src/common/data/`: columns, CSV loading (`loading.py`), sliding-window datasets (one file per class).
- `src/common/evaluation/checkpoint_eval.py`: unified PyTorch eval engine for all experiment families.
- `test_model_glumind/`: bundled GluMind checkpoint for reviewers (weights + metrics).
- `test_model_sugar_one/`: bundled SugarOne checkpoint for reviewers (weights + metrics).
- `test_data/livia_glumind_ready.csv`: self-contained demo CSV for quick end-to-end evaluation.
- `src/nf_baselines/`: NeuralForecast experiment (holdout suites + `glucose neuralforecast`); legacy `tune_nf_baselines_by_group.py` kept until parity.
- `src/glumind/eval_gluformer_val_test_masked.py`: GluFormer (Hugging Face) evaluation on val/test.
- `data/input/`: preferred location for local training/eval CSVs (see `docs/DATA.md`).
- `data/output/runs/`: default root for model run outputs (metrics, checkpoints, predictions).
- `data/output/marked_runs/`: curated run sets and analysis markdown files.
- `CROSS_MODEL_COMPARISON.md`: cross-model summary report.

## CLI reference

Durable platform + experiment CLI docs: **[docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)**.  
Layout / path migration: **[docs/MIGRATION.md](docs/MIGRATION.md)**.

Every script also supports built-in help:

| How to run | Help flag |
|------------|-----------|
| Installed console commands (see `pyproject.toml` `[project.scripts]`) | `uv run <command> --help` or `-h` where supported |
| Python entry files under `src/` | `uv run python src/.../script.py --help` or `-h` (argparse) |

Argparse-based CLIs (`train_glumind.py`, `tune_nf_baselines_by_group.py`, `eval_gluformer_val_test_masked.py`) print defaults in `--help` via `ArgumentDefaultsHelpFormatter` where configured. Typer apps list each option with `--help`.

### Platform — `glucose`

```bash
uv run glucose info
uv run glucose evaluate                    # defaults: src/glucose_evaluate.yaml
uv run glucose neuralforecast train --help
uv run glucose release pack --help
```

### `train-glumind` — `src/glumind/train_glumind.py`

`uv run train-glumind --help` or `uv run python src/glumind/train_glumind.py --help`

| Option | Meaning |
|--------|---------|
| `--csv` | Path to processed dataset CSV (required). |
| `--unique_id` | `sequence_id` or `user_id`: which column defines a series. |
| `--chunk_size` | Reserved for streaming chunk size (default 1_000_000). |
| `--max_train_series` | Cap number of training series; `0` = all. |
| `--max_eval_series` | Cap val/test series; `0` = all. |
| `--drop_interpolated` | Drop rows with `Event Type == Interpolated`. |
| `--mask_interpolated_targets` | Defined in CLI; not wired in the current training loop (no effect). |
| `--study_groups` | Comma-separated `Study Group` filter; empty = all. |
| `--split_scheme` | `classic` or `trainval_test_as_val` (train+val merged, test→val; no held-out test). |
| `--mode` | `global`, `per_group`, `cohort_wise`, or `continual`. |
| `--horizon` | Forecast length in steps (e.g. 12 = 60 min at 5 min). |
| `--input_steps` | History length in steps (e.g. 80). |
| `--d_model`, `--n_heads`, `--n_blocks`, `--ff_units`, `--dropout` | Architecture hyperparameters. |
| `--epochs`, `--batch_size` | Training length and batch size. |
| `--precision` | `fp32`, `bf16`, or `fp16` (mixed precision on CUDA when not fp32). |
| `--compile_mode` | `none`, `default`, `reduce-overhead`, `max-autotune` (`torch.compile`). |
| `--disable_tf32` | Turn off TF32 on CUDA. |
| `--num_workers` | DataLoader workers; `-1` auto (GPU: up to 8). |
| `--prefetch_factor` | Prefetch when `num_workers > 0`. |
| `--lr`, `--weight_decay` | AdamW optimizer. |
| `--patience` | Early stopping on val loss; `0` disables. |
| `--log_every` | Print progress every N epochs. |
| `--ckpt_every_n_epochs` | Save full checkpoint + val/test metrics under `checkpoints/epoch_NNNN/`; `0` off. |
| `--val_every_n_epochs` | Run validation every N epochs. |
| `--resume_from` | Path to `checkpoint.pt` (full state) to resume. |
| `--lwf_lambda` | Learning-without-forgetting weight in `continual` mode. |
| `--continual_order` | `default` or `reverse` study-group order. |
| `--continual_val_scope` | `current_group` or `all_groups` for val in continual mode. |
| `--device` | `cpu`, `mps`, or `cuda`. |
| `--seed` | RNG seed. |
| `--out_dir` | Base output directory for runs. |
| `--save_predictions` | Defined in CLI; not wired in the current training script (no effect). |

### `glucose evaluate` — unified PyTorch evaluation

`uv run glucose evaluate --help`

Central path for **GluMind**, **GluMind-Uni**, **SugarOne**, and **SugarJepa**. Loads architecture metadata and **`scalers.json`** from the run folder (falls back to re-fitting when the sidecar is missing), and reports **MAE, RMSE, MARD**. Multi-run comparison and plots are supported.

See [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) for the full flag table. Engine: `src/sugar_one/evaluate_model.py` (library).

### `download-glumind-hf` — `src/glumind/download_from_huggingface.py`

`uv run download-glumind-hf --help`

| Option | Meaning |
|--------|---------|
| `--repo-id` | Hugging Face model repo id, e.g. `OrgName/model-name`. |
| `--output-dir` | Local directory for downloaded files. |
| `--token` | Access token (private repos); empty for public. |
| `--revision` | Branch, tag, or commit (default `main`). |

Skips `checkpoints/` and `README.md` in the remote repo; downloads everything else.

### `upload_to_huggingface.py` (not a console script)

`uv run python src/glumind/upload_to_huggingface.py --help`

| Option | Meaning |
|--------|---------|
| `--model-dir` | Local run directory with weights and JSON metadata. |
| `--repo-name` | Repo name under the org, e.g. `glumind-global-h12`. |
| `--org` | Hugging Face organization name. |
| `--token` | Write token. |
| `--private` / `--public` | Create private repo (default public). |

### `tune_nf_baselines_by_group.py`

`uv run python src/nf_baselines/tune_nf_baselines_by_group.py -h`

| Option | Meaning |
|--------|---------|
| `--csv` | Dataset CSV (required). |
| `--split_scheme` | `classic` or `trainval_test_as_val`. |
| `--unique_id` | `sequence_id` or `user_id`. |
| `--model` | `tft`, `nhits`, `nbeatsx`, or `all`. |
| `--grid` | Optional JSON file overriding hyperparameter grids per model. |
| `--h_min` | Forecast horizon in **minutes** (default 60). |
| `--freq` | Pandas offset string, e.g. `5min`. |
| `--input_hours` | History length in hours for the model input window. |
| `--train_tail_val_hours` | Internal val tail per train series (used by NeuralForecast `val_size`). |
| `--max_steps`, `--val_check_steps` | PyTorch Lightning / NeuralForecast training steps. |
| `--batch_size`, `--valid_batch_size` | Batches. |
| `--windows_batch_size`, `--inference_windows_batch_size` | Window batching for NF. |
| `--step_size` | Sliding step between windows. |
| `--lr` | Learning rate. |
| `--device` | `cpu`, `mps`, or `cuda`. |
| `--seed` | Random seed. |
| `--chunk_size` | Pandas read chunk size for streaming. |
| `--max_train_series`, `--max_eval_series` | Subsample series; `0` = all. |
| `--max_points_per_series` | Truncate each series to the last N points after impute. |
| `--study_groups` | Comma-separated filter; empty uses all groups (unless global). |
| `--global_model` | One model on all study groups (no per-group runs). |
| `--out_dir` | Base output directory. |
| `--save_predictions` | Write `*_predictions.csv` per split. |
| `--ckpt_every_n_steps` | Checkpoint frequency in steps. |
| `--early_stop_patience` | Early stopping on `valid_loss`. |
| `--save_all_checkpoints` | Keep every checkpoint, not only best. |
| `--eval_checkpoints` | After training, eval each saved checkpoint. |
| `--train_event_type` | Optional: filter **train** rows by `Event Type`. |
| `--drop_interpolated` | Remove interpolated rows from all splits. |
| `--mask_interpolated_targets` | Keep history rows but drop interpolated **targets** from metrics. |

### `eval_gluformer_val_test_masked.py`

`uv run python src/glumind/eval_gluformer_val_test_masked.py -h`

| Option | Meaning |
|--------|---------|
| `--csv` | Dataset CSV (required). |
| `--unique_id` | `sequence_id` or `user_id`. |
| `--model_id` | Hugging Face model id (default `njeffrie/Gluformer`). |
| `--device` | `cpu`, `mps`, or `cuda`. |
| `--splits` | `val`, `test`, or `both`. |
| `--chunk_size` | Streaming read chunk size. |
| `--max_eval_series` | Subsample series; `0` = all. |
| `--max_points_per_series` | Truncate each series. |
| `--drop_interpolated` | Remove interpolated rows before eval. |
| `--mask_interpolated_targets` | Exclude interpolated **targets** from metrics. |
| `--out_dir` | Base run output directory. |
| `--save_predictions` | Save per-row predictions. |

### `src/glumind_uni/train_uniglumind.py` (GluMindUni)

Typer subcommand `train` (glucose-only model):

`uv run python src/glumind_uni/train_uniglumind.py train --help`

Options match `train-glumind` (same training modes and hyperparameters) except: glucose-only inputs; default `--out-dir` is `data/output/runs/glumind_uni`; device flag is `--device`. This Typer app does not expose `--chunk_size`, `--mask_interpolated_targets`, or `--save_predictions` (those exist on the argparse `train_glumind` CLI only).

### `src/sugar_one/train_sugar_one.py` (SugarOne)

Root command `main` (no subcommand name):

`uv run python src/sugar_one/train_sugar_one.py --help`

Same shape as GluMindUni: insulin/carb covariates, default `--out-dir` `data/output/runs/sugar_one`, `--csv` should be the loop + AI-READI joined CSV (see script docstring). Device: `--device`.

Expected loop-style columns (aliases are resolved automatically by `glucose evaluate`):

- `Glucose Value (mg/dL)` or `Glucose (mg/dL)`
- `Basal Rate (U/h)`
- `Bolus Insulin (U)`
- `Carbohydrates (g)`

### `tune-sugar-one` — `src/sugar_one/tune_sugar_one.py`

`uv run tune-sugar-one --help`

Random hyperparameter search for SugarOne (global mode only). Behaviour is driven by a TOML config:

| Option | Meaning |
|--------|---------|
| `--config`, `-c` | TOML config path (default: `src/sugar_one/tune_sugar_one_full.toml`). |
| `--device` | `cuda`, `cpu`, or `mps` (default `cuda`). |
| `--seed` | Override `.random_seed` from the config. |

Shipped configs: `tune_sugar_one_full.toml` (production search) and `tune_sugar_one_dev.toml` (smaller laptop search).

## Environment Setup

Python requirement:
- `>=3.12`

Install dependencies with `uv`:

```bash
uv sync
```

Run scripts with:

```bash
uv run python <script>.py ...
```

Use `uv run <installed-command> --help` or `uv run python <script>.py --help` / `-h` for options; the [CLI reference](#cli-reference) lists them in one place.

## Expected Dataset Columns

Core CSV columns expected by scripts:
- `sequence_id`
- `User ID`
- `Timestamp (YYYY-MM-DDThh:mm:ss)`
- `Recommended Split` (`train` / `val` / `test`)
- `Study Group`
- `Event Type`
- `Glucose Value (mg/dL)`
- `Heart Rate`
- `Step Count`

Loop / SugarOne columns (in addition to the core id/timestamp/split columns):

- `Glucose (mg/dL)` or `Glucose Value (mg/dL)`
- `Basal Rate (U/h)`
- `Bolus Insulin (U)`
- `Carbohydrates (g)`

## GluMind Training

Global mode example:

```bash
uv run python src/glumind/train_glumind.py \
  --csv data/input/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --mode global \
  --device cuda \
  --epochs 120 \
  --patience 20 \
  --batch_size 4096 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --out_dir data/output/runs/glumind
```

Continual mode example:

```bash
uv run python src/glumind/train_glumind.py \
  --csv data/input/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --mode continual \
  --lwf_lambda 0.2 \
  --device cuda \
  --epochs 80 \
  --patience 10 \
  --batch_size 2048 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --out_dir data/output/runs/glumind
```

Tune mode using test as validation:

```bash
uv run python src/glumind/train_glumind.py \
  --csv data/input/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --split_scheme trainval_test_as_val \
  --mode global \
  --device cuda \
  --epochs 120 \
  --patience 20 \
  --batch_size 4096 \
  --precision bf16 \
  --out_dir data/output/runs/glumind
```

Resume training from full checkpoint:

```bash
uv run python src/glumind/train_glumind.py \
  --csv data/input/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --mode global \
  --resume_from data/output/runs/glumind/<run_name>/last_checkpoint.pt \
  --epochs 250 \
  --device cuda
```

## SugarOne Training and Tuning

Train on loop + AI-READI joined data:

```bash
uv run python src/sugar_one/train_sugar_one.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv \
  --mode global \
  --device cuda \
  --epochs 120 \
  --patience 10 \
  --batch_size 256 \
  --out_dir data/output/runs/sugar_one
```

Production hyperparameter search:

```bash
uv run tune-sugar-one --device cuda
```

Use `-c src/sugar_one/tune_sugar_one_dev.toml` for a smaller dev search.

## NeuralForecast Baselines

Preferred path (sugarone-compatible **128 / 12 / stride-1** holdout):

```bash
uv run glucose neuralforecast --help
uv run glucose neuralforecast train --list-models
uv run glucose neuralforecast train \
  --data data/input/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --models NHITS \
  --global-model \
  --device auto \
  --max-steps 300 \
  --out-dir data/output/runs
```

Re-evaluate a saved bundle / merge per-model runs:

```bash
uv run glucose neuralforecast evaluate --run-dir <nf_run> --data <csv>
uv run glucose neuralforecast summarize-holdout --run-dir <run_a> --run-dir <run_b>
```

Legacy tuner (kept until parity is verified):

```bash
uv run python src/nf_baselines/tune_nf_baselines_by_group.py \
  --csv data/input/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --model nhits \
  --global_model \
  --device cuda \
  --mask_interpolated_targets \
  --max_steps 300 \
  --val_check_steps 50 \
  --ckpt_every_n_steps 50 \
  --early_stop_patience 6 \
  --save_all_checkpoints \
  --eval_checkpoints \
  --out_dir data/output/runs/nhits
```

Supported NF models via suites (`auto` / `baseline` / `recurrent`) or `--models NHITS,TFT,…` — see `src/nf_baselines/model_suites.yaml`.

## Inference releases (format 1.0)

Checksummed inference-only bundles live under `src/common/release/` (`model.safetensors` + contract JSON + `checksums.sha256`):

```bash
# Pack a training run (best_model.pt + meta + scalers.json) into a release bundle
uv run glucose release pack test_model_glumind --out temp_docs/my_bundle --release-id glumind-demo

uv run glucose release check <bundle_dir>
uv run glucose release publish <bundle_dir> --repo ORG/NAME
uv run glucose release pull --repo ORG/NAME --out <dir> --revision <sha_or_tag>
```

## GluFormer Evaluation

Evaluate val/test splits:

```bash
uv run python src/glumind/eval_gluformer_val_test_masked.py \
  --csv data/input/actual/with_complex_steps_processing/ai_ready_plus_type1_v2_val_only_in_test.csv \
  --device cuda \
  --splits both \
  --mask_interpolated_targets \
  --save_predictions \
  --out_dir data/output/runs/gluformer/ai_ready_plus_type1
```

## Checkpoints and Model Reuse

GluMind checkpoints are saved as:
- `best_model.pt` / `last_model.pt` (plain `state_dict`)
- `checkpoint.pt` / `last_checkpoint.pt` (full training state)

The architecture is now separated in:
- `src/glumind/glumind_model.py`

So you can load checkpoints without the full training script:

```python
import torch
from glumind.glumind_model import GluMindModel

model = GluMindModel(
    n_time_steps=80, n_features=3, d_model=32, n_heads=4,
    ff_units=128, n_blocks=3, prediction_horizon=12, dropout=0.1
)
state = torch.load("data/output/runs/.../best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

## Evaluate on `test_data/livia_glumind_ready.csv`

The repo ships reviewer checkpoint bundles and a demo CSV so you can run inference without private training data:

| Path | Role |
|------|------|
| `test_model_glumind/` | GluMind weights (`best_model.pt`, metadata, saved val/test metrics) |
| `test_model_sugar_one/` | SugarOne weights (same layout) |
| `test_data/livia_glumind_ready.csv` | Self-contained CGM sample (~140k rows) in GluMind CSV shape |

Use **`glucose evaluate`** for all custom PyTorch architectures. It reads run metadata, restores the checkpoint, loads **`scalers.json`** (train-fit MinMax params), and prints **MAE, RMSE, MARD**.

**Important for this demo file:**

- `livia_glumind_ready.csv` has **no** `Recommended Split` column — pass **`--test-split ''`** to evaluate all rows.
- Bundled `test_model_*` folders include **`scalers.json`** fitted on the original training CSVs — you do **not** need `--train-data` for correct scaling. To deliberately re-fit on the demo file (wrong for comparing to training-domain metrics), pass `--refit-scalers --train-data test_data/livia_glumind_ready.csv --allow-fit-on-eval`.
- The demo file has glucose (+ sparse HR/steps) but **no insulin/carb columns** — for SugarOne, pass **`--zero-cov`** so basal/bolus/carbs are zeroed after imputation.

Livia is type-1 personal data; numbers here are a **sanity check**, not a headline benchmark.

### GluMind (`test_model_glumind`)

```powershell
uv run glucose evaluate `
  --run-dir test_model_glumind `
  --model-type glumind `
  --data test_data/livia_glumind_ready.csv `
  --test-split "" `
  --batch-size 4096 `
  --no-plot
```

Model type can be omitted when `--run-dir` contains a GluMind checkpoint (`--model-type auto` detects embed_hr / embed_steps weights).

### SugarOne (`test_model_sugar_one`)

```powershell
uv run glucose evaluate `
  --run-dir test_model_sugar_one `
  --model-type sugar_one `
  --data test_data/livia_glumind_ready.csv `
  --zero-cov `
  --test-split "" `
  --batch-size 256 `
  --no-plot `
  --output-json docs/reports/milestone7_smoke_livia.json
```

With access to the full loop benchmark CSV, drop `--zero-cov` and point both `--data` and `--train-data` at `data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv` to reproduce in-domain test metrics (~12.4 MAE on the bundled SugarOne checkpoint). See `docs/GLUMIND_VS_SUGARONE_COMPARISON.md`.

For every flag, see [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md). To fetch GluMind weights from Hugging Face into a local folder, use `download-glumind-hf`.

## Outputs

Typical run artifacts:
- `val_metrics_overall.csv`
- `val_metrics_by_study_group.csv`
- `test_metrics_overall.csv`
- `test_metrics_by_study_group.csv`
- `tuning_meta.json`
- `config.json`
- `checkpoints/`

## Reports

Main analysis documents:
- `CROSS_MODEL_COMPARISON.md`
- `data/output/marked_runs/glumind/*/RUNS_ANALYSIS.md`
- `data/output/runs/nhits/RUNS_ANALYSIS.md`

## Notes

- In `trainval_test_as_val` mode, held-out test metrics are intentionally disabled.
- For quick validation after code changes, run smoke settings such as `--epochs 1 --max_train_series <small> --max_eval_series <small>`.
