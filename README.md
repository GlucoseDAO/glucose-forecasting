# GluMind Glucose Forecasting Project

This repository contains training, tuning, and comparison workflows for blood glucose forecasting on AI-READI-style datasets.

The project currently includes:
- `GluMind` (our architecture) training pipeline.
- NeuralForecast baselines (`NHITS`, `TFT`, `NBEATSx`) tuning pipeline.
- `GluFormer` evaluation script.
- Run analysis artifacts and cross-model comparison reports.

## Project Scope

- Forecast horizon: default `12` steps (`60` minutes at `5min` frequency).
- Main modalities used by GluMind: glucose, heart rate, steps.
- Main split mode `classic`: use train/val/test as provided.
- Main split mode `trainval_test_as_val`: merge train+val for training, use test as validation (no held-out test output).

## Repository Structure

- `scripts/glumind/train_glumind.py`: GluMind training/tuning entrypoint (also exposed as `train-glumind`).
- `scripts/glumind/glumind_model.py`: model architecture module (checkpoint-friendly).
- `scripts/glumind/evaluate_glumind.py`, `inference_glumind.py`, `download_from_huggingface.py`, `upload_to_huggingface.py`: evaluation, reproduction, Hub download/upload.
- `scripts/glumind_uni/train_uniglumind.py`: univariate GluMind variant (glucose-only).
- `scripts/glumind_ic/train_glumind_ic.py`: insulin/carb covariate variant (loop-style CSV).
- `scripts/tune_nf_baselines_by_group.py`: NeuralForecast baselines (NHITS, TFT, NBEATSx).
- `scripts/eval_gluformer_val_test_masked.py`: GluFormer (Hugging Face) evaluation on val/test.
- `runs/`: model run outputs (metrics, checkpoints, predictions).
- `marked_runs/`: curated run sets and analysis markdown files.
- `CROSS_MODEL_COMPARISON.md`: cross-model summary report.

## CLI reference

Every script supports **built-in help** when run with `uv`:

| How to run | Help flag |
|------------|-----------|
| Installed console commands (see `pyproject.toml` `[project.scripts]`) | `uv run <command> --help` or `-h` where supported |
| Python entry files | `uv run python scripts/.../script.py --help` or `-h` (argparse) |

Argparse-based CLIs (`train_glumind.py`, `tune_nf_baselines_by_group.py`, `eval_gluformer_val_test_masked.py`) print defaults in `--help` via `ArgumentDefaultsHelpFormatter` where configured. Typer apps list each option with `--help`.

### `train-glumind` — `scripts/glumind/train_glumind.py`

`uv run train-glumind --help` or `uv run python scripts/glumind/train_glumind.py --help`

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

### `evaluate-glumind` — `scripts/glumind/evaluate_glumind.py`

`uv run evaluate-glumind --help`

| Option | Meaning |
|--------|---------|
| `--registry-dir` | Folder with `_analysis_registry.csv`; picks lowest `val_mae` run. |
| `--run-dir` | Explicit run directory with `tuning_meta.json` / `config.json` and weights. Overrides registry. |
| `--checkpoint` | Specific `.pt` weights; still need `--run-dir` for architecture metadata. |
| `--test-csv` | CSV to score (required). |
| `--train-csv` | CSV to fit MinMax scalers; default from metadata. |
| `--test-split` | If the CSV has `Recommended Split`, keep only this value (e.g. `test`). |
| `--glucose-only` | Ablation: replace HR/steps with zeros or a fixed scaled value. |
| `--default-value` | With `--glucose-only`: `zero`, `mean`, or `median` for HR/steps replacement. |
| `--batch-size` | Override DataLoader batch size (default from metadata). |
| `--device` | Torch device (string, e.g. `cuda` or `cpu`). |

You must pass either `--registry-dir` or `--run-dir`.

### `inference-glumind` — `scripts/glumind/inference_glumind.py`

`uv run inference-glumind --help`

| Option | Meaning |
|--------|---------|
| `--run-dir` | Run directory with metadata and `best_model.pt` / `last_model.pt` (required). |
| `--mode` | `auto` (from `split_scheme` in metadata), `test`, or `val_as_test`. |
| `--glucose-only` | Ablation: zero or constant HR/steps in scaled space. |
| `--default-value` | `zero`, `mean`, or `median` (non-glucose channels). |
| `--device` | Torch device. |

Re-runs inference on the **training CSV** from metadata and compares to saved `val_metrics_overall.csv` / `test_metrics_overall.csv` when present.

### `download-glumind-hf` — `scripts/glumind/download_from_huggingface.py`

`uv run download-glumind-hf --help`

| Option | Meaning |
|--------|---------|
| `--repo-id` | Hugging Face model repo id, e.g. `OrgName/model-name`. |
| `--output-dir` | Local directory for downloaded files. |
| `--token` | Access token (private repos); empty for public. |
| `--revision` | Branch, tag, or commit (default `main`). |

Skips `checkpoints/` and `README.md` in the remote repo; downloads everything else.

### `upload_to_huggingface.py` (not a console script)

`uv run python scripts/glumind/upload_to_huggingface.py --help`

| Option | Meaning |
|--------|---------|
| `--model-dir` | Local run directory with weights and JSON metadata. |
| `--repo-name` | Repo name under the org, e.g. `glumind-global-h12`. |
| `--org` | Hugging Face organization name. |
| `--token` | Write token. |
| `--private` / `--public` | Create private repo (default public). |

### `tune_nf_baselines_by_group.py`

`uv run python scripts/tune_nf_baselines_by_group.py -h`

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

`uv run python scripts/eval_gluformer_val_test_masked.py -h`

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

### `scripts/glumind_uni/train_uniglumind.py` (GluMindUni)

Typer subcommand `train` (glucose-only model):

`uv run python scripts/glumind_uni/train_uniglumind.py train --help`

Options match `train-glumind` (same training modes and hyperparameters) except: glucose-only inputs; default `--out-dir` is `runs/glumind_uni`; device flag is `--device`. This Typer app does not expose `--chunk_size`, `--mask_interpolated_targets`, or `--save_predictions` (those exist on the argparse `train_glumind` CLI only).

### `scripts/glumind_ic/train_glumind_ic.py` (GluMindIC)

Root command `main` (no subcommand name):

`uv run python scripts/glumind_ic/train_glumind_ic.py --help`

Same shape as GluMindUni: insulin/carb covariates, default `--out-dir` `runs/glumind_ic`, `--csv` should be the loop + AI-READI joined CSV (see script docstring). Device: `--device`.

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

## GluMind Training

Global mode example:

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
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
  --out_dir runs/glumind
```

Continual mode example:

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
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
  --out_dir runs/glumind
```

Tune mode using test as validation:

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --split_scheme trainval_test_as_val \
  --mode global \
  --device cuda \
  --epochs 120 \
  --patience 20 \
  --batch_size 4096 \
  --precision bf16 \
  --out_dir runs/glumind
```

Resume training from full checkpoint:

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --mode global \
  --resume_from runs/glumind/<run_name>/last_checkpoint.pt \
  --epochs 250 \
  --device cuda
```

## NeuralForecast Baselines

NHITS example:

```bash
uv run python scripts/tune_nf_baselines_by_group.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
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
  --out_dir runs/nhits
```

Supported NF models in this repo:
- `nhits`
- `tft`
- `nbeatsx`

## GluFormer Evaluation

Evaluate val/test splits:

```bash
uv run python scripts/eval_gluformer_val_test_masked.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_plus_type1_v2_val_only_in_test.csv \
  --device cuda \
  --splits both \
  --mask_interpolated_targets \
  --save_predictions \
  --out_dir runs/gluformer/ai_ready_plus_type1
```

## Checkpoints and Model Reuse

GluMind checkpoints are saved as:
- `best_model.pt` / `last_model.pt` (plain `state_dict`)
- `checkpoint.pt` / `last_checkpoint.pt` (full training state)

The architecture is now separated in:
- `scripts/glumind/glumind_model.py`

So you can load checkpoints without the full training script:

```python
import torch
from scripts.glumind.glumind_model import GluMindModel

model = GluMindModel(
    n_time_steps=80, n_features=3, d_model=32, n_heads=4,
    ff_units=128, n_blocks=3, prediction_horizon=12, dropout=0.1
)
state = torch.load("runs/.../best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

## Evaluate on `test_data/livia_glumind_ready.csv`

The evaluation script loads metadata from the run folder (`tuning_meta.json` or `config.json`), restores `GluMindModel`, fits or applies scalers per script logic, and computes **MAE, RMSE, MARD** on the supplied CSV.

**Minimal command** (repo root; paths as requested):

```powershell
uv run python scripts/glumind/evaluate_glumind.py `
  --run-dir test_model `
  --test-csv test_data/livia_glumind_ready.csv
```

For every flag, see [CLI reference](#cli-reference) → **evaluate-glumind**. The script docstring at the top of `evaluate_glumind.py` also describes how to use `--registry-dir`, `--run-dir`, and `--checkpoint`. For Hub-downloaded artifacts, **`--run-dir test_model`** (or the folder you used with `download-glumind-hf`) is the usual choice.

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
- `marked_runs/glumind/*/RUNS_ANALYSIS.md`
- `runs/nhits/RUNS_ANALYSIS.md`

## Notes

- In `trainval_test_as_val` mode, held-out test metrics are intentionally disabled.
- For quick validation after code changes, run smoke settings such as `--epochs 1 --max_train_series <small> --max_eval_series <small>`.
