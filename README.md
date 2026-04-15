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

- `scripts/GluMind/train_glumind.py`: GluMind training/tuning entrypoint.
- `scripts/GluMind/glumind_model.py`: extracted model architecture module (checkpoint-friendly).
- `scripts/tune_nf_baselines_by_group.py`: NeuralForecast tuning.
- `scripts/eval_gluformer_val_test_masked.py`: GluFormer evaluation on val/test.
- `runs/`: model run outputs (metrics, checkpoints, predictions).
- `marked_runs/`: curated run sets and analysis markdown files.
- `CROSS_MODEL_COMPARISON.md`: cross-model summary report.

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
uv run python scripts/GluMind/train_glumind.py \
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
uv run python scripts/GluMind/train_glumind.py \
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
uv run python scripts/GluMind/train_glumind.py \
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
uv run python scripts/GluMind/train_glumind.py \
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
- `scripts/GluMind/glumind_model.py`

So you can load checkpoints without the full training script:

```python
import torch
from scripts.GluMind.glumind_model import GluMindModel

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
uv run scripts/glumind/evaluate_glumind.py `
  --run-dir test_model `
  --test-csv test_data/livia_glumind_ready.csv
```

**Useful options** (see `scripts/glumind/evaluate_glumind.py` Typer `main`):

- `--device cpu` — force CPU if CUDA is unavailable or undesired.
- `--train-csv <path>` — override the training CSV used for scaler fitting (default: path inside downloaded `tuning_meta.json` / metadata); use if you must point to a local file for scaler statistics.
- `--test-split test` — only if the CSV has `Recommended Split` and you want to filter rows.
- `--glucose-only` — ablation: zero (or mean/median) HR and steps for inference.

The script docstring at the top of `evaluate_glumind.py` documents three ways to point at a model (`--registry-dir`, `--run-dir`, `--checkpoint`); for Hugging Face–downloaded artifacts, **`--run-dir test_model`** is the intended path.

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
