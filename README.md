# Glucose Forecasting (GluMind / SugarOne)

Models that forecast blood glucose **60 minutes ahead** from CGM time series (optional pump or wearable covariates).

| Model | When to use | Inputs |
|-------|-------------|--------|
| **SugarOne** (main model) | Loop / insulin-pump data | glucose, basal, bolus, carbs |
| **GluMind** | Wearable / AI-READI-style data | glucose, heart rate, steps |

---

## Where to get the data

This repo does **not** ship the large licensed datasets. Prepare CSVs in the preprocessing project, then put them here:

1. **[glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing)** — download/convert raw CGM and pump exports into ML-ready CSVs  
2. Copy those files into **`data/input/`** in this repo (that folder is gitignored)

Typical files:

| File under `data/input/` | What it is |
|--------------------------|------------|
| `loop_ai_ready_joined2.csv` | Main Loop + AI-READI benchmark for SugarOne (~12M rows) |
| `loop_ai_ready_joined2_dev.csv` | Smaller subset for quick local runs |

More detail (schemas, join scripts): [docs/DATA.md](docs/DATA.md).

For a first try **without** those datasets, the repo includes a small demo CSV in `test_data/` and pretrained weights in `test_model_sugar_one/` / `test_model_glumind/`.

---

## Load weights and run on your data

Needs Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

### On your CSV (after you placed it in `data/input/`)

`--run-dir` is the folder with the weights (`best_model.pt` + `tuning_meta.json` / `config.json`).  
`--test-csv` is the data to score; `--train-csv` is used only to fit scalers (often the same file).

```bash
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv data/input/loop_ai_ready_joined2.csv \
  --train-csv data/input/loop_ai_ready_joined2.csv \
  --batch-size 256
```

This prints **MAE / RMSE / MARD**. With the bundled SugarOne weights on the full joined benchmark, expect about **12.4 MAE** on the test split.

To use weights you trained yourself, point `--run-dir` at that run folder under `runs/` instead of `test_model_sugar_one`.

### On the included demo CSV (no `data/input/` needed)

```bash
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv test_data/livia_sugar_one_ready.csv \
  --train-csv test_data/livia_sugar_one_ready.csv \
  --test-split '' \
  --batch-size 256
```

`--test-split ''` is required for this demo (it has no train/val/test labels). Scores on Livia are only a rough check, not a published benchmark.

GluMind weights / Hugging Face download: [How_to_run_checkpoint.md](How_to_run_checkpoint.md).

### If something fails

| Message | What to do |
|---------|------------|
| `CSV not found` | Put the file under `data/input/` or pass the correct `--test-csv` / `--train-csv` |
| `Evaluation dataframe is empty` | Your CSV has no `Recommended Split == test` rows — use `--test-split ''` to score all rows |
| Flags / options | `uv run evaluate-model --help` |

---

## Train a model (optional next step)

```bash
uv run python scripts/sugar_one/train_sugar_one.py \
  --csv data/input/loop_ai_ready_joined2_dev.csv \
  --mode global \
  --device cuda \
  --epochs 30 \
  --patience 3 \
  --batch-size 256 \
  --out-dir runs/sugar_one
```

Then evaluate with `--run-dir` pointing at the new folder under `runs/sugar_one/`.

Hyperparameter search: `uv run tune-sugar-one -c scripts/sugar_one/tune_sugar_one_dev.toml --device cuda`.

Results and milestone tables: [docs/MILESTONES.md](docs/MILESTONES.md), [docs/GLUMIND_VS_SUGARONE_COMPARISON.md](docs/GLUMIND_VS_SUGARONE_COMPARISON.md).

---

## What’s in this repo

| Path | What it is |
|------|------------|
| `test_model_sugar_one/`, `test_model_glumind/` | Pretrained weights you can load immediately |
| `test_data/` | Small demo CSVs |
| `data/input/` | Where **you** put ML-ready training/eval CSVs |
| `scripts/sugar_one/` | SugarOne training + `evaluate-model` |
| `scripts/glumind/` | GluMind training / evaluation |
| `scripts/loop_ai_ready/` | Helpers to join Loop + AI-READI CSVs |
| `docs/` | Data pipeline, milestones, comparison reports |
| `runs/` | Outputs from your training runs |

## CLI reference

Every script supports **built-in help** when run with `uv`:

| How to run | Help flag |
|------------|-----------|
| Installed console commands (see `pyproject.toml` `[project.scripts]`) | `uv run <command> --help` or `-h` where supported |
| Python entry files | `uv run python scripts/.../script.py --help` or `-h` (argparse) |

Argparse-based CLIs (`train_glumind.py`, `eval_gluformer_val_test_masked.py`) print defaults in `--help` via `ArgumentDefaultsHelpFormatter` where configured. Typer apps list each option with `--help`.

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

For cross-model evaluation (GluMind or SugarOne on any compatible CSV), prefer **`evaluate-model`** below.

### `evaluate-model` — `scripts/sugar_one/evaluate_model.py`

`uv run evaluate-model --help`

Unified evaluation for **GluMind** (HR + steps) and **SugarOne** (basal + bolus + carbs). Loads architecture metadata from the run folder, fits MinMax scalers on training rows, and reports **MAE, RMSE, MARD**.

| Option | Meaning |
|--------|---------|
| `--test-csv` | CSV to score (required). |
| `--run-dir` | Run directory with `tuning_meta.json` / `config.json` and `best_model.pt`. |
| `--registry-dir` | Folder with `_analysis_registry.csv`; picks lowest `val_mae` run. |
| `--checkpoint` | Explicit `.pt` weights; still need `--run-dir` for architecture metadata. |
| `--train-csv` | CSV for scaler fitting (default: `csv` from metadata). Override when the training file from metadata is not on disk. |
| `--model-type` | `auto` (detect from checkpoint), `glumind`, or `sugar_one`. |
| `--test-split` | Keep rows where `Recommended Split` equals this value (default `test`). Use `--test-split=''` to score all rows. |
| `--batch-size` | DataLoader batch size (default from metadata). |
| `--device` | Torch device (default `cuda` when available). |
| `--output-json` | Write metrics JSON for batch comparisons. |
| `--log-interval` | Seconds between inference progress logs (default `10`; `0` = first and last only). |
| `--zero-cov` | Zero all non-glucose covariates after imputation (glucose-only inference). Mutually exclusive with `--include-cov` / `--exclude-cov`. |
| `--include-cov` | Comma-separated covariates to keep; zero all other non-glucose channels (e.g. `basal,bolus`). |
| `--exclude-cov` | Comma-separated covariates to zero; keep the rest (e.g. `carbs`). |
| `--covariates` | Print covariate columns and fill stats for `--test-csv`; no checkpoint required. |

Full usage, ablation examples, and alias list: `scripts/sugar_one/README.md`.

You must pass either `--registry-dir` or `--run-dir` (unless using `--covariates` only).

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

### `glucose train --backend neuralforecast`

`uv run glucose train --backend neuralforecast --data DATA.csv --help`

| Option | Meaning |
|--------|---------|
| `--eval` | `holdout` (default) uses the CSV's fixed train/val/test splits for comparable per-cohort metrics. `cross-val` performs rolling cross-validation for model screening. |
| `--profile` | `auto` (default), `ai-readi`, or `loop`. Auto-detection selects HR/steps for AI-READI and basal/bolus/carbohydrates for Loop data. |
| `--models` | YAML model suite (`auto`, `baseline`, `recurrent`) or comma-separated concrete model names. |
| `--model-config` | Replacement YAML defining curated model suites. |
| `--device` | `auto` (default: CUDA, then MPS, then CPU), `cuda`, `mps`, or `cpu`. |
| `--global-model` | Train one model using all study groups. |
| `--max-steps`, `--max-train-series`, `--max-eval-series` | Training duration and real-data development limits. |
| `--plot` / `--no-plot` | Write interactive HTML and static PNG actual-versus-forecast charts (default on). |
| `--max-plot-series` | Number of representative sequences visualized per model (default 3). |
| `--list-models` | Show YAML suites and their resolved model names without training. |

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

### `scripts/sugar_one/train_sugar_one.py` (SugarOne)

Root command `main` (no subcommand name):

`uv run python scripts/sugar_one/train_sugar_one.py --help`

Same shape as GluMindUni: insulin/carb covariates, default `--out-dir` `runs/sugar_one`, `--csv` should be the joined Loop + AI-READI CSV (e.g. `data/input/loop_ai_ready_joined2.csv`; see [docs/DATA.md](docs/DATA.md)). Device: `--device`.

Expected loop-style columns (aliases are resolved automatically by `evaluate-model`):

- `Glucose Value (mg/dL)` or `Glucose (mg/dL)`
- `Basal Rate (U/h)`
- `Bolus Insulin (U)`
- `Carbohydrates (g)`

### `tune-sugar-one` — `scripts/sugar_one/tune_sugar_one.py`

`uv run tune-sugar-one --help`

Random hyperparameter search for SugarOne (global mode only). Behaviour is driven by a TOML config:

| Option | Meaning |
|--------|---------|
| `--config`, `-c` | TOML config path (default: `scripts/sugar_one/tune_sugar_one_full.toml`). |
| `--device` | `cuda`, `cpu`, or `mps` (default `cuda`). |
| `--seed` | Override `.random_seed` from the config. |

Shipped configs: `tune_sugar_one_full.toml` (production search) and `tune_sugar_one_dev.toml` (smaller laptop search).

## Expected Dataset Columns

Produce these CSVs with [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing), then place them under `data/input/` (see [docs/DATA.md](docs/DATA.md)).

GluMind / AI-READI-style columns:
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

Examples assume an AI-READI ML-ready CSV at `data/input/ai_ready_processed_dataset.csv` (from the preprocessing repo). Symlink or adjust `--csv` if you keep the older `data/actual/with_complex_steps_processing/` layout.

Global mode example:

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/input/ai_ready_processed_dataset.csv \
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
  --csv data/input/ai_ready_processed_dataset.csv \
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
  --csv data/input/ai_ready_processed_dataset.csv \
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
  --csv data/input/ai_ready_processed_dataset.csv \
  --mode global \
  --resume_from runs/glumind/<run_name>/last_checkpoint.pt \
  --epochs 250 \
  --device cuda
```

## SugarOne Training and Tuning

Train on the joined Loop + AI-READI benchmark (build with `scripts/loop_ai_ready/` after preprocessing):

```bash
uv run python scripts/sugar_one/train_sugar_one.py \
  --csv data/input/loop_ai_ready_joined2.csv \
  --mode global \
  --device cuda \
  --epochs 120 \
  --patience 10 \
  --batch_size 256 \
  --out_dir runs/sugar_one
```

Production hyperparameter search:

```bash
uv run tune-sugar-one --device cuda
```

Use `-c scripts/sugar_one/tune_sugar_one_dev.toml` for a smaller dev search.

## NeuralForecast Baselines

Run the default `auto` YAML suite on the development subset. The command detects the
Loop schema and uses basal rate, bolus insulin, and carbohydrates as historical
covariates. `--device auto` selects CUDA when available, otherwise MPS or CPU.

```bash
uv run glucose train \
  --backend neuralforecast \
  --data data/input/loop_ai_ready_joined2_dev.csv \
  --global-model \
  --max-steps 300 \
  --max-train-series 20 \
  --max-eval-series 10
```

Use `--eval holdout` (the default) for fixed CSV train/validation/test metrics that
can be compared with GluMind and SugarOne. Use `--eval cross-val` for rolling
cross-validation when screening the YAML model suite; those results are intentionally
stored separately and are not cohort-report metrics.

```bash
uv run glucose train \
  --backend neuralforecast \
  --eval cross-val \
  --data data/input/loop_ai_ready_joined2_dev.csv \
  --models auto \
  --max-steps 300 \
  --max-train-series 20
```

Curated suite membership lives in
`src/glucose_forecasting/backends/neuralforecast/model_suites.yaml`. `auto` includes
NHITS, NBEATSx, LSTM, TFT, TiDE, and xLSTM. Inspect it before training:

```bash
uv run glucose train --backend neuralforecast \
  --data data/input/loop_ai_ready_joined2_dev.csv --list-models
```

Each holdout run prints a Lightning progress bar plus a concise loss line after every
epoch. It also saves a reloadable NeuralForecast bundle, Lightning checkpoints,
metrics, predictions, plots, and both machine-readable and rendered structured logs:

```text
runs/nf_holdout/__ALL__/NHITS_<UTC_TIMESTAMP>/
├── checkpoints/                 # best Lightning checkpoint and last.ckpt
├── neuralforecast/              # nf.save() bundle for reuse
├── logs/
│   ├── training.json             # Eliot structured events (status, epoch loss, metrics)
│   └── training.log              # human-readable rendering of the same events
├── val_metrics_*.csv
├── test_metrics_*.csv
├── val_predictions.csv
├── test_predictions.csv
└── plots/
    ├── NHITS_val/sequence_*.html|png
    └── NHITS_test/sequence_*.html|png
```

Cross-validation additionally writes `cross_val_metrics_summary.csv`, per-model
metrics and prediction CSVs, interactive per-model charts, and
`plots/dashboard/model_comparison.html|png`.

## GluFormer Evaluation

Evaluate val/test splits:

```bash
uv run python scripts/eval_gluformer_val_test_masked.py \
  --csv data/input/ai_ready_plus_type1_v2_val_only_in_test.csv \
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

## Outputs

Typical run artifacts:
- `val_metrics_overall.csv`
- `val_metrics_by_study_group.csv`
- `test_metrics_overall.csv`
- `test_metrics_by_study_group.csv`
- `tuning_meta.json`
- `config.json`
- `checkpoints/`

## Reports and docs

| Doc | Contents |
|-----|----------|
| [docs/DATA.md](docs/DATA.md) | Preprocessing companion repo, local `data/input/` layout, joined benchmark |
| [docs/MILESTONES.md](docs/MILESTONES.md) | BGI M06/M07 summary + GluMindIC → SugarOne naming |
| [How_to_run_checkpoint.md](How_to_run_checkpoint.md) | Load weights and run on a CSV |
| [docs/CROSS_MODEL_COMPARISON_REPORT.md](docs/CROSS_MODEL_COMPARISON_REPORT.md) | GluMind vs NHITS / GluFormer (M06) |
| [CROSS_MODEL_COMPARISON.md](CROSS_MODEL_COMPARISON.md) | Detailed M06 numbers |
| [docs/GLUMIND_VS_SUGARONE_COMPARISON.md](docs/GLUMIND_VS_SUGARONE_COMPARISON.md) | GluMind vs SugarOne on joined benchmark (M07) |
| [docs/T1DM_COVARIATE_ABLATION_REPORT.md](docs/T1DM_COVARIATE_ABLATION_REPORT.md) | Basal/bolus/carb ablations |
| `reports/` / `marked_runs/` | Per-run analysis markdown |

## Notes

- In `trainval_test_as_val` mode, held-out test metrics are intentionally disabled.
- For a short training check after code changes, try `--epochs 1 --max_train_series <small> --max_eval_series <small>`.
- `--mask_interpolated_targets` / `--save_predictions` on `train-glumind` are defined in the CLI but not wired in the current training loop.
