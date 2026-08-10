# GluMind Inference & Evaluation

The `inference_glumind.py` script is used to evaluate trained GluMind models from their run directories. It automatically detects the correct evaluation mode (test vs. validation) and provides options for feature importance testing (e.g., glucose-only mode).

## Core Features

- **Auto-Detection**: Derives the evaluation mode (`test` or `val_as_test`) from `tuning_meta.json` using the `split_scheme`.
- **Reproducibility Check**: Compares the newly computed metrics against the saved metrics found in the run directory (`test_metrics_overall.csv` or `val_metrics_overall.csv`).
- **Feature Sensitivity**: Supports `--glucose-only` mode to evaluate model performance when non-glucose features (HR, Steps) are replaced with constant values.

## Usage

Run the script using `uv run`:

```bash
uv run src/glumind/inference_glumind.py --run-dir <path_to_run> [OPTIONS]
```

### Main Options

| Option | Description |
| :--- | :--- |
| `--run-dir` | **Required**. Path to the specific run directory (e.g., `marked_runs/glumind/ai_ready/run_name`). |
| `--mode` | Evaluation mode. Default is `auto`. Options: `auto`, `test`, `val_as_test`. |
| `--glucose-only` | If set, replaces all non-glucose features (HR, Steps) with a default value. |
| `--default-value` | Strategy for `--glucose-only`. Options: `zero` (default), `mean`, `median`. |
| `--device` | Torch device (`cpu` or `cuda`). Defaults to auto-detection. |

## Example Commands

### 1. Standard Evaluation (Auto Mode)
Evaluate a run using its original split scheme and compare with saved metrics:
```bash
uv run src/glumind/inference_glumind.py --run-dir "marked_runs/glumind/ai_ready/glumind_global_h12_20260222_194108"
```

### 2. Glucose-Only Sensitivity Test
Evaluate how the model performs without heart rate or step data by replacing them with their mean values from the training set:
```bash
uv run src/glumind/inference_glumind.py \
  --run-dir "marked_runs/glumind/type1_only/glumind_global_h12_20260225_120905" \
  --glucose-only \
  --default-value mean
```

### 3. Explicit Validation Check
Force evaluation on the validation split even if the run was a final test run:
```bash
uv run src/glumind/inference_glumind.py \
  --run-dir "marked_runs/glumind/ai_ready/glumind_global_h12_20260222_194108" \
  --mode val_as_test
```

## Internal Logic

1. **Metadata Loading**: Reads `tuning_meta.json` (or `config.json`) to determine model architecture (`d_model`, `n_heads`, etc.) and data parameters.
2. **Data Streaming**: Uses `load_splits_streaming` to handle large datasets efficiently.
3. **Mode Resolution**:
   - `split_scheme: classic` → evaluates on `test_df`.
   - `split_scheme: trainval_test_as_val` → evaluates on `val_df`.
4. **Metric Comparison**: Displays differences between reproduced metrics and the metrics originally saved during training to ensure consistency.
