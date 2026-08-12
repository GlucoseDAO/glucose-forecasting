# CLI reference

Prefer built-in help for exact flags:

| How to run | Help |
|------------|------|
| Console entry (`pyproject.toml` `[project.scripts]`) | `uv run <command> --help` |
| Module under `src/` | `uv run python src/.../script.py --help` (argparse) or Typer `--help` |

Argparse CLIs use `--snake_case`. Typer apps use `--kebab-case`.

Full worked examples also live in the root [README.md](../README.md).

---

## Platform CLI — `glucose`

```bash
uv run glucose --help
uv run glucose info
```

| Command | Role |
|---------|------|
| `info` | Package version, default runs root, evaluate config path |
| `evaluate` | **Unified** eval/compare for custom PyTorch runs (all model families) |
| `neuralforecast` | NF holdout train / evaluate / summarize |
| `release` | Pack / check / publish / pull inference bundles (format 1.0) |

There is **no** `glucose train` for custom PyTorch. Train with experiment CLIs (`train-glumind`, `train_sugar_one`, …).

### `glucose evaluate`

Central path for GluMind / GluMind-Uni / SugarOne / SugarJepa checkpoints (auto-detect or `--model-type`). Logic lives under `common/evaluation/checkpoint_eval.py`. Window datasets and CSV helpers live under `common/data/`.

```bash
uv run glucose evaluate --help
# Defaults from YAML (no --run-dir):
uv run glucose evaluate
# Single run re-inference:
uv run glucose evaluate --run-dir test_model_glumind --model-type glumind \
  --data test_data/livia_glumind_ready.csv --test-split "" --batch-size 4096 --no-plot
```

| Option | Meaning |
|--------|---------|
| `--run-dir` | Leaf run dir **or** container of runs (repeatable). Containers expand to best-by-val-MAE per model family. Default: `models[]` in YAML. |
| `--registry-dir` | Pick lowest `val_mae` from `_analysis_registry.csv` (single model). |
| `--checkpoint` | Explicit `.pt` weights (with `--run-dir` for meta). |
| `--data` | Eval CSV (default from YAML). Omit to use precomputed metrics when present. |
| `--train-data` | Legacy scaler fit CSV when `scalers.json` is missing. |
| `--label` | Label per `--run-dir` (repeatable). |
| `--out` | Comparison report directory (default `data/output/compare`). |
| `--config` | YAML defaults (`src/glucose_evaluate.yaml`). |
| `--model-type` | `auto` \| `glumind` \| `sugar_one` \| `glumind_uni` \| `sugar_jepa`. |
| `--test-split` | `Recommended Split` filter (default `test`; empty disables). |
| `--batch-size`, `--device` | Inference controls (`device`: `auto` \| `cuda` \| `mps` \| `cpu`). |
| `--zero-cov` / `--include-cov` / `--exclude-cov` | Covariate ablation. |
| `--refit-scalers` / `--allow-fit-on-eval` | Scaler fallback controls. |
| `--covariates` | Inspect CSV covariate columns and exit. |
| `--output-json` | Write metrics JSON for a single-run eval. |
| `--log-interval` | Progress log cadence (seconds). |
| `--plot` / `--no-plot` | Comparison charts under `--out`. |

YAML `models[]` entries may point at pinned demos (`test_model_glumind`) or containers such as `data/output/runs/nf_holdout`.

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
| Personalization suite | `src/personalization/*` | see console scripts in `pyproject.toml` |

Direct (no console script):

```bash
uv run python src/sugar_one/train_sugar_one.py --help
uv run python src/glumind_uni/train_uniglumind.py train --help
uv run python src/nf_baselines/tune_nf_baselines_by_group.py -h
uv run python src/glumind/eval_gluformer_val_test_masked.py -h   # external HF GluFormer baseline
```

---

## Quick smoke

```bash
uv run glucose --help
uv run glucose info
uv run glucose evaluate --run-dir test_model_glumind --model-type glumind \
  --data test_data/livia_glumind_ready.csv --test-split "" --batch-size 4096 --no-plot
uv run pytest -q
```
