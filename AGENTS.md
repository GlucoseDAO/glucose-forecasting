# AGENTS.md

This file provides guidance to coding agents (Cursor, Claude Code, Copilot, etc.) when working with code in this repository. Prefer this file over tool-specific names (`CLAUDE.md`, etc.).

## Temporary vs permanent artifacts

Put all intermediate reports, scratch notes, evaluation dumps, and other temporary information in `temp_docs/`. Put all temporary or one-off scripts in `temp_scripts/`.

- **One-shot / never-reuse scripts**: run them, then delete them. Do not leave them in the tree.
- **Potentially reusable but not part of the product surface**: keep them under `temp_scripts/` (e.g. one-off data prep, report generation, ad-hoc analysis).
- **Stay in the project codebase** only if the code is intentionally part of the product: CLI entry points, API surfaces, or modules imported by those (or by other first-class code / tests).

Examples:
- A script that prepares a dataset once → `temp_scripts/`
- A script that builds a milestone or interim report → `temp_scripts/`, outputs under `temp_docs/`
- Training/eval CLIs and shared libraries under `scripts/` that are imported by CLIs/tests → keep in the codebase

- Do not add intermediate markdown or scratch analysis under `docs/` or the repo root; `docs/` is for intentional, durable documentation. Training run outputs still go to `runs/` / `marked_runs/` as today.
- **Tests must not reference `temp_scripts/` or `temp_docs/`.** Those folders are gitignored and unavailable from a fresh clone. Do not import, invoke, or assert against temporary scripts or their outputs. If a script moves to `temp_scripts/`, delete the tests that covered it rather than skipping or path-hacking around the move.

## Project overview

Training, tuning, and evaluation pipelines for blood-glucose forecasting from CGM (continuous glucose monitor) data, on AI-READI-style and Loop-pump-style datasets. Several model variants exist as parallel experiments sharing similar training/eval scaffolding but different covariates:

- **GluMind** (`scripts/glumind/`) — glucose + heart rate + step count. Primary architecture (Farahmand et al., 2025b, arXiv:2509.18457): parallel cross-attention multimodal fusion + multi-scale self-attention, with optional LwF (learning-without-forgetting) for continual cross-cohort training.
- **GluMind-Uni** (`scripts/glumind_uni/`) — glucose-only variant of the same architecture.
- **SugarOne** (`scripts/sugar_one/`) — glucose + basal rate + bolus insulin + carbohydrates (Loop pump data), 3-way cross-attention with learnable softmax mixing weights (vs. GluMind's fixed 2-way averaging).
- **NeuralForecast baselines** (`scripts/tune_nf_baselines_by_group.py`) — NHITS / TFT / NBEATSx, glucose-only.
- **GluFormer** (`scripts/eval_gluformer_val_test_masked.py`) — evaluation only, against a pretrained Hugging Face model (`njeffrie/Gluformer`).

Forecast horizon defaults to 12 steps = 60 minutes at 5-minute sampling frequency.

## Commands

Environment (Python >=3.12, managed with `uv`):
```bash
uv sync
```

Run tests:
```bash
uv run pytest -q            # full suite (tests/ — 21 tests as of this writing)
uv run pytest tests/test_evaluate_model_covariates.py -q   # single file
uv run pytest tests/test_train_checkpoint_resume.py::test_checkpoint_stores_wait_and_resumes_next_epoch -q  # single test
```
No lint/format command is configured in `pyproject.toml`.

Installed console commands (defined in `pyproject.toml` `[project.scripts]`, all runnable as `uv run <name> --help`):
- `train-glumind` → `scripts/glumind/train_glumind.py:main` (argparse CLI)
- `evaluate-glumind` → `scripts/glumind/evaluate_glumind.py:app` (Typer; GluMind-only)
- `evaluate-model` → `scripts/sugar_one/evaluate_model.py:app` (Typer; **unified** GluMind + SugarOne eval — preferred for new work)
- `inference-glumind` → `scripts/glumind/inference_glumind.py:app`
- `tune-sugar-one` → `scripts/sugar_one/tune_sugar_one.py:app` (TOML-driven random hyperparameter search)
- `download-glumind-hf` → `scripts/glumind/download_from_huggingface.py:app`

Scripts without a console entry point are run directly, e.g.:
```bash
uv run python scripts/sugar_one/train_sugar_one.py --help          # Typer, no subcommand name
uv run python scripts/glumind_uni/train_uniglumind.py train --help # Typer, `train` subcommand
uv run python scripts/tune_nf_baselines_by_group.py -h              # argparse
uv run python scripts/eval_gluformer_val_test_masked.py -h          # argparse
uv run python scripts/glumind/upload_to_huggingface.py --help
```

Full flag reference and worked examples for every script live in the root `README.md` — read that before guessing at a flag name; it documents which flags exist per-CLI (argparse vs. Typer scripts have different flags even for equivalent features, e.g. `--csv` vs `--csv`, snake_case vs kebab-case).

Fast smoke test after code changes (no GPU, no full dataset needed):
```bash
uv run evaluate-model --run-dir test_model_glumind --model-type glumind \
  --test-csv test_data/livia_glumind_ready.csv --test-split "" --batch-size 4096
```
This uses the bundled reviewer checkpoint (`test_model_glumind/`), its **`scalers.json`**, and demo CSV (`test_data/livia_glumind_ready.csv`, ~140k rows, no `Recommended Split` column — always pass `--test-split ""` for it). For SugarOne against the same demo file, add `--zero-cov` since it has no insulin/carb columns. See README.md "Evaluate on `test_data/livia_glumind_ready.csv`" section for exact commands including the SugarOne case.

## Architecture

### Shared utilities: `scripts/common/`

As of the last refactor, model-agnostic logic that used to be duplicated across the training scripts (and re-imported cross-model, e.g. `evaluate_model.py` importing from `train_glumind.py`) now lives here. **New model variants or eval tools should use these instead of reimplementing.**

- `data_loading.py` — `load_splits_streaming`, `apply_split_scheme` (`classic` vs `trainval_test_as_val`), `impute_and_sort` (per-series forward/backward-fill for continuous signals like glucose/HR/basal, zero-fill for discrete signals like bolus/carbs), `limit_series`, `normalize_study_group_label`/`normalize_study_groups_column`, `resolve_num_workers`. Column names are passed in by the caller (`value_columns: dict[str, str]` mapping canonical → source CSV column) since each model variant reads a different CSV column set.
- `metrics.py` — `mae_rmse_mard` (MAE/RMSE/MARD — the metric triple used everywhere in this repo), `per_study_group_breakdown`, `overall_metrics_to_csv`.
- `checkpoint.py` — `save_full_checkpoint`/`load_full_checkpoint` (generalized across the three slightly different checkpoint shapes used by GluMind/SugarOne/GluMind-Uni via a `config_key` param — `"args"`, `"config"`, or `"cfg"`), `read_checkpoint_meta`, `update_latest_symlink`, `strip_compile_prefix` (strips the `_orig_mod.` prefix `torch.compile` adds to state_dict keys).
- `registry.py` — `find_best_run_dir` (reads `_analysis_registry.csv`, picks lowest `val_mae`), `load_run_meta` (reads `tuning_meta.json` or `config.json`), `resolve_checkpoint` (finds `best_model.pt`/`last_model.pt`), `resolve_csv_path`.
- `evaluation.py` — covariate alias/ablation machinery (maps derived from each family's `ModelFamilySpec.csv_column_aliases` / `covariate_aliases`; still exports `GLUMIND_COVARIATES`, `SUGAR_ONE_COVARIATES`, `COVARIATE_NAME_ALIASES`) and the shared inference loop (`_run_evaluate`), extracted from `evaluate_model.py`.
- `model_spec.py` — `ModelFamilySpec` Protocol + registry (`get_family_spec`, `detect_family_kind`). Concrete specs live beside each model (`scripts/glumind/glumind_spec.py`, `scripts/sugar_one/sugar_one_spec.py`, …). Architecture modules (`*_model.py`) stay torch-only for checkpoint reuse.
- `scalers.py` — schema-free `scalers.json` serialize/load (no kind→features whitelist; feature set comes from Spec or the file).

`train_glumind.py`, `train_sugar_one.py`, `train_uniglumind.py`, `evaluate_glumind.py`, and `evaluate_model.py` all import from `scripts/common/*` internally but **re-export the same names under their original locations** — e.g. `from scripts.glumind.train_glumind import load_splits_streaming` still works. Do not break this: `evaluate_model.py` still cross-imports several names from `train_glumind.py`/`train_sugar_one.py` by their original names, and `tests/test_evaluate_model_covariates.py` imports covariate helpers from `scripts.sugar_one.evaluate_model` directly.

### Model files — checkpoint-friendly, kept separate from training logic

- `scripts/glumind/glumind_model.py` — `GluMindModel`. 2-auxiliary parallel cross-attention (HR, steps) with fixed averaging + multi-scale self-attention.
- `scripts/sugar_one/sugar_one_model.py` — `SugarOneModel`. 3-auxiliary parallel cross-attention (basal, bolus, carbs) with **learnable softmax mixing weights** (the main architectural difference from GluMind).
- `scripts/glumind_uni/glumind_uni_model.py` — glucose-only variant, same block structure.

These are intentionally isolated from the training scripts so a checkpoint can be loaded with just the model file + `torch.load(..., weights_only=True)` — no training-script imports needed (see README.md "Checkpoints and Model Reuse" for a minimal load example). `PositionalEncoding` and much of `MultiScaleAttentionBlock` are near-duplicated between `glumind_model.py` and `sugar_one_model.py`; this has been left as-is (not deduplicated) to avoid touching model internals and risking checkpoint/behavior drift — treat these files as intentionally frozen unless a change is specifically requested.

### Checkpoints

Two kinds of weight files are saved per run, both plain `torch.save` of dicts (no pickled class references, so moving code around is safe as long as model `state_dict` key names don't change):
- `best_model.pt` / `last_model.pt` — plain `state_dict` only, loaded with `weights_only=True`.
- `checkpoint.pt` / `last_checkpoint.pt` — full training state (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, epoch, best_val_loss, config), loaded with `weights_only=False`, used to `--resume_from` a training run.

Architecture hyperparameters live separately in `tuning_meta.json` / `config.json` inside the run directory. **Train-fit MinMax scalers** are saved as sidecar **`scalers.json`** next to the weights (written at train time; eval prefers this over re-fitting from CSV). Legacy runs without the sidecar can be backfilled with `uv run python temp_scripts/migrate_scalers.py --run-dir <run>` when the training CSV is still available; otherwise eval falls back to CSV re-fit (`--train-csv` / metadata `csv`).

### Training modes

All training scripts (GluMind, SugarOne, GluMind-Uni) support the same four modes: `global` (one model, all study groups), `per_group` (independent model per study group), `cohort_wise`, and `continual` (sequential per-group training with optional LwF regularization via `--lwf_lambda` to reduce catastrophic forgetting across cohorts).

### CLI framework split

`train_glumind.py`, `tune_nf_baselines_by_group.py`, and `eval_gluformer_val_test_masked.py` use argparse; `train_sugar_one.py`, `train_uniglumind.py`, `evaluate_glumind.py`, `evaluate_model.py`, `tune_sugar_one.py` use Typer. This is a known inconsistency (not enforced) — check which framework a script uses before assuming flag syntax (argparse: `--snake_case`; Typer: `--kebab-case`).

### Data expectations

Core AI-READI CSV columns: `sequence_id`, `User ID`, `Timestamp (YYYY-MM-DDThh:mm:ss)`, `Recommended Split` (`train`/`val`/`test`), `Study Group`, `Event Type`, `Glucose Value (mg/dL)`, `Heart Rate`, `Step Count`.

Loop/SugarOne CSVs additionally/instead have: `Glucose (mg/dL)` (or `Glucose Value (mg/dL)`), `Basal Rate (U/h)`, `Bolus Insulin (U)`, `Carbohydrates (g)`.

`evaluate-model` resolves column aliases automatically and can zero out or ablate individual covariates at inference time (`--zero-cov`, `--include-cov`, `--exclude-cov`) for cross-model/cross-covariate comparison — this is the main tool for comparing GluMind vs. SugarOne on the same data.

One-off Loop+AI-READI join / sample scripts live under `temp_scripts/loop_ai_ready/` (not part of the training/eval pipeline). Run them as `uv run python temp_scripts/loop_ai_ready/<script>.py` from the repo root.

### Known dead/unwired flags

`--mask_interpolated_targets` and `--save_predictions` are defined in some CLIs' argument parsers but have no effect in the current training loop for `train_glumind.py` (per README.md) — don't assume they work without checking the specific script.

### Reports and run artifacts

`runs/` and `marked_runs/` hold training outputs (checkpoints, per-split metrics CSVs, `tuning_meta.json`). `marked_runs/` is curated/annotated subset with `RUNS_ANALYSIS.md` writeups per model/dataset combo. Durable comparison and milestone writeups live under `docs/` (e.g. `docs/GLUMIND_VS_SUGARONE_COMPARISON.md`, `docs/T1DM_COVARIATE_ABLATION_REPORT.md`). Intermediate / working reports go under `temp_docs/`. Root `CROSS_MODEL_COMPARISON.md` is the cross-model summary.
