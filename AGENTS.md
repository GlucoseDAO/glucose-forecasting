# AGENTS.md

This file provides guidance to coding agents (Cursor, Claude Code, Copilot, etc.) when working with code in this repository. Prefer this file over tool-specific names (`CLAUDE.md`, etc.).

## Before training or inference (mandatory)

**Never start a new training, evaluation, or inference job until you have checked that none is already running.** Stacking GPU/CPU jobs has repeatedly hung or stalled this machine.

1. Inspect Cursor terminals metadata under the project `terminals/` folder (or equivalent) for active commands matching train / evaluate / inference / `glucose evaluate` / `torch`.
2. On Windows, also scan processes, e.g.:
   ```powershell
   Get-CimInstance Win32_Process |
     Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'glucose|train_|evaluate|inference|torch|glumind|sugar_one' } |
     Select-Object ProcessId, Name, CommandLine
   ```
3. If a matching job is still running: **do not launch another**. Monitor or resume that job; only start a replacement after it has exited (or after the user explicitly asks to kill it).
4. Prefer one long-running ML job at a time. Do not “retry” by spawning a second `uv run glucose evaluate` / `train-*` while the first is still alive.

## Temporary vs permanent artifacts

Put all intermediate reports, scratch notes, evaluation dumps, and other temporary information in `temp_docs/`. Put all temporary or one-off scripts in `temp_scripts/`.

- **One-shot / never-reuse scripts**: run them, then delete them. Do not leave them in the tree.
- **Potentially reusable but not part of the product surface**: keep them under `temp_scripts/` (e.g. one-off data prep, report generation, ad-hoc analysis).
- **Stay in the project codebase** only if the code is intentionally part of the product: CLI entry points, API surfaces, or modules imported by those (or by other first-class code / tests).

Examples:
- A script that prepares a dataset once → `temp_scripts/`
- A script that builds a milestone or interim report → `temp_scripts/`, outputs under `temp_docs/`
- Training/eval CLIs and shared libraries under `src/` that are imported by CLIs/tests → keep in the codebase

- Do not add intermediate markdown or scratch analysis under `docs/` or the repo root; `docs/` is for intentional, durable documentation. Training run outputs go under `data/output/runs/` (not top-level `runs/`). See `docs/DATA.md`.
- **Tests must not reference `temp_scripts/` or `temp_docs/`.** Those folders are gitignored and unavailable from a fresh clone. Do not import, invoke, or assert against temporary scripts or their outputs. If a script moves to `temp_scripts/`, delete the tests that covered it rather than skipping or path-hacking around the move.

## Layout direction (platform adoption)

- Product code lives under `src/` as direct packages (`common`, `glumind`, `sugar_one`, `neuralforecast`, …). There is **no** `scripts/` tree and **no** nested `src/glucose_forecasting/` wrapper.
- Datasets live under `data/input/` (`actual/`, `loop_and_ai_ready/`, `personalization/`).
- Default run root is `data/output/runs/` (`common.paths.DEFAULT_RUNS_ROOT`); curated runs under `data/output/marked_runs/`.
- Top-level Typer app: `uv run glucose` (`src/cli.py`) — `info` + `evaluate` + `neuralforecast` + `release` (logic under `src/common/evaluation/`, `src/nf_baselines/`, `src/common/release/`). No `glucose train` for custom PyTorch; use experiment CLIs. NF holdout: `glucose neuralforecast train`. Release bundles: `glucose release check|publish|pull`.
- Implement adoption work **one phase at a time**; verify with `uv run pytest -q` and the demo `glucose evaluate` smoke before the next phase.
- Details: `temp_docs/ANTON_PR_COMPARISON_AND_REQUIREMENTS.md`.

## Project overview

Training, tuning, and evaluation pipelines for blood-glucose forecasting from CGM (continuous glucose monitor) data, on AI-READI-style and Loop-pump-style datasets. Several model variants exist as parallel experiments sharing similar training/eval scaffolding but different covariates:

- **GluMind** (`src/glumind/`) — glucose + heart rate + step count. Primary architecture (Farahmand et al., 2025b, arXiv:2509.18457): parallel cross-attention multimodal fusion + multi-scale self-attention, with optional LwF (learning-without-forgetting) for continual cross-cohort training.
- **GluMind-Uni** (`src/glumind_uni/`) — glucose-only variant of the same architecture.
- **SugarOne** (`src/sugar_one/`) — glucose + basal rate + bolus insulin + carbohydrates (Loop pump data), 3-way cross-attention with learnable softmax mixing weights (vs. GluMind's fixed 2-way averaging).
- **NeuralForecast baselines** (`src/nf_baselines/`) — preferred: sugarone-compatible holdout via `glucose neuralforecast` (128/12/stride-1); legacy tuner `tune_nf_baselines_by_group.py` kept until parity is verified.
- **GluFormer** (`src/glumind/eval_gluformer_val_test_masked.py`) — evaluation only, against a pretrained Hugging Face model (`njeffrie/Gluformer`).

Forecast horizon defaults to 12 steps = 60 minutes at 5-minute sampling frequency.

## Commands

Environment (Python >=3.12, managed with `uv`):
```bash
uv sync
```

Run tests:
```bash
uv run pytest -q            # full suite (tests/)
uv run pytest tests/test_evaluate_model_covariates.py -q   # single file
uv run pytest tests/test_train_checkpoint_resume.py::test_checkpoint_stores_wait_and_resumes_next_epoch -q  # single test
```


Platform / migration docs: `docs/CLI_REFERENCE.md`, `docs/MIGRATION.md`, `docs/DATA.md`.

Installed console commands (defined in `pyproject.toml` `[project.scripts]`, all runnable as `uv run <name> --help`):
- `glucose` → `src/cli.py:app` (platform CLI: `info`, `evaluate`, `neuralforecast`, `release`)
- `train-glumind` → `src/glumind/train_glumind.py:main` (argparse CLI)
- `tune-sugar-one` → `src/sugar_one/tune_sugar_one.py:app` (TOML-driven random hyperparameter search)
- `download-glumind-hf` → `src/glumind/download_from_huggingface.py:app`

Scripts without a console entry point are run directly, e.g.:
```bash
uv run python src/sugar_one/train_sugar_one.py --help          # Typer, no subcommand name
uv run python src/glumind_uni/train_uniglumind.py train --help # Typer, `train` subcommand
uv run python src/nf_baselines/tune_nf_baselines_by_group.py -h              # argparse
uv run python src/glumind/eval_gluformer_val_test_masked.py -h          # argparse
uv run python src/glumind/upload_to_huggingface.py --help
```

Full flag reference and worked examples for every script live in the root `README.md` — read that before guessing at a flag name; it documents which flags exist per-CLI (argparse vs. Typer scripts have different flags even for equivalent features, e.g. `--csv` vs `--csv`, snake_case vs kebab-case).

Fast smoke test after code changes (no GPU, no full dataset needed):
```bash
uv run glucose evaluate --run-dir test_model_glumind --model-type glumind \
  --data test_data/livia_glumind_ready.csv --test-split "" --batch-size 4096 --no-plot
```
This uses the bundled reviewer checkpoint (`test_model_glumind/`), its **`scalers.json`**, and demo CSV (`test_data/livia_glumind_ready.csv`, ~140k rows, no `Recommended Split` column — always pass `--test-split ""` for it). For SugarOne against the same demo file, add `--zero-cov` since it has no insulin/carb columns. See README.md and `docs/CLI_REFERENCE.md`.

## Architecture

### Shared utilities: `src/common/`

As of the last refactor, model-agnostic logic that used to be duplicated across the training scripts (and re-imported cross-model, e.g. `evaluate_model.py` importing from `train_glumind.py`) now lives here. **New model variants or eval tools should use these instead of reimplementing.**

- `data/loading.py` — `load_splits_streaming`, `apply_split_scheme` (`classic` vs `trainval_test_as_val`), `impute_and_sort` (per-series forward/backward-fill for continuous signals like glucose/HR/basal, zero-fill for discrete signals like bolus/carbs), `limit_series`, `normalize_study_group_label`/`normalize_study_groups_column`, `resolve_num_workers`. Column names are passed in by the caller (`value_columns: dict[str, str]` mapping canonical → source CSV column) since each model variant reads a different CSV column set.
- `metrics.py` — `mae_rmse_mard` (MAE/RMSE/MARD — the metric triple used everywhere in this repo), `per_study_group_breakdown`, `overall_metrics_to_csv`.
- `checkpoint.py` — `save_full_checkpoint`/`load_full_checkpoint` (generalized across the three slightly different checkpoint shapes used by GluMind/SugarOne/GluMind-Uni via a `config_key` param — `"args"`, `"config"`, or `"cfg"`), `read_checkpoint_meta`, `update_latest_symlink`, `strip_compile_prefix` (strips the `_orig_mod.` prefix `torch.compile` adds to state_dict keys).
- `registry.py` — `find_best_run_dir` (reads `_analysis_registry.csv`, picks lowest `val_mae`), `load_run_meta` (reads `tuning_meta.json` or `config.json`), `resolve_checkpoint` (finds `best_model.pt`/`last_model.pt`), `resolve_csv_path` (basename remap toward `data/input/` plus legacy→new path rewrites).
- `paths.py` — `DEFAULT_RUNS_ROOT` (`data/output/runs`), `DEFAULT_MARKED_RUNS_ROOT`, input dataset roots, legacy path rewrite helpers.
- `evaluation/` — covariate alias/ablation + shared inference loop (`core.py`); Phase-3 APIs (`runner`, `detect`, `comparison`, `pytorch`, `resolve_models`) used by `glucose evaluate`. Still exports `GLUMIND_COVARIATES`, `SUGAR_ONE_COVARIATES`, `COVARIATE_NAME_ALIASES`, `_run_evaluate`.
- `release/` — inference bundle format 1.0 (`manifest`/`config`/`preprocessor`/`metrics`/`provenance` + `model.safetensors` + SHA256); `glucose release pack|check|publish|pull`. Pack exports training run dirs into validated bundles.
- `model_spec.py` — `ModelFamilySpec` Protocol + registry (`get_family_spec`, `detect_family_kind`). Concrete specs live beside each model (`src/glumind/glumind_spec.py`, `src/sugar_one/sugar_one_spec.py`, …). Architecture modules (`*_model.py`) stay torch-only for checkpoint reuse.
- `scalers.py` — schema-free `scalers.json` serialize/load (no kind→features whitelist; feature set comes from Spec or the file).

`train_glumind.py`, `train_sugar_one.py`, `train_uniglumind.py`, and `common.evaluation.checkpoint_eval` (behind `glucose evaluate`) all import from `common.*`. Sliding-window datasets live in `common.data` (re-exported from train scripts). CSV loading: `common.data.loading`. Column constants: `common.data.columns`.

### Model files — checkpoint-friendly, kept separate from training logic

- `src/glumind/glumind_model.py` — `GluMindModel`. 2-auxiliary parallel cross-attention (HR, steps) with fixed averaging + multi-scale self-attention.
- `src/sugar_one/sugar_one_model.py` — `SugarOneModel`. 3-auxiliary parallel cross-attention (basal, bolus, carbs) with **learnable softmax mixing weights** (the main architectural difference from GluMind).
- `src/glumind_uni/glumind_uni_model.py` — glucose-only variant, same block structure.

These are intentionally isolated from the training scripts so a checkpoint can be loaded with just the model file + `torch.load(..., weights_only=True)` — no training-script imports needed (see README.md "Checkpoints and Model Reuse" for a minimal load example). `PositionalEncoding` and much of `MultiScaleAttentionBlock` are near-duplicated between `glumind_model.py` and `sugar_one_model.py`; this has been left as-is (not deduplicated) to avoid touching model internals and risking checkpoint/behavior drift — treat these files as intentionally frozen unless a change is specifically requested.

### Checkpoints

Two kinds of weight files are saved per run, both plain `torch.save` of dicts (no pickled class references, so moving code around is safe as long as model `state_dict` key names don't change):
- `best_model.pt` / `last_model.pt` — plain `state_dict` only, loaded with `weights_only=True`.
- `checkpoint.pt` / `last_checkpoint.pt` — full training state (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, epoch, best_val_loss, config), loaded with `weights_only=False`, used to `--resume_from` a training run.

Architecture hyperparameters live separately in `tuning_meta.json` / `config.json` inside the run directory. **Train-fit MinMax scalers** are saved as sidecar **`scalers.json`** next to the weights (written at train time; eval prefers this over re-fitting from CSV). Legacy runs without the sidecar can be backfilled with `uv run python temp_scripts/migrate_scalers.py --run-dir <run>` when the training CSV is still available; otherwise eval falls back to CSV re-fit (`--train-csv` / metadata `csv`).

### Training modes

All training scripts (GluMind, SugarOne, GluMind-Uni) support the same four modes: `global` (one model, all study groups), `per_group` (independent model per study group), `cohort_wise`, and `continual` (sequential per-group training with optional LwF regularization via `--lwf_lambda` to reduce catastrophic forgetting across cohorts).

### CLI framework split

`train_glumind.py`, `tune_nf_baselines_by_group.py`, and `eval_gluformer_val_test_masked.py` use argparse; `train_sugar_one.py`, `train_uniglumind.py`, `tune_sugar_one.py`, and `glucose` use Typer. This is a known inconsistency (not enforced) — check which framework a script uses before assuming flag syntax (argparse: `--snake_case`; Typer: `--kebab-case`).

### Data expectations

Core AI-READI CSV columns: `sequence_id`, `User ID`, `Timestamp (YYYY-MM-DDThh:mm:ss)`, `Recommended Split` (`train`/`val`/`test`), `Study Group`, `Event Type`, `Glucose Value (mg/dL)`, `Heart Rate`, `Step Count`.

Loop/SugarOne CSVs additionally/instead have: `Glucose (mg/dL)` (or `Glucose Value (mg/dL)`), `Basal Rate (U/h)`, `Bolus Insulin (U)`, `Carbohydrates (g)`.

`glucose evaluate` resolves column aliases automatically and can zero out or ablate individual covariates at inference time (`--zero-cov`, `--include-cov`, `--exclude-cov`) for cross-model/cross-covariate comparison — this is the main tool for comparing GluMind vs. SugarOne on the same data.

One-off Loop+AI-READI join / sample scripts live under `temp_scripts/loop_ai_ready/` (not part of the training/eval pipeline). Run them as `uv run python temp_scripts/loop_ai_ready/<script>.py` from the repo root.

### Known dead/unwired flags

`--mask_interpolated_targets` and `--save_predictions` are defined in some CLIs' argument parsers but have no effect in the current training loop for `train_glumind.py` (per README.md) — don't assume they work without checking the specific script.

### Reports and run artifacts

`data/output/runs/` holds training outputs (checkpoints, per-split metrics CSVs, `tuning_meta.json`). `data/output/marked_runs/` is a curated/annotated subset with `RUNS_ANALYSIS.md` writeups per model/dataset combo. Data layout and CSV remap rules: `docs/DATA.md`. Durable comparison and milestone writeups live under `docs/` (e.g. `docs/GLUMIND_VS_SUGARONE_COMPARISON.md`, `docs/T1DM_COVARIATE_ABLATION_REPORT.md`). Intermediate / working reports go under `temp_docs/`. Root `CROSS_MODEL_COMPARISON.md` is the cross-model summary.
