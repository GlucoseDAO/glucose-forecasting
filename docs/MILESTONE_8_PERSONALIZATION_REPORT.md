# Milestone 8 — Personalization and Fine-Tuning Analysis

**Status:** Tooling updated; experiment results pending GPU runs.  
**Base model:** [`test_model_sugar_one/`](../test_model_sugar_one/)  
**Package:** [`scripts/personalization/`](../scripts/personalization/)  
**Plan:** [`scripts/personalization/plan.md`](../scripts/personalization/plan.md)

## Summary

We personalize the global **SugarOne** model for individual users using **Learning without Forgetting (LwF)**: the frozen global checkpoint is the teacher while we adapt on one person's CGM and pump data.

The workflow is:

1. Prepare chronological splits  
2. Find best **LwF weight**, **learning rate**, and **weight_decay** on **all** personal train data (excluding val/test)  
3. With those fixed, measure how **number of personal train days** affects accuracy and find the **plateau**  
4. Validate on Loop holdout users — same hyperparameters and same days-curve analysis  
5. Aggregate results for the report  

Personal vs general data mixing was removed from the plan.

## Workflow

| Step | When | Script (CLI) | Purpose |
|------|------|--------------|---------|
| **1** | First, once | `prepare-personal-csv` | Chronological train/val/test for Livia and holdouts |
| **2** | Second, on Livia | `sweep-personal-hyperparams` | LwF × LR × weight_decay on **full** personal train data |
| **3** | Third, on Livia | `sweep-personal-data-size` | Days curve with fixed LwF/LR; plateau estimate |
| **4** | Fourth, holdouts | `validate-personal-holdouts` | Frozen Livia recipe + per-user days curves |
| **5** | Last | `temp_scripts/personalization/aggregate_results.py` | Merge all summaries |

Optional: `finetune-personal` for a single manual run (debug or one-off).

## Step 2 — LwF, learning rate, and weight_decay (full personal train)

Fine-tune on **all** personal train rows (val and test held out).

### GluMind LwF starting point

From [`reports/glumind/AI_READY_PLUS_TYPE_1_TUNED_MODELS_RUNS_ANALYSIS.md`](../reports/glumind/AI_READY_PLUS_TYPE_1_TUNED_MODELS_RUNS_ANALYSIS.md):

- Best continual run: `glumind_continual_h12_20260226_011733`
- Best **`lwf_lambda = 0.3`** (test MAE 12.1975)

Secondary reference (ai_ready only): `lwf_lambda = 0.2` in [`AI_READY_TUNED_MODELS_RUNS_ANALYSIS.md`](../reports/glumind/AI_READY_TUNED_MODELS_RUNS_ANALYSIS.md).

### Search grids

| Parameter | Base | Grid |
|-----------|------|------|
| **LwF lambda** | **0.3** (GluMind type-1) | `0.2, 0.25, 0.3, 0.35` |
| **Learning rate** | `4e-4` (`test_model_sugar_one`) | `2e-4, 4e-4, 8e-4` (0.5×, 1×, 2×) |
| **weight_decay** | **`3e-5`** (`0.00003`) | `1.5e-5, 3e-5, 6e-5` (0.5×, 1×, 2×) |
| **Patience** | From base model meta | `10` (fixed) |

**Output:** `runs/personalization/livia/sweeps/hyperparams/best_recipe.json`

## Step 3 — Personal train days (fixed recipe)

Use `lwf_lambda`, `lr`, and `weight_decay` from step 2. Sweep days: `1, 3, 7, 14, 30, 60, all`.

**Outputs:**

- `.../data_size/summary.csv` — learning curve  
- `.../data_size/plateau_analysis.json` — `plateau_day`, `optimal_day`  
- `.../best_recipe_with_days.json`

## Step 4 — Holdout validation

**Users:** `154`, `556`, `730`, `1017`, `1029`, `1082` (in `loop.csv`, **not** in `loop_ai_ready_joined2.csv`).

- **Phase A:** Full-data fine-tune with frozen Livia `lwf_lambda` + `lr`  
- **Phase B:** Days sweep per user with same hyperparameters  
- Compare each holdout's plateau curve to Livia (`validation_meta.json`)

## Commands

```bash
# 1) Prepare Livia
uv run prepare-personal-csv livia \
  --input data/personalization/livia_glumind_ic_ready_full.csv \
  --out-dir data/personalization/prepared

# 2) LwF + LR sweep (full train)
uv run sweep-personal-hyperparams \
  --base-run-dir test_model_sugar_one \
  --personal-csv data/personalization/prepared/livia_chronological.csv \
  --out-dir runs/personalization/livia/sweeps/hyperparams \
  --device cuda

# 3) Days sweep
uv run sweep-personal-data-size \
  --base-run-dir test_model_sugar_one \
  --personal-csv data/personalization/prepared/livia_chronological.csv \
  --recipe-json runs/personalization/livia/sweeps/hyperparams/best_recipe.json \
  --out-dir runs/personalization/livia/sweeps/data_size \
  --device cuda

# 4) Holdouts
uv run validate-personal-holdouts \
  --base-run-dir test_model_sugar_one \
  --recipe-json runs/personalization/livia/sweeps/hyperparams/best_recipe.json \
  --livia-data-size-summary runs/personalization/livia/sweeps/data_size/summary.csv \
  --loop-csv data/loop_and_ai_ready/loop.csv \
  --out-dir runs/personalization/holdout_validation \
  --device cuda

# 5) Aggregate
uv run python temp_scripts/personalization/aggregate_results.py \
  --root runs/personalization \
  --out temp_docs/reports/milestone8_personalization_summary.json
```

## Results (pending GPU runs)

### Livia — LwF × LR × weight_decay (full train)

| LwF | LR | weight_decay | Fine-tuned MAE |
|-----|-----|--------------|----------------|
| _pending_ | | | |

### Livia — days vs test MAE (fixed LwF/LR)

| Days | Zero-shot MAE | Fine-tuned MAE | Plateau? |
|------|---------------|----------------|----------|
| _pending_ | | | |

**Plateau day:** _pending_ (`plateau_analysis.json`)

### Holdouts — params transfer (full train, Livia recipe)

| User | Zero-shot MAE | Fine-tuned MAE | Improved? |
|------|---------------|----------------|-----------|
| 154 | _pending_ | | |
| 556 | | | |
| 730 | | | |
| 1017 | | | |
| 1029 | | | |
| 1082 | | | |

### Holdouts — days curve vs Livia

| User | Optimal days | Livia optimal days | Delta |
|------|--------------|-------------------|-------|
| _pending_ | | | |

Document in `holdout_validation/validation_meta.json`.

## Reproducibility

- [x] LwF teacher = frozen `test_model_sugar_one` weights  
- [x] LR grid anchored to base model `tuning_meta.json`  
- [x] Fixed personal test window across day sweeps  
- [x] Holdouts use frozen Livia recipe (no per-user hyperparam retuning)  
- [ ] Full GPU experiment runs completed  

## Out of scope

- Personal vs general data mix  
- GluMind HR/steps personalization  
- Architecture changes to SugarOne  
