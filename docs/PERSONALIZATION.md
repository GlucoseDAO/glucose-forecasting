# Personalization

Fine-tune the production **SugarOne** checkpoint on one person's CGM and pump data (glucose, basal, bolus, carbohydrates). This is the insulin + carbs personalization path — not GluMind HR/steps.

Package: `src/personalization/`. Console commands: `personal-*` (see [CLI reference](CLI_REFERENCE.md)). Technical results: [Personalization report](PERSONALIZATION_REPORT.md), [NeuralForecast](PERSONALIZATION_NF_REPORT.md), [SugarJEPA](PERSONALIZATION_JEPA_REPORT.md).

## Locked recipe

| Setting | Value | Why |
|---------|--------|-----|
| Base model | `fixtures/checkpoints/sugar_one_1.0/` | Production SugarOne global checkpoint |
| Method | Plain fine-tune (`lwf_lambda=0`) | ~10× faster than LwF; similar MAE |
| Train window stride | **6** (30 min at 5-min sampling) | ~6× fewer train windows; val/test stay stride 1 |
| Scalers | Base-run `scalers.json` | Short histories must stay in the pretrained input scale |
| Weight decay | `3e-5` | No test-MAE effect in sweeps |
| Patience | 3 | Early stopping on personal val |

Chronological split (every subject): last **25%** of the timeline is test; **15%** of the remainder is val; the rest is train. A day budget only shortens **train**.

## Model coverage

The pipeline is family-agnostic. Every CLI resolves the architecture, the dataset
window length, and the optimizer's param groups from `src/personalization/registry.py`,
keyed on the `model_type` in the base run's `tuning_meta.json` (or fingerprinted from
the checkpoint when that key is missing). No flag selects the model — point
`--base-run-dir` at the run you want to personalize.

| Family | Dataset window | Optimizer | Extra flags |
|--------|----------------|-----------|-------------|
| **SugarOne** | `input_steps` | one AdamW group | — |
| **SugarJepa2** | `max(input_steps, jepa_window)` | separate LR group for the JEPA encoder | `--freeze-jepa`, `--jepa-lr` |

`sugar_jepa` (the vendored-CGM-JEPA variant) is deliberately **not** registered: its
dataset yields a second `glucose_jepa` tensor, which the SugarOne fine-tune loop does
not pass on. Only families with plain `(x, y)` batches belong here.

Two consequences when personalizing SugarJepa2:

- **Longer windows shrink the usable population.** At `jepa_window=288` a person needs
  ≥300 contiguous rows to yield one training window, against 140 for SugarOne at
  `input_steps=128`; at 864 it is 876. Short subjects that fine-tune fine under SugarOne
  can drop out entirely, and the low end of a days sweep (1, 3 days) is where it shows
  first. Worth controlling for when comparing across JEPA windows.
- **`--freeze-jepa` is a second anti-forgetting knob, orthogonal to LwF.** LwF pulls the
  whole model toward the global teacher; freezing pins the glucose representation exactly
  and lets only the backbone adapt. Defaults to the base run's setting. Unlike in
  training, freezing is always safe here — the encoder comes from the base checkpoint,
  never from a random init.

By default the encoder's LR tracks the base run's `jepa_lr / lr` ratio, so an LR sweep
moves both param groups together rather than silently changing their balance; `--jepa-lr`
overrides with an absolute value. In the sweep CLIs both settings ride in the recipe /
params dict alongside `lwf_lambda` and `lr`.

## Demo: Livia + SugarOne

Tracked fixtures — no gitignored `data/input/` export required:

```bash
uv run personal-prepare livia
uv run personal-tune --dry-run
uv run personal-tune
```

Defaults:

- Input: `fixtures/livia_data/livia_sugar_one_ready.csv`
- Prepared CSV: `data/input/personalization/prepared/livia_chronological.csv`
- Checkpoint: `fixtures/checkpoints/sugar_one_1.0`
- Tuner config: `src/personalization/tune.toml` (LR grid `1e-4`, `2e-4`, `4e-4`)

Single run after prepare:

```bash
uv run personal-finetune --device cuda
```

`personal-prepare livia` drops any existing `Recommended Split` and assigns the chronological split above.

## Commands

| Command | Role |
|---------|------|
| `personal-prepare` | Chronological CSVs (`livia`, `holdouts`, `joined2-test`) |
| `personal-finetune` | One fine-tune (defaults: Livia + `sugar_one_1.0`, stride 6) |
| `personal-tune` | TOML grid + `leaderboard.csv` |
| `personal-sweep-days` | Train-days vs test MAE |
| `personal-plot` | Data-size charts |
| `personal-sweep-lr` | Holdout LR transfer vs Livia |
| `personal-sweep-lwf` | Independent LwF vs plain fine-tune |
| `personal-study` | Cohort data-size curves + personalization report |

```bash
uv run personal-prepare --help
uv run personal-finetune --help
uv run personal-tune --help
```

## Optional stride comparison

Sparse stride 6 is the default. To re-check dense vs sparse:

```bash
uv run personal-tune -c src/personalization/tune_window_stride.toml --dry-run
```

## Appendix — study / report CLIs

Research sweeps used for Milestone 8 (holdouts, joined2 study groups, LwF). Re-run report from on-disk results:

```bash
uv run personal-study --report-only
```

Results: [Personalization report](PERSONALIZATION_REPORT.md). Out of scope here: personal vs general data mix, GluMind personalization, SugarOne architecture changes.

NeuralForecast counterpart (continue-fit, no LwF/LR search): `uv run personal-nf-study` — [PERSONALIZATION_NF_REPORT.md](PERSONALIZATION_NF_REPORT.md). SugarJEPA day-budget write-up: [PERSONALIZATION_JEPA_REPORT.md](PERSONALIZATION_JEPA_REPORT.md).
