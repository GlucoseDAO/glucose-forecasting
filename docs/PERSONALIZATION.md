# Personalization

Fine-tune the production **SugarOne** checkpoint on one person's CGM and pump data (glucose, basal, bolus, carbohydrates). This is the insulin + carbs personalization path — not GluMind HR/steps.

Package: `src/personalization/`. Console commands: `personal-*` (see [CLI reference](CLI_REFERENCE.md)). Technical results: [Milestone 8 report](MILESTONE_8_PERSONALIZATION_REPORT.md).

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
| `personal-study` | Cohort data-size curves + Milestone 8 report |

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

Results: [Milestone 8 report](MILESTONE_8_PERSONALIZATION_REPORT.md). Out of scope here: personal vs general data mix, GluMind personalization, SugarOne architecture changes.
