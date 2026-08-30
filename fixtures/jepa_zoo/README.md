# JEPA model zoo

Every encoder, forecaster, and metrics file behind the JEPA results, copied out of the
gitignored `runs/` and `data/output/` trees so the numbers stay auditable after those are
cleaned. `MANIFEST.csv` is the machine-readable index: source run dir, weight file,
SHA-256 prefix, architecture, and headline test metrics for each artifact.

## Layout

```
encoders/<name>/      encoder_best.pt, config.json, best_info.json, pretrain_metrics.csv
forecasters/<name>/   last_model.pt [+ best_model.pt], config.json, tuning_meta.json,
                      scalers.json, {test,val}_metrics_{overall,by_study_group}.csv
metrics/encoder_pca/        probe_metrics.csv, probe_config.json, pca_w{288,864,2016}.png
metrics/personalization/    mae_by_day_budget.{csv,tex}
metrics/patient_weighted/   comparison_overall.csv
```

Optimizer state (`last_checkpoint.pt`) and per-epoch plot dirs were deliberately left out.

## Scoring an artifact

```bash
uv run glucose evaluate --run-dir fixtures/jepa_zoo/forecasters/sugar_jepa-288 \
  --model-type auto --data data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv --no-plot
```

All 11 checkpoints load under `strict=True` with `--model-type auto`.

## Edits made to the copies

Two stale fields were corrected so the archived runs load at all. The original values are
kept alongside under `*_original` keys in the same `tuning_meta.json`.

- `model_type: "sugar_jepa"` -> `"sugar_jepa2"` on the four `sugar_jepa-*` forecasters.
  These are `SugarJepaModel2` checkpoints written before the family was named; their own
  `scalers.json` already said `sugar_jepa2`, so `--model-type auto` aborted on the
  mismatch. Not applied to `sugar_jepa-cgm`, which really is the v1 vendored family.
- `jepa_weights_dir: "scripts/sugar_jepa/pretrained/cgm_jepa"` ->
  `"src/sugar_jepa/pretrained/cgm_jepa"` on `sugar_jepa-cgm`, a pre-`scripts/`->`src/`
  restructure path that no longer resolves.

`sugar_jepa-864/scalers.json` was taken from run `..._20260826_005309`; the run that
produced the metrics (`..._153705`) is a `resume_from` continuation of it and did not
rewrite its own scalers.

## Caveats that affect how these numbers should be read

1. **The published test metrics were computed in bf16 autocast on CUDA** (`precision:
   bf16` in every `config.json`). Re-scoring the same weights through `glucose evaluate`
   in fp32 moves MAE by ~0.3-0.5%: `sugar_jepa-288` reports 11.3685 but re-scores to
   11.3175 (`last_model.pt`) / 11.3991 (`best_model.pt`). Differences smaller than that
   are not resolvable -- notably `sugar_jepa-288` (11.3685) vs `sugar_jepa-cgm` (11.4101),
   a 0.36% gap.
2. **`test_metrics_*.csv` describes the final-epoch model, not the best-validation one.**
   The trainer scores the live model returned by `train_loop` and never reloads
   `best_model.pt`. For `sugar_jepa-288` best-val was epoch 4 and training ran to 7, so
   `last_model.pt` is the artifact the published row belongs to. Both are shipped where
   both exist.
3. **Each forecaster is scored on its own window set.** Lookback determines how many
   windows survive, so the per-run `test_metrics_overall.csv` files are not directly
   comparable: 1,667,437 windows at lookback 128 against 967,521 at 864.
   `metrics/patient_weighted/comparison_overall.csv` is the like-for-like version --
   all models on the 967,521 windows / 336 patients they share.
4. **The runs are not hyperparameter-matched.** `epochs_run` in `MANIFEST.csv` is 30 for
   the 128/128-64/288 forecasters, 50 for 864, and 200 for the SugarOne baseline;
   `freeze_jepa` is true only for 864; the baseline also uses `input_steps: 80` (not 128)
   and `lr: 1e-3` (not 4e-4).

## Not archived here

- **SugarJepa on `jepa-2016`** -- trained on another machine. Its encoder is present
  (`encoders/jepa-2016`), but no forecaster run dir, weights, or per-study-group CSV
  exists locally.
- **Patient-identification probing** -- no code or run artifacts for it exist on any
  branch of this repo; only the reported accuracies.
- **`jepa-864` / `jepa-2016` personalization sweeps** -- only the aggregated
  `metrics/personalization/mae_by_day_budget.csv` rows survive locally; the per-subject
  run dirs under `data/output/runs/personalization/` cover `sugarone`, `jepa128-64`,
  `jepa128`, and `jepa288` only.
- **PCA probe for `jepa-128` / `jepa-128-64`** -- `metrics/encoder_pca/probe_metrics.csv`
  covers w288, w864, and w2016 only.
