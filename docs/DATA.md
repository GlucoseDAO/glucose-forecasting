# Data layout

This repo **does not produce** training CSVs. CGM / pump tables come from the companion
preprocessing pipeline: [GlucoseDAO/glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing).

## Recommended local layout

```text
data/
  input/          # place CSVs here for training and eval (preferred)
  output/
    runs/         # default root for all new training / tuning / personalization runs
  processed/      # optional intermediate artifacts
  cache/          # optional caches
  personalization/  # prepared personal CSVs (existing workflows)
  loop_and_ai_ready/  # legacy / existing joined Loop + AI-READI tables
  actual/             # legacy / existing AI-READI-style tables
```

Bundled demos stay under `test_data/` (not under `data/`).

## Run outputs

**Default output root:** `data/output/runs/` (constant `scripts.common.paths.DEFAULT_RUNS_ROOT`).

Examples:

- `data/output/runs/glumind/`
- `data/output/runs/sugar_one/`
- `data/output/runs/personalization/`

Do **not** use top-level `runs/` as a destination for new work. Curated historical
artifacts may still live under `marked_runs/` (read-only reference).

## CSV path resolution

`scripts.common.registry.resolve_csv_path` remaps checkpoint metadata paths that
point at another machine. Preferred lookup for a basename is `data/input/<name>`,
then known legacy folders under `data/`. Prefer putting shared CSVs in `data/input/`.

For demos, pass explicit paths under `test_data/` (see root `README.md`).
