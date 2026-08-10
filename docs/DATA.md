# Data layout

This repo **does not produce** training CSVs. CGM / pump tables come from the companion
preprocessing pipeline: [GlucoseDAO/glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing).

## Recommended local layout

```text
data/
  input/                    # all training / eval CSVs
    actual/                 # AI-READI-style tables
    loop_and_ai_ready/      # joined Loop + AI-READI tables
    personalization/        # prepared personal / holdout CSVs
  output/
    runs/                   # default root for training / tuning / personalization runs
    marked_runs/            # curated historical runs + RUNS_ANALYSIS.md
  processed/                # optional intermediate artifacts
  cache/                    # optional caches
```

Bundled demos stay under `test_data/` (not under `data/`).

## Run outputs

**Default output root:** `data/output/runs/` (constant `common.paths.DEFAULT_RUNS_ROOT`).

Examples:

- `data/output/runs/glumind/`
- `data/output/runs/sugar_one/`
- `data/output/runs/personalization/`

Do **not** use top-level `runs/` or `marked_runs/` as destinations. Curated historical
artifacts live under `data/output/marked_runs/` (read-only reference).

## CSV path resolution

`common.registry.resolve_csv_path` remaps checkpoint metadata paths that
point at another machine or at pre-move locations. Preferred lookup for a basename is
`data/input/<name>`, then known folders under `data/input/` (`loop_and_ai_ready/`,
`actual/...`, `personalization/`). Legacy prefixes such as `data/loop_and_ai_ready/`,
`data/actual/`, `runs/`, and `marked_runs/` are rewritten to the layout above when resolving.

For demos, pass explicit paths under `test_data/` (see root `README.md`).
