# CLAUDE.md

This file is kept in sync with `AGENTS.md`. See `AGENTS.md` for the canonical, detailed version. Key points reproduced below.

## Before training or inference (mandatory)

**Never start a new training, evaluation, or inference job until you have checked that none is already running.** Stacking GPU/CPU jobs has repeatedly hung or stalled this machine. Prefer one long-running ML job at a time.

## Temporary vs permanent artifacts

Put all intermediate reports, scratch notes, evaluation dumps, and other temporary information in `temp_docs/`. Put all temporary or one-off scripts in `temp_scripts/`. Do not add intermediate markdown under `docs/` or the repo root.

## Project overview

Training, tuning, and evaluation pipelines for blood-glucose forecasting from CGM data, on AI-READI-style and Loop-pump-style datasets:

- **GluMind** (`src/glumind/`) — glucose + heart rate + step count. Parallel cross-attention + multi-scale self-attention.
- **GluMind-Uni** (`src/glumind_uni/`) — glucose-only variant.
- **SugarOne** (`src/sugar_one/`) — glucose + basal rate + bolus insulin + carbohydrates (Loop pump data), 3-way cross-attention with learnable softmax mixing weights.
- **SugarJEPA** (`src/sugar_jepa/`) — SugarOne + frozen CGM-JEPA auxiliary stream.
- **NeuralForecast baselines** (`src/nf_baselines/`) — N-HiTS and other baselines.

Forecast horizon defaults to 12 steps = 60 minutes at 5-minute sampling frequency.

## Key conventions

- Product code lives under `src/` as direct packages. There is **no** `scripts/` tree.
- New models need a Typer CLI entry point.
- Always use `device=auto` — never hardcode `cpu` when GPU may be available.
- Training artifacts go under `data/output/runs/`.
- Tracked demo CSVs and reviewer checkpoints live under `fixtures/`.

## Commands

```bash
uv sync                          # install dependencies
uv run pytest -q                 # run tests
uv run glucose evaluate --help   # platform evaluation CLI
```

Fast smoke test:
```bash
uv run glucose evaluate --run-dir fixtures/checkpoints/glumind_1.0 --model-type glumind \
  --data fixtures/livia_data/livia_glumind_ready.csv --test-split "" --batch-size 4096 --no-plot
```

## Manuscript writing

The manuscript lives in `docs/manuscript/`. This is a scratch workspace for drafting the paper.

### Naming

- **Sugar I** = presentation/public name for the GluMind architecture (`src/glumind/`)
- **SugarOne** = insulin pump extension (`src/sugar_one/`)
- **SugarJEPA** = SugarOne + frozen CGM-JEPA auxiliary (`src/sugar_jepa/`)
- Code uses `glumind` / `sugar_one` / `sugar_jepa` internally

### Files

- `docs/manuscript/manuscript.tex` — LaTeX source (EASRP 2026 template, 8-page main text limit)
- `docs/manuscript/manuscript.md` — Human-readable Markdown version (auto-generated)
- `docs/manuscript/references.bib` — BibTeX references (`plainnat` style)
- `docs/manuscript/easrp2026.sty` — Style file placeholder (replace with official when available)

### Workflow

1. **Edit** `docs/manuscript/manuscript.tex`
2. **Compile** PDF:
   ```bash
   cd docs/manuscript && pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex
   ```
3. **Generate** Markdown:
   ```bash
   uv run manuscript docs/manuscript/manuscript.tex -o docs/manuscript/manuscript.md
   ```
4. After editing .tex, always regenerate both PDF and .md

### Fact-checking

- Verify all claims against actual run data in `docs/` reports and `data/output/`
- Use exact numbers from `*_metrics_overall.csv` / `*_metrics_by_study_group.csv` files
- Cross-reference `docs/presentation/PRESENTATION_NOTES.md` for fact-checked claims and known discrepancies
- The abstract PDF is at `docs/presentation/RoBioinfo2026_Abstract_Zaharia_et_al .pdf`
- Papers for context are downloaded to `data/cache/for_manuscript/` (gitignored)
- The 8-page limit applies to main text only; references and appendix are unlimited

### Related repositories (add to workspace for full manuscript context)

- [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing) — CGM preprocessing pipeline (9 device formats, 50+ datasets)
- [sugar-sugar](https://github.com/GlucoseDAO/sugar-sugar) — Human benchmarking web app (Sugar-Sugar study, ethics ref A 2026-0064)
- [cgm_format](https://github.com/GlucoseDAO/cgm_format) — Individual CGM sensor format parsing

For full architecture details, checkpoint format, CLI reference, data expectations, and more, see `AGENTS.md`.
