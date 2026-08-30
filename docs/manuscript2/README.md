# Manuscript prototyping

This folder is a scratch workspace for drafting the paper: outlines, LaTeX, figures, and notes we iterate on together. It is not a finished submission and is not the project's durable documentation.

Use it to try section structure, wording, and citations before anything lands in a venue-specific Overleaf project or a camera-ready PDF. Durable writeups and results stay in `docs/` (comparison reports, CLI, data layout). Training artifacts stay under `data/output/`.

`template.tex` is the European AI Summer Research (Anonymous Conference 2026) starter: anonymous 8-page A4 main text, compile with `pdflatex` (or Overleaf) and keep `easrp2026.sty` next to the `.tex` file.

Build Markdown (for editing context) and PDF:

```bash
uv run manuscript template      # template.tex → template.md + template.pdf
uv run manuscript manuscript    # manuscript.tex → manuscript.md + manuscript.pdf
```


# Manuscript writing process

The idea is that we use ClaudeCode and Cursor to write the manuscript. To let it have the context we make it in this ( https://anonymous.4open.science/r/glucose-forecasting ) repository and we also add other relevant repositories to the cursor/vscode workspace. 
In particular:
* https://anonymous.4open.science/r/glucose-data-processing for data processing
* https://anonymous.4open.science/r/cgm-format for processing individual sensor information (optional)
* https://anonymous.4open.science/r/sugar-sugar to tell about the glucose prediction game (optional)

Some of the state of the art papers (note: we can probably have a separate list of sources for related work section and download it to data/.cache folder):
* gluformer https://github.com/mrsergazinov/gluformer
* another gluformer https://github.com/Guylu/GluFormer
* glumind (we based partially on them) https://arxiv.org/abs/2509.18457
* benchmarks:
    ** https://github.com/IrinaStatsLab/GlucoBench
    ** https://pmc.ncbi.nlm.nih.gov/articles/PMC13321326/
    ** https://github.com/JHU-CDHAI/EventGlucoseBench
* glucose JEPA:
    https://arxiv.org/abs/2605.00933

we use data/cache/for_manuscript to download required files for writing. And those files are gitignored