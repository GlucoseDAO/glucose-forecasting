"""Build Markdown and PDF from manuscript LaTeX sources."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pypandoc

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"
_MANUSCRIPT_DIR_RE = re.compile(r"^manuscript(\d+)?$")
LOG_TAIL_LINES = 40
TECTONIC_MISSING = "tectonic not found. Install dev dependencies with: uv sync"


def latest_manuscript_dir(docs_dir: Path = DOCS_DIR) -> Path:
    """Return the highest-numbered ``manuscript`` / ``manuscriptN`` directory.

    Unnumbered ``manuscript`` is version 1. ``manuscript2`` beats it; ``manuscript10``
    beats ``manuscript2``.
    """
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"{docs_dir} not found")

    candidates: list[tuple[int, Path]] = []
    for child in docs_dir.iterdir():
        if not child.is_dir():
            continue
        version = _manuscript_dir_version(child.name)
        if version is not None:
            candidates.append((version, child))
    if not candidates:
        raise FileNotFoundError(f"no manuscript directory under {docs_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _manuscript_dir_version(name: str) -> int | None:
    match = _MANUSCRIPT_DIR_RE.fullmatch(name)
    if match is None:
        return None
    suffix = match.group(1)
    return int(suffix) if suffix is not None else 1


def manuscript_tex(docs_dir: Path = DOCS_DIR) -> Path:
    return latest_manuscript_dir(docs_dir) / "manuscript.tex"


def template_tex(docs_dir: Path = DOCS_DIR) -> Path:
    return latest_manuscript_dir(docs_dir) / "template.tex"


_LATEX_SYMBOLS: dict[str, str] = {
    r"\sim": "~",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\Delta": "Δ",
    r"\lambda": "λ",
    r"\alpha": "α",
    r"\beta": "β",
    r"\sigma": "σ",
    r"\times": "×",
    r"\cdot": "·",
    r"\pm": "±",
    r"\infty": "∞",
    r"\neq": "≠",
    r"\approx": "≈",
}

_MATH_PATTERN = re.compile(r"\$`([^`]*)`\$")


def _latex_math_to_text(match: re.Match[str]) -> str:
    """Convert a single pandoc GFM math span to plain text."""
    expr = match.group(1)
    for cmd, repl in _LATEX_SYMBOLS.items():
        expr = expr.replace(cmd, repl)
    expr = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"\\text\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"\\boldsymbol\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"(\d+)\s*\\times\s*10\^\{([^}]*)\}", r"\1×10^\2", expr)
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.replace("\\,", " ")
    return expr


def _postprocess_markdown(text: str) -> str:
    """Clean up pandoc GFM output: flatten math spans, strip comment tags."""
    text = _MATH_PATTERN.sub(_latex_math_to_text, text)
    text = text.replace("<!-- -->", "")
    return text


def latex_to_markdown(source: Path, output: Path | None = None) -> Path:
    """Convert a LaTeX file to Markdown next to the source (or to ``output``)."""
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")

    dest = output if output is not None else source.with_suffix(".md")
    dest.parent.mkdir(parents=True, exist_ok=True)

    bib = source.parent / "references.bib"
    extra = ["--wrap=none"]
    if bib.exists():
        extra += ["--citeproc", f"--bibliography={bib}"]

    pypandoc.convert_file(
        str(source),
        "gfm",
        outputfile=str(dest),
        extra_args=extra,
    )
    raw = dest.read_text(encoding="utf-8")
    dest.write_text(_postprocess_markdown(raw), encoding="utf-8")
    return dest


def latex_to_pdf(source: Path, output: Path | None = None) -> Path:
    """Compile a LaTeX file to PDF with the venv ``tectonic`` binary."""
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")

    dest = output if output is not None else source.with_suffix(".pdf")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tectonic = _resolve_tectonic()
    proc = _run(
        [
            str(tectonic),
            "--keep-logs",
            "--outdir",
            str(dest.parent.resolve()),
            source.name,
        ],
        source.parent,
    )
    if proc.returncode != 0:
        raise RuntimeError(_tectonic_error(source, dest.parent, proc))

    built = dest.parent / f"{source.stem}.pdf"
    if not built.exists():
        raise RuntimeError(f"tectonic did not produce {built}")
    if dest.resolve() != built.resolve():
        shutil.copy2(built, dest)
    return dest


def _resolve_tectonic() -> Path:
    venv_bin = Path(sys.executable).resolve().parent
    for name in ("tectonic", "tecto"):
        candidate = venv_bin / name
        if candidate.is_file():
            return candidate
    found = shutil.which("tectonic") or shutil.which("tecto")
    if found is None:
        raise FileNotFoundError(TECTONIC_MISSING)
    return Path(found)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _tectonic_error(
    source: Path,
    outdir: Path,
    proc: subprocess.CompletedProcess[str],
) -> str:
    log = outdir / f"{source.stem}.log"
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-LOG_TAIL_LINES:])
    else:
        tail = (proc.stderr or proc.stdout or "").strip()
    return f"tectonic failed for {source.name}\n{tail}"
