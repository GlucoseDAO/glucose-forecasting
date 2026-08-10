#!/usr/bin/env python3
"""Shared filesystem path defaults for the project."""
from __future__ import annotations

from pathlib import Path

# Preferred roots after the data/input + data/output layout adoption.
DEFAULT_INPUT_ROOT: Path = Path("data") / "input"
DEFAULT_OUTPUT_ROOT: Path = Path("data") / "output"
# Single output root for training / tuning / personalization runs.
# Do not use top-level ``runs/`` as a default destination.
DEFAULT_RUNS_ROOT: Path = DEFAULT_OUTPUT_ROOT / "runs"
DEFAULT_MARKED_RUNS_ROOT: Path = DEFAULT_OUTPUT_ROOT / "marked_runs"

# Dataset family folders live under ``data/input/``.
DEFAULT_ACTUAL_DATA_ROOT: Path = DEFAULT_INPUT_ROOT / "actual"
DEFAULT_LOOP_AI_READY_ROOT: Path = DEFAULT_INPUT_ROOT / "loop_and_ai_ready"
DEFAULT_PERSONALIZATION_DATA_ROOT: Path = DEFAULT_INPUT_ROOT / "personalization"

# Legacy relative prefixes rewritten to the new layout (metadata, registries, CLIs).
_LEGACY_PREFIX_REWRITES: tuple[tuple[str, str], ...] = (
    ("data/actual/", "data/input/actual/"),
    ("data/loop_and_ai_ready/", "data/input/loop_and_ai_ready/"),
    ("data/personalization/", "data/input/personalization/"),
    ("marked_runs/", "data/output/marked_runs/"),
    ("runs/", "data/output/runs/"),
)


def normalize_relpath_text(path_value: str | Path) -> str:
    """Normalize separators and strip a leading ``./`` for prefix matching."""
    return str(path_value).replace("\\", "/").lstrip("./")


def rewrite_legacy_relpath(path_value: str | Path) -> Path:
    """Rewrite a known legacy project-relative path to the current layout.

    Paths that are already under the new layout (or unrelated) are returned unchanged.
    Only rewrites when the legacy prefix is present; does not invent paths.
    """
    text = normalize_relpath_text(path_value)
    # Already under the new runs root — do not rewrite the inner ``runs/`` segment.
    if text.startswith("data/output/runs/") or text.startswith("data/output/marked_runs/"):
        return Path(text)
    if text.startswith("data/input/"):
        return Path(text)

    for old, new in _LEGACY_PREFIX_REWRITES:
        if text.startswith(old):
            return Path(new + text[len(old) :])
    return Path(text)


def legacy_relpath_candidates(path_value: str | Path) -> list[Path]:
    """Return unique candidate relative paths: as-given, then legacy rewrite if different."""
    original = Path(normalize_relpath_text(path_value))
    rewritten = rewrite_legacy_relpath(path_value)
    if rewritten == original:
        return [original]
    return [original, rewritten]


def resolve_project_path(path_value: str | Path, project_root: Path | None = None) -> Path:
    """Resolve a user path, rewriting known legacy prefixes when the original is missing.

    Prefers an existing path. Falls back to the rewritten layout under ``project_root``
    (or CWD) so CLI flags like ``--run-dir marked_runs/...`` keep working after the move.
    """
    path = Path(path_value)
    if path.exists():
        return path

    bases: list[Path] = []
    if project_root is not None:
        bases.append(project_root)
    bases.append(Path.cwd())

    seen: set[Path] = set()
    for base in bases:
        for rel in legacy_relpath_candidates(path):
            candidate = rel if rel.is_absolute() else base / rel
            try:
                key = candidate.resolve()
            except OSError:
                key = candidate
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists():
                return candidate

    rewritten = rewrite_legacy_relpath(path)
    if rewritten != Path(normalize_relpath_text(path)):
        if project_root is not None and not rewritten.is_absolute():
            return project_root / rewritten
        return rewritten
    return path
