#!/usr/bin/env python3
"""Shared run-registry / checkpoint-resolution helpers.

Extracted from duplicated verbatim code in ``evaluate_glumind.py`` and
``evaluate_model.py``: locating the best run from an analysis registry CSV,
loading its metadata, and resolving the checkpoint / CSV paths to evaluate.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def find_best_run_dir(registry_dir: Path, project_root: Path) -> tuple[Path, dict]:
    """Parse _analysis_registry.csv and return (step_dir, row) for lowest val_mae."""
    registry_csv = registry_dir / "_analysis_registry.csv"
    if not registry_csv.exists():
        typer.echo(f"Error: _analysis_registry.csv not found in {registry_dir}", err=True)
        raise typer.Exit(1)

    best_row: dict | None = None
    best_mae: float = float("inf")

    with open(registry_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_mae_str = row.get("val_mae", "").strip()
            if not val_mae_str:
                continue
            val_mae = float(val_mae_str)
            if val_mae < best_mae:
                best_mae = val_mae
                best_row = row

    if best_row is None:
        typer.echo("Error: No valid rows with val_mae found in the registry.", err=True)
        raise typer.Exit(1)

    run_dir_rel = Path(best_row["run_dir"])
    # run_dir in the registry is relative to the project root
    run_dir_abs = project_root / run_dir_rel

    # For continual runs there is a final_step subdirectory
    final_step = best_row.get("final_step", "").strip()
    if final_step:
        step_dir = run_dir_abs / final_step
    else:
        step_dir = run_dir_abs

    typer.echo(
        f"Best run (val_mae={best_mae:.6f}): {run_dir_rel}"
        + (f"  step={final_step}" if final_step else "")
    )
    return step_dir, best_row


def load_run_meta(run_dir: Path) -> dict:
    for name in ("tuning_meta.json", "config.json"):
        p = run_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    typer.echo(f"Error: No metadata file (tuning_meta.json / config.json) in {run_dir}", err=True)
    raise typer.Exit(1)


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)
        return checkpoint

    for name in ("best_model.pt", "last_model.pt"):
        p = run_dir / name
        if p.exists():
            return p

    typer.echo(f"Error: No model weights (best_model.pt / last_model.pt) found in {run_dir}", err=True)
    raise typer.Exit(1)


def _csv_basename(csv_value: str | Path) -> str:
    """Basename that works for POSIX and Windows-style paths in metadata."""
    text = str(csv_value).replace("\\", "/").rstrip("/")
    return Path(text).name


def resolve_csv_path(csv_value: str | Path, project_root: Path) -> Path:
    """Resolve a training/eval CSV path, including cross-machine absolute paths.

    Tries, in order:
    1. Absolute path as given (when absolute)
    2. ``project_root / csv_value``
    3. Basename under preferred local folders (``data/input/``, then legacy data dirs)
    4. Suffix starting at a ``data/`` component (other-machine absolutes)
    5. Unique basename match under ``project_root / data``
    6. Relative CWD path (last resort)
    """
    csv_path = Path(csv_value)
    candidates: list[Path] = []
    if csv_path.is_absolute():
        candidates.append(csv_path)
    candidates.append(project_root / csv_path)

    name = _csv_basename(csv_value)
    if name:
        candidates.extend(
            [
                project_root / "data" / "input" / name,
                project_root / "data" / "loop_and_ai_ready" / name,
                project_root / "data" / "actual" / "with_complex_steps_processing" / name,
            ]
        )

    parts = Path(str(csv_value).replace("\\", "/")).parts
    if "data" in parts:
        idx = list(parts).index("data")
        candidates.append(project_root.joinpath(*parts[idx:]))

    if name:
        data_root = project_root / "data"
        if data_root.is_dir():
            hits = list(data_root.rglob(name))
            if len(hits) == 1:
                candidates.append(hits[0])
            elif len(hits) > 1:
                # Prefer an earlier explicit candidate if it exists; otherwise fail clearly.
                preferred = [
                    project_root / "data" / "input" / name,
                    project_root / "data" / "loop_and_ai_ready" / name,
                    project_root / "data" / "actual" / "with_complex_steps_processing" / name,
                ]
                if not any(p.is_file() for p in preferred):
                    typer.echo(
                        f"Error: Ambiguous CSV basename {name!r} under data/: "
                        + ", ".join(str(h.relative_to(project_root)) for h in hits[:5]),
                        err=True,
                    )
                    raise typer.Exit(1)

    if not csv_path.is_absolute():
        candidates.append(csv_path)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            if candidate != csv_path and not csv_path.exists():
                typer.echo(f"Note: using {candidate} (resolved from missing {csv_value})")
            return candidate

    typer.echo(f"Error: CSV not found: {csv_value}", err=True)
    typer.echo(
        "Hint: pass an explicit --train-csv / --test-csv / --csv under data/input/ "
        "(see docs/DATA.md). Demo smoke tests should use test_data/ with --train-csv.",
        err=True,
    )
    raise typer.Exit(1)


def try_resolve_csv_path(csv_value: str | Path, project_root: Path) -> Path | None:
    """Like ``resolve_csv_path`` but returns None instead of exiting."""
    try:
        return resolve_csv_path(csv_value, project_root)
    except typer.Exit:
        return None
