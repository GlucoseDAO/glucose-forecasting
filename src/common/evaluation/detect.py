#!/usr/bin/env python3
"""Detect what kind of artifacts a run directory contains."""
from __future__ import annotations

from pathlib import Path

from common.evaluation.types import RunDirKind
from common.paths import resolve_project_path


def is_leaf_run_dir(path: Path) -> bool:
    """Return True when ``path`` itself is an evaluable run folder (not a container)."""
    if not path.is_dir():
        return False
    has_metrics = any(path.glob("*_metrics_overall.csv"))
    has_weights = (path / "best_model.pt").is_file() or (path / "last_model.pt").is_file()
    has_meta = (path / "tuning_meta.json").is_file() or (path / "config.json").is_file()
    has_nf = (path / "neuralforecast").is_dir() and (path / "run_config.json").is_file()
    return has_metrics or (has_weights and has_meta) or has_nf


def detect_run_dir(run_dir: Path | str, project_root: Path | None = None) -> RunDirKind:
    """Classify a run directory as custom PyTorch, NF, precomputed metrics, or unknown."""
    root = Path.cwd() if project_root is None else project_root
    path = resolve_project_path(run_dir, root)
    if not path.is_dir():
        return RunDirKind.UNKNOWN

    has_nf = (path / "neuralforecast").is_dir() and (path / "run_config.json").is_file()
    if has_nf:
        return RunDirKind.NEURALFORECAST

    has_weights = (path / "best_model.pt").is_file() or (path / "last_model.pt").is_file()
    has_meta = (path / "tuning_meta.json").is_file() or (path / "config.json").is_file()
    if has_weights and has_meta:
        return RunDirKind.CUSTOM_PYTORCH

    metric_hits = list(path.glob("*_metrics_overall.csv"))
    if metric_hits:
        return RunDirKind.PRECOMPUTED
    return RunDirKind.UNKNOWN
