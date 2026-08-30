#!/usr/bin/env python3
"""Expand evaluate ``models[]`` / ``--run-dir`` entries into concrete run folders.

A path may be either:
- a leaf run directory (checkpoint / NF bundle / precomputed metrics), or
- a container of such runs (e.g. ``data/output/runs/nf_holdout``).

Containers are expanded to the best run per model family, ranked by MAE
(prefer ``val_metrics_overall.csv``, else ``test_metrics_overall.csv``) —
same ranking signal used via registry ``validation_metric``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from common.evaluation.config import ModelEvalSpec
from common.evaluation.detect import detect_run_dir, is_leaf_run_dir
from common.evaluation.readers import read_selection_mae
from common.evaluation.types import RunDirKind
from common.paths import resolve_project_path

# NF holdout dirs look like ``NHITS_20260811T160526Z``.
_NF_RUN_NAME = re.compile(r"^(?P<model>.+)_(?P<ts>\d{8}T\d{6}Z)$")

_SKIP_DIR_NAMES = frozenset(
    {
        "checkpoints",
        "neuralforecast",
        "summaries",
        "evaluations",
        "lightning_logs",
        "__pycache__",
    }
)


@dataclass(frozen=True, slots=True)
class RankedRun:
    """One candidate leaf run with its ranking MAE and model family key."""

    run_dir: Path
    model_key: str
    mae: float


def infer_model_key(run_dir: Path) -> str:
    """Stable family name used to group competing runs of the same model."""
    match = _NF_RUN_NAME.match(run_dir.name)
    if match:
        return match.group("model")

    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            models = payload.get("models")
            if isinstance(models, str) and models.strip() and "," not in models:
                text = models.strip()
                if text.lower() not in {"auto", "baseline", "recurrent"}:
                    return text

    for meta_name in ("tuning_meta.json", "config.json"):
        meta_path = run_dir / meta_name
        if not meta_path.is_file():
            continue
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for key in ("model_type", "model", "architecture"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return run_dir.name


def iter_leaf_run_dirs(root: Path) -> list[Path]:
    """Depth-first leaf runs under ``root`` (or ``[root]`` when root itself is a leaf)."""
    if is_leaf_run_dir(root):
        return [root]

    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if path.name in _SKIP_DIR_NAMES:
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        if is_leaf_run_dir(path):
            found.append(path)
    return found


def select_best_runs_by_mae(root: Path) -> list[RankedRun]:
    """Pick the lowest-MAE leaf run for each model family under ``root``."""
    best: dict[str, RankedRun] = {}
    for run_dir in iter_leaf_run_dirs(root):
        mae = read_selection_mae(run_dir)
        if mae is None:
            continue
        key = infer_model_key(run_dir)
        candidate = RankedRun(run_dir=run_dir, model_key=key, mae=mae)
        current = best.get(key)
        if current is None or candidate.mae < current.mae:
            best[key] = candidate
        elif candidate.mae == current.mae and str(candidate.run_dir) < str(current.run_dir):
            best[key] = candidate
    return [best[key] for key in sorted(best)]


def expand_model_specs(
    specs: list[ModelEvalSpec],
    *,
    project_root: Path | None = None,
) -> list[ModelEvalSpec]:
    """Expand container paths into best-per-model leaf specs; leave leaves as-is."""
    root = project_root or Path.cwd()
    expanded: list[ModelEvalSpec] = []
    for spec in specs:
        resolved = resolve_project_path(spec.run_dir, root)
        if not resolved.is_dir():
            raise FileNotFoundError(f"Run path does not exist: {resolved}")

        if is_leaf_run_dir(resolved):
            expanded.append(
                ModelEvalSpec(
                    run_dir=resolved,
                    label=spec.label,
                    model_type=spec.model_type,
                    zero_cov=spec.zero_cov,
                    include_cov=spec.include_cov,
                    exclude_cov=spec.exclude_cov,
                    batch_size=spec.batch_size,
                )
            )
            continue

        ranked = select_best_runs_by_mae(resolved)
        if not ranked:
            raise ValueError(
                f"No scored run directories found under {resolved}; "
                "expected leaf runs with val/test_metrics_overall.csv"
            )
        for item in ranked:
            label = item.model_key if spec.label is None else f"{spec.label}/{item.model_key}"
            # NF / precomputed containers keep auto; pytorch leaves keep the parent type.
            model_type = spec.model_type
            kind = detect_run_dir(item.run_dir, root)
            if kind == RunDirKind.NEURALFORECAST:
                model_type = "auto"
            expanded.append(
                ModelEvalSpec(
                    run_dir=item.run_dir,
                    label=label,
                    model_type=model_type,
                    zero_cov=spec.zero_cov,
                    include_cov=spec.include_cov,
                    exclude_cov=spec.exclude_cov,
                    batch_size=spec.batch_size,
                )
            )
    return expanded
