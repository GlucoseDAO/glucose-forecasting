"""Shared helpers for personalization sweep runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from scripts.common.registry import load_run_meta
from scripts.personalization.constants import DEFAULT_LR_MULTIPLIERS


def write_summary(rows: list[dict[str, Any]], out_dir: Path, name: str = "summary") -> Path:
    """Write sweep rows as CSV + JSON; return CSV path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"
    if not rows:
        pl.DataFrame([]).write_csv(csv_path)
    else:
        pl.DataFrame(rows).write_csv(csv_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return csv_path


def flatten_metrics(prefix: str, metrics: dict[str, Any] | None) -> dict[str, float | None]:
    if not metrics:
        return {f"{prefix}_mae": None, f"{prefix}_rmse": None, f"{prefix}_mard": None}
    return {
        f"{prefix}_mae": metrics.get("mae"),
        f"{prefix}_rmse": metrics.get("rmse"),
        f"{prefix}_mard": metrics.get("mard"),
    }


def pick_best_row(rows: list[dict[str, Any]], metric_key: str = "ft_test_mae") -> dict[str, Any] | None:
    """Return row with lowest non-null metric."""
    best: dict[str, Any] | None = None
    best_val = float("inf")
    for row in rows:
        val = row.get(metric_key)
        if val is None:
            continue
        fval = float(val)
        if fval < best_val:
            best_val = fval
            best = row
    return best


def write_best_recipe(path: Path, recipe: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)


def load_best_recipe(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict recipe in {path}")
    return data


def load_base_training_meta(base_run_dir: Path) -> dict[str, Any]:
    """Read ``tuning_meta.json`` / ``config.json`` from the global checkpoint."""
    return load_run_meta(Path(base_run_dir))


def lr_grid_from_base(
    base_run_dir: Path,
    multipliers: tuple[float, ...] = DEFAULT_LR_MULTIPLIERS,
) -> list[float]:
    """Build fine-tune LR grid as multipliers of the base model training LR."""
    meta = load_base_training_meta(base_run_dir)
    base_lr = float(meta.get("lr", 4e-4))
    return [base_lr * float(m) for m in multipliers]


def weight_decay_grid(
    multipliers: tuple[float, ...] | None = None,
    *,
    base_weight_decay: float | None = None,
) -> list[float]:
    """Build weight_decay grid as multipliers of the default/base value (3e-5)."""
    from scripts.personalization.constants import (
        DEFAULT_WEIGHT_DECAY,
        DEFAULT_WEIGHT_DECAY_MULTIPLIERS,
    )

    mults = multipliers if multipliers is not None else DEFAULT_WEIGHT_DECAY_MULTIPLIERS
    base = float(base_weight_decay if base_weight_decay is not None else DEFAULT_WEIGHT_DECAY)
    return [base * float(m) for m in mults]


def default_patience_from_base(base_run_dir: Path) -> int:
    meta = load_base_training_meta(base_run_dir)
    return int(meta.get("patience", 10))


def estimate_plateau_day(
    rows: list[dict[str, Any]],
    *,
    metric_key: str = "ft_test_mae",
    min_improvement: float = 0.05,
) -> dict[str, Any]:
    """Estimate plateau day from a data-size curve (sorted by personal_days).

    Returns dict with ``plateau_day``, ``optimal_day`` (best MAE), and per-step deltas.
    """
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get(metric_key) is not None]
    if not ok_rows:
        return {"plateau_day": None, "optimal_day": None, "steps": []}

    def _day_sort_key(row: dict[str, Any]) -> float:
        d = row.get("personal_days", "all")
        if d == "all":
            return float("inf")
        return float(d)

    ordered = sorted(ok_rows, key=_day_sort_key)
    steps: list[dict[str, Any]] = []
    prev_mae: float | None = None
    plateau_day: str | int | None = None
    optimal_day: str | int | None = None
    best_mae = float("inf")

    for row in ordered:
        mae = float(row[metric_key])
        day = row.get("personal_days", "all")
        delta = None if prev_mae is None else mae - prev_mae
        steps.append({"personal_days": day, "mae": mae, "delta_mae": delta})
        if mae < best_mae:
            best_mae = mae
            optimal_day = day
        if (
            plateau_day is None
            and delta is not None
            and abs(delta) < min_improvement
        ):
            plateau_day = day
        prev_mae = mae

    if plateau_day is None and len(ordered) >= 2:
        plateau_day = ordered[-1].get("personal_days")

    return {
        "plateau_day": plateau_day,
        "optimal_day": optimal_day,
        "best_mae": best_mae,
        "steps": steps,
    }
