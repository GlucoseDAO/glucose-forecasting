"""Shared helpers for personalization sweep runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from common.registry import load_run_meta
from personalization.constants import DEFAULT_LR_MULTIPLIERS


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
    from personalization.constants import (
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


def build_holdout_lr_comparison(
    rows: list[dict[str, Any]],
    *,
    livia_reference_lr: float,
    metric_key: str = "ft_test_mae",
) -> list[dict[str, Any]]:
    """Per-user optimal LR vs Livia reference; notes on divergence."""
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        uid = str(row.get("user_id", ""))
        if not uid:
            continue
        by_user.setdefault(uid, []).append(row)

    comparison: list[dict[str, Any]] = []
    for uid in sorted(by_user):
        user_rows = by_user[uid]
        best = pick_best_row(user_rows, metric_key=metric_key)
        if best is None:
            continue
        optimal_lr = float(best["lr"])
        ratio = optimal_lr / livia_reference_lr if livia_reference_lr > 0 else None
        if optimal_lr == livia_reference_lr:
            divergence = "same"
            note = f"Optimal LR matches Livia ({livia_reference_lr:g})."
        elif optimal_lr < livia_reference_lr:
            divergence = "lower"
            note = (
                f"Optimal LR {optimal_lr:g} is below Livia ({livia_reference_lr:g}); "
                f"ratio={ratio:g} — slower/ more conservative fine-tune preferred."
            )
        else:
            divergence = "higher"
            note = (
                f"Optimal LR {optimal_lr:g} is above Livia ({livia_reference_lr:g}); "
                f"ratio={ratio:g} — faster adaptation preferred."
            )

        grid_maes = {
            float(r["lr"]): float(r[metric_key])
            for r in user_rows
            if r.get("lr") is not None and r.get(metric_key) is not None
        }
        livia_mae_at_ref = grid_maes.get(livia_reference_lr)
        optimal_mae = float(best[metric_key])
        mae_delta_vs_livia_lr = (
            optimal_mae - livia_mae_at_ref
            if livia_mae_at_ref is not None
            else None
        )

        comparison.append(
            {
                "user_id": uid,
                "subject": best.get("subject"),
                "livia_reference_lr": livia_reference_lr,
                "optimal_lr": optimal_lr,
                "optimal_ft_test_mae": optimal_mae,
                "livia_lr_ft_test_mae": livia_mae_at_ref,
                "mae_delta_optimal_minus_livia_lr": mae_delta_vs_livia_lr,
                "lr_ratio_vs_livia": ratio,
                "divergence": divergence,
                "note": note,
                "lr_grid_maes": grid_maes,
                "run_dir": best.get("run_dir"),
            }
        )
    return comparison


def holdout_combo_out_dir(out_dir: Path, subject: str, lr: float) -> Path:
    return out_dir / subject / f"lr{lr:g}"


def holdout_run_dir(out_dir: Path, subject: str, lr: float) -> Path:
    label = f"lr{lr:g}"
    return holdout_combo_out_dir(out_dir, subject, lr) / f"{subject}_{label}"


def personalization_run_complete(run_dir: Path) -> bool:
    metrics_path = run_dir / "personalization_metrics.json"
    if not metrics_path.is_file():
        return False
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    return results.get("finetuned_test") is not None


def holdout_run_complete(run_dir: Path) -> bool:
    return personalization_run_complete(run_dir)


def data_size_run_dir(out_dir: Path, subject: str, day_label: str) -> Path:
    return out_dir / f"days_{day_label}" / f"{subject}_days_{day_label}"


def holdout_row_from_metrics(
    run_dir: Path,
    *,
    user_id: str,
    subject: str,
    lwf_lambda: float,
    weight_decay: float,
    patience: int,
    epochs: int,
) -> dict[str, Any] | None:
    metrics_path = run_dir / "personalization_metrics.json"
    if not metrics_path.is_file():
        return None
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = results.get("config", {})
    if results.get("finetuned_test") is None:
        return None
    return {
        "user_id": user_id,
        "subject": subject,
        "lwf_lambda": lwf_lambda,
        "lr": float(cfg.get("lr", 0)),
        "weight_decay": weight_decay,
        "patience": patience,
        "epochs": epochs,
        "personal_days": "all",
        "run_dir": str(run_dir),
        "status": "ok",
        **flatten_metrics("zs_test", results.get("zero_shot_test")),
        **flatten_metrics("ft_test", results.get("finetuned_test")),
        **flatten_metrics("ft_val", results.get("finetuned_val")),
    }


def data_size_row_from_metrics(
    run_dir: Path,
    *,
    subject: str,
    day_label: str,
    lwf_lambda: float,
    lr: float,
    weight_decay: float,
    patience: int,
) -> dict[str, Any] | None:
    metrics_path = run_dir / "personalization_metrics.json"
    if not metrics_path.is_file():
        return None
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    if results.get("finetuned_test") is None:
        return None
    return {
        "subject": subject,
        "personal_days": day_label,
        "lwf_lambda": lwf_lambda,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "run_dir": str(run_dir),
        "status": "ok",
        **flatten_metrics("zs_test", results.get("zero_shot_test")),
        **flatten_metrics("ft_test", results.get("finetuned_test")),
        **flatten_metrics("ft_val", results.get("finetuned_val")),
    }
