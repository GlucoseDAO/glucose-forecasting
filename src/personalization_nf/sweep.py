"""Data-size sweep: personal train days vs test MAE for one subject×model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from common.console import safe_echo
from personalization.plots import plot_data_size_curve
from personalization.splits import load_train_span_days
from personalization.sweep_utils import (
    flatten_metrics,
    should_skip_day_budget,
    write_summary,
)
from personalization_nf.adapt import adapt_run_complete, run_adapt
from personalization_nf.constants import DATA_SIZE_DAYS, METRICS_FILENAME
from personalization_nf.data import day_label
from personalization_nf.discover import NfHoldoutRun


ProgressCallback = Callable[[str, list[dict[str, Any]]], None]


def parse_days_grid(raw: str | None) -> list[int | None]:
    if raw is None:
        items = list(DATA_SIZE_DAYS)
    else:
        items = [part.strip() for part in raw.split(",") if part.strip()]
    out: list[int | None] = []
    for item in items:
        if str(item).lower() == "all":
            out.append(None)
        else:
            out.append(int(item))
    return out


def data_size_run_dir(out_dir: Path, subject: str, label: str) -> Path:
    return out_dir / f"days_{label}" / f"{subject}_days_{label}"


def row_from_results(
    *,
    subject: str,
    model_key: str,
    label: str,
    run_dir: Path,
    results: dict[str, Any],
) -> dict[str, Any]:
    cfg = results.get("config") if isinstance(results.get("config"), dict) else {}
    return {
        "subject": subject,
        "model_key": model_key,
        "personal_days": label,
        "run_dir": str(run_dir),
        "status": "ok",
        "train_span_days": cfg.get("train_span_days"),
        "used_train_days": cfg.get("used_train_days"),
        "max_steps": cfg.get("max_steps"),
        "val_size": cfg.get("val_size"),
        "patience": cfg.get("patience"),
        "protocol": cfg.get("protocol"),
        **flatten_metrics("zs_test", results.get("zero_shot_test")),
        **flatten_metrics("ft_test", results.get("finetuned_test")),
        **flatten_metrics("ft_val", results.get("finetuned_val")),
    }


def row_from_disk(
    run_dir: Path,
    *,
    subject: str,
    model_key: str,
    label: str,
) -> dict[str, Any] | None:
    path = run_dir / METRICS_FILENAME
    if not path.is_file():
        return None
    results = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(results, dict) or results.get("finetuned_test") is None:
        return None
    row = row_from_results(
        subject=subject,
        model_key=model_key,
        label=label,
        run_dir=run_dir,
        results=results,
    )
    return row


def run_data_size_sweep(
    *,
    holdout: NfHoldoutRun,
    personal_csv: Path,
    out_dir: Path,
    subject: str,
    days_grid: list[int | None],
    device: str,
    skip_completed: bool = True,
    dry_run: bool = False,
    plot: bool = True,
    on_progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Run or resume a subject×model day-budget sweep. Safe to re-invoke."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_span = load_train_span_days(personal_csv)
    if train_span is not None:
        safe_echo(f"Train span: {train_span:.1f} days ({personal_csv})")

    completed_rows: list[dict[str, Any]] = []
    pending: list[tuple[str, int | None]] = []
    for day_budget in days_grid:
        label = day_label(day_budget)
        if should_skip_day_budget(day_budget, train_span):
            safe_echo(
                f"Skipping days={label}: budget covers full train span "
                f"({train_span:.1f}d); using days=all instead"
            )
            continue
        run_dir = data_size_run_dir(out_dir, subject, label)
        if skip_completed and adapt_run_complete(run_dir):
            row = row_from_disk(
                run_dir, subject=subject, model_key=holdout.model_key, label=label
            )
            if row is not None:
                if train_span is not None and row.get("train_span_days") is None:
                    row["train_span_days"] = train_span
                    row["used_train_days"] = (
                        train_span if label == "all" else min(float(label), train_span)
                    )
                completed_rows.append(row)
            continue
        pending.append((label, day_budget))

    safe_echo(
        f"{holdout.model_key} {subject}: completed={len(completed_rows)} pending={len(pending)}"
    )
    if dry_run:
        for label, _day in pending:
            safe_echo(f"  would run days={label}")
        if completed_rows:
            _finalize_summary(completed_rows, out_dir, subject=subject, plot=plot)
        return completed_rows

    rows: list[dict[str, Any]] = list(completed_rows)
    for label, day_budget in pending:
        safe_echo(f"===== {holdout.model_key} {subject} days={label} =====")
        run_dir = data_size_run_dir(out_dir, subject, label)
        try:
            written, results = run_adapt(
                holdout=holdout,
                personal_csv=personal_csv,
                out_dir=run_dir,
                subject=subject,
                personal_days=day_budget,
                device=device,
                skip_completed=skip_completed,
                subject_model_dir=out_dir,
            )
        except ValueError as exc:
            safe_echo(f"Skipping days={label}: {exc}", err=True)
            rows.append(
                {
                    "subject": subject,
                    "model_key": holdout.model_key,
                    "personal_days": label,
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            _finalize_summary(rows, out_dir, subject=subject, plot=plot)
            if on_progress is not None:
                on_progress(subject, rows)
            continue
        rows.append(
            row_from_results(
                subject=subject,
                model_key=holdout.model_key,
                label=label,
                run_dir=written,
                results=results,
            )
        )
        _finalize_summary(rows, out_dir, subject=subject, plot=plot)
        if on_progress is not None:
            on_progress(subject, rows)

    _finalize_summary(rows, out_dir, subject=subject, plot=plot)
    return rows


def _finalize_summary(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    subject: str,
    plot: bool,
) -> None:
    summary_path = write_summary(rows, out_dir)
    safe_echo(f"Wrote {summary_path}")
    if not plot or not any(row.get("status") == "ok" for row in rows):
        return
    chart_path = out_dir / "data_size_curve.png"
    try:
        plot_data_size_curve(
            rows,
            out_png=chart_path,
            title=f"{subject} — personal train days vs test MAE (60 days)",
            subject=subject,
            mode="max_days",
            max_days=60.0,
        )
        safe_echo(f"Wrote {chart_path}")
    except ValueError as exc:
        safe_echo(f"Skip 60-day chart: {exc}", err=True)
