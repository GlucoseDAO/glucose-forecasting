#!/usr/bin/env python3
"""Step 3: sweep personal train-day budgets using the best LR recipe.

After hyperparameters are fixed on full personal train data, measure how test
MAE improves as more days of personal data are used for fine-tuning. Estimates
plateau day for optimal dataset size.

Protocol (must match Step-2 tune):
- plain fine-tune, sparse stride, recipe LR/wd/patience
- ``precision`` from recipe (default ``bf16`` — same as ``personalization_tune.toml``)
- scalers fitted on the **full** personal train split; day limit only for train windows
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from common.console import init_cli_console, safe_echo
from common.paths import DEFAULT_RUNS_ROOT
from personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_DATA_SIZE_DAYS,
    DEFAULT_FT_PATIENCE,
    DEFAULT_PERSONAL_LWF_LAMBDA,
    DEFAULT_SEED,
    DEFAULT_TRAIN_WINDOW_STRIDE,
    DEFAULT_VAL_EVERY_N_EPOCHS,
)
from personalization.finetune import run_finetune
from personalization.leaderboard import find_resume_checkpoint
from personalization.splits import load_train_span_days
from personalization.sweep_utils import (
    archive_legacy_scaler_runs,
    data_size_row_from_metrics,
    data_size_run_dir,
    estimate_plateau_day,
    flatten_metrics,
    load_best_recipe,
    personalization_run_complete,
    should_skip_day_budget,
    write_best_recipe,
    write_summary,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DEFAULT_PRECISION: str = "bf16"


def _parse_days_grid(raw: str | None) -> list[int | None]:
    if raw is None:
        items = list(DEFAULT_DATA_SIZE_DAYS)
    else:
        items = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[int | None] = []
    for item in items:
        if str(item).lower() == "all":
            out.append(None)
        else:
            out.append(int(item))
    return out


def _data_size_params(
    *,
    base_run_dir: Path,
    personal_csv: Path,
    lwf: float,
    lr: float,
    weight_decay: float,
    patience: int,
    epochs: int,
    batch_size: int,
    day_budget: int | None,
    precision: str,
) -> dict[str, Any]:
    return {
        "base_run_dir": str(base_run_dir.resolve()),
        "personal_csv": str(personal_csv.resolve()),
        "lwf_lambda": lwf,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "epochs": epochs,
        "batch_size": batch_size,
        "personal_days": day_budget,
        "train_window_stride": DEFAULT_TRAIN_WINDOW_STRIDE,
        "val_every_n_epochs": DEFAULT_VAL_EVERY_N_EPOCHS,
        "precision": precision,
        "eval_zero_shot": True,
        "refit_scalers_on_personal": False,
    }


def _seed_day_from_run(
    *,
    source_run: Path,
    dest_run: Path,
    subject: str,
    day_label: str,
    lwf: float,
    lr: float,
    weight_decay: float,
    patience: int,
) -> dict[str, Any]:
    """Copy a finished personalization run into the data-size layout (e.g. days=all)."""
    if not personalization_run_complete(source_run):
        raise ValueError(f"source run incomplete: {source_run}")
    dest_run.parent.mkdir(parents=True, exist_ok=True)
    if dest_run.exists():
        shutil.rmtree(dest_run)
    shutil.copytree(source_run, dest_run)
    row = data_size_row_from_metrics(
        dest_run,
        subject=subject,
        day_label=day_label,
        lwf_lambda=lwf,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )
    if row is None:
        raise ValueError(f"failed to read seeded metrics from {dest_run}")
    row["seeded_from"] = str(source_run)
    return row


def _row_from_results(
    *,
    subject: str,
    label: str,
    lwf: float,
    lr: float,
    weight_decay: float,
    patience: int,
    run_dir: Path,
    results: dict[str, Any],
) -> dict[str, Any]:
    cfg = results.get("config") if isinstance(results.get("config"), dict) else {}
    return {
        "subject": subject,
        "personal_days": label,
        "lwf_lambda": lwf,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "run_dir": str(run_dir),
        "status": "ok",
        "train_span_days": cfg.get("train_span_days"),
        "used_train_days": cfg.get("used_train_days"),
        "scaler_source": cfg.get("scaler_source"),
        "refit_scalers_on_personal": cfg.get("refit_scalers_on_personal"),
        **flatten_metrics("zs_test", results.get("zero_shot_test")),
        **flatten_metrics("ft_test", results.get("finetuned_test")),
        **flatten_metrics("ft_val", results.get("finetuned_val")),
    }


def run_data_size_sweep(
    *,
    base_run_dir: Path,
    personal_csv: Path,
    out_dir: Path,
    recipe: dict[str, Any],
    days_grid: list[int | None],
    subject: str,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    precision: str,
    skip_completed: bool = True,
    dry_run: bool = False,
    plot: bool = True,
    seed_all_from: Path | None = None,
    archive_legacy: bool = True,
    on_progress: Any | None = None,
) -> list[dict[str, Any]]:
    """Run or resume a data-size sweep. Safe to re-invoke after interruption."""
    out_dir = Path(out_dir)
    if archive_legacy and not dry_run:
        archived = archive_legacy_scaler_runs(out_dir)
        if archived is not None:
            safe_echo(f"Archived personal-scaler runs to {archived}")

    out_dir.mkdir(parents=True, exist_ok=True)
    lwf = float(recipe.get("lwf_lambda", DEFAULT_PERSONAL_LWF_LAMBDA))
    lr = float(recipe.get("lr", 4e-4))
    wd = float(recipe.get("weight_decay", 3e-5))
    patience = int(recipe.get("patience", DEFAULT_FT_PATIENCE))
    train_span = load_train_span_days(personal_csv)
    if train_span is not None:
        safe_echo(f"Train span: {train_span:.1f} days ({personal_csv})")

    completed_rows: list[dict[str, Any]] = []
    pending: list[tuple[str, int | None]] = []

    for day_budget in days_grid:
        label = "all" if day_budget is None else str(day_budget)
        if should_skip_day_budget(day_budget, train_span):
            safe_echo(
                f"Skipping days={label}: budget covers full train span "
                f"({train_span:.1f}d); using days=all instead"
            )
            continue
        run_dir = data_size_run_dir(out_dir, subject, label)

        if (
            label == "all"
            and seed_all_from is not None
            and not (skip_completed and personalization_run_complete(run_dir))
        ):
            if dry_run:
                safe_echo(f"  days=all would seed from {seed_all_from}")
                continue
            if not personalization_run_complete(seed_all_from):
                safe_echo(
                    f"Not seeding days=all from {seed_all_from} "
                    "(incomplete or personal-scaler run)"
                )
            else:
                safe_echo(f"Seeding days=all from {seed_all_from}")
                row = _seed_day_from_run(
                    source_run=seed_all_from,
                    dest_run=run_dir,
                    subject=subject,
                    day_label=label,
                    lwf=lwf,
                    lr=lr,
                    weight_decay=wd,
                    patience=patience,
                )
                completed_rows.append(row)
                continue

        if skip_completed and personalization_run_complete(run_dir):
            row = data_size_row_from_metrics(
                run_dir,
                subject=subject,
                day_label=label,
                lwf_lambda=lwf,
                lr=lr,
                weight_decay=wd,
                patience=patience,
            )
            if row is not None:
                if train_span is not None and row.get("train_span_days") is None:
                    row["train_span_days"] = train_span
                    if label != "all":
                        row["used_train_days"] = min(float(label), train_span)
                    else:
                        row["used_train_days"] = train_span
                completed_rows.append(row)
            continue
        pending.append((label, day_budget))

    safe_echo(f"Completed day budgets: {len(completed_rows)}")
    safe_echo(f"Pending day budgets: {len(pending)}")

    if dry_run:
        safe_echo("\n--- Pending ---")
        for label, day_budget in pending:
            combo_out = out_dir / f"days_{label}"
            resume_ckpt = find_resume_checkpoint(
                combo_out,
                _data_size_params(
                    base_run_dir=base_run_dir,
                    personal_csv=personal_csv,
                    lwf=lwf,
                    lr=lr,
                    weight_decay=wd,
                    patience=patience,
                    epochs=epochs,
                    batch_size=batch_size,
                    day_budget=day_budget,
                    precision=precision,
                ),
            )
            resume_note = f" resume={resume_ckpt}" if resume_ckpt else ""
            safe_echo(f"  days={label}{resume_note}")
        if completed_rows:
            _finalize_summary(completed_rows, out_dir, recipe, plot=plot)
        return completed_rows

    rows: list[dict[str, Any]] = list(completed_rows)

    for label, day_budget in pending:
        safe_echo(f"\n===== {subject} data-size days={label} =====")
        combo_out = out_dir / f"days_{label}"
        params = _data_size_params(
            base_run_dir=base_run_dir,
            personal_csv=personal_csv,
            lwf=lwf,
            lr=lr,
            weight_decay=wd,
            patience=patience,
            epochs=epochs,
            batch_size=batch_size,
            day_budget=day_budget,
            precision=precision,
        )
        resume_ckpt = find_resume_checkpoint(combo_out, params)
        if resume_ckpt is not None:
            safe_echo(f"  Resume checkpoint: {resume_ckpt}")
        try:
            run_dir, results = run_finetune(
                base_run_dir=base_run_dir,
                personal_csv=personal_csv,
                out_dir=combo_out,
                run_name=f"{subject}_days_{label}",
                personal_days=day_budget,
                lwf_lambda=lwf,
                epochs=epochs,
                lr=lr,
                weight_decay=wd,
                patience=patience,
                batch_size=batch_size,
                seed=seed,
                device=device,
                precision=precision,
                eval_zero_shot=True,
                resume_from=resume_ckpt,
                refit_scalers_on_personal=False,
            )
        except ValueError as exc:
            safe_echo(f"Skipping days={label}: {exc}", err=True)
            rows.append(
                {
                    "subject": subject,
                    "personal_days": label,
                    "lwf_lambda": lwf,
                    "lr": lr,
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            if on_progress is not None:
                on_progress(subject, rows)
            continue

        rows.append(
            _row_from_results(
                subject=subject,
                label=label,
                lwf=lwf,
                lr=lr,
                weight_decay=wd,
                patience=patience,
                run_dir=run_dir,
                results=results,
            )
        )
        _finalize_summary(rows, out_dir, recipe, plot=plot)
        if on_progress is not None:
            on_progress(subject, rows)

    _finalize_summary(rows, out_dir, recipe, plot=plot)
    return rows


@app.command()
def main(
    base_run_dir: Path = typer.Option(
        Path(DEFAULT_BASE_RUN_DIR),
        "--base-run-dir",
    ),
    personal_csv: Path = typer.Option(..., "--personal-csv"),
    out_dir: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "data_size",
        "--out-dir",
    ),
    recipe_json: Path = typer.Option(
        ...,
        "--recipe-json",
        help="best_recipe.json from hyperparameter sweep (LwF + LR).",
    ),
    days: Optional[str] = typer.Option(
        None,
        "--days",
        help="Comma-separated day budgets, e.g. '1,3,7,14,30,60,all'.",
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    precision: Optional[str] = typer.Option(
        None,
        "--precision",
        help="Override recipe precision (default: recipe or bf16 to match tune).",
    ),
    seed_all_from: Optional[Path] = typer.Option(
        None,
        "--seed-all-from",
        help="Copy this finished run as days=all (use Step-2 best tune run).",
    ),
    subject: str = typer.Option("livia", "--subject"),
    skip_completed: bool = typer.Option(
        True,
        "--skip-completed/--no-skip-completed",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Rebuild summary/chart from completed runs on disk.",
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="After sweep, write data_size_curve.png from summary.csv.",
    ),
) -> None:
    """Run data-size learning curve with fixed LwF/LR; estimate plateau."""
    init_cli_console()
    recipe = load_best_recipe(recipe_json)
    lwf = float(recipe.get("lwf_lambda", DEFAULT_PERSONAL_LWF_LAMBDA))
    lr = float(recipe.get("lr", 4e-4))
    wd = float(recipe.get("weight_decay", 3e-5))
    patience = int(recipe.get("patience", DEFAULT_FT_PATIENCE))
    recipe_epochs = int(epochs if epochs is not None else recipe.get("epochs", 30))
    resolved_precision = str(
        precision if precision is not None else recipe.get("precision", DEFAULT_PRECISION)
    )

    safe_echo(
        f"Using recipe: lwf={lwf} lr={lr} weight_decay={wd} patience={patience} "
        f"precision={resolved_precision} from {recipe_json}"
    )

    grid = _parse_days_grid(days)

    if report_only:
        rows: list[dict[str, Any]] = []
        for day_budget in grid:
            label = "all" if day_budget is None else str(day_budget)
            run_dir = data_size_run_dir(out_dir, subject, label)
            row = data_size_row_from_metrics(
                run_dir,
                subject=subject,
                day_label=label,
                lwf_lambda=lwf,
                lr=lr,
                weight_decay=wd,
                patience=patience,
            )
            if row is not None:
                rows.append(row)
        if not rows:
            raise ValueError(f"No completed data-size runs under {out_dir}")
        _finalize_summary(rows, out_dir, recipe, plot=plot)
        return

    run_data_size_sweep(
        base_run_dir=base_run_dir,
        personal_csv=personal_csv,
        out_dir=out_dir,
        recipe=recipe,
        days_grid=grid,
        subject=subject,
        epochs=recipe_epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        precision=resolved_precision,
        skip_completed=skip_completed,
        dry_run=dry_run,
        plot=plot,
        seed_all_from=seed_all_from,
    )


def _finalize_summary(
    rows: list[dict[str, Any]],
    out_dir: Path,
    recipe: dict[str, Any],
    *,
    plot: bool,
) -> None:
    summary_path = write_summary(rows, out_dir)
    plateau_info = estimate_plateau_day(rows)
    plateau_path = out_dir / "plateau_analysis.json"
    with plateau_path.open("w", encoding="utf-8") as f:
        json.dump(plateau_info, f, indent=2)

    safe_echo(f"Wrote {summary_path}")
    safe_echo(
        f"Plateau estimate: optimal_day={plateau_info.get('optimal_day')} "
        f"plateau_day={plateau_info.get('plateau_day')}"
    )

    recipe_out = {
        **recipe,
        "sweep": "data_size",
        "plateau_day": plateau_info.get("plateau_day"),
        "optimal_day": plateau_info.get("optimal_day"),
        "best_mae": plateau_info.get("best_mae"),
    }
    write_best_recipe(out_dir / "best_from_data_size.json", recipe_out)
    subject_root = out_dir.parents[1] if out_dir.name == "data_size" else out_dir
    write_best_recipe(subject_root / "best_recipe_with_days.json", recipe_out)

    if plot and any(r.get("status") == "ok" for r in rows):
        from personalization.plot_data_size_curve import plot_data_size_curve

        chart_path = out_dir / "data_size_curve.png"
        subject_name = str(rows[0].get("subject", "subject"))
        try:
            plot_data_size_curve(
                rows,
                out_png=chart_path,
                title=f"{subject_name} — personal train days vs test MAE (60 days)",
                subject=subject_name,
                mode="max_days",
                max_days=60.0,
            )
            safe_echo(f"Wrote {chart_path}")
        except ValueError as exc:
            safe_echo(f"Skip 60-day chart: {exc}", err=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
