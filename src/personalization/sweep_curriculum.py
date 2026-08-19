#!/usr/bin/env python3
"""Legacy sequential Livia curricula (student chained across day budgets).

The real-world decision is “given N days, should I fine-tune from the global
model?” That protocol lives in ``sweep_lwf.py`` (independent init + global
teacher). This module is kept for the older chained-weight experiment only.

- ``plain``: sequential student init, ``lwf_lambda=0``
- ``lwf_decay``: sequential student init, decaying λ on 1–14 days, 0 from 30

The LwF teacher was the frozen global model; the mistake was chaining the
*student* from the previous shorter-day checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

import typer

from common.console import init_cli_console, safe_echo
from common.paths import DEFAULT_RUNS_ROOT
from personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_FT_PATIENCE,
    DEFAULT_SEED,
    LWF_CURRICULUM_ZERO_FROM_DAYS,
    LWF_DECAY_SCHEDULE,
)
from personalization.finetune import run_finetune
from personalization.leaderboard import find_resume_checkpoint
from personalization.splits import load_train_span_days
from personalization.sweep_data_size import (
    _data_size_params,
    _finalize_summary,
    _parse_days_grid,
    _row_from_results,
)
from personalization.sweep_utils import (
    data_size_row_from_metrics,
    data_size_run_dir,
    load_best_recipe,
    personalization_run_complete,
    should_skip_day_budget,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

CurriculumKind = Literal["plain", "lwf_decay"]

KIND_SUBJECT: dict[CurriculumKind, str] = {
    "plain": "livia_curr_plain",
    "lwf_decay": "livia_curr_lwf",
}

KIND_OUT_DIR: dict[CurriculumKind, Path] = {
    "plain": DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "curriculum_plain",
    "lwf_decay": DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "curriculum_lwf",
}


def decaying_lwf_lambda(
    day_budget: int | None,
    *,
    schedule: dict[int, float] = LWF_DECAY_SCHEDULE,
    zero_from_days: int = LWF_CURRICULUM_ZERO_FROM_DAYS,
) -> float:
    """High LwF on short histories; 0 from ``zero_from_days`` onward (incl. all)."""
    if day_budget is None or day_budget >= zero_from_days:
        return 0.0
    if day_budget in schedule:
        return float(schedule[day_budget])
    span = float(zero_from_days)
    return round(max(schedule.values()) * max(0.0, 1.0 - day_budget / span), 4)


def lwf_for_kind(kind: CurriculumKind, day_budget: int | None) -> float:
    if kind == "plain":
        return 0.0
    return decaying_lwf_lambda(day_budget)


def run_curriculum_sweep(
    *,
    kind: CurriculumKind,
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
    sequential: bool = True,
    skip_completed: bool = True,
    dry_run: bool = False,
    plot: bool = True,
) -> list[dict[str, Any]]:
    """Run one curriculum along the personal-days grid (resume-safe)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = float(recipe.get("lr", 4e-4))
    wd = float(recipe.get("weight_decay", 3e-5))
    patience = int(recipe.get("patience", DEFAULT_FT_PATIENCE))
    train_span = load_train_span_days(personal_csv)
    if train_span is not None:
        safe_echo(f"Train span: {train_span:.1f} days ({personal_csv})")
    safe_echo(
        f"Curriculum kind={kind} sequential={sequential} "
        f"lr={lr} precision={precision} device={device}"
    )

    rows: list[dict[str, Any]] = []
    prev_best: Path | None = None

    for day_budget in days_grid:
        label = "all" if day_budget is None else str(day_budget)
        if should_skip_day_budget(day_budget, train_span):
            safe_echo(
                f"Skipping days={label}: budget covers full train span "
                f"({train_span:.1f}d); using days=all instead"
            )
            continue
        lwf = lwf_for_kind(kind, day_budget)
        run_dir = data_size_run_dir(out_dir, subject, label)
        best_path = run_dir / "best_model.pt"

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
                row["curriculum"] = kind
                row["sequential"] = sequential
                rows.append(row)
            if best_path.is_file():
                prev_best = best_path
            continue

        init_from = prev_best if sequential else None
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
        resume_ckpt = find_resume_checkpoint(out_dir / f"days_{label}", params)
        if dry_run:
            init_note = f" init={init_from}" if init_from else ""
            resume_note = f" resume={resume_ckpt}" if resume_ckpt else ""
            safe_echo(f"  days={label} lwf={lwf:g}{init_note}{resume_note}")
            continue

        safe_echo(f"\n===== {subject} {kind} days={label} lwf={lwf:g} =====")
        if resume_ckpt is not None:
            safe_echo(f"  Resume checkpoint: {resume_ckpt}")
        elif init_from is not None:
            safe_echo(f"  Student init from previous budget: {init_from}")

        run_dir, results = run_finetune(
            base_run_dir=base_run_dir,
            personal_csv=personal_csv,
            out_dir=out_dir / f"days_{label}",
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
            init_weights_from=None if resume_ckpt is not None else init_from,
            refit_scalers_on_personal=False,
        )
        row = _row_from_results(
            subject=subject,
            label=label,
            lwf=lwf,
            lr=lr,
            weight_decay=wd,
            patience=patience,
            run_dir=run_dir,
            results=results,
        )
        row["curriculum"] = kind
        row["sequential"] = sequential
        rows.append(row)
        if (run_dir / "best_model.pt").is_file():
            prev_best = run_dir / "best_model.pt"
        _finalize_summary(rows, out_dir, recipe, plot=plot)

    if not dry_run:
        _finalize_summary(rows, out_dir, recipe, plot=plot)
    return rows


@app.command()
def main(
    kind: str = typer.Option(
        "both",
        "--kind",
        help="plain | lwf-decay | both",
    ),
    base_run_dir: Path = typer.Option(Path(DEFAULT_BASE_RUN_DIR), "--base-run-dir"),
    personal_csv: Path = typer.Option(
        Path("data/input/personalization/prepared/livia_chronological.csv"),
        "--personal-csv",
    ),
    recipe_json: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "livia" / "best_recipe.json",
        "--recipe-json",
    ),
    days: Optional[str] = typer.Option(None, "--days"),
    sequential: bool = typer.Option(True, "--sequential/--independent"),
    device: str = typer.Option("cuda", "--device"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    report_only: bool = typer.Option(False, "--report-only"),
) -> None:
    """Sequential Livia curricula (plain and/or decaying LwF)."""
    init_cli_console()
    raw = kind.strip().lower().replace("_", "-")
    kinds: list[CurriculumKind]
    if raw in {"both", "all"}:
        kinds = ["plain", "lwf_decay"]
    elif raw in {"plain"}:
        kinds = ["plain"]
    elif raw in {"lwf-decay", "lwf", "decay"}:
        kinds = ["lwf_decay"]
    else:
        raise typer.BadParameter("kind must be plain, lwf-decay, or both")

    recipe = load_best_recipe(recipe_json)
    precision = str(recipe.get("precision", "bf16"))
    epochs = int(recipe.get("epochs", 30))
    days_grid = _parse_days_grid(days)

    if report_only:
        from personalization.report import write_milestone8_report

        write_milestone8_report()
        return

    for item in kinds:
        run_curriculum_sweep(
            kind=item,
            base_run_dir=base_run_dir,
            personal_csv=personal_csv,
            out_dir=KIND_OUT_DIR[item],
            recipe=recipe,
            days_grid=days_grid,
            subject=KIND_SUBJECT[item],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            precision=precision,
            sequential=sequential,
            skip_completed=skip_completed,
            dry_run=dry_run,
            plot=True,
        )
    if not dry_run:
        from personalization.report import write_milestone8_report

        write_milestone8_report()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
