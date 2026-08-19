#!/usr/bin/env python3
"""Resumable Milestone 8 Phase 4 runner (data-size curves + interim report).

Runs the frozen Livia recipe with base-run scalers for Livia, Loop quality
holdouts, and two joined2 test users per study group. Safe to Ctrl+C and
re-invoke: completed base-scaler day budgets are skipped; partial runs resume
from ``last_checkpoint.pt``. The interim report is rewritten after every
finished day budget.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer

from common.console import init_cli_console, safe_echo
from common.paths import DEFAULT_RUNS_ROOT
from personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_SEED,
)
from personalization.cohort import PHASE4_SUBJECTS, Phase4Subject
from personalization.report import collect_data_size_series, write_milestone8_report
from personalization.sweep_data_size import _parse_days_grid, run_data_size_sweep
from personalization.sweep_utils import (
    data_size_run_dir,
    load_best_recipe,
    personalization_run_complete,
    write_summary,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

STATUS_PATH = DEFAULT_RUNS_ROOT / "personalization" / "phase4_status.json"
STATUS_MD_PATH = DEFAULT_RUNS_ROOT / "personalization" / "phase4_status.md"
DEFAULT_RECIPE = DEFAULT_RUNS_ROOT / "personalization" / "livia" / "best_recipe.json"
DEFAULT_ROOT = DEFAULT_RUNS_ROOT / "personalization"


def _subject_out_dir(root: Path, subject: str) -> Path:
    return root / subject / "sweeps" / "data_size"


def _norm_filter_token(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "")


def _spec_matches_filter(spec: Phase4Subject, wanted: set[str]) -> bool:
    tokens = {
        spec.subject,
        spec.user_id,
        spec.cohort,
        spec.study_group,
        spec.subject.removeprefix("loop_"),
        spec.subject.removeprefix("ai_ready_"),
    }
    return any(_norm_filter_token(token) in wanted for token in tokens)


def _write_status(payload: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat()
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Phase 4 personalization recalc status",
        "",
        f"Updated: {payload['updated_at']}",
        f"Current: `{payload.get('current') or 'idle'}`",
        "",
        "## Subjects",
        "",
    ]
    for entry in payload.get("subjects", []):
        done = entry.get("completed_days") or []
        pending = entry.get("pending_days") or []
        lines.append(
            f"- **{entry.get('subject')}**: completed {done}; pending {pending}"
        )
    lines.extend(["", "Re-run:", "", "```bash", "uv run run-personal-phase4 --device cuda", "```", ""])
    STATUS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _params_rows_from_series(root: Path) -> list[dict[str, Any]]:
    """Phase A table: full-train (`days=all`) row per subject."""
    rows: list[dict[str, Any]] = []
    for spec in PHASE4_SUBJECTS:
        run_dir = data_size_run_dir(_subject_out_dir(root, spec.subject), spec.subject, "all")
        metrics_path = run_dir / "personalization_metrics.json"
        if not personalization_run_complete(run_dir):
            continue
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = results.get("config") if isinstance(results.get("config"), dict) else {}
        zs = results.get("zero_shot_test") or {}
        ft = results.get("finetuned_test") or {}
        delta = None
        if zs.get("mae") is not None and ft.get("mae") is not None:
            delta = float(ft["mae"]) - float(zs["mae"])
        rows.append(
            {
                "user_id": spec.user_id,
                "subject": spec.subject,
                "phase": "params_validation",
                "status": "ok",
                "lr": cfg.get("lr"),
                "personal_days": "all",
                "train_span_days": cfg.get("train_span_days"),
                "run_dir": str(run_dir),
                "zs_test_mae": zs.get("mae"),
                "ft_test_mae": ft.get("mae"),
                "delta_mae_ft_minus_zs": delta,
                "improved": delta is not None and delta < 0,
            }
        )
    return rows


def run_phase4(
    *,
    base_run_dir: Path,
    recipe_json: Path,
    root: Path,
    device: str,
    days: str | None,
    batch_size: int,
    seed: int,
    dry_run: bool,
    report_only: bool,
    skip_completed: bool,
    archive_legacy: bool,
    subjects_filter: list[str] | None,
) -> None:
    init_cli_console()
    recipe = load_best_recipe(recipe_json)
    precision = str(recipe.get("precision", "bf16"))
    epochs = int(recipe.get("epochs", 30))
    days_grid = _parse_days_grid(days)
    wanted = (
        {_norm_filter_token(s) for s in subjects_filter} if subjects_filter else None
    )

    selected: list[Phase4Subject] = [
        spec
        for spec in PHASE4_SUBJECTS
        if wanted is None or _spec_matches_filter(spec, wanted)
    ]
    if not selected:
        raise ValueError(f"No subjects matched filter {subjects_filter}")

    done_names = {spec.subject for spec, _rows in collect_data_size_series(root)}
    pending_names = [s.subject for s in selected if s.subject not in done_names]

    status: dict[str, Any] = {
        "recipe_json": str(recipe_json),
        "device": device,
        "days": [ "all" if d is None else d for d in days_grid ],
        "current": "report" if report_only else None,
        "pending_subjects": pending_names,
        "subjects": [],
    }

    def refresh_report() -> None:
        params_rows = _params_rows_from_series(root)
        if params_rows:
            write_summary(params_rows, root / "holdout_validation" / "params", name="summary")
        write_milestone8_report(root=root, status=status)
        _write_status(status)

    if report_only:
        refresh_report()
        return

    safe_echo(
        f"Phase 4 recalc: {len(selected)} subjects, recipe lr={recipe.get('lr')} "
        f"precision={precision} device={device}"
    )
    safe_echo("Re-invoke this command after Ctrl+C; completed day budgets are skipped.")

    for spec in selected:
        uid, subject, csv_path = spec.user_id, spec.subject, spec.csv
        out_dir = _subject_out_dir(root, subject)
        status["current"] = f"{subject}"
        status["pending_subjects"] = [
            s.subject for s in selected if s.subject not in {e.get("subject") for e in status["subjects"]} and s.subject != subject
        ]
        _write_status(status)
        if not csv_path.exists():
            safe_echo(f"Missing CSV for {subject}: {csv_path}", err=True)
            status["subjects"].append(
                {"subject": subject, "user_id": uid, "status": "skipped", "error": f"missing {csv_path}"}
            )
            continue

        def on_progress(_subj: str, _rows: list[dict[str, Any]]) -> None:
            refresh_report()

        rows = run_data_size_sweep(
            base_run_dir=base_run_dir,
            personal_csv=csv_path,
            out_dir=out_dir,
            recipe=recipe,
            days_grid=days_grid,
            subject=subject,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            precision=precision,
            skip_completed=skip_completed,
            dry_run=dry_run,
            plot=True,
            seed_all_from=None,
            archive_legacy=archive_legacy,
            on_progress=on_progress,
        )
        completed_days = [str(r.get("personal_days")) for r in rows if r.get("status") == "ok"]
        status["subjects"].append(
            {
                "subject": subject,
                "user_id": uid,
                "status": "ok",
                "completed_days": completed_days,
                "n_ok": len(completed_days),
            }
        )
        status["pending_subjects"] = [
            s.subject for s in selected if s.subject not in {e.get("subject") for e in status["subjects"]}
        ]
        refresh_report()

    status["current"] = "done"
    refresh_report()
    safe_echo("Phase 4 sweep finished (or dry-run complete).")


@app.command()
def main(
    base_run_dir: Path = typer.Option(Path(DEFAULT_BASE_RUN_DIR), "--base-run-dir"),
    recipe_json: Path = typer.Option(DEFAULT_RECIPE, "--recipe-json"),
    root: Path = typer.Option(DEFAULT_ROOT, "--root"),
    device: str = typer.Option("cuda", "--device"),
    days: Optional[str] = typer.Option(None, "--days"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    report_only: bool = typer.Option(False, "--report-only"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    archive_legacy: bool = typer.Option(True, "--archive-legacy/--no-archive-legacy"),
    subjects: Optional[str] = typer.Option(
        None,
        "--subjects",
        help="Comma-separated subjects, user ids, or cohort (livia, joined2_test, quality_holdout).",
    ),
) -> None:
    """Recalculate data-size curves with base scalers; resume-safe overnight runner."""
    filter_list = (
        [s.strip() for s in subjects.split(",") if s.strip()] if subjects else None
    )
    try:
        run_phase4(
            base_run_dir=base_run_dir,
            recipe_json=recipe_json,
            root=root,
            device=device,
            days=days,
            batch_size=batch_size,
            seed=seed,
            dry_run=dry_run,
            report_only=report_only,
            skip_completed=skip_completed,
            archive_legacy=archive_legacy,
            subjects_filter=filter_list,
        )
    except KeyboardInterrupt:
        safe_echo("Interrupted — re-run the same command to resume.", err=True)
        raise typer.Exit(130) from None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
