#!/usr/bin/env python3
"""Resumable NeuralForecast personalization study (data-size curves + report).

Continue-fits each ``nf_holdout`` global bundle on the same chronological
personal CSVs as SugarOne personalization. No LwF and no LR search.
Re-invoke after Ctrl+C: completed day budgets are skipped.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import typer

from common.console import init_cli_console, safe_echo
from personalization.cohort import PHASE4_SUBJECTS, Phase4Subject
from personalization_nf.constants import (
    DEFAULT_NF_HOLDOUT_ROOT,
    DEFAULT_NF_PERSONALIZATION_ROOT,
    DEFAULT_REPORT_PATH,
)
from personalization_nf.discover import discover_holdout_runs, parse_model_filter
from personalization_nf.report import write_personalization_nf_report
from personalization_nf.sweep import parse_days_grid, run_data_size_sweep

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

STATUS_PATH = DEFAULT_NF_PERSONALIZATION_ROOT / "study_status.json"
STATUS_MD_PATH = DEFAULT_NF_PERSONALIZATION_ROOT / "study_status.md"


def _subject_model_out_dir(root: Path, subject: str, model_key: str) -> Path:
    return root / subject / model_key


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
        "# NeuralForecast personalization study status",
        "",
        f"Updated: {payload['updated_at']}",
        f"Current: `{payload.get('current') or 'idle'}`",
        "",
        "## Jobs",
        "",
    ]
    for entry in payload.get("jobs", []):
        done = entry.get("completed_days") or []
        lines.append(
            f"- **{entry.get('model_key')} / {entry.get('subject')}**: "
            f"completed {done}"
        )
    lines.extend(
        [
            "",
            "Re-run:",
            "",
            "```bash",
            "uv run personal-nf-study --device auto",
            "```",
            "",
        ]
    )
    STATUS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_study(
    *,
    holdout_root: Path,
    root: Path,
    device: str,
    days: str | None,
    models_filter: str | None,
    subjects_filter: list[str] | None,
    dry_run: bool,
    report_only: bool,
    skip_completed: bool,
    report_path: Path,
) -> None:
    init_cli_console()
    wanted_models = parse_model_filter(models_filter)
    holdouts = discover_holdout_runs(holdout_root, models=wanted_models)
    holdouts = sorted(holdouts, key=lambda item: (item.val_mae, item.model_key))
    days_grid = parse_days_grid(days)
    wanted_subjects = (
        {_norm_filter_token(token) for token in subjects_filter}
        if subjects_filter
        else None
    )
    selected: list[Phase4Subject] = [
        spec
        for spec in PHASE4_SUBJECTS
        if wanted_subjects is None or _spec_matches_filter(spec, wanted_subjects)
    ]
    if not selected:
        raise ValueError(f"No subjects matched filter {subjects_filter}")

    status: dict[str, Any] = {
        "holdout_root": str(holdout_root),
        "device": device,
        "models": [item.model_key for item in holdouts],
        "days": ["all" if day is None else day for day in days_grid],
        "current": "report" if report_only else None,
        "jobs": [],
    }

    def refresh_report() -> None:
        write_personalization_nf_report(
            root=root,
            holdouts=holdouts,
            report_path=report_path,
            status=status,
        )
        _write_status(status)

    if report_only:
        refresh_report()
        return

    safe_echo(
        f"NF personalization: {len(selected)} subjects × {len(holdouts)} models, "
        f"device={device}"
    )
    safe_echo("Re-invoke after Ctrl+C; completed day budgets are skipped.")

    for spec in selected:
        if not spec.csv.exists():
            safe_echo(f"Missing CSV for {spec.subject}: {spec.csv}", err=True)
            status["jobs"].append(
                {
                    "subject": spec.subject,
                    "model_key": "*",
                    "status": "skipped",
                    "error": f"missing {spec.csv}",
                }
            )
            continue
        for holdout in holdouts:
            job_name = f"{holdout.model_key}/{spec.subject}"
            status["current"] = job_name
            _write_status(status)
            out_dir = _subject_model_out_dir(root, spec.subject, holdout.model_key)

            def on_progress(_subject: str, _rows: list[dict[str, Any]]) -> None:
                refresh_report()

            rows = run_data_size_sweep(
                holdout=holdout,
                personal_csv=spec.csv,
                out_dir=out_dir,
                subject=spec.subject,
                days_grid=days_grid,
                device=device,
                skip_completed=skip_completed,
                dry_run=dry_run,
                plot=True,
                on_progress=on_progress,
            )
            completed_days = [
                str(row.get("personal_days")) for row in rows if row.get("status") == "ok"
            ]
            status["jobs"].append(
                {
                    "subject": spec.subject,
                    "model_key": holdout.model_key,
                    "status": "ok",
                    "completed_days": completed_days,
                    "n_ok": len(completed_days),
                }
            )
            refresh_report()

    status["current"] = "done"
    refresh_report()
    safe_echo("Study sweep finished (or dry-run complete).")


@app.command()
def main(
    holdout_root: Path = typer.Option(DEFAULT_NF_HOLDOUT_ROOT, "--holdout-root"),
    root: Path = typer.Option(DEFAULT_NF_PERSONALIZATION_ROOT, "--root"),
    device: str = typer.Option("auto", "--device"),
    days: Optional[str] = typer.Option(None, "--days"),
    models: Optional[str] = typer.Option(
        None, "--models", help="Comma-separated model names (default: all holdout models)."
    ),
    subjects: Optional[str] = typer.Option(
        None,
        "--subjects",
        help="Comma-separated subjects, user ids, or cohort names.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    report_only: bool = typer.Option(False, "--report-only"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    report_path: Path = typer.Option(DEFAULT_REPORT_PATH, "--report-path"),
) -> None:
    """Zero-shot vs continue-fit data-size curves for NeuralForecast models."""
    filter_list = (
        [token.strip() for token in subjects.split(",") if token.strip()]
        if subjects
        else None
    )
    try:
        run_study(
            holdout_root=holdout_root,
            root=root,
            device=device,
            days=days,
            models_filter=models,
            subjects_filter=filter_list,
            dry_run=dry_run,
            report_only=report_only,
            skip_completed=skip_completed,
            report_path=report_path,
        )
    except KeyboardInterrupt:
        safe_echo("Interrupted — re-run the same command to resume.", err=True)
        raise typer.Exit(130) from None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
