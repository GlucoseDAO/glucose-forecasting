#!/usr/bin/env python3
"""Independent LwF data-size sweeps (teacher = global sugar_one_1.0).

Real-world question: given N days of one user, does fine-tuning help or hurt?
Each day budget is a *new* run from the global checkpoint. The LwF teacher is
always ``fixtures/checkpoints/sugar_one_1.0`` (never a shorter-day student).

Kinds:
- ``decay``: λ=0.5/0.4/0.3/0.2 on 1/3/7/14 days; λ=0 from 30 days (copy the
  existing independent λ=0 data-size run).
- ``const``: λ=0.1 on every day budget, including 30/60/all.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import typer

from common.console import init_cli_console, safe_echo
from common.paths import DEFAULT_RUNS_ROOT
from personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_FT_PATIENCE,
    DEFAULT_SEED,
    LWF_CONST_LAMBDA,
    decaying_lwf_lambda,
)
from personalization.finetune import run_finetune
from personalization.leaderboard import find_resume_checkpoint
from personalization.splits import load_train_span_days
from personalization.sweep_data_size import (
    _data_size_params,
    _finalize_summary,
    _parse_days_grid,
    _row_from_results,
    _seed_day_from_run,
)
from personalization.sweep_utils import (
    data_size_row_from_metrics,
    data_size_run_dir,
    load_best_recipe,
    personalization_run_complete,
    should_skip_day_budget,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

LwfKind = Literal["decay", "const"]

STATUS_PATH = DEFAULT_RUNS_ROOT / "personalization" / "lwf_indep_status.json"
STATUS_MD_PATH = DEFAULT_RUNS_ROOT / "personalization" / "lwf_indep_status.md"
DEFAULT_RECIPE = DEFAULT_RUNS_ROOT / "personalization" / "livia" / "best_recipe.json"


@dataclass(frozen=True)
class LwfPerson:
    name: str
    csv: Path
    independent_subject: str
    independent_dir: Path
    decay_subject: str
    decay_dir: Path
    const_subject: str
    const_dir: Path

    def method_sweeps(self) -> tuple[tuple[str, str], ...]:
        root = DEFAULT_RUNS_ROOT / "personalization"
        return (
            (self.independent_subject, self.independent_dir.relative_to(root).as_posix()),
            (self.decay_subject, self.decay_dir.relative_to(root).as_posix()),
            (self.const_subject, self.const_dir.relative_to(root).as_posix()),
        )


PERSONS: tuple[LwfPerson, ...] = (
    LwfPerson(
        name="livia",
        csv=Path("data/input/personalization/prepared/livia_chronological.csv"),
        independent_subject="livia",
        independent_dir=DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "data_size",
        decay_subject="livia_lwf_decay",
        decay_dir=DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "lwf_decay_indep",
        const_subject="livia_lwf_01",
        const_dir=DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "lwf_const_0.1",
    ),
    LwfPerson(
        name="154",
        csv=Path("data/input/personalization/holdouts/loop_154_chronological.csv"),
        independent_subject="loop_154",
        independent_dir=DEFAULT_RUNS_ROOT / "personalization" / "loop_154" / "sweeps" / "data_size",
        decay_subject="loop_154_lwf_decay",
        decay_dir=DEFAULT_RUNS_ROOT / "personalization" / "loop_154" / "sweeps" / "lwf_decay_indep",
        const_subject="loop_154_lwf_01",
        const_dir=DEFAULT_RUNS_ROOT / "personalization" / "loop_154" / "sweeps" / "lwf_const_0.1",
    ),
)


def lwf_for_independent_kind(kind: LwfKind, day_budget: int | None) -> float:
    if kind == "const":
        return float(LWF_CONST_LAMBDA)
    return decaying_lwf_lambda(day_budget)


def _write_status(payload: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat()
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Independent LwF overnight status",
        "",
        f"Updated: {payload['updated_at']}",
        f"Current: `{payload.get('current') or 'idle'}`",
        "",
        "## Jobs",
        "",
    ]
    for entry in payload.get("jobs", []):
        lines.append(
            f"- **{entry.get('person')} / {entry.get('kind')}**: "
            f"completed {entry.get('completed_days') or []}"
        )
    lines.extend(
        [
            "",
            "Re-run:",
            "",
            "```bash",
            "uv run personal-sweep-lwf --device cuda",
            "```",
            "",
        ]
    )
    STATUS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_independent_lwf_sweep(
    *,
    person: LwfPerson,
    kind: LwfKind,
    base_run_dir: Path,
    recipe: dict[str, Any],
    days_grid: list[int | None],
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    precision: str,
    skip_completed: bool = True,
    dry_run: bool = False,
    plot: bool = True,
    on_progress: Any | None = None,
) -> list[dict[str, Any]]:
    """One person × one λ policy. Student and teacher both start from global."""
    out_dir = person.decay_dir if kind == "decay" else person.const_dir
    subject = person.decay_subject if kind == "decay" else person.const_subject
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = float(recipe.get("lr", 4e-4))
    wd = float(recipe.get("weight_decay", 3e-5))
    patience = int(recipe.get("patience", DEFAULT_FT_PATIENCE))
    train_span = load_train_span_days(person.csv)
    if train_span is not None:
        safe_echo(f"Train span: {train_span:.1f} days ({person.csv})")
    safe_echo(
        f"Independent LwF person={person.name} kind={kind} "
        f"teacher=sugar_one_1.0 sequential_init=False lr={lr} device={device}"
    )

    rows: list[dict[str, Any]] = []
    for day_budget in days_grid:
        label = "all" if day_budget is None else str(day_budget)
        if should_skip_day_budget(day_budget, train_span):
            safe_echo(
                f"Skipping days={label}: budget covers full train span "
                f"({train_span:.1f}d); using days=all instead"
            )
            continue
        lwf = lwf_for_independent_kind(kind, day_budget)
        run_dir = data_size_run_dir(out_dir, subject, label)

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
                row["lwf_kind"] = kind
                row["seeded_independent"] = lwf == 0.0 and kind == "decay"
                rows.append(row)
            continue

        if kind == "decay" and lwf == 0.0:
            source = data_size_run_dir(
                person.independent_dir, person.independent_subject, label
            )
            if dry_run:
                safe_echo(f"  days={label} lwf=0 seed from {source}")
                continue
            safe_echo(f"Seeding days={label} from independent λ=0 run {source}")
            row = _seed_day_from_run(
                source_run=source,
                dest_run=run_dir,
                subject=subject,
                day_label=label,
                lwf=0.0,
                lr=lr,
                weight_decay=wd,
                patience=patience,
            )
            row["lwf_kind"] = kind
            row["seeded_independent"] = True
            rows.append(row)
            _finalize_summary(rows, out_dir, recipe, plot=plot)
            if on_progress is not None:
                on_progress(subject, rows)
            continue

        params = _data_size_params(
            base_run_dir=base_run_dir,
            personal_csv=person.csv,
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
            resume_note = f" resume={resume_ckpt}" if resume_ckpt else ""
            safe_echo(
                f"  days={label} lwf={lwf:g} init=global teacher=global{resume_note}"
            )
            continue

        safe_echo(
            f"\n===== {subject} independent LwF days={label} lwf={lwf:g} "
            f"init=global teacher=global ====="
        )
        if resume_ckpt is not None:
            safe_echo(f"  Resume checkpoint: {resume_ckpt}")

        run_dir, results = run_finetune(
            base_run_dir=base_run_dir,
            personal_csv=person.csv,
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
            init_weights_from=None,
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
        row["lwf_kind"] = kind
        row["seeded_independent"] = False
        rows.append(row)
        _finalize_summary(rows, out_dir, recipe, plot=plot)
        if on_progress is not None:
            on_progress(subject, rows)

    if not dry_run:
        _finalize_summary(rows, out_dir, recipe, plot=plot)
    return rows


@app.command()
def main(
    subjects: str = typer.Option(
        "livia,154",
        "--subjects",
        help="Comma-separated: livia, 154.",
    ),
    kind: str = typer.Option(
        "both",
        "--kind",
        help="decay | const | both",
    ),
    base_run_dir: Path = typer.Option(Path(DEFAULT_BASE_RUN_DIR), "--base-run-dir"),
    recipe_json: Path = typer.Option(DEFAULT_RECIPE, "--recipe-json"),
    days: Optional[str] = typer.Option(None, "--days"),
    device: str = typer.Option("cuda", "--device"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
    report_only: bool = typer.Option(False, "--report-only"),
) -> None:
    """Overnight independent LwF: Livia then user 154. Resume-safe."""
    init_cli_console()
    wanted = {s.strip().lower() for s in subjects.split(",") if s.strip()}
    selected = [p for p in PERSONS if p.name in wanted]
    if not selected:
        raise typer.BadParameter(f"No persons matched {subjects}")

    raw_kind = kind.strip().lower()
    kinds: list[LwfKind]
    if raw_kind in {"both", "all"}:
        kinds = ["decay", "const"]
    elif raw_kind in {"decay", "lwf-decay", "curriculum"}:
        kinds = ["decay"]
    elif raw_kind in {"const", "plain", "0.1"}:
        kinds = ["const"]
    else:
        raise typer.BadParameter("kind must be decay, const, or both")

    from personalization.report import write_milestone8_report

    if report_only:
        write_milestone8_report()
        return

    recipe = load_best_recipe(recipe_json)
    precision = str(recipe.get("precision", "bf16"))
    epochs = int(recipe.get("epochs", 30))
    days_grid = _parse_days_grid(days)
    status: dict[str, Any] = {
        "device": device,
        "current": None,
        "jobs": [],
    }

    def refresh() -> None:
        write_milestone8_report(status=status)
        _write_status(status)

    safe_echo(
        f"Independent LwF overnight: {len(selected)} person(s), kinds={kinds}, "
        f"teacher={base_run_dir}"
    )
    safe_echo("Re-invoke after Ctrl+C; completed day budgets are skipped.")

    for person in selected:
        for item in kinds:
            status["current"] = f"{person.name}/{item}"
            _write_status(status)
            rows = run_independent_lwf_sweep(
                person=person,
                kind=item,
                base_run_dir=base_run_dir,
                recipe=recipe,
                days_grid=days_grid,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                device=device,
                precision=precision,
                skip_completed=skip_completed,
                dry_run=dry_run,
                plot=True,
                on_progress=lambda _s, _r: refresh(),
            )
            completed = [str(r.get("personal_days")) for r in rows if r.get("status") == "ok"]
            status["jobs"].append(
                {
                    "person": person.name,
                    "kind": item,
                    "completed_days": completed,
                    "n_ok": len(completed),
                }
            )
            refresh()

    status["current"] = "done"
    refresh()
    safe_echo("Independent LwF sweep finished (or dry-run complete).")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
