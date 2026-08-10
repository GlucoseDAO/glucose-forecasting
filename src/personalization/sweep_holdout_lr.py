#!/usr/bin/env python3
"""Sweep learning rate on Loop holdout users and compare to Livia optimum.

After Livia Step-2 finds the best LR (currently 2e-4), run the same LR grid on
each holdout person to see whether the optimal LR transfers or diverges.

Default grid: ``0.0001, 0.0002, 0.0004`` (plain fine-tune, fixed wd, sparse stride).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from common.console import init_cli_console, safe_echo
from common.paths import DEFAULT_RUNS_ROOT
from personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_FT_PATIENCE,
    DEFAULT_HOLDOUT_LR_GRID,
    DEFAULT_PERSONAL_LWF_LAMBDA,
    DEFAULT_SEED,
    DEFAULT_TRAIN_WINDOW_STRIDE,
    DEFAULT_VAL_EVERY_N_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    HOLDOUT_LR_DEFERRED_USERS,
    HOLDOUT_LR_PILOT_USERS,
    LIVIA_REFERENCE_LR,
    LOOP_HOLDOUT_QUALITY_USERS,
)
from personalization.finetune import run_finetune
from personalization.leaderboard import find_resume_checkpoint
from personalization.sweep_utils import (
    build_holdout_lr_comparison,
    flatten_metrics,
    holdout_combo_out_dir,
    holdout_row_from_metrics,
    holdout_run_complete,
    holdout_run_dir,
    write_summary,
)
from personalization.validate_holdouts import _ensure_holdout_csv

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _parse_lr_grid(raw: str | None) -> list[float]:
    if raw is None:
        return list(DEFAULT_HOLDOUT_LR_GRID)
    return [float(p.strip()) for p in raw.split(",") if p.strip()]


def _holdout_params(
    *,
    base_run_dir: Path,
    personal_csv: Path,
    lwf: float,
    lr: float,
    weight_decay: float,
    patience: int,
    epochs: int,
    batch_size: int,
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
        "personal_days": None,
        "train_window_stride": DEFAULT_TRAIN_WINDOW_STRIDE,
        "val_every_n_epochs": DEFAULT_VAL_EVERY_N_EPOCHS,
        "precision": "fp32",
        "eval_zero_shot": True,
    }


def _collect_holdout_rows(
    *,
    out_dir: Path,
    user_ids: list[str],
    lr_values: list[float],
    lwf: float,
    weight_decay: float,
    patience: int,
    epochs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uid in user_ids:
        subject = f"loop_{uid}"
        for lr in lr_values:
            run_dir = holdout_run_dir(out_dir, subject, lr)
            if not holdout_run_complete(run_dir):
                continue
            row = holdout_row_from_metrics(
                run_dir,
                user_id=uid,
                subject=subject,
                lwf_lambda=lwf,
                weight_decay=weight_decay,
                patience=patience,
                epochs=epochs,
            )
            if row is not None:
                rows.append(row)
    return rows


def _write_sweep_status(
    *,
    out_dir: Path,
    pilot_users: list[str],
    deferred_users: list[str],
    lr_values: list[float],
    all_rows: list[dict[str, Any]],
    livia_reference_lr: float,
) -> None:
    completed_by_user: dict[str, list[float]] = {}
    for row in all_rows:
        if row.get("status") != "ok":
            continue
        uid = str(row["user_id"])
        completed_by_user.setdefault(uid, []).append(float(row["lr"]))

    status = {
        "livia_reference_lr": livia_reference_lr,
        "lr_grid": lr_values,
        "pilot_users": pilot_users,
        "deferred_users": deferred_users,
        "pilot_complete": {
            uid: sorted(completed_by_user.get(uid, [])) for uid in pilot_users
        },
        "deferred_pending": list(deferred_users),
        "note": "Deferred users reserved for follow-up after interim report.",
    }
    status_path = out_dir / "sweep_status.json"
    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    md_path = out_dir / "sweep_status.md"
    lines = [
        "# Holdout LR sweep status",
        "",
        f"**Livia reference LR:** {livia_reference_lr:g}",
        f"**LR grid:** `{lr_values}`",
        "",
        "## Pilot users (interim report)",
        "",
    ]
    for uid in pilot_users:
        done = completed_by_user.get(uid, [])
        state = "complete" if len(done) == len(lr_values) else f"{len(done)}/{len(lr_values)} LRs"
        lines.append(f"- **{uid}**: {state} — completed LRs: `{done}`")
    lines.extend(["", "## Deferred (next week)", ""])
    for uid in deferred_users:
        lines.append(f"- **{uid}**: not started")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _write_holdout_reports(
    *,
    all_rows: list[dict[str, Any]],
    out_dir: Path,
    livia_reference_lr: float,
    lr_values: list[float],
    lwf: float,
    weight_decay: float,
    patience: int,
    epochs: int,
    pilot_users: list[str],
    deferred_users: list[str],
) -> None:
    summary_path = write_summary(all_rows, out_dir, name="summary")
    comparison = build_holdout_lr_comparison(
        all_rows,
        livia_reference_lr=livia_reference_lr,
    )
    comparison_path = out_dir / "lr_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    notes_path = out_dir / "lr_divergence_notes.md"
    lines = [
        "# Holdout LR sweep — divergence from Livia",
        "",
        f"Livia reference LR (Step 2): **{livia_reference_lr:g}**",
        f"Holdout grid: `{lr_values}`",
        "",
        "## Per-user summary",
        "",
    ]
    n_same = 0
    n_lower = 0
    n_higher = 0
    for entry in comparison:
        div = entry["divergence"]
        if div == "same":
            n_same += 1
        elif div == "lower":
            n_lower += 1
        else:
            n_higher += 1
        lines.append(
            f"- **User {entry['user_id']}**: optimal LR **{entry['optimal_lr']:g}** "
            f"(ft_test_mae={entry['optimal_ft_test_mae']:.4f}) — "
            f"{entry['note']}"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Same as Livia: {n_same}/{len(comparison)}",
            f"- Lower than Livia: {n_lower}/{len(comparison)}",
            f"- Higher than Livia: {n_higher}/{len(comparison)}",
            "",
        ]
    )
    notes_path.write_text("\n".join(lines), encoding="utf-8")

    best_per_user_recipes: list[dict[str, Any]] = []
    for entry in comparison:
        best_per_user_recipes.append(
            {
                "user_id": entry["user_id"],
                "subject": entry["subject"],
                "lwf_lambda": lwf,
                "lr": entry["optimal_lr"],
                "weight_decay": weight_decay,
                "patience": patience,
                "epochs": epochs,
                "livia_reference_lr": livia_reference_lr,
                "lr_divergence": entry["divergence"],
                "ft_test_mae": entry["optimal_ft_test_mae"],
                "run_dir": entry.get("run_dir"),
            }
        )
    if best_per_user_recipes:
        recipe_path = out_dir / "best_recipe_per_user.json"
        with recipe_path.open("w", encoding="utf-8") as f:
            json.dump(best_per_user_recipes, f, indent=2)

    safe_echo(f"\nWrote {summary_path}")
    safe_echo(f"Wrote {comparison_path}")
    safe_echo(f"Wrote {notes_path}")
    _write_sweep_status(
        out_dir=out_dir,
        pilot_users=pilot_users,
        deferred_users=deferred_users,
        lr_values=lr_values,
        all_rows=all_rows,
        livia_reference_lr=livia_reference_lr,
    )
    safe_echo(f"Wrote {out_dir / 'sweep_status.json'}")
    safe_echo("\n--- LR divergence vs Livia ---")
    for entry in comparison:
        safe_echo(f"  user {entry['user_id']}: {entry['note']}")


@app.command()
def main(
    base_run_dir: Path = typer.Option(
        Path(DEFAULT_BASE_RUN_DIR),
        "--base-run-dir",
    ),
    loop_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop.csv"),
        "--loop-csv",
    ),
    holdout_dir: Path = typer.Option(
        Path("data/personalization/holdouts"),
        "--holdout-dir",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "holdout_lr_sweep",
        "--out-dir",
    ),
    livia_reference_lr: float = typer.Option(
        LIVIA_REFERENCE_LR,
        "--livia-lr",
        help="Livia optimal LR from Step-2 (reference for divergence notes).",
    ),
    lr_grid: Optional[str] = typer.Option(
        None,
        "--lr-grid",
        help="Comma-separated LR values (default: 0.0001,0.0002,0.0004).",
    ),
    users: Optional[str] = typer.Option(None, "--users"),
    pilot_only: bool = typer.Option(
        False,
        "--pilot-only",
        help=f"Only pilot users {HOLDOUT_LR_PILOT_USERS} (3 users for interim report).",
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Rebuild summary/notes from completed runs on disk; no training.",
    ),
    weight_decay: float = typer.Option(DEFAULT_WEIGHT_DECAY, "--weight-decay"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    test_fraction: float = typer.Option(0.25, "--test-fraction"),
    val_fraction_of_remainder: float = typer.Option(0.15, "--val-fraction-of-remainder"),
    skip_prepare: bool = typer.Option(False, "--skip-prepare"),
    skip_completed: bool = typer.Option(
        True,
        "--skip-completed/--no-skip-completed",
        help="Skip user×LR combos that already have finetuned_test metrics.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Fine-tune each holdout user across an LR grid; compare optimum to Livia."""
    init_cli_console()
    out_dir.mkdir(parents=True, exist_ok=True)
    lr_values = _parse_lr_grid(lr_grid)
    patience = DEFAULT_FT_PATIENCE
    lwf = DEFAULT_PERSONAL_LWF_LAMBDA

    user_list = (
        [u.strip() for u in users.split(",") if u.strip()]
        if users
        else list(HOLDOUT_LR_PILOT_USERS if pilot_only else LOOP_HOLDOUT_QUALITY_USERS)
    )
    pilot_users = list(HOLDOUT_LR_PILOT_USERS)
    deferred_users = list(HOLDOUT_LR_DEFERRED_USERS)

    safe_echo(f"Livia reference LR: {livia_reference_lr:g}")
    safe_echo(f"Holdout LR grid: {lr_values}")
    safe_echo(f"Users: {user_list}")
    if set(user_list).issubset(set(pilot_users)):
        safe_echo(f"Deferred for later: {deferred_users}")

    if report_only:
        all_rows = _collect_holdout_rows(
            out_dir=out_dir,
            user_ids=list(pilot_users),
            lr_values=lr_values,
            lwf=lwf,
            weight_decay=weight_decay,
            patience=patience,
            epochs=epochs,
        )
        _write_holdout_reports(
            all_rows=all_rows,
            out_dir=out_dir,
            livia_reference_lr=livia_reference_lr,
            lr_values=lr_values,
            lwf=lwf,
            weight_decay=weight_decay,
            patience=patience,
            epochs=epochs,
            pilot_users=pilot_users,
            deferred_users=deferred_users,
        )
        return

    pending: list[tuple[str, str, Path, float]] = []
    completed_rows: list[dict[str, Any]] = []

    for uid in user_list:
        subject = f"loop_{uid}"
        try:
            if skip_prepare:
                personal_csv = holdout_dir / f"loop_{uid}_chronological.csv"
                if not personal_csv.exists():
                    raise ValueError(f"Missing prepared CSV: {personal_csv}")
            else:
                if not loop_csv.exists():
                    raise ValueError(f"loop CSV not found: {loop_csv}")
                personal_csv = _ensure_holdout_csv(
                    loop_csv,
                    uid,
                    holdout_dir,
                    test_fraction,
                    val_fraction_of_remainder,
                )

            for lr in lr_values:
                run_dir = holdout_run_dir(out_dir, subject, lr)
                if skip_completed and holdout_run_complete(run_dir):
                    row = holdout_row_from_metrics(
                        run_dir,
                        user_id=uid,
                        subject=subject,
                        lwf_lambda=lwf,
                        weight_decay=weight_decay,
                        patience=patience,
                        epochs=epochs,
                    )
                    if row is not None:
                        completed_rows.append(row)
                    continue
                pending.append((uid, subject, personal_csv, lr))
        except (ValueError, FileNotFoundError) as exc:
            safe_echo(f"Skipping user {uid}: {exc}", err=True)

    safe_echo(f"Completed combos: {len(completed_rows)}")
    safe_echo(f"Pending combos: {len(pending)}")

    if dry_run:
        safe_echo("\n--- Pending ---")
        for uid, subject, _csv, lr in pending:
            combo_out = holdout_combo_out_dir(out_dir, subject, lr)
            resume_ckpt = find_resume_checkpoint(
                combo_out,
                _holdout_params(
                    base_run_dir=base_run_dir,
                    personal_csv=_csv,
                    lwf=lwf,
                    lr=lr,
                    weight_decay=weight_decay,
                    patience=patience,
                    epochs=epochs,
                    batch_size=batch_size,
                ),
            )
            resume_note = f" resume={resume_ckpt}" if resume_ckpt else ""
            safe_echo(f"  user={uid} lr={lr:g}{resume_note}")
        if completed_rows:
            _write_holdout_reports(
                all_rows=completed_rows,
                out_dir=out_dir,
                livia_reference_lr=livia_reference_lr,
                lr_values=lr_values,
                lwf=lwf,
                weight_decay=weight_decay,
                patience=patience,
                epochs=epochs,
                pilot_users=pilot_users,
                deferred_users=deferred_users,
            )
        return

    all_rows: list[dict[str, Any]] = list(completed_rows)

    for uid, subject, personal_csv, lr in pending:
        label = f"lr{lr:g}"
        combo_out = holdout_combo_out_dir(out_dir, subject, lr)
        safe_echo(f"\n===== holdout user={uid} {label} =====")
        params = _holdout_params(
            base_run_dir=base_run_dir,
            personal_csv=personal_csv,
            lwf=lwf,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            epochs=epochs,
            batch_size=batch_size,
        )
        resume_ckpt = find_resume_checkpoint(combo_out, params)
        if resume_ckpt is not None:
            safe_echo(f"  Resume checkpoint: {resume_ckpt}")
        try:
            run_dir, results = run_finetune(
                base_run_dir=base_run_dir,
                personal_csv=personal_csv,
                out_dir=combo_out,
                run_name=f"{subject}_{label}",
                personal_days=None,
                lwf_lambda=lwf,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                batch_size=batch_size,
                seed=seed,
                device=device,
                eval_zero_shot=True,
                resume_from=resume_ckpt,
            )
        except ValueError as exc:
            safe_echo(f"  Failed {label}: {exc}", err=True)
            all_rows.append(
                {
                    "user_id": uid,
                    "subject": subject,
                    "lr": lr,
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            continue

        all_rows.append(
            {
                "user_id": uid,
                "subject": subject,
                "lwf_lambda": lwf,
                "lr": lr,
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
        )

    _write_holdout_reports(
        all_rows=all_rows,
        out_dir=out_dir,
        livia_reference_lr=livia_reference_lr,
        lr_values=lr_values,
        lwf=lwf,
        weight_decay=weight_decay,
        patience=patience,
        epochs=epochs,
        pilot_users=pilot_users,
        deferred_users=deferred_users,
    )

    n_ok_users = len({r["user_id"] for r in all_rows if r.get("status") == "ok"})
    safe_echo(f"\nCompleted LR runs for {n_ok_users} user(s).")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
