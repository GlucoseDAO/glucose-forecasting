#!/usr/bin/env python3
"""TOML-driven personalization fine-tune runner with leaderboard.

Control parameter grids or explicit [[runs]] in a config file; track completed
combinations in state.json and leaderboard.csv (same pattern as tune_sugar_one).

Default config: src/personalization/personalization_tune.toml

Examples:
  uv run tune-personal
  uv run tune-personal -c src/personalization/personalization_tune_window_stride.toml
  uv run tune-personal --list
  uv run tune-personal --dry-run
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import tomllib
import typer

from common.console import init_cli_console, safe_echo
from personalization.constants import DEFAULT_PERSONAL_LWF_LAMBDA, DEFAULT_TRAIN_WINDOW_STRIDE
from personalization.finetune import run_finetune
from personalization.leaderboard import (
    STATE_VERSION,
    STATUS_FAILED,
    STATUS_RUNNING,
    atomic_write_json,
    build_run_combos,
    combo_hash,
    completed_hashes,
    finalize_trial_from_results,
    find_resume_checkpoint,
    grid_combo_hashes,
    import_existing_runs,
    load_json_if_exists,
    resolve_path,
    trial_record_from_run,
    write_leaderboard_csv,
)

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="Run personalization fine-tunes from TOML; track trials in leaderboard.csv.",
)

DEFAULT_CONFIG_FILENAME = "personalization_tune.toml"
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _SCRIPT_DIR / DEFAULT_CONFIG_FILENAME


def resolve_config_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise typer.BadParameter(f"Config file not found: {path}")
        return path
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    raise typer.BadParameter(
        f"Config not found. Pass --config PATH or place {DEFAULT_CONFIG_FILENAME} in {_SCRIPT_DIR}."
    )


def load_user_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    cfg = tomllib.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid TOML root in {path}")
    return cfg


def _run_name_for_combo(params: dict[str, Any], index: int) -> str:
    if params.get("name"):
        return str(params["name"])
    parts: list[str] = []
    lwf = float(params.get("lwf_lambda", DEFAULT_PERSONAL_LWF_LAMBDA))
    if lwf > 0.0:
        parts.append(f"lwf{lwf:g}")
    parts.append(f"lr{params.get('lr', 0):g}")
    stride = params.get("train_window_stride", DEFAULT_TRAIN_WINDOW_STRIDE)
    if stride != 1:
        parts.append(f"stride{stride}")
    days = params.get("personal_days")
    if days is not None:
        parts.append(f"d{days}")
    return f"run_{index:03d}_{'_'.join(parts)}"


def _existing_running_trial(
    trials: list[dict[str, Any]],
    combo_hash_val: str,
) -> dict[str, Any] | None:
    for trial in trials:
        if trial.get("combo_hash") == combo_hash_val and trial.get("status") == STATUS_RUNNING:
            return trial
    return None


def _print_leaderboard(leaderboard_path: Path) -> None:
    if not leaderboard_path.exists():
        safe_echo(f"No leaderboard yet: {leaderboard_path}")
        return
    text = leaderboard_path.read_text(encoding="utf-8")
    safe_echo(f"Leaderboard: {leaderboard_path}")
    safe_echo(text.rstrip())


def _ensure_state(state_path: Path, config_path: Path) -> dict[str, Any]:
    existing = load_json_if_exists(state_path)
    if existing is not None:
        return existing
    state = {
        "version": STATE_VERSION,
        "config_path": str(config_path.resolve()),
        "trials": [],
    }
    atomic_write_json(state_path, state)
    return state


@app.command()
def main(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="TOML config (default: src/personalization/personalization_tune.toml).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show pending/completed combos without training.",
    ),
    list_only: bool = typer.Option(
        False,
        "--list",
        help="Print leaderboard.csv and exit.",
    ),
    import_existing: bool = typer.Option(
        True,
        "--import-existing/--no-import-existing",
        help="Import finished runs from output_dir into state/leaderboard.",
    ),
    max_runs: int = typer.Option(
        0,
        "--max-runs",
        help="Max new runs this invocation (0 = all pending).",
    ),
) -> None:
    """Run fine-tunes from TOML config; update state.json + leaderboard.csv."""
    init_cli_console()
    config_path = resolve_config_path(config)
    user_cfg = load_user_config(config_path)

    paths = user_cfg["paths"]
    tune = user_cfg.get("tune", {})
    out_root = resolve_path(str(paths["output_dir"]))
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "state.json"
    leaderboard_path = out_root / "leaderboard.csv"

    combos = build_run_combos(user_cfg)
    active_hashes = grid_combo_hashes(user_cfg)

    if list_only:
        state = _ensure_state(state_path, config_path)
        trials = list(state.get("trials", []))
        if import_existing:
            n_imported = import_existing_runs(out_root, trials)
            if n_imported:
                state["trials"] = trials
                atomic_write_json(state_path, state)
                safe_echo(f"Imported {n_imported} existing run(s).")
        write_leaderboard_csv(leaderboard_path, trials, active_combo_hashes=active_hashes)
        _print_leaderboard(leaderboard_path)
        return

    state = _ensure_state(state_path, config_path)
    trials: list[dict[str, Any]] = list(state.get("trials", []))

    if import_existing:
        n_imported = import_existing_runs(out_root, trials)
        if n_imported:
            safe_echo(f"Imported {n_imported} existing run(s) into state.")
            state["trials"] = trials
            atomic_write_json(state_path, state)
            write_leaderboard_csv(
                leaderboard_path, trials, active_combo_hashes=active_hashes
            )

    done = completed_hashes(trials, active_combo_hashes=active_hashes)
    skip_completed = bool(tune.get("skip_completed", True))

    pending: list[tuple[int, dict[str, Any]]] = []
    for i, params in enumerate(combos, start=1):
        ch = combo_hash(params)
        if skip_completed and ch in done:
            continue
        pending.append((i, params))

    safe_echo(f"Config: {config_path}")
    safe_echo(f"Output: {out_root}")
    safe_echo(f"Total combos in config: {len(combos)}")
    safe_echo(f"Already completed: {len(done)}")
    safe_echo(f"Pending: {len(pending)}")

    if dry_run:
        safe_echo("\n--- Pending runs ---")
        for i, params in pending:
            safe_echo(f"  [{i}] hash={combo_hash(params)} {_run_name_for_combo(params, i)}")
        safe_echo("\n--- Completed (leaderboard) ---")
        write_leaderboard_csv(
            leaderboard_path, trials, active_combo_hashes=active_hashes
        )
        _print_leaderboard(leaderboard_path)
        return

    if not pending:
        safe_echo("Nothing to run. See leaderboard:")
        write_leaderboard_csv(
            leaderboard_path, trials, active_combo_hashes=active_hashes
        )
        _print_leaderboard(leaderboard_path)
        return

    limit = max_runs if max_runs > 0 else len(pending)
    started = 0

    for i, params in pending:
        if started >= limit:
            break
        run_name = _run_name_for_combo(params, i)
        ch = combo_hash(params)
        record = _existing_running_trial(trials, ch)
        if record is None:
            next_index = max((int(t.get("run_index", 0)) for t in trials), default=0) + 1
            record = trial_record_from_run(run_index=next_index, params=params, run_name=run_name)
            trials.append(record)
        else:
            record["run_name"] = run_name
            safe_echo(f"Resuming interrupted trial: {run_name}")
        state["trials"] = trials
        atomic_write_json(state_path, state)

        resume_ckpt = find_resume_checkpoint(out_root, params)
        safe_echo(
            f"\n===== run {record['run_index']}: {run_name} "
            f"(hash={record['combo_hash']}) ====="
        )
        if resume_ckpt is not None:
            safe_echo(f"Resume checkpoint: {resume_ckpt}")
        try:
            run_dir, results = run_finetune(
                base_run_dir=resolve_path(str(params["base_run_dir"])),
                personal_csv=resolve_path(str(params["personal_csv"])),
                out_dir=out_root,
                run_name=run_name,
                personal_days=params.get("personal_days"),
                lwf_lambda=float(params.get("lwf_lambda", DEFAULT_PERSONAL_LWF_LAMBDA)),
                lr=float(params["lr"]) if params.get("lr") is not None else None,
                weight_decay=float(params["weight_decay"])
                if params.get("weight_decay") is not None
                else None,
                patience=int(params["patience"]) if params.get("patience") is not None else None,
                val_every_n_epochs=int(params["val_every_n_epochs"])
                if params.get("val_every_n_epochs") is not None
                else None,
                epochs=int(params.get("epochs", 30)),
                batch_size=int(params.get("batch_size", 256)),
                train_window_stride=int(
                    params.get("train_window_stride", DEFAULT_TRAIN_WINDOW_STRIDE)
                ),
                seed=int(params.get("seed", 43)),
                device=str(params.get("device", "cpu")),
                precision=str(params.get("precision", "fp32")),
                num_workers=int(params.get("num_workers", -1)),
                eval_zero_shot=bool(params.get("eval_zero_shot", True)),
                resume_from=resume_ckpt,
            )
            finalize_trial_from_results(record, run_dir=run_dir, results=results)
        except ValueError as exc:
            record["status"] = STATUS_FAILED
            record["error"] = str(exc)
            record["finished_at"] = record.get("started_at")
            safe_echo(f"FAILED: {exc}", err=True)
        except Exception as exc:
            record["status"] = STATUS_FAILED
            record["error"] = f"{exc}\n{traceback.format_exc()}"
            record["finished_at"] = record.get("started_at")
            safe_echo(f"FAILED: {exc}", err=True)

        state["trials"] = trials
        atomic_write_json(state_path, state)
        write_leaderboard_csv(
            leaderboard_path, trials, active_combo_hashes=active_hashes
        )
        started += 1

    safe_echo(f"\nLeaderboard: {leaderboard_path}")
    write_leaderboard_csv(leaderboard_path, trials, active_combo_hashes=active_hashes)
    _print_leaderboard(leaderboard_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
