#!/usr/bin/env python3
"""
Random-search tuner for GluMindIC (global training only).

One code path; behaviour comes entirely from the TOML config file.
Shipped configs: tune_glumind_ic_dev.toml (laptop) and tune_glumind_ic_full.toml (production).
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import tomllib
import torch
import typer

from scripts.glumind_ic import train_glumind_ic as tg
from scripts.glumind_ic.console_log import echo_plain

app = typer.Typer(
    name="tune-glumind-ic",
    add_completion=False,
    help="Random hyperparameter search for GluMindIC (global mode).",
)

STATE_VERSION = 1

# Default config when --config is omitted: file next to this script, else cwd fallback.
DEFAULT_CONFIG_FILENAME = "tune_glumind_ic_dev.toml"
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
    cwd_fallback = Path.cwd() / "scripts" / "glumind_ic" / DEFAULT_CONFIG_FILENAME
    if cwd_fallback.is_file():
        return cwd_fallback.resolve()
    raise typer.BadParameter(
        f"Config not found. Pass --config PATH or place {DEFAULT_CONFIG_FILENAME} in "
        f"{_SCRIPT_DIR} (or scripts/glumind_ic/ under the repo root)."
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def canonical_params_for_hash(params: Mapping[str, Any]) -> dict[str, Any]:
    """Stable dict for hashing (sorted keys, rounded floats)."""
    out: dict[str, Any] = {}
    for key in sorted(params):
        val = params[key]
        if isinstance(val, float):
            out[key] = round(val, 12)
        elif isinstance(val, bool):
            out[key] = val
        elif isinstance(val, int):
            out[key] = val
        elif isinstance(val, str):
            out[key] = val
        else:
            out[key] = val
    return out


def combo_hash(params: Mapping[str, Any]) -> str:
    blob = json.dumps(canonical_params_for_hash(params), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_user_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    cfg = tomllib.loads(raw.decode("utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid TOML root in {path}")
    return cfg


def resolve_csv_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def prepare_frames(
    csv_path: Path,
    *,
    unique_id: str,
    drop_interpolated: bool,
    study_groups: str,
    split_scheme: str,
    max_train_series: int,
    max_eval_series: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df, val_df, test_df = tg.load_splits_streaming(
        csv_path, unique_id, drop_interpolated
    )
    train_df = tg.normalize_study_groups_column(train_df)
    val_df = tg.normalize_study_groups_column(val_df)
    test_df = tg.normalize_study_groups_column(test_df)

    if study_groups.strip():
        group_list = [
            tg.normalize_study_group_label(g.strip())
            for g in study_groups.split(",")
            if g.strip()
        ]
        train_df = train_df.filter(pl.col("study_group").is_in(group_list))
        val_df = val_df.filter(pl.col("study_group").is_in(group_list))
        test_df = test_df.filter(pl.col("study_group").is_in(group_list))

    train_df, val_df, test_df = tg.apply_split_scheme(train_df, val_df, test_df, split_scheme)

    if max_train_series > 0:
        train_df = tg.limit_series(train_df, max_train_series)
    if max_eval_series > 0:
        val_df = tg.limit_series(val_df, max_eval_series)
        test_df = tg.limit_series(test_df, max_eval_series)

    train_df = tg.impute_and_sort(train_df)
    val_df = tg.impute_and_sort(val_df)
    test_df = tg.impute_and_sort(test_df)
    return train_df, val_df, test_df


def make_device(device_name: str) -> torch.device:
    if device_name == "mps" and not torch.backends.mps.is_available():
        echo_plain("MPS not available, falling back to CPU.")
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        echo_plain("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device


def setup_cuda_flags(disable_tf32: bool, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        echo_plain("TF32 enabled.")


def read_split_metrics(run_dir: Path, split: str) -> dict[str, float] | None:
    path = run_dir / f"{split}_metrics_overall.csv"
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if df.is_empty():
        return None
    row = df.row(0, named=True)
    return {
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "mard": float(row["mard"]),
    }


def sample_from_space(rng: random.Random, space: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items()}


def derive_rng(master_seed: int, draw_index: int) -> random.Random:
    """Deterministic RNG stream slot — safe for resume via persisted next_draw_index."""
    digest = hashlib.sha256(f"{master_seed}:{draw_index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def merge_defaults_and_sample(
    defaults: Mapping[str, Any],
    sampled: Mapping[str, Any],
) -> dict[str, Any]:
    """Config [defaults], then random draw from [tune.space] (sample wins on overlap)."""
    return {**dict(defaults), **dict(sampled)}


def format_trial_params(params: Mapping[str, Any]) -> str:
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return " | ".join(parts)


def runtime_from_tune_section(tune: Mapping[str, Any]) -> dict[str, Any]:
    """Training loop scalars from [tune] (everything except nested [tune.space])."""
    return {
        "epochs": int(tune["epochs"]),
        "patience": int(tune["patience"]),
        "log_every": int(tune["log_every"]),
        "val_every_n_epochs": int(tune["val_every_n_epochs"]),
        "ckpt_every_n_epochs": int(tune["ckpt_every_n_epochs"]),
        "precision": str(tune["precision"]),
        "compile_mode": str(tune["compile_mode"]),
        "disable_tf32": bool(tune["disable_tf32"]),
        "num_workers": int(tune["num_workers"]),
        "prefetch_factor": int(tune["prefetch_factor"]),
        "batch_log_every": int(tune.get("batch_log_every", 0)),
        "eval_batch_log_every": int(tune.get("eval_batch_log_every", 0)),
    }


def build_train_cfg(
    *,
    csv_resolved: Path,
    dataset_cfg: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
    device_name: str,
    seed: int,
    out_dir: Path,
    resume_from: str,
) -> dict[str, Any]:
    p = params
    r = runtime
    return {
        "csv": str(csv_resolved),
        "unique_id": dataset_cfg["unique_id"],
        "drop_interpolated": dataset_cfg["drop_interpolated"],
        "study_groups": dataset_cfg["study_groups"],
        "split_scheme": dataset_cfg["split_scheme"],
        "mode": "global",
        "horizon": int(p["horizon"]),
        "input_steps": int(p["input_steps"]),
        "d_model": int(p["d_model"]),
        "n_heads": int(p["n_heads"]),
        "n_blocks": int(p["n_blocks"]),
        "ff_units": int(p["ff_units"]),
        "dropout": float(p["dropout"]),
        "epochs": int(r["epochs"]),
        "batch_size": int(p["batch_size"]),
        "precision": str(r["precision"]),
        "compile_mode": str(r["compile_mode"]),
        "disable_tf32": bool(r["disable_tf32"]),
        "num_workers": int(r["num_workers"]),
        "prefetch_factor": int(r["prefetch_factor"]),
        "lr": float(p["lr"]),
        "weight_decay": float(p["weight_decay"]),
        "patience": int(r["patience"]),
        "log_every": int(r["log_every"]),
        "ckpt_every_n_epochs": int(r["ckpt_every_n_epochs"]),
        "val_every_n_epochs": int(r["val_every_n_epochs"]),
        "batch_log_every": int(r.get("batch_log_every", 0)),
        "eval_batch_log_every": int(r.get("eval_batch_log_every", 0)),
        "resume_from": resume_from,
        "lwf_lambda": 0.5,
        "continual_order": "default",
        "continual_val_scope": "current_group",
        "device": device_name,
        "seed": seed,
        "out_dir": str(out_dir),
    }


def run_one_global_trial(
    *,
    cfg: dict[str, Any],
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    device: torch.device,
    run_name: str,
    out_root: Path,
) -> Path:
    t_ds = time.perf_counter()
    train_ds, val_ds, test_ds = tg.build_datasets(
        train_df, val_df, test_df, cfg["input_steps"], cfg["horizon"]
    )
    echo_plain(
        f"  Dataset windows built in {time.perf_counter() - t_ds:.1f}s | "
        f"train={len(train_ds):,} val={len(val_ds) if val_ds else 0:,} "
        f"test={len(test_ds) if test_ds else 0:,}"
    )
    model = tg.make_model(**tg._model_kwargs(cfg), device=device)
    tg.run_train_and_eval(
        model,
        train_ds,
        val_ds,
        test_ds,
        cfg,
        device,
        run_name,
        out_root,
    )
    return out_root / run_name


def trial_params_from_record(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Merged hyperparameters (legacy state used separate capacity_fixed)."""
    params = dict(trial.get("params", {}))
    params.update(trial.get("capacity_fixed", {}))
    return params


def leaderboard_row_from_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    params = trial_params_from_record(trial)
    row: dict[str, Any] = {
        "trial_index": int(trial["trial_index"]),
        "combo_hash": str(trial["combo_hash"]),
        "val_mae": float(trial["val_mae"]),
        "val_rmse": float(trial["val_rmse"]),
        "val_mard": float(trial["val_mard"]),
        "seconds": float(trial["duration_seconds"]),
    }
    row.update(params)
    return row


def write_leaderboard_csv(path: Path, trials_ok: list[Mapping[str, Any]]) -> None:
    """Rewrite leaderboard from all successful trials (sorted by val_mae)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trials_ok:
        if path.exists():
            path.unlink()
        return
    rows = [leaderboard_row_from_trial(t) for t in trials_ok]
    pl.DataFrame(rows).sort("val_mae").write_csv(path)


def marginal_counts(top_trials: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key in keys:
        counts: dict[str, int] = {}
        for t in top_trials:
            raw = trial_params_from_record(t).get(key)
            label = str(raw)
            counts[label] = counts.get(label, 0) + 1
        out[key] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return out


def write_tune_report(
    *,
    report_path: Path,
    config_path: Path,
    trials_ok: list[dict[str, Any]],
    timing_stats: dict[str, Any],
    report_param_keys: list[str] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# GluMindIC tuning report")
    lines.append("")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append(f"- Successful trials: **{len(trials_ok)}**")
    lines.append(
        f"- Total wall time (success): **{timing_stats.get('total_seconds_success', 0.0):.1f}s** "
        f"| mean **{timing_stats.get('mean_seconds_per_trial', 0.0):.1f}s** / trial"
    )
    lines.append("")
    lines.append("## Leaderboard (sorted by val MAE)")
    lines.append("")
    if trials_ok:
        if report_param_keys is None:
            report_param_keys = sorted(
                {k for t in trials_ok for k in t.get("params", {}).keys()}
            )
        rows = []
        for t in sorted(trials_ok, key=lambda x: float(x["val_mae"])):
            row: dict[str, Any] = {
                "trial": t["trial_index"],
                "val_mae": float(t["val_mae"]),
                "val_rmse": float(t["val_rmse"]),
                "val_mard": float(t["val_mard"]),
                "seconds": round(float(t["duration_seconds"]), 2),
            }
            for key in report_param_keys:
                row[key] = trial_params_from_record(t).get(key)
            rows.append(row)
        buf = io.StringIO()
        pl.DataFrame(rows).write_csv(buf)
        lines.append("```csv")
        lines.append(buf.getvalue().rstrip("\n"))
        lines.append("```")
    else:
        lines.append("_No successful trials yet._")
    lines.append("")
    lines.append("## Parameter prioritization (top quartile of successful trials)")
    lines.append("")
    sorted_ok = sorted(trials_ok, key=lambda x: float(x["val_mae"]))
    q = max(1, len(sorted_ok) // 4)
    top_q = sorted_ok[:q]
    keys = report_param_keys if report_param_keys else []
    if top_q and keys:
        marg = marginal_counts(top_q, keys)
        lines.append(
            "Counts show how often each discrete value appeared among the "
            f"best **{len(top_q)}** runs."
        )
        lines.append("")
        for key in keys:
            lines.append(f"### `{key}`")
            lines.append("")
            for val, cnt in marg[key].items():
                lines.append(f"- `{val}` -> **{cnt}**")
            lines.append("")
    else:
        lines.append("_No data._")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def timing_from_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [t for t in trials if t.get("status") == "ok"]
    total = sum(float(t["duration_seconds"]) for t in ok)
    n = len(ok)
    mean = total / n if n else 0.0
    return {
        "total_seconds_success": total,
        "mean_seconds_per_trial": mean,
        "count_success": n,
    }


def tune_loop(
    *,
    user_cfg: dict[str, Any],
    config_path: Path,
    device_name: str,
    seed_override: int | None,
) -> None:
    paths = user_cfg["paths"]
    dataset = user_cfg["dataset"]
    tune = user_cfg["tune"]
    space_raw = tune["space"]
    param_defaults: dict[str, Any] = dict(user_cfg.get("defaults", {}))
    runtime = runtime_from_tune_section(tune)
    resume_from = str(tune.get("resume_from", ""))

    csv_path = resolve_csv_path(str(paths["csv"]))
    out_root = resolve_csv_path(str(paths["output_dir"]))
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "state.json"
    leaderboard_path = out_root / "leaderboard.csv"
    report_path = out_root / "tune_report.md"

    space: dict[str, list[Any]] = {k: list(v) for k, v in space_raw.items()}
    if not space:
        raise typer.BadParameter("[tune.space] must not be empty.")

    report_param_keys = sorted({*space.keys(), *param_defaults.keys()})
    n_trials_target = int(tune["n_trials"])
    max_draws = int(tune["max_random_draws"])
    rng_seed = int(seed_override if seed_override is not None else tune["random_seed"])

    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)

    device = make_device(device_name)
    setup_cuda_flags(bool(tune["disable_tf32"]), device)

    echo_plain(f"Config: {config_path}")
    echo_plain(f"Device: {device}")
    echo_plain(f"CSV: {csv_path}")
    echo_plain(f"Output: {out_root}")
    searched = sorted(space.keys())
    fixed_keys = sorted(k for k in param_defaults if k not in space)
    echo_plain(f"Search space ({len(searched)} keys): {', '.join(searched)}")
    if fixed_keys:
        echo_plain(f"Defaults (not in .space): {', '.join(fixed_keys)}")

    train_df, val_df, test_df = prepare_frames(
        csv_path,
        unique_id=str(dataset["unique_id"]),
        drop_interpolated=bool(dataset["drop_interpolated"]),
        study_groups=str(dataset["study_groups"]),
        split_scheme=str(dataset["split_scheme"]),
        max_train_series=int(dataset["max_train_series"]),
        max_eval_series=int(dataset["max_eval_series"]),
    )
    echo_plain(
        f"Frames: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,} rows"
    )

    state = load_json_if_exists(state_path)
    if state is None:
        state = {
            "version": STATE_VERSION,
            "config_path": str(config_path.resolve()),
            "random_seed": rng_seed,
            "target_trials": n_trials_target,
            "trials": [],
        }
        atomic_write_json(state_path, state)

    trials: list[dict[str, Any]] = state["trials"]
    tried_hashes = {str(t["combo_hash"]) for t in trials}

    successes = [t for t in trials if t.get("status") == "ok"]
    if len(successes) >= n_trials_target:
        echo_plain(
            f"Already completed {len(successes)} >= target {n_trials_target}. Regenerating report."
        )
        stats = timing_from_trials(trials)
        write_leaderboard_csv(leaderboard_path, successes)
        write_tune_report(
            report_path=report_path,
            config_path=config_path,
            trials_ok=successes,
            timing_stats=stats,
            report_param_keys=report_param_keys,
        )
        echo_plain(f"Report: {report_path}")
        echo_plain(f"Leaderboard: {leaderboard_path}")
        return

    attempts = 0
    next_index = max((int(t["trial_index"]) for t in trials), default=-1) + 1
    draw_idx = int(state.get("next_draw_index", 0))

    while True:
        successes = [t for t in trials if t.get("status") == "ok"]
        if len(successes) >= n_trials_target:
            break
        if attempts >= max_draws:
            break
        attempts += 1

        trial_rng = derive_rng(rng_seed, draw_idx)
        draw_idx += 1
        sampled = sample_from_space(trial_rng, space)
        trial_params = merge_defaults_and_sample(param_defaults, sampled)
        ch = combo_hash(trial_params)
        if ch in tried_hashes:
            state["next_draw_index"] = draw_idx
            atomic_write_json(state_path, state)
            continue
        tried_hashes.add(ch)

        cfg = build_train_cfg(
            csv_resolved=csv_path,
            dataset_cfg={
                "unique_id": dataset["unique_id"],
                "drop_interpolated": dataset["drop_interpolated"],
                "study_groups": dataset["study_groups"],
                "split_scheme": dataset["split_scheme"],
            },
            params=trial_params,
            runtime=runtime,
            device_name=device_name,
            seed=rng_seed,
            out_dir=out_root,
            resume_from=resume_from,
        )

        short = ch[:8]
        run_name = f"trial_{next_index:04d}_{short}"
        echo_plain("")
        echo_plain("=" * 72)
        echo_plain(f"Trial {next_index} | hash={ch[:16]}... | {format_trial_params(trial_params)}")
        echo_plain("=" * 72)

        record: dict[str, Any] = {
            "trial_index": next_index,
            "combo_hash": ch,
            "params": trial_params,
            "run_name": run_name,
            "status": "failed",
            "error": None,
            "duration_seconds": 0.0,
            "val_mae": None,
            "val_rmse": None,
            "val_mard": None,
            "run_dir": str(out_root / run_name),
        }

        t0 = time.perf_counter()
        try:
            run_dir = run_one_global_trial(
                cfg=cfg,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                device=device,
                run_name=run_name,
                out_root=out_root,
            )
            metrics = read_split_metrics(run_dir, "val")
            elapsed = time.perf_counter() - t0
            if metrics is None:
                raise RuntimeError("Missing val_metrics_overall.csv after training.")
            record["status"] = "ok"
            record["duration_seconds"] = elapsed
            record["val_mae"] = metrics["mae"]
            record["val_rmse"] = metrics["rmse"]
            record["val_mard"] = metrics["mard"]

            stats = timing_from_trials(trials + [record])
            ok_n = len([t for t in trials + [record] if t.get("status") == "ok"])
            echo_plain(
                f"OK Done in {elapsed:.1f}s | val MAE={metrics['mae']:.4f} | "
                f"mean ok-trial time={stats['mean_seconds_per_trial']:.1f}s | "
                f"completed {ok_n}/{n_trials_target}"
            )
            eta_remain = stats["mean_seconds_per_trial"] * (n_trials_target - ok_n)
            unit = "min" if eta_remain < 7200 else "h"
            eta_val = eta_remain / 60 if unit == "min" else eta_remain / 3600
            echo_plain(f"  ETA (~remaining trials): {eta_val:.1f} {unit}")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            record["duration_seconds"] = elapsed
            record["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            echo_plain(f"FAIL Trial failed after {elapsed:.1f}s: {exc}")

        trials.append(record)
        state["trials"] = trials
        state["next_draw_index"] = draw_idx
        atomic_write_json(state_path, state)

        successes = [t for t in trials if t.get("status") == "ok"]
        write_leaderboard_csv(leaderboard_path, successes)
        stats_now = timing_from_trials(trials)
        write_tune_report(
            report_path=report_path,
            config_path=config_path,
            trials_ok=successes,
            timing_stats=stats_now,
            report_param_keys=report_param_keys,
        )
        echo_plain(f"  Updated leaderboard + report: {leaderboard_path}")

        next_index += 1

    successes = [t for t in trials if t.get("status") == "ok"]
    if len(successes) < n_trials_target:
        echo_plain(
            f"\nStopped: max_random_draws={max_draws} attempts exhausted "
            f"with only {len(successes)}/{n_trials_target} successes."
        )
        raise typer.Exit(code=1)

    echo_plain("\nTuning complete.")
    echo_plain(f"Report: {report_path}")
    echo_plain(f"Leaderboard: {leaderboard_path}")


@app.callback(invoke_without_command=True)
def cli(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help=f"TOML config (default: {DEFAULT_CONFIG_FILENAME} next to this script).",
    ),
    device_name: str = typer.Option(
        "cuda",
        "--device",
        help="cuda | cpu | mps",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Override [tune].random_seed.",
    ),
) -> None:
    """Random hyperparameter search for GluMindIC (global mode)."""
    config_path = resolve_config_path(config)
    user_cfg = load_user_config(config_path)
    if "tune" not in user_cfg or "space" not in user_cfg.get("tune", {}):
        raise typer.BadParameter("Config must define [tune] and [tune.space].")
    tune_loop(
        user_cfg=user_cfg,
        config_path=config_path,
        device_name=device_name,
        seed_override=seed,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
