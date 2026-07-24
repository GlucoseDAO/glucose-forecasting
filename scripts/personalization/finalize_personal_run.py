#!/usr/bin/env python3
"""Finalize a personalization run after training stopped (eval + metrics only)."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import torch
import typer

from scripts.common.checkpoint import strip_compile_prefix
from scripts.common.console import init_cli_console, safe_echo
from scripts.common.data_loading import resolve_num_workers
from scripts.personalization.constants import SUGAR_ONE_VALUE_COLUMNS
from scripts.personalization.finetune import (
    _eval_split,
    _load_saved_run_config,
    _load_split_frames,
    _metrics_dict,
)
from scripts.personalization.registry import load_base_checkpoint
from scripts.sugar_one.train_sugar_one import SugarOneWindowDataset

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _read_overall_csv(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    df = pl.read_csv(path)
    if df.is_empty():
        return None
    row = df.row(0, named=True)
    return _metrics_dict(float(row["mae"]), float(row["rmse"]), float(row["mard"]))


def finalize_personal_run(run_dir: Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load ``best_model.pt``, run val/test eval, write ``personalization_metrics.json``."""
    run_dir = run_dir.resolve()
    cfg = _load_saved_run_config(run_dir)
    best_path = run_dir / "best_model.pt"
    if not best_path.is_file():
        raise ValueError(f"best_model.pt not found in {run_dir}")

    personal_csv = Path(cfg["personal_csv"])
    base_run_dir = Path(cfg["base_run_dir"])
    personal_days = cfg.get("personal_days")
    if personal_days is not None:
        personal_days = int(personal_days)

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model, base_meta, resolved_type, _ckpt_path = load_base_checkpoint(
        base_run_dir, model_type=cfg.get("model_type"), device=torch_device
    )
    state = torch.load(best_path, map_location=torch_device, weights_only=True)
    model.load_state_dict(strip_compile_prefix(state))

    input_steps = int(cfg.get("input_steps", base_meta.get("input_steps", 128)))
    horizon = int(cfg.get("horizon", base_meta.get("horizon", 12)))
    batch_size = int(cfg.get("batch_size", 256))
    train_window_stride = int(cfg.get("train_window_stride", 1))
    workers = resolve_num_workers(int(cfg.get("num_workers", 0)), torch_device)
    precision = str(cfg.get("precision", "fp32"))
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    p_train_full, p_val, p_test = _load_split_frames(personal_csv)
    # Match finetune.py: scalers from full train; day limit only for train windows.
    scaler_ds = SugarOneWindowDataset(
        p_train_full, input_steps, horizon, fit_scalers=True, window_stride=1
    )
    p_train = p_train_full
    if personal_days is not None:
        t0 = p_train_full.select(pl.col("ds").min()).item()
        t_end = t0 + timedelta(days=personal_days)
        p_train = p_train_full.filter(pl.col("ds") < t_end)

    def _make_ds(df: pl.DataFrame, *, window_stride: int) -> SugarOneWindowDataset | None:
        if df.is_empty():
            return None
        return SugarOneWindowDataset(
            df,
            input_steps,
            horizon,
            scaler_glucose=scaler_ds.scaler_glucose,
            scaler_basal=scaler_ds.scaler_basal,
            scaler_bolus=scaler_ds.scaler_bolus,
            scaler_carbs=scaler_ds.scaler_carbs,
            window_stride=window_stride,
        )

    personal_val_ds = _make_ds(p_val, window_stride=1)
    personal_test_ds = _make_ds(p_test, window_stride=1)

    t0 = time.perf_counter()
    safe_echo(f"Finalizing {run_dir.name} on {torch_device} (best_model.pt epoch {cfg.get('best_epoch', '?')})")

    results: dict[str, Any] = {"config": cfg}
    zs = _read_overall_csv(run_dir / "zero_shot_test_metrics_overall.csv")
    if zs is not None:
        results["zero_shot_test"] = zs

    results["finetuned_val"] = _eval_split(
        model,
        personal_val_ds,
        scaler_ds,
        torch_device,
        batch_size,
        workers,
        run_dir,
        "val",
        use_amp,
        amp_dtype,
    )
    results["finetuned_test"] = _eval_split(
        model,
        personal_test_ds,
        scaler_ds,
        torch_device,
        batch_size,
        workers,
        run_dir,
        "test",
        use_amp,
        amp_dtype,
    )

    wall_time_s = float(cfg.get("wall_time_s", 0) or 0) + (time.perf_counter() - t0)
    results["wall_time_s"] = wall_time_s
    cfg["wall_time_s"] = wall_time_s
    cfg["end_time"] = datetime.now().isoformat()
    cfg["value_columns"] = dict(SUGAR_ONE_VALUE_COLUMNS)

    with (run_dir / "personalization_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with (run_dir / "tuning_meta.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    ft_test = results.get("finetuned_test")
    if isinstance(ft_test, dict):
        safe_echo(f"Done. test MAE={ft_test['mae']:.4f}")
    return results


@app.command()
def main(
    run_dir: Path = typer.Argument(..., help="Run directory containing best_model.pt"),
    device: str = typer.Option("cpu", "--device"),
) -> None:
    """Write personalization_metrics.json from an existing best checkpoint."""
    init_cli_console()
    finalize_personal_run(run_dir, device=device)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
