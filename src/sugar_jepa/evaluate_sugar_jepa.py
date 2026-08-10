#!/usr/bin/env python3
"""
Standalone evaluation CLI for SugarJepa run directories.

Not wired into the unified `evaluate-model` CLI (src/sugar_one/evaluate_model.py)
— this is a deliberately separate script so the SugarJepa proof-of-concept
doesn't touch any existing, shared evaluation code. Reuses the same
model-reconstruction pattern as evaluate_model.py (src/common/registry.py)
and the training script's own dataset/eval/metrics functions (no need to
duplicate them — src/sugar_jepa/train_sugar_jepa.py already has an
`evaluate()` that handles this model's (x, glucose_jepa, y) batches).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
import typer

from common.data_loading import impute_and_sort as _common_impute_and_sort
from common.registry import load_run_meta, resolve_checkpoint, resolve_csv_path
from common.checkpoint import strip_compile_prefix
from common.scalers import SCALERS_FILENAME, load_scalers, resolve_scalers_path, save_scalers_for_run
from sugar_jepa.sugar_jepa_model import SugarJepaModel
from sugar_jepa.train_sugar_jepa import (
    SugarJepaWindowDataset,
    compute_and_print_metrics,
    evaluate,
    load_splits_streaming,
)

import torch.nn as nn
from torch.utils.data import DataLoader

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    run_dir: Path = typer.Option(..., "--run-dir", help="Run directory with tuning_meta.json/config.json + best_model.pt."),
    test_csv: Path = typer.Option(..., "--test-csv", help="CSV to evaluate on."),
    train_csv: Path | None = typer.Option(
        None, "--train-csv", help="CSV to fit scalers on when scalers.json is absent."
    ),
    refit_scalers: bool = typer.Option(
        False, "--refit-scalers", help="Ignore scalers.json and re-fit from train CSV."
    ),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Explicit .pt weights (default: best_model.pt)."),
    test_split: str = typer.Option("test", "--test-split", help="Recommended Split value to filter to; '' disables filtering."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Default: from the run's config."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Torch device for inference."),
) -> None:
    run_dir = run_dir.resolve()
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_run_meta(run_dir)
    ckpt_path = resolve_checkpoint(run_dir, checkpoint)

    typer.echo(f"Run directory: {run_dir}")
    typer.echo(f"Checkpoint   : {ckpt_path}")

    test_path = resolve_csv_path(test_csv, project_root)
    typer.echo(f"Test CSV  (evaluation): {test_path}")

    sidecar = None if refit_scalers else resolve_scalers_path(run_dir, cfg)
    if sidecar is not None:
        kind, scalers, _ = load_scalers(sidecar)
        if kind is not None and kind != "sugar_jepa":
            typer.echo(f"Error: scalers.json kind={kind!r}, expected sugar_jepa.", err=True)
            raise typer.Exit(1)
        for required in ("glucose", "basal", "bolus", "carbs", "glucose_jepa"):
            if required not in scalers:
                typer.echo(
                    f"Error: scalers.json missing feature {required!r}.",
                    err=True,
                )
                raise typer.Exit(1)
        typer.echo(f"Loaded scalers from: {sidecar}")
        scaler_glucose = scalers["glucose"]
        scaler_basal = scalers["basal"]
        scaler_bolus = scalers["bolus"]
        scaler_carbs = scalers["carbs"]
        scaler_glucose_jepa = scalers["glucose_jepa"]
    else:
        train_path = resolve_csv_path(train_csv or cfg["csv"], project_root)
        typer.echo(f"Train CSV (scaler fit): {train_path}")
        train_df, _, _ = load_splits_streaming(train_path, cfg["unique_id"], cfg["drop_interpolated"])
        train_df = _common_impute_and_sort(
            train_df, ffill_bfill_columns=["glucose", "basal"], zero_fill_columns=["bolus", "carbs"],
        )
        train_ds = SugarJepaWindowDataset(
            train_df, cfg["input_steps"], cfg["horizon"], cfg["jepa_window"], fit_scalers=True,
        )
        scaler_glucose = train_ds.scaler_glucose
        scaler_basal = train_ds.scaler_basal
        scaler_bolus = train_ds.scaler_bolus
        scaler_carbs = train_ds.scaler_carbs
        scaler_glucose_jepa = train_ds.scaler_glucose_jepa
        save_scalers_for_run(
            run_dir,
            kind="sugar_jepa",
            scalers={
                "glucose": scaler_glucose,
                "basal": scaler_basal,
                "bolus": scaler_bolus,
                "carbs": scaler_carbs,
                "glucose_jepa": scaler_glucose_jepa,
            },
            provenance={"csv": str(train_path), "source": "legacy_refit"},
        )
        typer.echo(f"Wrote {SCALERS_FILENAME} (legacy re-fit).")

    dev = torch.device(device)
    resolved_batch_size = batch_size or cfg.get("batch_size", 256)

    model = SugarJepaModel(
        n_time_steps=cfg["input_steps"],
        n_features=4,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_units=cfg["ff_units"],
        n_blocks=cfg["n_blocks"],
        prediction_horizon=cfg["horizon"],
        dropout=cfg["dropout"],
        jepa_weights_dir=cfg["jepa_weights_dir"],
        jepa_patch_size=cfg["jepa_patch_size"],
        jepa_freeze=not cfg.get("finetune_jepa", False),
    ).to(dev)

    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    state = strip_compile_prefix(state)
    model.load_state_dict(state)
    model.eval()

    test_train_df, test_val_df, test_test_df = load_splits_streaming(
        test_path, cfg["unique_id"], cfg["drop_interpolated"]
    )
    if test_split == "":
        eval_df = pl.concat([test_train_df, test_val_df, test_test_df])
    elif test_split == "train":
        eval_df = test_train_df
    elif test_split == "val":
        eval_df = test_val_df
    else:
        eval_df = test_test_df
    if eval_df is None or eval_df.is_empty():
        typer.echo(f"Error: no rows for --test-split={test_split!r} in {test_path}", err=True)
        raise typer.Exit(1)
    eval_df = _common_impute_and_sort(
        eval_df, ffill_bfill_columns=["glucose", "basal"], zero_fill_columns=["bolus", "carbs"],
    )

    eval_ds = SugarJepaWindowDataset(
        eval_df, cfg["input_steps"], cfg["horizon"], cfg["jepa_window"],
        scaler_glucose=scaler_glucose,
        scaler_basal=scaler_basal,
        scaler_bolus=scaler_bolus,
        scaler_carbs=scaler_carbs,
        scaler_glucose_jepa=scaler_glucose_jepa,
    )
    typer.echo(f"Evaluation windows: {len(eval_ds):,}")

    loader = DataLoader(eval_ds, batch_size=resolved_batch_size, shuffle=False)
    loss_fn = nn.MSELoss()
    _, true_arr, pred_arr = evaluate(model, loader, loss_fn, dev, split_label=test_split or "all")
    compute_and_print_metrics(true_arr, pred_arr, scaler_glucose, test_split or "all", run_dir, eval_ds)


if __name__ == "__main__":
    app()
