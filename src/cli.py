#!/usr/bin/env python3
"""Thin top-level ``glucose`` Typer CLI.

Experiment training stays on per-model entry points (``train-glumind``,
``uv run python src/sugar_one/train_sugar_one.py``, ...). This app exposes
shared evaluate/compare under ``common.evaluation``.

Defaults live in ``src/glucose_evaluate.yaml`` (override with ``--config`` / flags).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import typer

from common.evaluation.config import (
    ModelEvalSpec,
    default_config_path,
    load_evaluate_config,
)
from common.evaluation.resolve_models import expand_model_specs
from common.evaluation.runner import evaluate_and_compare, evaluate_run_dir
from common.paths import DEFAULT_RUNS_ROOT
from nf_baselines.cli import app as neuralforecast_app

DEFAULT_CONFIG_HINT = "src/glucose_evaluate.yaml"

app = typer.Typer(
    name="glucose",
    help=(
        "Glucose forecasting platform CLI. "
        "Train via experiment CLIs (train-glumind, train_sugar_one, ...); "
        "NeuralForecast via `glucose neuralforecast`; "
        "evaluate/compare via `glucose evaluate` (defaults: glucose_evaluate.yaml)."
    ),
    add_completion=False,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)
app.add_typer(neuralforecast_app, name="neuralforecast")


def _package_version() -> str:
    try:
        return version("glucose-forecasting")
    except PackageNotFoundError:
        return "0.1.0"


@app.command("info")
def info() -> None:
    """Print package version and default paths."""
    cfg_path = default_config_path()
    typer.echo(f"glucose-forecasting {_package_version()}")
    typer.echo(f"default runs root: {DEFAULT_RUNS_ROOT}")
    typer.echo(f"evaluate config: {cfg_path}")
    typer.echo("train: use experiment CLIs (train-glumind, train_sugar_one, ...)")
    typer.echo("neuralforecast: glucose neuralforecast --help")
    typer.echo("evaluate: glucose evaluate --help")


@app.command("evaluate")
def evaluate(
    run_dir: list[Path] = typer.Option(
        [],
        "--run-dir",
        help="Run directory (repeat for multi-run comparison). Default: models[] from YAML.",
    ),
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Evaluation CSV. Default from YAML. Omit to read precomputed metrics.",
    ),
    train_data: Optional[Path] = typer.Option(
        None,
        "--train-data",
        help="Optional CSV for legacy scaler fitting when scalers.json is absent.",
    ),
    label: list[str] = typer.Option(
        [],
        "--label",
        help="Optional label per --run-dir (repeatable).",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Comparison report directory (default from YAML / data/output/compare).",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help=f"YAML defaults file (default: {DEFAULT_CONFIG_HINT}).",
    ),
    model_type: Optional[str] = typer.Option(
        None,
        "--model-type",
        help="Global model type auto|glumind|sugar_one (YAML / per-model overrides).",
    ),
    test_split: Optional[str] = typer.Option(
        None,
        "--test-split",
        help="Recommended Split value; empty string disables filtering.",
    ),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    device: Optional[str] = typer.Option(None, "--device", help="auto | cuda | mps | cpu"),
    zero_cov: Optional[bool] = typer.Option(
        None,
        "--zero-cov/--no-zero-cov",
        help="Zero non-glucose covariates (global; per-model YAML can override).",
    ),
    include_cov: Optional[str] = typer.Option(None, "--include-cov"),
    exclude_cov: Optional[str] = typer.Option(None, "--exclude-cov"),
    plot: Optional[bool] = typer.Option(
        None,
        "--plot/--no-plot",
        help="Write comparison charts under --out (default from YAML: true).",
    ),
) -> None:
    """Evaluate one or more run directories (defaults from glucose_evaluate.yaml)."""
    try:
        cfg = load_evaluate_config(config)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error loading evaluate config: {exc}", err=True)
        raise typer.Exit(1) from exc

    resolved_data = data if data is not None else cfg.data
    resolved_train = train_data if train_data is not None else cfg.train_data
    resolved_out = out if out is not None else cfg.out
    resolved_device = device if device is not None else cfg.device
    resolved_batch = batch_size if batch_size is not None else cfg.batch_size
    resolved_plot = cfg.plot if plot is None else plot
    resolved_model_type = cfg.model_type if model_type is None else model_type
    if resolved_model_type not in ("auto", "glumind", "sugar_one"):
        typer.echo(
            f"Error: --model-type must be auto|glumind|sugar_one, got {resolved_model_type!r}.",
            err=True,
        )
        raise typer.Exit(1)

    if test_split is None:
        resolved_split = cfg.test_split
    else:
        resolved_split = test_split if test_split else None

    global_zero = cfg.zero_cov if zero_cov is None else zero_cov
    global_include = include_cov if include_cov is not None else cfg.include_cov
    global_exclude = exclude_cov if exclude_cov is not None else cfg.exclude_cov

    models: list[ModelEvalSpec]
    if run_dir:
        models = []
        for idx, path in enumerate(run_dir):
            lbl = label[idx] if idx < len(label) else None
            models.append(
                ModelEvalSpec(
                    run_dir=path,
                    label=lbl,
                    model_type=resolved_model_type,  # type: ignore[arg-type]
                    zero_cov=global_zero,
                    include_cov=global_include,
                    exclude_cov=global_exclude,
                    batch_size=resolved_batch,
                )
            )
    else:
        if not cfg.models:
            typer.echo(
                "Error: provide --run-dir or define models[] in the YAML config.",
                err=True,
            )
            raise typer.Exit(1)
        models = list(cfg.models)
        if model_type is not None:
            models = [
                ModelEvalSpec(
                    run_dir=m.run_dir,
                    label=m.label,
                    model_type=resolved_model_type,  # type: ignore[arg-type]
                    zero_cov=m.zero_cov,
                    include_cov=m.include_cov,
                    exclude_cov=m.exclude_cov,
                    batch_size=m.batch_size,
                )
                for m in models
            ]
        if zero_cov is not None:
            models = [
                ModelEvalSpec(
                    run_dir=m.run_dir,
                    label=m.label,
                    model_type=m.model_type,
                    zero_cov=global_zero,
                    include_cov=m.include_cov,
                    exclude_cov=m.exclude_cov,
                    batch_size=m.batch_size,
                )
                for m in models
            ]
        if batch_size is not None:
            models = [
                ModelEvalSpec(
                    run_dir=m.run_dir,
                    label=m.label,
                    model_type=m.model_type,
                    zero_cov=m.zero_cov,
                    include_cov=m.include_cov,
                    exclude_cov=m.exclude_cov,
                    batch_size=resolved_batch,
                )
                for m in models
            ]

    try:
        models = expand_model_specs(models)
    except (OSError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error expanding run paths: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Config: {config or default_config_path()}")
    typer.echo(f"Models: {', '.join(m.label or m.run_dir.name for m in models)}")
    if resolved_data is not None:
        typer.echo(f"Data  : {resolved_data}")
    typer.echo(f"Out   : {resolved_out}")
    typer.echo(f"Plot  : {resolved_plot}")

    if len(models) == 1 and out is None and not resolved_plot:
        one_batch = models[0].batch_size if models[0].batch_size is not None else resolved_batch
        result = evaluate_run_dir(
            models[0].run_dir,
            data=resolved_data,
            train_data=resolved_train,
            label=models[0].label,
            device=resolved_device,
            model_type=models[0].model_type,
            test_split=resolved_split,
            batch_size=one_batch,
            zero_cov=models[0].zero_cov,
            include_cov=models[0].include_cov,
            exclude_cov=models[0].exclude_cov,
        )
        primary = result.primary_overall()
        if primary is None:
            typer.echo("Error: No metrics produced.", err=True)
            raise typer.Exit(1)
        typer.echo(
            f"{result.model_name}: MAE={primary.mae:.4f} "
            f"RMSE={primary.rmse:.4f} MARD={primary.mard:.4f}%"
        )
        return

    results, report_dir = evaluate_and_compare(
        models=models,
        data=resolved_data,
        train_data=resolved_train,
        output_dir=resolved_out,
        device=resolved_device,
        test_split=resolved_split,
        batch_size=resolved_batch,
        plot=resolved_plot,
    )
    for result in results:
        primary = result.primary_overall()
        if primary is None:
            typer.echo(f"{result.model_name}: (no metrics)")
            continue
        typer.echo(
            f"{result.model_name}: MAE={primary.mae:.4f} "
            f"RMSE={primary.rmse:.4f} MARD={primary.mard:.4f}%"
        )
    if report_dir is not None:
        typer.echo(f"Comparison report: {report_dir}")


if __name__ == "__main__":
    app()
