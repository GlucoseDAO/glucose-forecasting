"""Typer commands for NeuralForecast holdout train / evaluate / summarize."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from pydantic import ValidationError

from common.paths import DEFAULT_RUNS_ROOT
from nf_baselines.config import NeuralForecastRunConfig, load_model_suites
from nf_baselines.evaluations.holdout import (
    run_holdout,
    run_loaded_holdout,
    summarize_holdout_runs,
)

app = typer.Typer(
    name="neuralforecast",
    help="Train and evaluate NeuralForecast baselines (sugarone-compatible 128/12/1 holdout).",
    add_completion=False,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)

SUGARONE_INPUT_HOURS = 128 * 5 / 60


@app.command("train")
def train(
    data: Annotated[
        Optional[Path],
        typer.Option(help="Labeled CSV data file (required unless --list-models)."),
    ] = None,
    holdout_protocol: Annotated[
        str,
        typer.Option(
            help=(
                "Holdout scoring: sugarone-compatible uses dense stride-1 128/12 windows; "
                "dense supports experimental geometry; tail is legacy final-horizon."
            )
        ),
    ] = "sugarone-compatible",
    profile: Annotated[
        str, typer.Option(help="Data profile: auto, ai-readi, or loop.")
    ] = "auto",
    models: Annotated[
        str, typer.Option(help="YAML suite or comma-separated model names; defaults to auto.")
    ] = "auto",
    model_config: Annotated[
        Optional[Path], typer.Option(help="Optional replacement model-suites YAML.")
    ] = None,
    device: Annotated[
        str, typer.Option(help="auto prefers CUDA, then MPS, then CPU.")
    ] = "auto",
    out_dir: Annotated[Path, typer.Option(help="Directory for run artifacts.")] = DEFAULT_RUNS_ROOT,
    unique_id: Annotated[
        str, typer.Option(help="Series identifier: sequence_id or user_id.")
    ] = "sequence_id",
    split_scheme: Annotated[
        str, typer.Option(help="classic or trainval_test_as_val.")
    ] = "classic",
    global_model: Annotated[
        bool, typer.Option("--global-model/--per-group", help="Train one model on all study groups.")
    ] = False,
    study_group: Annotated[
        list[str], typer.Option(help="Study group(s) to train; repeat this option.")
    ] = [],
    max_steps: Annotated[int, typer.Option(help="Maximum optimization steps.")] = 2000,
    h_minutes: Annotated[int, typer.Option(help="Forecast horizon in minutes.")] = 60,
    freq: Annotated[str, typer.Option(help="Sampling frequency, e.g. 5min.")] = "5min",
    input_hours: Annotated[
        float,
        typer.Option(help="Input context in hours; 10.6667 gives SugarOne's 128 steps at 5min."),
    ] = SUGARONE_INPUT_HOURS,
    train_tail_val_hours: Annotated[
        float, typer.Option(help="Internal train-tail validation length in hours.")
    ] = 24.0,
    val_check_steps: Annotated[int, typer.Option(help="Validation check interval.")] = 400,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 8,
    valid_batch_size: Annotated[int, typer.Option(help="Validation batch size.")] = 8,
    windows_batch_size: Annotated[int, typer.Option(help="Training window batch size.")] = 256,
    inference_windows_batch_size: Annotated[
        int, typer.Option(help="Inference window batch size.")
    ] = 256,
    step_size: Annotated[
        int,
        typer.Option(help="Sliding training-window step; 1 matches SugarOne."),
    ] = 1,
    learning_rate: Annotated[float, typer.Option(help="Learning rate.")] = 1e-3,
    max_train_series: Annotated[
        int, typer.Option(help="Cap training series; useful for smoke runs.")
    ] = 0,
    max_eval_series: Annotated[int, typer.Option(help="Cap evaluation series.")] = 0,
    max_points_per_series: Annotated[
        int, typer.Option(help="Keep the latest N points in each series; zero keeps all.")
    ] = 0,
    drop_interpolated: Annotated[
        bool, typer.Option(help="Remove interpolated rows from every split.")
    ] = False,
    mask_interpolated_targets: Annotated[
        bool, typer.Option(help="Keep interpolated history but exclude targets from metrics.")
    ] = False,
    list_models: Annotated[
        bool, typer.Option("--list-models", help="Show YAML suites, then exit.")
    ] = False,
) -> None:
    """Train NeuralForecast models with fixed-split holdout evaluation."""
    try:
        suites, suite_yaml = load_model_suites(model_config)
        if list_models:
            for name, suite in suites.suites.items():
                typer.echo(
                    f"{name}: profiles={','.join(suite.profiles)} models="
                    f"{','.join(model.value for model in suite.models)}"
                )
            return
        if data is None:
            typer.echo("Error: --data is required unless --list-models.", err=True)
            raise typer.Exit(1)
        run_config = NeuralForecastRunConfig(
            csv=data,
            profile=profile,  # type: ignore[arg-type]
            models=models,
            model_config_path=model_config,
            evaluation="holdout",
            holdout_protocol=holdout_protocol,  # type: ignore[arg-type]
            device=device,  # type: ignore[arg-type]
            out_dir=out_dir,
            unique_id=unique_id,  # type: ignore[arg-type]
            split_scheme=split_scheme,  # type: ignore[arg-type]
            global_model=global_model,
            study_groups=tuple(study_group),
            max_steps=max_steps,
            h_minutes=h_minutes,
            freq=freq,
            input_hours=input_hours,
            train_tail_val_hours=train_tail_val_hours,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            step_size=step_size,
            learning_rate=learning_rate,
            max_train_series=max_train_series,
            max_eval_series=max_eval_series,
            max_points_per_series=max_points_per_series,
            drop_interpolated=drop_interpolated,
            mask_interpolated_targets=mask_interpolated_targets,
            plot=False,
        )
        run_dirs = run_holdout(run_config, suites=suites, suites_yaml=suite_yaml)
    except (OSError, ValueError, RuntimeError, ValidationError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    for run_dir in run_dirs:
        typer.echo(f"NeuralForecast run written to {run_dir}")


@app.command("evaluate")
def evaluate_neuralforecast(
    run_dir: Annotated[Path, typer.Option(help="Completed NeuralForecast model run directory.")],
    data: Annotated[Path, typer.Option(help="Labeled CSV to evaluate using the source split settings.")],
    out: Annotated[
        Optional[Path],
        typer.Option(help="New output directory; defaults to RUN_DIR/evaluations/<UTC timestamp>."),
    ] = None,
    device: Annotated[
        Optional[str],
        typer.Option(help="Optional inference device override: auto, cpu, cuda, or mps."),
    ] = None,
    max_eval_series: Annotated[
        Optional[int],
        typer.Option(help="Optional cap on evaluated series for a diagnostic run."),
    ] = None,
) -> None:
    """Re-evaluate a saved NeuralForecast bundle without fitting."""
    config_path = run_dir / "run_config.json"
    bundle_dir = run_dir / "neuralforecast"
    if not config_path.is_file():
        typer.echo(f"Error: run config not found: {config_path}", err=True)
        raise typer.Exit(1)
    if not bundle_dir.is_dir():
        typer.echo(f"Error: saved NeuralForecast bundle not found: {bundle_dir}", err=True)
        raise typer.Exit(1)
    if not data.is_file():
        typer.echo(f"Error: CSV data file not found: {data}", err=True)
        raise typer.Exit(1)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("holdout_protocol") is None:
            payload["holdout_protocol"] = "dense"
        payload["csv"] = str(data)
        if device is not None:
            payload["device"] = device
        if max_eval_series is not None:
            payload["max_eval_series"] = max_eval_series
        config = NeuralForecastRunConfig.model_validate(payload)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = out if out is not None else run_dir / "evaluations" / timestamp
        written_dir = run_loaded_holdout(config, bundle_dir=bundle_dir, run_dir=output_dir)
    except (OSError, ValueError, RuntimeError, ValidationError, json.JSONDecodeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"NeuralForecast evaluation written to {written_dir}")


@app.command("summarize-holdout")
def summarize_neuralforecast_holdout(
    run_dir: Annotated[
        list[Path],
        typer.Option("--run-dir", help="Completed per-model holdout run; repeat for each model."),
    ],
    out: Annotated[
        Optional[Path],
        typer.Option(help="Summary directory; defaults below the shared group directory."),
    ] = None,
) -> None:
    """Combine compatible per-model holdout runs without retraining."""
    if not run_dir:
        typer.echo("Error: provide at least one --run-dir.", err=True)
        raise typer.Exit(1)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = out if out is not None else run_dir[0].parent / "summaries" / timestamp
    try:
        written_dir = summarize_holdout_runs(run_dir, output_dir=output_dir, plot=False)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"NeuralForecast holdout summary written to {written_dir}")
