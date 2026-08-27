#!/usr/bin/env python3
"""High-level evaluate / compare entry points for the glucose CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import typer

from common.evaluation.comparison import write_comparison_report
from common.evaluation.config import ModelEvalSpec
from common.evaluation.detect import detect_run_dir
from common.evaluation.device import resolve_torch_device
from common.evaluation.pytorch import evaluate_pytorch_run
from common.evaluation.readers import read_precomputed_split_metrics
from common.evaluation.resolve_models import expand_model_specs
from common.evaluation.types import RunDirKind, SingleModelResult
from common.paths import resolve_project_path

ModelTypeName = Literal["auto", "glumind", "sugar_one", "glumind_uni", "sugar_jepa", "sugar_jepa2"]


def evaluate_run_dir(
    run_dir: Path | str,
    *,
    data: Path | str | None = None,
    train_data: Path | str | None = None,
    label: str | None = None,
    device: str = "auto",
    model_type: ModelTypeName = "auto",
    test_split: str | None = "test",
    batch_size: int | None = None,
    zero_cov: bool = False,
    include_cov: str | None = None,
    exclude_cov: str | None = None,
    refit_scalers: bool = False,
    allow_fit_on_eval: bool = False,
    log_interval: float = 10.0,
    checkpoint: Path | None = None,
    project_root: Path | None = None,
    force_rerun: bool = False,
) -> SingleModelResult:
    """Evaluate one run directory (re-inference when ``data`` is set)."""
    root = project_root or Path.cwd()
    resolved = resolve_project_path(run_dir, root)
    if not resolved.is_dir():
        typer.echo(f"Error: Run directory does not exist: {resolved}", err=True)
        raise typer.Exit(1)

    kind = detect_run_dir(resolved, root)
    name = label or resolved.name

    # NF / precomputed-only runs: use saved metrics (live NF reload is separate CLI).
    if kind in (RunDirKind.NEURALFORECAST, RunDirKind.PRECOMPUTED):
        splits = read_precomputed_split_metrics(resolved)
        if splits:
            if data is not None and kind == RunDirKind.NEURALFORECAST:
                typer.echo(
                    f"Note: using precomputed metrics for NeuralForecast run {resolved} "
                    "(pass `glucose neuralforecast evaluate` for live re-eval)."
                )
            return SingleModelResult(
                model_name=name,
                run_dir=resolved,
                kind=kind,
                split_results=splits,
                model_type=None,
            )
        if kind == RunDirKind.NEURALFORECAST:
            typer.echo(
                f"Error: NeuralForecast run has no metrics CSVs: {resolved}. "
                "Re-run holdout eval or use `glucose neuralforecast evaluate`.",
                err=True,
            )
            raise typer.Exit(1)

    if data is None and not force_rerun:
        if kind == RunDirKind.CUSTOM_PYTORCH:
            splits = read_precomputed_split_metrics(resolved)
            if splits:
                typer.echo(
                    f"Note: no --data provided; using precomputed metrics in {resolved}"
                )
                return SingleModelResult(
                    model_name=name,
                    run_dir=resolved,
                    kind=kind,
                    split_results=splits,
                    model_type=None,
                )
        typer.echo(
            "Error: No --data CSV and no precomputed *_metrics_overall.csv found. "
            "Pass --data to run inference.",
            err=True,
        )
        raise typer.Exit(1)

    if data is None:
        typer.echo("Error: --data is required to re-run inference.", err=True)
        raise typer.Exit(1)

    if kind == RunDirKind.UNKNOWN and not (
        (resolved / "best_model.pt").exists() or (resolved / "last_model.pt").exists()
    ):
        typer.echo(
            f"Error: Unsupported run directory for re-inference: {resolved}",
            err=True,
        )
        raise typer.Exit(1)

    data_path = resolve_project_path(data, root)
    train_path = resolve_project_path(train_data, root) if train_data is not None else None
    return evaluate_pytorch_run(
        resolved,
        test_csv=data_path,
        train_csv=train_path,
        checkpoint=checkpoint,
        model_type=model_type,
        test_split=test_split,
        batch_size=batch_size,
        device=resolve_torch_device(device),
        zero_cov=zero_cov,
        include_cov=include_cov,
        exclude_cov=exclude_cov,
        refit_scalers=refit_scalers,
        allow_fit_on_eval=allow_fit_on_eval,
        log_interval=log_interval,
        label=name,
        project_root=root,
    )


def evaluate_and_compare(
    run_dirs: Sequence[Path | str] | None = None,
    *,
    models: Sequence[ModelEvalSpec] | None = None,
    data: Path | str | None = None,
    train_data: Path | str | None = None,
    labels: Sequence[str] | None = None,
    output_dir: Path | str | None = None,
    device: str = "auto",
    model_type: ModelTypeName = "auto",
    test_split: str | None = "test",
    batch_size: int | None = None,
    zero_cov: bool = False,
    include_cov: str | None = None,
    exclude_cov: str | None = None,
    plot: bool = False,
    project_root: Path | None = None,
) -> tuple[list[SingleModelResult], Path | None]:
    """Evaluate multiple runs and optionally write a comparison report."""
    specs: list[ModelEvalSpec]
    if models is not None:
        specs = list(models)
    elif run_dirs:
        specs = []
        for idx, run_dir in enumerate(run_dirs):
            label = None
            if labels is not None and idx < len(labels):
                label = labels[idx]
            specs.append(
                ModelEvalSpec(
                    run_dir=Path(run_dir),
                    label=label,
                    model_type=model_type,
                    zero_cov=zero_cov,
                    include_cov=include_cov,
                    exclude_cov=exclude_cov,
                    batch_size=batch_size,
                )
            )
    else:
        typer.echo("Error: Provide at least one --run-dir or config models[].", err=True)
        raise typer.Exit(1)

    if not specs:
        typer.echo("Error: Provide at least one --run-dir or config models[].", err=True)
        raise typer.Exit(1)

    try:
        specs = expand_model_specs(specs, project_root=project_root)
    except (OSError, ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error expanding run paths: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Comparing {len(specs)} run(s) after best-per-model expansion.")
    for spec in specs:
        typer.echo(f"  - {spec.label or spec.run_dir.name}: {spec.run_dir}")

    results: list[SingleModelResult] = []
    for spec in specs:
        spec_batch = spec.batch_size if spec.batch_size is not None else batch_size
        results.append(
            evaluate_run_dir(
                spec.run_dir,
                data=data,
                train_data=train_data,
                label=spec.label,
                device=device,
                model_type=spec.model_type,
                test_split=test_split,
                batch_size=spec_batch,
                zero_cov=spec.zero_cov,
                include_cov=spec.include_cov,
                exclude_cov=spec.exclude_cov,
                project_root=project_root,
            )
        )

    report_dir: Path | None = None
    if output_dir is not None or len(results) > 1:
        out = Path(output_dir) if output_dir is not None else Path("data/output/compare")
        report_dir = write_comparison_report(results, out, plot=plot)
    return results, report_dir
