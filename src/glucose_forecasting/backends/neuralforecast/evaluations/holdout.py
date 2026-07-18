"""Fixed-split held-out NeuralForecast evaluation."""
from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import torch
from eliot import start_action

from glucose_forecasting.backends.neuralforecast.adapter import (
    PreparedSplits,
    filter_minimum_length,
    prepare_splits,
)
from glucose_forecasting.backends.neuralforecast.benchmark import (
    calculate_metrics,
    to_neuralforecast_frame,
)
from glucose_forecasting.backends.neuralforecast.catalog import (
    ModelDefinition,
    create_model,
    resolve_models,
)
from glucose_forecasting.backends.neuralforecast.config import (
    ModelSuiteConfig,
    NeuralForecastRunConfig,
)
from glucose_forecasting.backends.neuralforecast.plotting import write_prediction_charts
from glucose_forecasting.backends.neuralforecast.telemetry import (
    EpochProgressCallback,
    announce,
    configure_run_logs,
)
from glucose_forecasting.common.data_loading import limit_series


def resolve_device(requested: str) -> str:
    """Select a supported accelerator, preferring CUDA in automatic mode."""
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if not available[requested]:
        raise ValueError(f"requested device {requested!r} is not available")
    return requested


def run_holdout(
    config: NeuralForecastRunConfig,
    *,
    suites: ModelSuiteConfig,
    suites_yaml: str,
) -> list[Path]:
    """Fit selected models and evaluate fixed validation/test CSV splits."""
    if config.evaluation != "holdout":
        raise ValueError("run_holdout requires evaluation='holdout'")
    splits = prepare_splits(
        config.csv,
        profile_name=config.profile,
        unique_id_choice=config.unique_id,
        split_scheme=config.split_scheme,
        drop_interpolated=config.drop_interpolated,
        max_train_series=config.max_train_series,
        max_points_per_series=config.max_points_per_series,
    )
    resolved_profile = splits.profile.name
    models = _resolve_selected_models(config, suites, resolved_profile)
    device = resolve_device(config.device)
    horizon, input_size, validation_size = _window_sizes(config)
    prepared = PreparedSplits(
        profile=splits.profile,
        train=filter_minimum_length(splits.train, input_size + validation_size + horizon),
        validation=filter_minimum_length(splits.validation, input_size + horizon),
        test=filter_minimum_length(splits.test, input_size + horizon),
    )
    if prepared.train.is_empty():
        raise ValueError("no training series remain after minimum-length filtering")

    groups = _select_groups(prepared, config)
    runs: list[Path] = []
    for group in groups:
        group_splits = _for_group(prepared, group)
        for definition in models:
            runs.append(
                _fit_one(
                    config,
                    definition=definition,
                    splits=group_splits,
                    group=group,
                    device=device,
                    horizon=horizon,
                    input_size=input_size,
                    validation_size=validation_size,
                    suites_yaml=suites_yaml,
                )
            )
    return runs


def _fit_one(
    config: NeuralForecastRunConfig,
    *,
    definition: ModelDefinition,
    splits: PreparedSplits,
    group: str,
    device: str,
    horizon: int,
    input_size: int,
    validation_size: int,
    suites_yaml: str,
) -> Path:
    from neuralforecast import NeuralForecast
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.out_dir / "nf_holdout" / group / f"{definition.name.value}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = configure_run_logs(run_dir)
    announce(
        f"Training {definition.name.value} for group {group} on {device}.",
        model=definition.name.value,
        group=group,
        device=device,
        max_steps=config.max_steps,
        log_directory=str(logs_dir),
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="step-{step}",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer_kwargs: dict[str, Any] = {
        "accelerator": device,
        "devices": 1,
        "default_root_dir": str(run_dir),
        "callbacks": [
            checkpoint,
            EarlyStopping(monitor="valid_loss", mode="min", patience=10),
            EpochProgressCallback(model_name=definition.name.value),
        ],
        "enable_checkpointing": True,
        "logger": False,
        "enable_progress_bar": True,
    }
    model = create_model(
        definition,
        horizon=horizon,
        input_size=input_size,
        max_steps=config.max_steps,
        historical_exogenous=splits.profile.historical_exogenous,
        trainer_kwargs=trainer_kwargs,
        learning_rate=config.learning_rate,
        val_check_steps=config.val_check_steps,
        batch_size=config.batch_size,
        valid_batch_size=config.valid_batch_size,
        windows_batch_size=config.windows_batch_size,
        inference_windows_batch_size=config.inference_windows_batch_size,
        step_size=config.step_size,
    )
    neuralforecast = NeuralForecast(models=[model], freq=config.freq)
    with start_action(
        action_type="neuralforecast_fit",
        model=definition.name.value,
        group=group,
        max_steps=config.max_steps,
    ):
        neuralforecast.fit(
            df=to_neuralforecast_frame(splits.train, splits.profile.historical_exogenous),
            val_size=validation_size,
        )
    announce(
        f"Finished training {definition.name.value}; saving model bundle.",
        model=definition.name.value,
        group=group,
    )
    neuralforecast.save(str(run_dir / "neuralforecast"), overwrite=True)
    for split_name, frame in (("val", splits.validation), ("test", splits.test)):
        announce(
            f"Evaluating {definition.name.value} on {split_name}.",
            model=definition.name.value,
            group=group,
            split=split_name,
        )
        evaluated = _evaluate_split(
            neuralforecast,
            split_name=split_name,
            frame=frame,
            profile_exogenous=splits.profile.historical_exogenous,
            horizon=horizon,
            input_size=input_size,
            run_dir=run_dir,
            config=config,
        )
        if config.plot and evaluated is not None:
            announce(
                f"Writing {split_name} charts for {definition.name.value}.",
                model=definition.name.value,
                group=group,
                split=split_name,
            )
            write_prediction_charts(
                frame,
                evaluated,
                model_name=f"{definition.name.value}_{split_name}",
                output_dir=run_dir,
                max_sequences=config.max_plot_series,
                title_prefix=f"{split_name.title()} — ",
            )
    (run_dir / "run_config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_dir / "model_suites.yaml").write_text(suites_yaml, encoding="utf-8")
    announce(
        f"Completed {definition.name.value}; artifacts: {run_dir}",
        model=definition.name.value,
        group=group,
        run_directory=str(run_dir),
    )
    return run_dir


def _evaluate_split(
    neuralforecast: Any,
    *,
    split_name: str,
    frame: pl.DataFrame,
    profile_exogenous: tuple[str, ...],
    horizon: int,
    input_size: int,
    run_dir: Path,
    config: NeuralForecastRunConfig,
) -> pl.DataFrame | None:
    frame = limit_series(frame, config.max_eval_series)
    frame = filter_minimum_length(frame, input_size + horizon)
    if frame.is_empty():
        return None
    truth = frame.group_by("unique_id", maintain_order=True).tail(horizon)
    history = frame.join(
        truth.select(["unique_id", "ds"]),
        on=["unique_id", "ds"],
        how="anti",
    )
    predictions = neuralforecast.predict(
        df=to_neuralforecast_frame(history, profile_exogenous)
    )
    prediction_columns = [column for column in predictions.columns if column not in {"unique_id", "ds"}]
    if len(prediction_columns) != 1:
        raise RuntimeError("expected exactly one prediction column for one selected model")
    predicted = pl.from_pandas(predictions).rename({prediction_columns[0]: "yhat"})
    evaluated = truth.join(predicted, on=["unique_id", "ds"], how="inner")
    if config.mask_interpolated_targets:
        evaluated = evaluated.filter(pl.col("event_type") != "Interpolated")
    if evaluated.is_empty():
        return None
    metrics = calculate_metrics(evaluated)
    announce(
        (
            f"{split_name} metrics: MAE={metrics.overall.mae:.3f}, "
            f"RMSE={metrics.overall.rmse:.3f}, MARD={metrics.overall.mard:.3f}%."
        ),
        split=split_name,
        mae=metrics.overall.mae,
        rmse=metrics.overall.rmse,
        mard=metrics.overall.mard,
    )
    pl.DataFrame(
        [{
            "mae": metrics.overall.mae,
            "rmse": metrics.overall.rmse,
            "mard": metrics.overall.mard,
        }]
    ).write_csv(run_dir / f"{split_name}_metrics_overall.csv")
    metrics.by_study_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")
    if config.save_predictions or config.plot:
        evaluated.write_csv(run_dir / f"{split_name}_predictions.csv")
    return evaluated


def _resolve_selected_models(
    config: NeuralForecastRunConfig,
    suites: ModelSuiteConfig,
    profile: str,
) -> tuple[ModelDefinition, ...]:
    suite_models = {
        name: suite.models
        for name, suite in suites.suites.items()
        if profile in suite.profiles
    }
    if config.models in suites.suites and config.models not in suite_models:
        raise ValueError(f"model suite {config.models!r} is not compatible with {profile!r}")
    return resolve_models(config.models, suite_models=suite_models)


def _window_sizes(config: NeuralForecastRunConfig) -> tuple[int, int, int]:
    step_minutes = int(pd.Timedelta(config.freq).total_seconds() // 60)
    if config.h_minutes % step_minutes:
        raise ValueError(f"h_minutes={config.h_minutes} is not divisible by freq={config.freq}")
    return (
        config.h_minutes // step_minutes,
        round(config.input_hours * 60 / step_minutes),
        round(config.train_tail_val_hours * 60 / step_minutes),
    )


def _select_groups(splits: PreparedSplits, config: NeuralForecastRunConfig) -> tuple[str, ...]:
    if config.global_model:
        return ("__ALL__",)
    all_groups = tuple(sorted(splits.train["study_group"].unique().cast(pl.String).to_list()))
    selected = set(config.study_groups)
    groups = tuple(group for group in all_groups if not selected or group in selected)
    if not groups:
        raise ValueError("no study groups selected after filtering")
    return groups


def _for_group(splits: PreparedSplits, group: str) -> PreparedSplits:
    if group == "__ALL__":
        return splits
    return PreparedSplits(
        profile=splits.profile,
        train=splits.train.filter(pl.col("study_group") == group),
        validation=splits.validation.filter(pl.col("study_group") == group),
        test=splits.test.filter(pl.col("study_group") == group),
    )
