"""Reusable primitives for fixed-split NeuralForecast holdout evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from common.metrics import mae_rmse_mard

_BASE_COLUMNS = ("unique_id", "ds", "y")
_METRIC_COLUMNS: dict[str, pl.DataType] = {
    "study_group": pl.String,
    "n_points": pl.UInt32,
    "mae": pl.Float64,
    "rmse": pl.Float64,
    "mard": pl.Float64,
}


@dataclass(frozen=True)
class RegressionMetrics:
    """MAE, RMSE, and MARD values in the original glucose scale."""

    mae: float
    rmse: float
    mard: float


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Overall and per-study-group metrics for one evaluated split."""

    overall: RegressionMetrics
    by_study_group: pl.DataFrame


def to_neuralforecast_frame(
    split: pl.DataFrame,
    historical_exogenous: Sequence[str] = (),
) -> pl.DataFrame:
    """Select and sort normalized split columns for NeuralForecast fit/predict."""
    columns = (*_BASE_COLUMNS, *historical_exogenous)
    _require_columns(split, columns)

    if len(set(columns)) != len(columns):
        raise ValueError("historical_exogenous must not repeat NeuralForecast columns.")

    return split.select(columns).sort(["unique_id", "ds"])


def calculate_metrics(
    predictions: pl.DataFrame,
    *,
    target_column: str = "y",
    prediction_column: str = "yhat",
    study_group_column: str = "study_group",
) -> BenchmarkMetrics:
    """Calculate holdout overall and per-study-group MAE/RMSE/MARD."""
    _require_columns(
        predictions, (target_column, prediction_column, study_group_column)
    )
    if predictions.is_empty():
        raise ValueError("Cannot calculate metrics from an empty prediction frame.")

    mae, rmse, mard = mae_rmse_mard(
        predictions[target_column].to_numpy(),
        predictions[prediction_column].to_numpy(),
    )
    overall = RegressionMetrics(mae=mae, rmse=rmse, mard=mard)
    groups = predictions.partition_by(study_group_column, as_dict=True)
    rows = []
    for group, group_frame in groups.items():
        g_mae, g_rmse, g_mard = mae_rmse_mard(
            group_frame[target_column].to_numpy(),
            group_frame[prediction_column].to_numpy(),
        )
        rows.append(
            {
                "study_group": str(group[0] if isinstance(group, tuple) else group),
                "n_points": group_frame.height,
                "mae": g_mae,
                "rmse": g_rmse,
                "mard": g_mard,
            }
        )
    by_study_group = (
        pl.DataFrame(rows, schema=_METRIC_COLUMNS).sort("mae")
        if rows
        else pl.DataFrame(schema=_METRIC_COLUMNS)
    )
    return BenchmarkMetrics(overall=overall, by_study_group=by_study_group)


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ValueError(f"Frame is missing required columns: {formatted}")
