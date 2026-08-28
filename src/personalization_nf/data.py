"""Load chronological personal CSVs into NeuralForecast prepared splits."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from nf_baselines.adapter import PreparedSplits, filter_minimum_length, prepare_splits
from nf_baselines.config import NeuralForecastRunConfig, frequency_minutes
from personalization.splits import calendar_span_days, load_train_span_days


def window_sizes_from_config(config: NeuralForecastRunConfig) -> tuple[int, int, int]:
    """Return ``(horizon, input_size, train_tail_val_size)`` in steps."""
    step_minutes = frequency_minutes(config.freq)
    if config.h_minutes % step_minutes:
        raise ValueError(f"h_minutes={config.h_minutes} is not divisible by freq={config.freq}")
    return (
        config.h_minutes // step_minutes,
        round(config.input_hours * 60 / step_minutes),
        round(config.train_tail_val_hours * 60 / step_minutes),
    )


def load_personal_splits(
    personal_csv: Path,
    *,
    source_config: NeuralForecastRunConfig,
) -> PreparedSplits:
    """Prepare a personal chronological CSV with the source holdout profile."""
    return prepare_splits(
        personal_csv,
        profile_name=source_config.profile,
        unique_id_choice=source_config.unique_id,
        split_scheme=source_config.split_scheme,
        drop_interpolated=source_config.drop_interpolated,
        max_train_series=0,
        max_points_per_series=0,
    )


def limit_train_calendar_days(
    train: pl.DataFrame,
    personal_days: int | None,
    *,
    time_col: str = "ds",
) -> pl.DataFrame:
    """Keep the first ``personal_days`` of train (val/test are not passed in)."""
    if personal_days is None:
        return train
    if personal_days <= 0:
        raise ValueError(f"personal_days must be positive, got {personal_days}")
    if train.is_empty():
        return train
    t0 = train.select(pl.col(time_col).min()).item()
    if not isinstance(t0, datetime):
        t0 = datetime.fromisoformat(str(t0))
    t_end = t0 + timedelta(days=personal_days)
    limited = train.filter(pl.col(time_col) < t_end)
    if limited.is_empty():
        return train.sort(time_col).head(1)
    return limited.sort(["unique_id", time_col])


def span_days(frame: pl.DataFrame, *, time_col: str = "ds") -> float | None:
    """Inclusive calendar span of a prepared split."""
    if frame.is_empty():
        return None
    t_min = frame.select(pl.col(time_col).min()).item()
    t_max = frame.select(pl.col(time_col).max()).item()
    if t_min is None or t_max is None:
        return None
    return calendar_span_days(t_min, t_max)


def choose_val_size(
    train: pl.DataFrame,
    *,
    input_size: int,
    horizon: int,
    configured_val_size: int,
    val_tail_fraction: float,
) -> int:
    """Train-tail val length that still leaves one input+horizon window.

    NeuralForecast ``val_df`` requires equal-length series, which personal
    CSVs do not have. Early stopping therefore uses ``val_size`` on the train
    frame. Short day budgets cap the tail so training is not emptied.
    """
    if train.is_empty() or configured_val_size <= 0:
        return 0
    lengths = train.group_by("unique_id").len()
    min_len = int(lengths["len"].min())
    leftover = min_len - input_size - horizon
    if leftover < horizon:
        return 0
    fractional = max(horizon, int(round(min_len * val_tail_fraction)))
    return min(configured_val_size, leftover, fractional)


def filter_train_for_fit(
    train: pl.DataFrame,
    *,
    input_size: int,
    horizon: int,
    val_size: int,
) -> pl.DataFrame:
    """Drop series too short to yield one train window after the val tail."""
    return filter_minimum_length(train, input_size + val_size + horizon)


def day_label(personal_days: int | None) -> str:
    return "all" if personal_days is None else str(personal_days)


def train_span_for_csv(personal_csv: Path, train: pl.DataFrame) -> float | None:
    """Prefer split_meta.json; fall back to the prepared train frame."""
    from_meta = load_train_span_days(personal_csv)
    if from_meta is not None:
        return from_meta
    return span_days(train)


def used_train_days(train: pl.DataFrame, personal_days: int | None) -> float | None:
    span = span_days(train)
    if personal_days is None:
        return span
    if span is None:
        return float(personal_days)
    return min(float(personal_days), span)


def metrics_to_dict(metrics: Any) -> dict[str, float]:
    overall = metrics.overall
    return {
        "mae": float(overall.mae),
        "rmse": float(overall.rmse),
        "mard": float(overall.mard),
    }
