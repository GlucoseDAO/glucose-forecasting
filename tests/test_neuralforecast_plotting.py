"""Tests for NeuralForecast interactive prediction charts."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from glucose_forecasting.backends.neuralforecast.plotting import (
    write_comparison_dashboard,
    write_prediction_charts,
)


def _actual_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["series-1"] * 4,
            "ds": ["2026-01-01T00:00:00", "2026-01-01T00:05:00", "2026-01-01T00:10:00", "2026-01-01T00:15:00"],
            "y": [100.0, 110.0, 120.0, 115.0],
        }
    ).with_columns(pl.col("ds").str.to_datetime())


def test_prediction_charts_write_interactive_html(tmp_path: Path) -> None:
    actual = _actual_frame()
    predictions = actual.tail(2).with_columns(pl.Series("yhat", [118.0, 116.0]))

    paths = write_prediction_charts(
        actual,
        predictions,
        model_name="NHITS",
        output_dir=tmp_path,
    )

    assert len(paths) == 1
    assert paths[0].is_file()
    assert "Actual glucose" in paths[0].read_text(encoding="utf-8")


def test_comparison_dashboard_writes_html(tmp_path: Path) -> None:
    actual = _actual_frame()
    predictions = actual.tail(2).with_columns(pl.Series("NHITS", [118.0, 116.0]))

    path = write_comparison_dashboard(
        actual,
        {"NHITS": predictions},
        output_dir=tmp_path,
    )

    assert path.is_file()
    assert "NeuralForecast model comparison" in path.read_text(encoding="utf-8")
