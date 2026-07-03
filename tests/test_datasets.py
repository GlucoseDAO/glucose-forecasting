"""Unit tests for the sliding-window Dataset classes:
GlucoseWindowDataset (train_glumind.py) and SugarOneWindowDataset (train_sugar_one.py).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from scripts.glumind.train_glumind import GlucoseWindowDataset
from scripts.sugar_one.train_sugar_one import SugarOneWindowDataset


def _glumind_df(n_rows_per_series: dict[str, int]) -> pl.DataFrame:
    rows = []
    for uid, n in n_rows_per_series.items():
        for i in range(n):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": i,
                    "glucose": 100.0 + i,
                    "hr": 70.0 + i,
                    "steps": float(i),
                    "study_group": "T1DM",
                }
            )
    return pl.DataFrame(rows)


def _sugar_one_df(n_rows_per_series: dict[str, int]) -> pl.DataFrame:
    rows = []
    for uid, n in n_rows_per_series.items():
        for i in range(n):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": i,
                    "glucose": 100.0 + i,
                    "basal": 1.0,
                    "bolus": 2.0 if i % 5 == 0 else 0.0,
                    "carbs": 10.0 if i % 7 == 0 else 0.0,
                    "study_group": "T1DM",
                }
            )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# GlucoseWindowDataset
# ---------------------------------------------------------------------------


def test_glucose_window_dataset_window_count() -> None:
    input_steps, horizon = 4, 2
    window_len = input_steps + horizon
    n_rows = 10
    df = _glumind_df({"a": n_rows})
    ds = GlucoseWindowDataset(df, input_steps, horizon, fit_scalers=True)
    assert len(ds) == n_rows - window_len + 1


def test_glucose_window_dataset_skips_short_series(capsys: pytest.CaptureFixture) -> None:
    input_steps, horizon = 8, 2
    window_len = input_steps + horizon  # 10
    df = _glumind_df({"short": 5, "long": 15})
    ds = GlucoseWindowDataset(df, input_steps, horizon, fit_scalers=True)
    # Only "long" produces windows: 15 - 10 + 1 = 6
    assert len(ds) == 15 - window_len + 1
    assert "short" not in ds.series_ids


def test_glucose_window_dataset_getitem_shapes_and_scaling() -> None:
    input_steps, horizon = 4, 2
    df = _glumind_df({"a": 10})
    ds = GlucoseWindowDataset(df, input_steps, horizon, fit_scalers=True)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (input_steps, 3)
    assert y.shape == (horizon,)
    # Scaled into [0, 1] via the fitted MinMaxScaler (glucose is monotonic 100..109).
    assert x.min() >= 0.0 - 1e-6
    assert x.max() <= 1.0 + 1e-6
    # First and last raw glucose values map to 0 and 1 respectively.
    assert x[0, 0].item() == pytest.approx(0.0, abs=1e-5)


def test_glucose_window_dataset_reuses_scaler_not_refit() -> None:
    input_steps, horizon = 4, 2
    train_df = _glumind_df({"train": 10})
    val_df = _glumind_df({"val": 8})  # different value range than train would fit differently only if refit

    train_ds = GlucoseWindowDataset(train_df, input_steps, horizon, fit_scalers=True)
    val_ds = GlucoseWindowDataset(
        val_df, input_steps, horizon,
        scaler_glucose=train_ds.scaler_glucose,
        scaler_hr=train_ds.scaler_hr,
        scaler_steps=train_ds.scaler_steps,
        fit_scalers=False,
    )
    # val reuses train's scaler object exactly (not a refit clone).
    assert val_ds.scaler_glucose is train_ds.scaler_glucose
    # Manually verify the transform matches applying train's scaler directly.
    raw_val_glucose = val_df.sort(["unique_id", "ds"])["glucose"].to_numpy()
    expected = train_ds.scaler_glucose.transform(raw_val_glucose.reshape(-1, 1)).ravel()
    x0, _ = val_ds[0]
    assert x0[0, 0].item() == pytest.approx(float(expected[0]), abs=1e-5)


# ---------------------------------------------------------------------------
# SugarOneWindowDataset
# ---------------------------------------------------------------------------


def test_sugar_one_window_dataset_window_count() -> None:
    input_steps, horizon = 4, 2
    window_len = input_steps + horizon
    n_rows = 12
    df = _sugar_one_df({"a": n_rows})
    ds = SugarOneWindowDataset(df, input_steps, horizon, fit_scalers=True)
    assert len(ds) == n_rows - window_len + 1


def test_sugar_one_window_dataset_skips_short_series() -> None:
    input_steps, horizon = 8, 2
    window_len = input_steps + horizon
    df = _sugar_one_df({"short": 4, "long": 20})
    ds = SugarOneWindowDataset(df, input_steps, horizon, fit_scalers=True)
    assert len(ds) == 20 - window_len + 1
    assert "short" not in ds.series_ids


def test_sugar_one_window_dataset_getitem_shapes_and_scaling() -> None:
    input_steps, horizon = 4, 2
    df = _sugar_one_df({"a": 10})
    ds = SugarOneWindowDataset(df, input_steps, horizon, fit_scalers=True)
    x, y = ds[0]
    assert x.shape == (input_steps, 4)
    assert y.shape == (horizon,)
    assert x.min() >= 0.0 - 1e-6
    assert x.max() <= 1.0 + 1e-6


def test_sugar_one_window_dataset_reuses_scaler_not_refit() -> None:
    input_steps, horizon = 4, 2
    train_df = _sugar_one_df({"train": 10})
    val_df = _sugar_one_df({"val": 8})

    train_ds = SugarOneWindowDataset(train_df, input_steps, horizon, fit_scalers=True)
    val_ds = SugarOneWindowDataset(
        val_df, input_steps, horizon,
        scaler_glucose=train_ds.scaler_glucose,
        scaler_basal=train_ds.scaler_basal,
        scaler_bolus=train_ds.scaler_bolus,
        scaler_carbs=train_ds.scaler_carbs,
        fit_scalers=False,
    )
    assert val_ds.scaler_glucose is train_ds.scaler_glucose
    raw_val_glucose = val_df.sort(["unique_id", "ds"])["glucose"].to_numpy()
    expected = train_ds.scaler_glucose.transform(raw_val_glucose.reshape(-1, 1)).ravel()
    x0, _ = val_ds[0]
    assert x0[0, 0].item() == pytest.approx(float(expected[0]), abs=1e-5)
