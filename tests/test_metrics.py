"""Unit tests for scripts/common/metrics.py."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import MinMaxScaler

from scripts.common.metrics import mae_rmse_mard, overall_metrics_to_csv, per_study_group_breakdown


def test_mae_rmse_mard_hand_computed() -> None:
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([110.0, 180.0, 60.0])
    # err = [-10, 20, -10]
    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)
    assert mae == pytest.approx((10 + 20 + 10) / 3)
    assert rmse == pytest.approx(math.sqrt((100 + 400 + 100) / 3))
    # mard = mean(|err|/|true|) * 100 = mean([0.1, 0.1, 0.2]) * 100
    assert mard == pytest.approx((0.1 + 0.1 + 0.2) / 3 * 100)


def test_mae_rmse_mard_zero_glucose_mard_is_nan() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)
    assert mae == pytest.approx(1.5)
    assert math.isnan(mard)


def test_mae_rmse_mard_single_value() -> None:
    y_true = np.array([100.0])
    y_pred = np.array([90.0])
    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)
    assert mae == pytest.approx(10.0)
    assert rmse == pytest.approx(10.0)
    assert mard == pytest.approx(10.0)


def test_mae_rmse_mard_mixed_zero_and_nonzero() -> None:
    # zero entries are excluded from MARD but included in MAE/RMSE.
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)
    assert mae == pytest.approx((5 + 10) / 2)
    assert mard == pytest.approx(10.0)  # only the nonzero entry contributes


def test_per_study_group_breakdown_correct_and_sorted() -> None:
    scaler = MinMaxScaler().fit(np.array([[0.0], [200.0]]))
    # Two groups: "A" has small error, "B" has large error -> A should sort first.
    true_scaled = scaler.transform(np.array([[100.0], [100.0], [100.0], [100.0]])).ravel()
    pred_scaled = scaler.transform(np.array([[101.0], [99.0], [150.0], [50.0]])).ravel()
    groups = ["B", "B", "A", "A"]

    out = per_study_group_breakdown(true_scaled, pred_scaled, scaler, groups)
    assert out is not None
    # Explicitly check sort order by mae ascending.
    maes = out["mae"].to_list()
    assert maes == sorted(maes)
    assert out.filter(pl.col("study_group") == "B")["n_windows"][0] == 2
    assert out.filter(pl.col("study_group") == "A")["n_windows"][0] == 2


def test_per_study_group_breakdown_length_mismatch_returns_none() -> None:
    scaler = MinMaxScaler().fit(np.array([[0.0], [200.0]]))
    true_scaled = np.array([0.5, 0.5, 0.5])
    pred_scaled = np.array([0.5, 0.5, 0.5])
    groups = ["A", "B"]  # mismatched length
    out = per_study_group_breakdown(true_scaled, pred_scaled, scaler, groups)
    assert out is None


def test_overall_metrics_to_csv_writes_expected_columns(tmp_path: Path) -> None:
    overall_metrics_to_csv(1.5, 2.5, 3.5, tmp_path, "val")
    out_path = tmp_path / "val_metrics_overall.csv"
    assert out_path.exists()
    df = pl.read_csv(out_path)
    assert df.columns == ["mae", "rmse", "mard"]
    assert df["mae"][0] == pytest.approx(1.5)
    assert df["rmse"][0] == pytest.approx(2.5)
    assert df["mard"][0] == pytest.approx(3.5)
