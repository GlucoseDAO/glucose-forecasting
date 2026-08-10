"""Integration tests for src/common/evaluation.py's _load_csv_flexible and
_run_evaluate — the two biggest previously-untested, highest-risk pieces of
the evaluation pipeline (a shape/wiring regression here is otherwise silent).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.evaluation import _load_csv_flexible, _run_evaluate
from glumind.glumind_model import GluMindModel

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"
COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"


def _write_csv(path: Path, header_glucose_col: str, include_hr: bool, include_steps: bool) -> None:
    base = datetime(2020, 1, 1)
    rows = []
    for i in range(10):
        row = {
            COL_SEQ: "a",
            COL_USER: "u1",
            "Timestamp (YYYY-MM-DDThh:mm:ss)": (base + timedelta(minutes=5 * i)).strftime(TS_FORMAT),
            header_glucose_col: 100.0 + i,
            COL_SPLIT: "test",
            COL_GROUP: "T1DM",
            "Event Type": "EGV",
        }
        if include_hr:
            row["Heart Rate"] = 70.0 + i
        if include_steps:
            row["Step Count"] = float(i)
        rows.append(row)
    pl.DataFrame(rows).write_csv(path)


def _load(csv_path: Path, **overrides) -> pl.DataFrame:
    kwargs = dict(
        csv_path=csv_path,
        model_kind="glumind",
        unique_id_choice="sequence_id",
        drop_interpolated=False,
        eval_split=None,
        train_only=False,
        col_seq=COL_SEQ,
        col_user=COL_USER,
        col_split=COL_SPLIT,
        col_group=COL_GROUP,
        ts_aliases=["Timestamp (YYYY-MM-DDThh:mm:ss)", "Timestamp"],
        ts_format=TS_FORMAT,
    )
    kwargs.update(overrides)
    return _load_csv_flexible(**kwargs)


def test_load_csv_flexible_all_covariates_present(tmp_path: Path) -> None:
    csv_path = tmp_path / "full.csv"
    _write_csv(csv_path, "Glucose Value (mg/dL)", include_hr=True, include_steps=True)
    df = _load(csv_path)
    assert len(df) == 10
    assert set(["unique_id", "ds", "glucose", "hr", "steps", "study_group", "split"]).issubset(df.columns)
    assert df["hr"].null_count() == 0
    assert df["hr"].to_list()[0] == pytest.approx(70.0)


def test_load_csv_flexible_missing_covariate_filled_with_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_steps.csv"
    _write_csv(csv_path, "Glucose Value (mg/dL)", include_hr=True, include_steps=False)
    df = _load(csv_path)
    assert (df["steps"] == 0.0).all()
    assert (df["hr"] != 0.0).any()  # hr was actually present, unaffected


def test_load_csv_flexible_alias_resolution_picks_right_header(tmp_path: Path) -> None:
    # "Glucose (mg/dL)" is the SugarOne-style alias but should still resolve
    # for glumind since it's in GLUMIND_COVARIATES["glucose"] alias list.
    csv_path = tmp_path / "alt_glucose_header.csv"
    _write_csv(csv_path, "Glucose (mg/dL)", include_hr=True, include_steps=True)
    df = _load(csv_path)
    assert len(df) == 10
    assert df["glucose"].to_list()[0] == pytest.approx(100.0)


def test_load_csv_flexible_eval_split_filters_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed_split.csv"
    base = datetime(2020, 1, 1)
    rows = []
    for i in range(6):
        rows.append(
            {
                COL_SEQ: "a",
                COL_USER: "u1",
                "Timestamp (YYYY-MM-DDThh:mm:ss)": (base + timedelta(minutes=5 * i)).strftime(TS_FORMAT),
                "Glucose Value (mg/dL)": 100.0 + i,
                COL_SPLIT: "train" if i < 3 else "test",
                COL_GROUP: "T1DM",
                "Event Type": "EGV",
                "Heart Rate": 70.0,
                "Step Count": 1.0,
            }
        )
    pl.DataFrame(rows).write_csv(csv_path)
    df = _load(csv_path, eval_split="test")
    assert len(df) == 3


# ---------------------------------------------------------------------------
# _run_evaluate
# ---------------------------------------------------------------------------


def test_run_evaluate_prediction_count_and_finiteness() -> None:
    torch.manual_seed(0)
    model = GluMindModel(
        n_time_steps=8, n_features=3, d_model=8, n_heads=2, ff_units=16,
        n_blocks=1, prediction_horizon=2, dropout=0.0,
    )
    n_windows = 13
    x = torch.randn(n_windows, 8, 3)
    y = torch.randn(n_windows, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    true_arr, pred_arr = _run_evaluate(model, loader, "cpu", n_windows=n_windows, log_interval_s=9999.0)

    assert true_arr.shape == (n_windows, 2)
    assert pred_arr.shape == (n_windows, 2)
    assert np.isfinite(true_arr).all()
    assert np.isfinite(pred_arr).all()
