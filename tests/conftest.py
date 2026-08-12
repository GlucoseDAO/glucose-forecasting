"""Shared fixtures for the glucose-forecasting test suite.

Generalizes the synthetic-CSV pattern from ``_write_loop_ic_csv()`` in
``tests/test_tune_sugar_one_smoke.py`` and adds tiny in-memory Polars
DataFrame builders + tiny model factories so individual test files don't
re-duplicate this setup.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch

from glumind.glumind_model import GluMindModel
from glumind_uni.glumind_uni_model import GluMindUniModel
from sugar_one.sugar_one_model import SugarOneModel

# ---------------------------------------------------------------------------
# Tiny model dims shared across tests (fast forward/backward passes on CPU).
# ---------------------------------------------------------------------------
TINY_D_MODEL = 8
TINY_N_HEADS = 2
TINY_N_BLOCKS = 1
TINY_FF_UNITS = 16
TINY_INPUT_STEPS = 8
TINY_HORIZON = 2


@pytest.fixture
def tiny_glumind_model() -> GluMindModel:
    return GluMindModel(
        n_time_steps=TINY_INPUT_STEPS,
        n_features=3,
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        ff_units=TINY_FF_UNITS,
        n_blocks=TINY_N_BLOCKS,
        prediction_horizon=TINY_HORIZON,
        dropout=0.0,
    )


@pytest.fixture
def tiny_sugar_one_model() -> SugarOneModel:
    return SugarOneModel(
        n_time_steps=TINY_INPUT_STEPS,
        n_features=4,
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        ff_units=TINY_FF_UNITS,
        n_blocks=TINY_N_BLOCKS,
        prediction_horizon=TINY_HORIZON,
        dropout=0.0,
    )


@pytest.fixture
def tiny_glumind_uni_model() -> GluMindUniModel:
    return GluMindUniModel(
        n_time_steps=TINY_INPUT_STEPS,
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        ff_units=TINY_FF_UNITS,
        n_blocks=TINY_N_BLOCKS,
        prediction_horizon=TINY_HORIZON,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Tiny in-memory Polars DataFrames matching the "canonical" post-load schema
# (unique_id, ds, glucose, ..., study_group, split, event_type) used by
# common.data.loading and common.evaluation.
# ---------------------------------------------------------------------------


def _timestamps(n: int, start: datetime | None = None, step_minutes: int = 5) -> list[datetime]:
    base = start or datetime(2020, 1, 1, 0, 0, 0)
    return [base + timedelta(minutes=step_minutes * i) for i in range(n)]


def tiny_glumind_df() -> pl.DataFrame:
    """Small canonical-schema DataFrame with glucose/hr/steps, 2 series, a gap."""
    rows: list[dict[str, object]] = []

    def add_series(uid: str, split: str, group: str, n: int, glucose0: float, gap_idx: int | None = None):
        ts = _timestamps(n)
        for i in range(n):
            glucose = None if gap_idx is not None and i == gap_idx else glucose0 + i * 0.5
            rows.append(
                {
                    "unique_id": uid,
                    "ds": ts[i],
                    "glucose": glucose,
                    "hr": 70.0 + i * 0.1,
                    "steps": 10.0 * i,
                    "study_group": group,
                    "split": split,
                    "event_type": "EGV",
                }
            )

    add_series("s-train-a", "train", "T1DM", 20, 100.0, gap_idx=5)
    add_series("s-val-b", "val", "Healthy", 15, 110.0)
    add_series("s-test-c", "test", "T1DM", 15, 105.0)

    return pl.DataFrame(rows)


def tiny_sugar_one_df() -> pl.DataFrame:
    """Small canonical-schema DataFrame with glucose/basal/bolus/carbs, 2-3 series, gaps."""
    rows: list[dict[str, object]] = []

    def add_series(uid: str, split: str, group: str, n: int, glucose0: float, gap_idx: int | None = None):
        ts = _timestamps(n)
        for i in range(n):
            glucose = None if gap_idx is not None and i == gap_idx else glucose0 + i * 0.5
            basal = None if gap_idx is not None and i == gap_idx else 1.0
            rows.append(
                {
                    "unique_id": uid,
                    "ds": ts[i],
                    "glucose": glucose,
                    "basal": basal,
                    "bolus": 2.0 if i % 7 == 0 else None,
                    "carbs": 15.0 if i % 9 == 0 else None,
                    "study_group": group,
                    "split": split,
                    "event_type": "EGV",
                }
            )

    add_series("s-train-a", "train", "T1DM", 20, 100.0, gap_idx=5)
    add_series("s-val-b", "val", "Insulin-T2DM", 15, 110.0)
    add_series("s-test-c", "test", "T1DM", 15, 105.0)

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic CSV writers (raw source-column shape, generalizing
# ``_write_loop_ic_csv`` from test_tune_sugar_one_smoke.py).
# ---------------------------------------------------------------------------


def write_glumind_csv(
    path: Path,
    *,
    series: list[tuple[str, str, str, int, float]] | None = None,
) -> None:
    """Write a GluMind-shape CSV (glucose/hr/steps columns) train_glumind.py expects.

    ``series``: list of (unique_id, split, study_group, n_rows, glucose0).
    """
    series = series or [
        ("g-train-a", "train", "T1DM", 40, 100.0),
        ("g-val-b", "val", "Healthy", 30, 110.0),
        ("g-test-c", "test", "T1DM", 25, 105.0),
    ]
    rows: list[dict[str, object]] = []
    for uid, split, group, n, glucose0 in series:
        ts = _timestamps(n)
        for i in range(n):
            rows.append(
                {
                    "sequence_id": uid,
                    "User ID": uid,
                    "Timestamp (YYYY-MM-DDThh:mm:ss)": ts[i].strftime("%Y-%m-%dT%H:%M:%S"),
                    "Event Type": "EGV",
                    "Study Group": group,
                    "Glucose Value (mg/dL)": glucose0 + i * 0.5,
                    "Heart Rate": 70.0 + i * 0.1,
                    "Step Count": 10.0 * i,
                    "Recommended Split": split,
                }
            )
    pl.DataFrame(rows).write_csv(path)


def write_sugar_one_csv(
    path: Path,
    *,
    series: list[tuple[str, str, str, int, float]] | None = None,
) -> None:
    """Write a SugarOne-shape CSV (glucose/basal/bolus/carbs columns)."""
    series = series or [
        ("s-train-a", "train", "T1DM", 40, 100.0),
        ("s-val-b", "val", "Insulin-T2DM", 30, 110.0),
        ("s-test-c", "test", "T1DM", 25, 105.0),
    ]
    rows: list[dict[str, object]] = []
    for uid, split, group, n, glucose0 in series:
        ts = _timestamps(n)
        for i in range(n):
            rows.append(
                {
                    "sequence_id": uid,
                    "Timestamp": ts[i].strftime("%Y-%m-%dT%H:%M:%S"),
                    "Event Type": "EGV",
                    "User ID": uid,
                    "Glucose (mg/dL)": glucose0 + i * 0.5,
                    "Basal Rate (U/h)": "1.0",
                    "Bolus Insulin (U)": "2.0" if i % 7 == 0 else "",
                    "Carbohydrates (g)": "15.0" if i % 9 == 0 else "",
                    "Recommended Split": split,
                    "Study Group": group,
                }
            )
    pl.DataFrame(rows).write_csv(path)
