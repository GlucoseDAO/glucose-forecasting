"""Unit tests for scripts/common/data_loading.py."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from scripts.common.data_loading import (
    apply_split_scheme,
    impute_and_sort,
    limit_series,
    load_splits_streaming,
    normalize_study_group_label,
    normalize_study_groups_column,
    resolve_num_workers,
)

from tests.conftest import write_glumind_csv


# ---------------------------------------------------------------------------
# load_splits_streaming
# ---------------------------------------------------------------------------


def test_load_splits_streaming_partitions_and_renames(tmp_path: Path) -> None:
    csv_path = tmp_path / "glumind.csv"
    write_glumind_csv(
        csv_path,
        series=[
            ("a", "train", "T1DM", 5, 100.0),
            ("b", "val", "Healthy", 4, 110.0),
            ("c", "test", "T1DM", 3, 105.0),
        ],
    )

    train_df, val_df, test_df = load_splits_streaming(
        csv_path,
        "sequence_id",
        drop_interpolated=False,
        col_seq="sequence_id",
        col_user="User ID",
        col_ts="Timestamp (YYYY-MM-DDThh:mm:ss)",
        col_split="Recommended Split",
        col_group="Study Group",
        col_event="Event Type",
        value_columns={"glucose": "Glucose Value (mg/dL)", "hr": "Heart Rate", "steps": "Step Count"},
        ts_format="%Y-%m-%dT%H:%M:%S",
    )

    assert len(train_df) == 5
    assert len(val_df) == 4
    assert len(test_df) == 3
    # Renamed/canonical columns present.
    for df in (train_df, val_df, test_df):
        assert set(["unique_id", "ds", "study_group", "split", "event_type", "glucose", "hr", "steps"]).issubset(
            set(df.columns)
        )
    assert train_df["glucose"].dtype == pl.Float32


def test_load_splits_streaming_drop_interpolated(tmp_path: Path) -> None:
    csv_path = tmp_path / "glumind.csv"
    write_glumind_csv(csv_path, series=[("a", "train", "T1DM", 6, 100.0)])

    # Manually inject one Interpolated row by rewriting the CSV.
    df = pl.read_csv(csv_path)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("Interpolated")).otherwise(pl.col("Event Type")).alias("Event Type")
    )
    df.write_csv(csv_path)

    train_df, _, _ = load_splits_streaming(
        csv_path,
        "sequence_id",
        drop_interpolated=True,
        col_seq="sequence_id",
        col_user="User ID",
        col_ts="Timestamp (YYYY-MM-DDThh:mm:ss)",
        col_split="Recommended Split",
        col_group="Study Group",
        col_event="Event Type",
        value_columns={"glucose": "Glucose Value (mg/dL)", "hr": "Heart Rate", "steps": "Step Count"},
        ts_format="%Y-%m-%dT%H:%M:%S",
    )
    assert len(train_df) == 5  # one row dropped

    train_df_kept, _, _ = load_splits_streaming(
        csv_path,
        "sequence_id",
        drop_interpolated=False,
        col_seq="sequence_id",
        col_user="User ID",
        col_ts="Timestamp (YYYY-MM-DDThh:mm:ss)",
        col_split="Recommended Split",
        col_group="Study Group",
        col_event="Event Type",
        value_columns={"glucose": "Glucose Value (mg/dL)", "hr": "Heart Rate", "steps": "Step Count"},
        ts_format="%Y-%m-%dT%H:%M:%S",
    )
    assert len(train_df_kept) == 6


# ---------------------------------------------------------------------------
# apply_split_scheme
# ---------------------------------------------------------------------------


def _splits() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = pl.DataFrame({"unique_id": ["a"], "ds": [1], "split": ["train"]})
    val = pl.DataFrame({"unique_id": ["b"], "ds": [2], "split": ["val"]})
    test = pl.DataFrame({"unique_id": ["c"], "ds": [3], "split": ["test"]})
    return train, val, test


def test_apply_split_scheme_classic_is_noop() -> None:
    train, val, test = _splits()
    out_train, out_val, out_test = apply_split_scheme(train, val, test, "classic")
    assert out_train is train
    assert out_val is val
    assert out_test is test


def test_apply_split_scheme_trainval_test_as_val() -> None:
    train, val, test = _splits()
    out_train, out_val, out_test = apply_split_scheme(train, val, test, "trainval_test_as_val")
    assert len(out_train) == 2  # train + val merged
    assert set(out_train["unique_id"].to_list()) == {"a", "b"}
    assert out_val.equals(test)
    assert out_test.is_empty()


def test_apply_split_scheme_unknown_raises() -> None:
    train, val, test = _splits()
    with pytest.raises(ValueError, match="Unknown split_scheme"):
        apply_split_scheme(train, val, test, "bogus_scheme")


def test_apply_split_scheme_empty_test_raises() -> None:
    train, val, test = _splits()
    with pytest.raises(ValueError, match="non-empty test split"):
        apply_split_scheme(train, val, test.clear(), "trainval_test_as_val")


# ---------------------------------------------------------------------------
# impute_and_sort
# ---------------------------------------------------------------------------


def test_impute_and_sort_ffill_bfill_per_series_no_crossover() -> None:
    # Series "a" has an internal gap at ds=2 that should forward-fill from ds=1.
    # Series "b" has a leading null that should back-fill from its own first value.
    df = pl.DataFrame(
        {
            "unique_id": ["a", "a", "a", "b", "b"],
            "ds": [1, 2, 3, 1, 2],
            "glucose": [10.0, None, 30.0, None, 50.0],
        }
    )
    out = impute_and_sort(df, ffill_bfill_columns=["glucose"])
    out = out.sort(["unique_id", "ds"])
    a_vals = out.filter(pl.col("unique_id") == "a")["glucose"].to_list()
    b_vals = out.filter(pl.col("unique_id") == "b")["glucose"].to_list()
    assert a_vals == [10.0, 10.0, 30.0]  # forward-filled from a's own row, not b's
    assert b_vals == [50.0, 50.0]  # back-filled from b's own row, not a's


def test_impute_and_sort_zero_fill_no_carryover() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["a", "a", "a"],
            "ds": [1, 2, 3],
            "bolus": [5.0, None, None],
        }
    )
    out = impute_and_sort(df, zero_fill_columns=["bolus"]).sort("ds")
    assert out["bolus"].to_list() == [5.0, 0.0, 0.0]


def test_impute_and_sort_all_null_column_falls_back_to_zero() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "glucose": [None, None],
        }
    )
    out = impute_and_sort(df, ffill_bfill_columns=["glucose"]).sort("ds")
    assert out["glucose"].to_list() == [0.0, 0.0]


def test_impute_and_sort_empty_df_is_noop() -> None:
    df = pl.DataFrame({"unique_id": [], "ds": [], "glucose": []})
    out = impute_and_sort(df, ffill_bfill_columns=["glucose"])
    assert out.is_empty()


# ---------------------------------------------------------------------------
# limit_series
# ---------------------------------------------------------------------------


def test_limit_series_caps_to_first_n() -> None:
    df = pl.DataFrame({"unique_id": ["a", "a", "b", "b", "c", "c"], "val": [1, 2, 3, 4, 5, 6]})
    out = limit_series(df, 2)
    assert set(out["unique_id"].unique().to_list()) == {"a", "b"}
    assert len(out) == 4


def test_limit_series_non_positive_is_noop() -> None:
    df = pl.DataFrame({"unique_id": ["a", "b"], "val": [1, 2]})
    out = limit_series(df, 0)
    assert out is df
    out2 = limit_series(df, -5)
    assert out2 is df


# ---------------------------------------------------------------------------
# normalize_study_group_label / normalize_study_groups_column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("prediabetes", "Pre-T2DM"),
        ("Pre-Diabetes", "Pre-T2DM"),
        ("oral_medication", "Oral-T2DM"),
        ("insulin_dependent", "Insulin-T2DM"),
        ("healthy", "Healthy"),
    ],
)
def test_normalize_study_group_label_aliases(raw: str, expected: str) -> None:
    assert normalize_study_group_label(raw) == expected


def test_normalize_study_group_label_passthrough_unmapped() -> None:
    assert normalize_study_group_label("T1DM") == "T1DM"
    assert normalize_study_group_label("SomeUnknownLabel") == "SomeUnknownLabel"


def test_normalize_study_groups_column() -> None:
    df = pl.DataFrame({"study_group": ["prediabetes", "T1DM", "healthy"]})
    out = normalize_study_groups_column(df)
    assert out["study_group"].to_list() == ["Pre-T2DM", "T1DM", "Healthy"]


def test_normalize_study_groups_column_empty_df_is_noop() -> None:
    df = pl.DataFrame({"study_group": []})
    out = normalize_study_groups_column(df)
    assert out is df


# ---------------------------------------------------------------------------
# resolve_num_workers
# ---------------------------------------------------------------------------


def test_resolve_num_workers_explicit_passthrough() -> None:
    assert resolve_num_workers(4, torch.device("cpu")) == 4
    assert resolve_num_workers(0, torch.device("cuda")) == 0


def test_resolve_num_workers_auto_cpu_is_zero() -> None:
    assert resolve_num_workers(-1, torch.device("cpu")) == 0


def test_resolve_num_workers_auto_cuda_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 32)
    n = resolve_num_workers(-1, torch.device("cuda"))
    assert n == 8  # capped at 8

    monkeypatch.setattr("os.cpu_count", lambda: 2)
    n2 = resolve_num_workers(-1, torch.device("cuda"))
    assert n2 == 2  # max(2, 2//2)=2
