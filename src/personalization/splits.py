"""Chronological personal-data split helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from personalization.constants import (
    COL_SPLIT,
    COL_TS,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION_OF_REMAINDER,
    TS_FORMAT,
)


def _ensure_datetime(df: pl.DataFrame, ts_col: str = COL_TS) -> pl.DataFrame:
    dtype = df.schema[ts_col]
    if dtype == pl.Datetime or str(dtype).startswith("Datetime"):
        return df
    return df.with_columns(
        pl.col(ts_col).str.to_datetime(TS_FORMAT, strict=False).alias(ts_col)
    )


def chronological_split_labels(
    df: pl.DataFrame,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction_of_remainder: float = DEFAULT_VAL_FRACTION_OF_REMAINDER,
    ts_col: str = COL_TS,
) -> pl.DataFrame:
    """Assign train/val/test by chronological position within the full frame.

    Rows must already be filtered to a single person (or a contiguous personal
    timeline). Labels are assigned by row order after sorting on ``ts_col``.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if not 0.0 <= val_fraction_of_remainder < 1.0:
        raise ValueError(
            f"val_fraction_of_remainder must be in [0, 1), got {val_fraction_of_remainder}"
        )

    work = _ensure_datetime(df, ts_col).sort(ts_col)
    n = work.height
    if n < 3:
        raise ValueError(f"Need at least 3 rows for chronological split, got {n}")

    n_test = max(1, int(round(n * test_fraction)))
    n_remain = n - n_test
    n_val = max(1, int(round(n_remain * val_fraction_of_remainder))) if n_remain > 1 else 0
    if n_val >= n_remain:
        n_val = max(0, n_remain - 1)
    n_train = n_remain - n_val
    if n_train < 1:
        raise ValueError(
            f"Chronological split left no train rows (n={n}, test={n_test}, val={n_val})"
        )

    labels = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    return work.with_columns(pl.Series(COL_SPLIT, labels))


def limit_train_days(
    df: pl.DataFrame,
    personal_days: int | None,
    *,
    ts_col: str = COL_TS,
    split_col: str = COL_SPLIT,
) -> pl.DataFrame:
    """Keep only the first ``personal_days`` of the train split (by calendar time).

    Val/test rows are unchanged. If ``personal_days`` is None, return ``df`` as-is.
    """
    if personal_days is None:
        return df
    if personal_days <= 0:
        raise ValueError(f"personal_days must be positive, got {personal_days}")

    work = _ensure_datetime(df, ts_col)
    train = work.filter(pl.col(split_col) == "train")
    other = work.filter(pl.col(split_col) != "train")
    if train.is_empty():
        return work

    t0 = train.select(pl.col(ts_col).min()).item()
    if not isinstance(t0, datetime):
        t0 = datetime.fromisoformat(str(t0))
    t_end = t0 + timedelta(days=personal_days)
    train_limited = train.filter(pl.col(ts_col) < t_end)
    if train_limited.is_empty():
        # Fall back to at least the first train row if day window is empty
        # (e.g. sparse sampling); keep earliest row only.
        train_limited = train.head(1)
    return pl.concat([train_limited, other], how="vertical").sort(ts_col)


def calendar_span_days(start: datetime, end: datetime) -> float:
    """Inclusive calendar span in days between two timestamps."""
    return max(0.0, (end - start).total_seconds() / 86400.0)


def parse_timestamp(raw: str | datetime) -> datetime:
    if isinstance(raw, datetime):
        return raw
    text = str(raw).strip()
    for fmt in (TS_FORMAT, "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(text)


def train_span_days_from_split_meta(meta: dict[str, Any]) -> float | None:
    """Train calendar span in days from a ``split_meta.json`` payload."""
    part = meta.get("train")
    if not isinstance(part, dict):
        return None
    start_raw = part.get("start")
    end_raw = part.get("end")
    if not start_raw or not end_raw:
        return None
    return calendar_span_days(parse_timestamp(str(start_raw)), parse_timestamp(str(end_raw)))


def find_split_meta_path(personal_csv: Path) -> Path | None:
    """Locate split_meta next to a prepared personal CSV."""
    stem = personal_csv.stem
    candidates = [
        personal_csv.with_name(f"{stem}_split_meta.json"),
        personal_csv.with_name(f"{stem.replace('_chronological', '')}_split_meta.json"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_train_span_days(personal_csv: Path) -> float | None:
    meta_path = find_split_meta_path(personal_csv)
    if meta_path is None:
        return None
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        return None
    return train_span_days_from_split_meta(meta)


def split_meta(df: pl.DataFrame, *, ts_col: str = COL_TS) -> dict[str, Any]:
    """Summarize chronological split for reproducibility artifacts."""
    work = _ensure_datetime(df, ts_col)
    out: dict[str, Any] = {"n_rows": work.height}
    for split in ("train", "val", "test"):
        part = work.filter(pl.col(COL_SPLIT) == split)
        if part.is_empty():
            out[split] = {"n_rows": 0}
            continue
        t_min = part.select(pl.col(ts_col).min()).item()
        t_max = part.select(pl.col(ts_col).max()).item()
        out[split] = {
            "n_rows": part.height,
            "start": str(t_min),
            "end": str(t_max),
            "span_days": calendar_span_days(t_min, t_max),
        }
    return out


def write_split_meta(path: Path, meta: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
