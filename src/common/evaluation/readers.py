#!/usr/bin/env python3
"""Read precomputed metrics CSVs from a run directory."""
from __future__ import annotations

import csv
from pathlib import Path

from common.evaluation.types import RegressionMetrics, SplitMetrics


def _read_overall_csv(path: Path) -> RegressionMetrics | None:
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None:
        return None
    try:
        return RegressionMetrics(
            mae=float(row["mae"]),
            rmse=float(row["rmse"]),
            mard=float(row["mard"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_precomputed_split_metrics(run_dir: Path) -> dict[str, SplitMetrics]:
    """Load ``{test,val,...}_metrics_overall.csv`` when present."""
    mapping = {
        "test": "test_metrics_overall.csv",
        "val": "val_metrics_overall.csv",
        "val_as_test": "val_metrics_overall.csv",
    }
    out: dict[str, SplitMetrics] = {}
    for split, name in mapping.items():
        metrics = _read_overall_csv(run_dir / name)
        if metrics is None:
            continue
        # Prefer explicit test/val keys; skip alias duplicate if already loaded.
        if split == "val_as_test" and "val" in out:
            continue
        out[split if split != "val_as_test" else "val"] = SplitMetrics(overall=metrics)
    return out
