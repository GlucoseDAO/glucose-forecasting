#!/usr/bin/env python3
"""Typed results for unified evaluation / comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl


class RunDirKind(str, Enum):
    CUSTOM_PYTORCH = "custom_pytorch"
    PRECOMPUTED = "precomputed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mard: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "mard": self.mard}


@dataclass
class SplitMetrics:
    overall: RegressionMetrics
    by_study_group: pl.DataFrame | None = None


@dataclass
class SingleModelResult:
    model_name: str
    run_dir: Path
    kind: RunDirKind
    split_results: dict[str, SplitMetrics] = field(default_factory=dict)
    model_type: str | None = None
    checkpoint: Path | None = None
    test_csv: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def primary_overall(self, prefer: tuple[str, ...] = ("test", "val", "all")) -> RegressionMetrics | None:
        for key in prefer:
            if key in self.split_results:
                return self.split_results[key].overall
        if self.split_results:
            return next(iter(self.split_results.values())).overall
        return None
