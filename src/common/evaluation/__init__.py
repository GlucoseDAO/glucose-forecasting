#!/usr/bin/env python3
"""Shared evaluation package: covariates, inference helpers, and glucose evaluate APIs.

Backward-compatible imports (``from common.evaluation import X``) re-export the
legacy helpers from ``core``. Newer Phase-3 APIs live in sibling modules
(``types``, ``runner``, ``comparison``, …).
"""
from __future__ import annotations

from common.evaluation.core import (
    COL_EVENT,
    COVARIATE_NAME_ALIASES,
    DEFAULT_INFERENCE_LOG_INTERVAL_S,
    GLUMIND_COVARIATES,
    ModelKind,
    SUGAR_ONE_COVARIATES,
    _alias_to_canonical,
    _canonical_feature_cols,
    _covariate_map,
    _format_duration,
    _load_csv_flexible,
    _non_glucose_covariate_cols,
    _parse_covariate_names,
    _pick_header_column,
    _resolve_covariate_zeroing,
    _run_evaluate,
    _split_cov_arg,
    _zero_covariates,
)
from common.evaluation.device import resolve_torch_device
from common.evaluation.types import (
    RegressionMetrics,
    RunDirKind,
    SingleModelResult,
    SplitMetrics,
)

__all__ = [
    "COL_EVENT",
    "COVARIATE_NAME_ALIASES",
    "DEFAULT_INFERENCE_LOG_INTERVAL_S",
    "GLUMIND_COVARIATES",
    "ModelKind",
    "RegressionMetrics",
    "RunDirKind",
    "SUGAR_ONE_COVARIATES",
    "SingleModelResult",
    "SplitMetrics",
    "_alias_to_canonical",
    "_canonical_feature_cols",
    "_covariate_map",
    "_format_duration",
    "_load_csv_flexible",
    "_non_glucose_covariate_cols",
    "_parse_covariate_names",
    "_pick_header_column",
    "_resolve_covariate_zeroing",
    "_run_evaluate",
    "_split_cov_arg",
    "_zero_covariates",
    "resolve_torch_device",
]
