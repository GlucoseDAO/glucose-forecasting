#!/usr/bin/env python3
"""GluMind family metadata (features, CSV map, build, scaler extract)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from scripts.common.model_spec import arch_hparams_from_meta, register_family_spec
from scripts.common.scalers import ScalerLike, extract_scalers_from_dataset
from scripts.glumind.glumind_model import GluMindModel

COL_GLU = "Glucose Value (mg/dL)"
COL_HR = "Heart Rate"
COL_STEPS = "Step Count"


@dataclass(frozen=True)
class GluMindFamilySpec:
    kind: str = "glumind"
    feature_names: Sequence[str] = ("glucose", "hr", "steps")
    n_features: int = 3
    value_columns: Mapping[str, str] = field(
        default_factory=lambda: {
            "glucose": COL_GLU,
            "hr": COL_HR,
            "steps": COL_STEPS,
        }
    )
    csv_column_aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "glucose": ("Glucose Value (mg/dL)", "Glucose (mg/dL)"),
            "hr": ("Heart Rate",),
            "steps": ("Step Count",),
        }
    )
    covariate_aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "hr": ("hr", "heart_rate", "heart rate", "heartrate"),
            "steps": ("steps", "step", "step_count", "step count", "stepcount"),
        }
    )
    fingerprint_keys: Sequence[str] = ("embed_hr.weight", "embed_steps.weight")

    def build_model(self, meta: Mapping[str, Any], device: torch.device) -> nn.Module:
        arch = arch_hparams_from_meta(meta)
        model = GluMindModel(
            n_time_steps=arch["input_steps"],
            n_features=self.n_features,
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            ff_units=arch["ff_units"],
            n_blocks=arch["n_blocks"],
            prediction_horizon=arch["horizon"],
            dropout=arch["dropout"],
        )
        return model.to(device)

    def extract_scalers(self, dataset: Any) -> dict[str, ScalerLike]:
        return extract_scalers_from_dataset(dataset, feature_names=self.feature_names)


GLUMIND_SPEC = GluMindFamilySpec()
register_family_spec(GLUMIND_SPEC)
