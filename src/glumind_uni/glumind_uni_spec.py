#!/usr/bin/env python3
"""GluMind-Uni family metadata (glucose-only)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from common.model_spec import arch_hparams_from_meta, register_family_spec
from common.scalers import ScalerLike, extract_scalers_from_dataset
from glumind_uni.glumind_uni_model import GluMindUniModel

COL_GLU = "Glucose Value (mg/dL)"


@dataclass(frozen=True)
class GluMindUniFamilySpec:
    kind: str = "glumind_uni"
    feature_names: Sequence[str] = ("glucose",)
    n_features: int = 1
    value_columns: Mapping[str, str] = field(
        default_factory=lambda: {"glucose": COL_GLU}
    )
    csv_column_aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "glucose": ("Glucose Value (mg/dL)", "Glucose (mg/dL)"),
        }
    )
    covariate_aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    # Uni shares embed_glucose with other families; detect only via explicit meta.
    fingerprint_keys: Sequence[str] = ()

    def build_model(self, meta: Mapping[str, Any], device: torch.device) -> nn.Module:
        arch = arch_hparams_from_meta(meta)
        model = GluMindUniModel(
            n_time_steps=arch["input_steps"],
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


GLUMIND_UNI_SPEC = GluMindUniFamilySpec()
register_family_spec(GLUMIND_UNI_SPEC)
