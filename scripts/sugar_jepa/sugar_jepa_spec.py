#!/usr/bin/env python3
"""SugarJEPA family metadata (SugarOne channels + JEPA glucose scaler)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from scripts.common.model_spec import arch_hparams_from_meta, register_family_spec
from scripts.common.scalers import ScalerLike, extract_scalers_from_dataset
from scripts.sugar_jepa.sugar_jepa_model import SugarJepaModel
from scripts.sugar_one.sugar_one_spec import COL_BASAL, COL_BOLUS, COL_CARB, COL_GLU


@dataclass(frozen=True)
class SugarJepaFamilySpec:
    kind: str = "sugar_jepa"
    feature_names: Sequence[str] = (
        "glucose",
        "basal",
        "bolus",
        "carbs",
        "glucose_jepa",
    )
    n_features: int = 4
    value_columns: Mapping[str, str] = field(
        default_factory=lambda: {
            "glucose": COL_GLU,
            "basal": COL_BASAL,
            "bolus": COL_BOLUS,
            "carbs": COL_CARB,
        }
    )
    csv_column_aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "glucose": ("Glucose Value (mg/dL)", "Glucose (mg/dL)"),
            "basal": ("Basal Rate (U/h)",),
            "bolus": ("Bolus Insulin (U)",),
            "carbs": ("Carbohydrates (g)",),
        }
    )
    covariate_aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "basal": ("basal", "basal_rate", "basal rate", "basalrate"),
            "bolus": (
                "bolus",
                "bolus_insulin",
                "bolus insulin",
                "insulin",
                "bolusinsulin",
            ),
            "carbs": (
                "carbs",
                "carb",
                "carbohydrates",
                "carbohydrate",
                "carbohydrate_g",
            ),
        }
    )
    fingerprint_keys: Sequence[str] = ("jepa_encoder.proj.weight",)

    def build_model(self, meta: Mapping[str, Any], device: torch.device) -> nn.Module:
        arch = arch_hparams_from_meta(meta)
        model = SugarJepaModel(
            n_time_steps=arch["input_steps"],
            n_features=self.n_features,
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            ff_units=arch["ff_units"],
            n_blocks=arch["n_blocks"],
            prediction_horizon=arch["horizon"],
            dropout=arch["dropout"],
            jepa_weights_dir=str(
                meta.get("jepa_weights_dir", "scripts/sugar_jepa/pretrained/cgm_jepa")
            ),
            jepa_patch_size=int(meta.get("jepa_patch_size", 12)),
            jepa_freeze=not bool(meta.get("finetune_jepa", False)),
        )
        return model.to(device)

    def extract_scalers(self, dataset: Any) -> dict[str, ScalerLike]:
        return extract_scalers_from_dataset(dataset, feature_names=self.feature_names)


SUGAR_JEPA_SPEC = SugarJepaFamilySpec()
register_family_spec(SUGAR_JEPA_SPEC)
