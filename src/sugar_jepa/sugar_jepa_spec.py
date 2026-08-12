#!/usr/bin/env python3
"""SugarJEPA family metadata (SugarOne channels + JEPA glucose scaler)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from common.data.columns import COL_BASAL, COL_BOLUS, COL_CARB, COL_GLU
from common.model_spec import (
    arch_hparams_from_meta,
    infer_batch_jepa,
    register_family_spec,
)
from common.scalers import ScalerLike, extract_scalers_from_dataset
from sugar_jepa.sugar_jepa_model import SugarJepaModel


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
    ffill_bfill_columns: Sequence[str] = ("glucose", "basal")
    zero_fill_columns: Sequence[str] = ("bolus", "carbs")

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
                meta.get("jepa_weights_dir", "src/sugar_jepa/pretrained/cgm_jepa")
            ),
            jepa_patch_size=int(meta.get("jepa_patch_size", 12)),
            jepa_freeze=not bool(meta.get("finetune_jepa", False)),
        )
        return model.to(device)

    def extract_scalers(self, dataset: Any) -> dict[str, ScalerLike]:
        return extract_scalers_from_dataset(dataset, feature_names=self.feature_names)

    def build_window_dataset(
        self,
        df: Any,
        *,
        input_steps: int,
        horizon: int,
        scalers: Mapping[str, ScalerLike] | None = None,
        fit_scalers: bool = False,
        window_stride: int = 1,
        meta: Mapping[str, Any] | None = None,
    ) -> Any:
        from common.data import SugarJepaWindowDataset

        _ = window_stride
        jepa_window = int((meta or {}).get("jepa_window", 288))
        if fit_scalers or scalers is None:
            return SugarJepaWindowDataset(
                df, input_steps, horizon, jepa_window, fit_scalers=True
            )
        return SugarJepaWindowDataset(
            df,
            input_steps,
            horizon,
            jepa_window,
            scaler_glucose=scalers["glucose"],  # type: ignore[arg-type]
            scaler_basal=scalers["basal"],  # type: ignore[arg-type]
            scaler_bolus=scalers["bolus"],  # type: ignore[arg-type]
            scaler_carbs=scalers["carbs"],  # type: ignore[arg-type]
            scaler_glucose_jepa=scalers["glucose_jepa"],  # type: ignore[arg-type]
            fit_scalers=False,
        )

    def infer_batch(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return infer_batch_jepa(model, batch, device)


SUGAR_JEPA_SPEC = SugarJepaFamilySpec()
register_family_spec(SUGAR_JEPA_SPEC)
