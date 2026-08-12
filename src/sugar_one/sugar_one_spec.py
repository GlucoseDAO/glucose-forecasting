#!/usr/bin/env python3
"""SugarOne family metadata (features, CSV map, build, scaler extract)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from common.data.columns import COL_BASAL, COL_BOLUS, COL_CARB, COL_GLU
from common.model_spec import arch_hparams_from_meta, infer_batch_xy, register_family_spec
from common.scalers import ScalerLike, extract_scalers_from_dataset
from sugar_one.sugar_one_model import SugarOneModel


@dataclass(frozen=True)
class SugarOneFamilySpec:
    kind: str = "sugar_one"
    feature_names: Sequence[str] = ("glucose", "basal", "bolus", "carbs")
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
    fingerprint_keys: Sequence[str] = (
        "embed_basal.weight",
        "embed_bolus.weight",
        "embed_carbs.weight",
    )
    ffill_bfill_columns: Sequence[str] = ("glucose", "basal")
    zero_fill_columns: Sequence[str] = ("bolus", "carbs")

    def build_model(self, meta: Mapping[str, Any], device: torch.device) -> nn.Module:
        arch = arch_hparams_from_meta(meta)
        model = SugarOneModel(
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
        from common.data import SugarOneWindowDataset

        _ = meta
        if fit_scalers or scalers is None:
            return SugarOneWindowDataset(
                df,
                input_steps,
                horizon,
                fit_scalers=True,
                window_stride=window_stride,
            )
        return SugarOneWindowDataset(
            df,
            input_steps,
            horizon,
            scaler_glucose=scalers["glucose"],  # type: ignore[arg-type]
            scaler_basal=scalers["basal"],  # type: ignore[arg-type]
            scaler_bolus=scalers["bolus"],  # type: ignore[arg-type]
            scaler_carbs=scalers["carbs"],  # type: ignore[arg-type]
            fit_scalers=False,
            window_stride=window_stride,
        )

    def infer_batch(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return infer_batch_xy(model, batch, device)


SUGAR_ONE_SPEC = SugarOneFamilySpec()
register_family_spec(SUGAR_ONE_SPEC)
