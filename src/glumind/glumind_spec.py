#!/usr/bin/env python3
"""GluMind family metadata (features, CSV map, build, scaler extract)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from common.data.columns import COL_GLU_VALUE, COL_HR, COL_STEPS
from common.model_spec import arch_hparams_from_meta, infer_batch_xy, register_family_spec
from common.scalers import ScalerLike, extract_scalers_from_dataset
from glumind.glumind_model import GluMindModel

COL_GLU = COL_GLU_VALUE


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
    exclude_keys: Sequence[str] = ()
    ffill_bfill_columns: Sequence[str] = ("glucose", "hr", "steps")
    zero_fill_columns: Sequence[str] = ()

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
        from common.data import GlucoseWindowDataset

        _ = window_stride, meta
        if fit_scalers or scalers is None:
            return GlucoseWindowDataset(df, input_steps, horizon, fit_scalers=True)
        return GlucoseWindowDataset(
            df,
            input_steps,
            horizon,
            scaler_glucose=scalers["glucose"],  # type: ignore[arg-type]
            scaler_hr=scalers["hr"],  # type: ignore[arg-type]
            scaler_steps=scalers["steps"],  # type: ignore[arg-type]
            fit_scalers=False,
        )

    def infer_batch(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return infer_batch_xy(model, batch, device)


GLUMIND_SPEC = GluMindFamilySpec()
register_family_spec(GLUMIND_SPEC)
