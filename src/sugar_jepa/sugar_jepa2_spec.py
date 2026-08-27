#!/usr/bin/env python3
"""SugarJEPA-2 family metadata — SugarOne channels, our own JEPA encoder.

Sibling of :mod:`sugar_jepa.sugar_jepa_spec`, which describes the variant built
on the *vendored, pretrained* CGM-JEPA encoder. This one describes
:class:`~sugar_jepa.sugar_jepa_model.SugarJepaModel2`, whose encoder we train
ourselves (`jepa_pretrain.py`).

The two differ in their data contract, which is why they need separate specs:

* ``sugar_jepa`` feeds the JEPA branch a **second tensor** (its own scaler,
  ``SugarJepaWindowDataset`` yields ``(x, glucose_jepa, y)``).
* ``sugar_jepa2`` keeps SugarOne's plain ``(x, y)`` and reads a **single, longer
  window** — ``max(input_steps, jepa_window)`` steps — out of which the model
  slices each branch's trailing view. So the dataset is literally
  ``SugarOneWindowDataset``, just built at the longer lookback, and there is no
  fifth scaler.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from common.data.columns import COL_BASAL, COL_BOLUS, COL_CARB, COL_GLU
from common.model_spec import (
    arch_hparams_from_meta,
    infer_batch_xy,
    register_family_spec,
)
from common.scalers import ScalerLike, extract_scalers_from_dataset
from sugar_jepa.sugar_jepa_model import SugarJepaModel2

# The CLI's default --jepa-window. Not the fallback used here: the trainer always
# records the value it used, so a run whose metadata lacks the key predates the
# flag and ran at the backbone's window. See jepa2_window().
CLI_DEFAULT_JEPA_WINDOW = 288  # 24h at 5-min sampling


def jepa2_window(input_steps: int, meta: Mapping[str, Any] | None) -> int:
    """The JEPA branch's own lookback for a run.

    Mirrors ``SugarJepaModel2.__init__``, where ``jepa_window=None`` means "same
    window as the backbone" — so a missing or null key resolves to ``input_steps``,
    *not* to the CLI's 288 default. Runs from before the flag existed have no key
    at all, and building them at 288 fails the strict weight load with a
    ``pos_enc.pe`` shape mismatch (36 patches against the checkpoint's 16).
    """
    raw = (meta or {}).get("jepa_window")
    if raw is None:
        return input_steps
    return int(raw)


def jepa2_lookback(input_steps: int, meta: Mapping[str, Any] | None) -> int:
    """Steps the dataset must emit per sample: whichever view is longer.

    Mirrors ``SugarJepaModel2.lookback``; a window built at ``input_steps`` alone
    would be rejected by the model's own forward-pass length check.
    """
    return max(input_steps, jepa2_window(input_steps, meta))


@dataclass(frozen=True)
class SugarJepa2FamilySpec:
    kind: str = "sugar_jepa2"
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
    # Conv1d patchify — unique to our encoder; the vendored one has
    # `jepa_encoder.encoder.*` / `jepa_encoder.proj.*` instead.
    fingerprint_keys: Sequence[str] = ("jepa_encoder.patch_embed.weight",)
    exclude_keys: Sequence[str] = ()
    ffill_bfill_columns: Sequence[str] = ("glucose", "basal")
    zero_fill_columns: Sequence[str] = ("bolus", "carbs")

    def build_model(self, meta: Mapping[str, Any], device: torch.device) -> nn.Module:
        arch = arch_hparams_from_meta(meta)
        model = SugarJepaModel2(
            n_time_steps=arch["input_steps"],
            n_features=self.n_features,
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            ff_units=arch["ff_units"],
            n_blocks=arch["n_blocks"],
            prediction_horizon=arch["horizon"],
            dropout=arch["dropout"],
            jepa_window=jepa2_window(arch["input_steps"], meta),
            jepa_patch_size=int(meta.get("jepa_patch_size", 8)),
            jepa_embed_dim=int(meta.get("jepa_embed_dim", 96)),
            jepa_layers=int(meta.get("jepa_layers", 3)),
            jepa_heads=int(meta.get("jepa_heads", 6)),
            jepa_norm=str(meta.get("jepa_norm", "instance")),
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

        lookback = jepa2_lookback(input_steps, meta)
        if fit_scalers or scalers is None:
            return SugarOneWindowDataset(
                df,
                lookback,
                horizon,
                fit_scalers=True,
                window_stride=window_stride,
            )
        return SugarOneWindowDataset(
            df,
            lookback,
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


SUGAR_JEPA2_SPEC = SugarJepa2FamilySpec()
register_family_spec(SUGAR_JEPA2_SPEC)
