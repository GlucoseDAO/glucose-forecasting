#!/usr/bin/env python3
"""SugarOne sliding-window dataset (glucose, basal, bolus, carbs)."""
from __future__ import annotations

import polars as pl
from sklearn.preprocessing import MinMaxScaler

from common.data.columns import SUGAR_ONE_CHANNELS
from common.data.multichannel import MultichannelWindowDataset


class SugarOneWindowDataset(MultichannelWindowDataset):
    """SugarOne channels: glucose, basal, bolus, carbs."""

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        scaler_glucose: MinMaxScaler | None = None,
        scaler_basal: MinMaxScaler | None = None,
        scaler_bolus: MinMaxScaler | None = None,
        scaler_carbs: MinMaxScaler | None = None,
        fit_scalers: bool = False,
        window_stride: int = 1,
    ) -> None:
        scalers: dict[str, MinMaxScaler] | None = None
        if not fit_scalers and scaler_glucose is not None:
            if scaler_basal is None or scaler_bolus is None or scaler_carbs is None:
                raise ValueError(
                    "scaler_basal, scaler_bolus, and scaler_carbs required when reusing scalers"
                )
            scalers = {
                "glucose": scaler_glucose,
                "basal": scaler_basal,
                "bolus": scaler_bolus,
                "carbs": scaler_carbs,
            }
        super().__init__(
            df,
            input_steps,
            horizon,
            SUGAR_ONE_CHANNELS,
            scalers=scalers,
            fit_scalers=fit_scalers or scalers is None,
            window_stride=window_stride,
        )
