#!/usr/bin/env python3
"""GluMind sliding-window dataset (glucose, hr, steps)."""
from __future__ import annotations

import polars as pl
from sklearn.preprocessing import MinMaxScaler

from common.data.columns import GLUMIND_CHANNELS
from common.data.multichannel import MultichannelWindowDataset


class GlucoseWindowDataset(MultichannelWindowDataset):
    """GluMind channels: glucose, hr, steps."""

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        scaler_glucose: MinMaxScaler | None = None,
        scaler_hr: MinMaxScaler | None = None,
        scaler_steps: MinMaxScaler | None = None,
        fit_scalers: bool = False,
    ) -> None:
        scalers: dict[str, MinMaxScaler] | None = None
        if not fit_scalers and scaler_glucose is not None:
            if scaler_hr is None or scaler_steps is None:
                raise ValueError("scaler_hr and scaler_steps required when reusing scalers")
            scalers = {
                "glucose": scaler_glucose,
                "hr": scaler_hr,
                "steps": scaler_steps,
            }
        super().__init__(
            df,
            input_steps,
            horizon,
            GLUMIND_CHANNELS,
            scalers=scalers,
            fit_scalers=fit_scalers or scalers is None,
        )
