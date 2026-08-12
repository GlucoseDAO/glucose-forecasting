#!/usr/bin/env python3
"""GluMind-Uni sliding-window dataset (glucose only)."""
from __future__ import annotations

import polars as pl
from sklearn.preprocessing import MinMaxScaler

from common.data.columns import GLUMIND_UNI_CHANNELS
from common.data.multichannel import MultichannelWindowDataset


class GlucoseUniWindowDataset(MultichannelWindowDataset):
    """Glucose-only GluMind-Uni windows."""

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        scaler_glucose: MinMaxScaler | None = None,
        fit_scalers: bool = False,
    ) -> None:
        scalers: dict[str, MinMaxScaler] | None = None
        if not fit_scalers and scaler_glucose is not None:
            scalers = {"glucose": scaler_glucose}
        super().__init__(
            df,
            input_steps,
            horizon,
            GLUMIND_UNI_CHANNELS,
            scalers=scalers,
            fit_scalers=fit_scalers or scalers is None,
        )
