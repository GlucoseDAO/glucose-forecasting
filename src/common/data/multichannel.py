#!/usr/bin/env python3
"""Parameterized multichannel sliding-window dataset ``(x, y)``."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from common.data.columns import TARGET_CHANNEL


class MultichannelWindowDataset(Dataset):
    """Lazy sliding-window dataset: ``x`` is ``(input_steps, C)``, ``y`` is ``(horizon,)``.

    Stores only scaled per-series arrays; windows are sliced in ``__getitem__``.
    Target channel (default ``glucose``) is always the forecast target.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        channels: Sequence[str],
        *,
        scalers: Mapping[str, MinMaxScaler] | None = None,
        fit_scalers: bool = False,
        window_stride: int = 1,
        target_channel: str = TARGET_CHANNEL,
    ) -> None:
        if not channels:
            raise ValueError("channels must be non-empty")
        if target_channel not in channels:
            raise ValueError(
                f"target_channel {target_channel!r} must be one of {list(channels)}"
            )
        if window_stride < 1:
            raise ValueError(f"window_stride must be >= 1, got {window_stride}")

        self.input_steps = input_steps
        self.horizon = horizon
        self.channels = tuple(channels)
        self.target_channel = target_channel
        self.window_stride = window_stride
        window_len = input_steps + horizon

        missing = [c for c in self.channels if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        raw_by_channel: dict[str, list[np.ndarray]] = {c: [] for c in self.channels}
        uids: list = []
        sgroups: list[str] = []
        for (uid_val,), grp in df.sort(["unique_id", "ds"]).group_by(
            ["unique_id"], maintain_order=True
        ):
            uids.append(uid_val)
            sgroups.append(grp["study_group"][0])
            for channel in self.channels:
                raw_by_channel[channel].append(grp[channel].to_numpy())

        fitted: dict[str, MinMaxScaler] = {}
        if fit_scalers or scalers is None:
            for channel in self.channels:
                all_vals = np.concatenate(raw_by_channel[channel]).reshape(-1, 1)
                fitted[channel] = MinMaxScaler().fit(all_vals)
        else:
            for channel in self.channels:
                if channel not in scalers:
                    raise ValueError(f"Missing scaler for channel {channel!r}")
                fitted[channel] = scalers[channel]

        for channel, scaler in fitted.items():
            setattr(self, f"scaler_{channel}", scaler)

        self._series: dict[str, list[np.ndarray]] = {c: [] for c in self.channels}
        self._index: list[tuple[int, int]] = []
        self.series_ids: list = []
        self.study_groups: list[str] = []

        n_skipped = 0
        n_series = len(uids)
        for i in range(n_series):
            scaled: dict[str, np.ndarray] = {}
            for channel in self.channels:
                raw = raw_by_channel[channel][i]
                scaled[channel] = (
                    fitted[channel]
                    .transform(raw.reshape(-1, 1))
                    .ravel()
                    .astype(np.float32)
                )
                self._series[channel].append(scaled[channel])

            n_windows = len(scaled[target_channel]) - window_len + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            for start in range(0, n_windows, window_stride):
                self._index.append((i, start))
                self.series_ids.append(uids[i])
                self.study_groups.append(sgroups[i])

        if n_skipped > 0:
            print(
                f"  Note: Skipped {n_skipped} series/segments shorter than "
                f"{window_len} steps."
            )

        # Backward-compatible aliases used by older GluMind/SugarOne code paths.
        if "glucose" in self._series:
            self._series_g = self._series["glucose"]
        if "hr" in self._series:
            self._series_h = self._series["hr"]
        if "steps" in self._series:
            self._series_s = self._series["steps"]
        if "basal" in self._series:
            self._series_b = self._series["basal"]
        if "bolus" in self._series:
            self._series_bo = self._series["bolus"]
        if "carbs" in self._series:
            self._series_c = self._series["carbs"]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        series_idx, start = self._index[idx]
        end = start + self.input_steps
        stacks = [self._series[c][series_idx][start:end] for c in self.channels]
        x = np.stack(stacks, axis=-1)
        y = self._series[self.target_channel][series_idx][end : end + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)
