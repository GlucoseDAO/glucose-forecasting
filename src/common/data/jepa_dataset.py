#!/usr/bin/env python3
"""SugarJepa dual-view sliding-window dataset (x, glucose_jepa, y).

Kept separate from ``MultichannelWindowDataset`` because samples are a 3-tuple
with an independent JEPA lookback and a StandardScaler glucose view.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
import typer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


class SugarJepaWindowDataset(Dataset):
    """Lazy sliding-window dataset for SugarJepa.

    Each sample provides two views ending at the same point in time ("now"):
      x:            (input_steps, 4) — [glucose, basal, bolus, carbs], MinMax-scaled.
      glucose_jepa: (jepa_window,)   — glucose only, z-score normalized.
      y:            (horizon,)       — future glucose, MinMax-scaled.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        jepa_window: int,
        scaler_glucose: MinMaxScaler | None = None,
        scaler_basal: MinMaxScaler | None = None,
        scaler_bolus: MinMaxScaler | None = None,
        scaler_carbs: MinMaxScaler | None = None,
        scaler_glucose_jepa: StandardScaler | None = None,
        fit_scalers: bool = False,
    ) -> None:
        self.input_steps = input_steps
        self.horizon = horizon
        self.jepa_window = jepa_window
        self.lookback = max(input_steps, jepa_window)
        window_len = self.lookback + horizon

        raw_glucose: list[np.ndarray] = []
        raw_basal: list[np.ndarray] = []
        raw_bolus: list[np.ndarray] = []
        raw_carbs: list[np.ndarray] = []
        uids: list = []
        sgroups: list[str] = []

        for (uid_val,), grp in (
            df.sort(["unique_id", "ds"]).group_by(["unique_id"], maintain_order=True)
        ):
            uids.append(uid_val)
            sgroups.append(grp["study_group"][0])
            raw_glucose.append(grp["glucose"].to_numpy())
            raw_basal.append(grp["basal"].to_numpy())
            raw_bolus.append(grp["bolus"].to_numpy())
            raw_carbs.append(grp["carbs"].to_numpy())

        if fit_scalers or scaler_glucose is None:
            all_g = np.concatenate(raw_glucose).reshape(-1, 1)
            all_b = np.concatenate(raw_basal).reshape(-1, 1)
            all_bo = np.concatenate(raw_bolus).reshape(-1, 1)
            all_c = np.concatenate(raw_carbs).reshape(-1, 1)
            self.scaler_glucose = MinMaxScaler().fit(all_g)
            self.scaler_basal = MinMaxScaler().fit(all_b)
            self.scaler_bolus = MinMaxScaler().fit(all_bo)
            self.scaler_carbs = MinMaxScaler().fit(all_c)
            self.scaler_glucose_jepa = StandardScaler().fit(all_g)
        else:
            if (
                scaler_basal is None
                or scaler_bolus is None
                or scaler_carbs is None
                or scaler_glucose_jepa is None
            ):
                raise ValueError("All SugarJepa scalers required when fit_scalers=False")
            self.scaler_glucose = scaler_glucose
            self.scaler_basal = scaler_basal
            self.scaler_bolus = scaler_bolus
            self.scaler_carbs = scaler_carbs
            self.scaler_glucose_jepa = scaler_glucose_jepa

        self._series_g: list[np.ndarray] = []
        self._series_g_jepa: list[np.ndarray] = []
        self._series_b: list[np.ndarray] = []
        self._series_bo: list[np.ndarray] = []
        self._series_c: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []
        self.series_ids: list = []
        self.study_groups: list[str] = []

        n_skipped = 0
        for i, (uid, sg, rg, rb, rbo, rc) in enumerate(
            zip(uids, sgroups, raw_glucose, raw_basal, raw_bolus, raw_carbs)
        ):
            g = self.scaler_glucose.transform(rg.reshape(-1, 1)).ravel().astype(np.float32)
            g_jepa = (
                self.scaler_glucose_jepa.transform(rg.reshape(-1, 1)).ravel().astype(np.float32)
            )
            b = self.scaler_basal.transform(rb.reshape(-1, 1)).ravel().astype(np.float32)
            bo = self.scaler_bolus.transform(rbo.reshape(-1, 1)).ravel().astype(np.float32)
            c = self.scaler_carbs.transform(rc.reshape(-1, 1)).ravel().astype(np.float32)
            self._series_g.append(g)
            self._series_g_jepa.append(g_jepa)
            self._series_b.append(b)
            self._series_bo.append(bo)
            self._series_c.append(c)
            n_windows = len(g) - window_len + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            for start in range(n_windows):
                self._index.append((i, start))
                self.series_ids.append(uid)
                self.study_groups.append(sg)

        if n_skipped > 0:
            typer.echo(
                f"  Note: Skipped {n_skipped} series shorter than {window_len} steps "
                f"(lookback={self.lookback} = max(input_steps={input_steps}, "
                f"jepa_window={jepa_window})."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        si, start = self._index[idx]
        g = self._series_g[si]
        g_jepa = self._series_g_jepa[si]
        b = self._series_b[si]
        bo = self._series_bo[si]
        c = self._series_c[si]

        now = start + self.lookback
        x_start = now - self.input_steps
        jepa_start = now - self.jepa_window

        x = np.stack(
            [g[x_start:now], b[x_start:now], bo[x_start:now], c[x_start:now]],
            axis=-1,
        )
        jepa = g_jepa[jepa_start:now]
        y = g[now : now + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(jepa), torch.from_numpy(y)
