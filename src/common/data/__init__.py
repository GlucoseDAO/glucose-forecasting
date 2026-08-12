#!/usr/bin/env python3
"""Shared data primitives: columns, CSV loading, sliding-window datasets."""
from __future__ import annotations

from common.data.columns import (
    COL_BASAL,
    COL_BOLUS,
    COL_CARB,
    COL_EVENT,
    COL_GLU,
    COL_GLU_VALUE,
    COL_GROUP,
    COL_HR,
    COL_SEQ,
    COL_SPLIT,
    COL_STEPS,
    COL_TS,
    COL_TS_SHORT,
    COL_USER,
    FEAT_BASAL,
    FEAT_BOLUS,
    FEAT_CARBS,
    FEAT_GLUCOSE,
    FEAT_GLUCOSE_JEPA,
    FEAT_HR,
    FEAT_STEPS,
    GLUMIND_CHANNELS,
    GLUMIND_UNI_CHANNELS,
    SUGAR_ONE_CHANNELS,
    TARGET_CHANNEL,
    TS_FORMAT,
)
from common.data.glumind_dataset import GlucoseWindowDataset
from common.data.glumind_uni_dataset import GlucoseUniWindowDataset
from common.data.jepa_dataset import SugarJepaWindowDataset
from common.data.multichannel import MultichannelWindowDataset
from common.data.sugar_one_dataset import SugarOneWindowDataset

__all__ = [
    "COL_BASAL",
    "COL_BOLUS",
    "COL_CARB",
    "COL_EVENT",
    "COL_GLU",
    "COL_GLU_VALUE",
    "COL_GROUP",
    "COL_HR",
    "COL_SEQ",
    "COL_SPLIT",
    "COL_STEPS",
    "COL_TS",
    "COL_TS_SHORT",
    "COL_USER",
    "FEAT_BASAL",
    "FEAT_BOLUS",
    "FEAT_CARBS",
    "FEAT_GLUCOSE",
    "FEAT_GLUCOSE_JEPA",
    "FEAT_HR",
    "FEAT_STEPS",
    "GLUMIND_CHANNELS",
    "GLUMIND_UNI_CHANNELS",
    "SUGAR_ONE_CHANNELS",
    "TARGET_CHANNEL",
    "TS_FORMAT",
    "GlucoseUniWindowDataset",
    "GlucoseWindowDataset",
    "MultichannelWindowDataset",
    "SugarJepaWindowDataset",
    "SugarOneWindowDataset",
]
