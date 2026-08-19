"""Shared constants for SugarOne personalization (insulin + carbs)."""
from __future__ import annotations

from pathlib import Path
from typing import Final

from common.paths import (
    DEFAULT_PERSONALIZATION_DATA_ROOT,
    DEFAULT_SUGAR_ONE_CHECKPOINT,
    LIVIA_SUGAR_ONE_CSV,
)

# Production personalization: global SugarOne checkpoint + Livia demo CSV.
DEFAULT_BASE_RUN_DIR: Final[str] = DEFAULT_SUGAR_ONE_CHECKPOINT.as_posix()
DEFAULT_LIVIA_SOURCE_CSV: Final[Path] = LIVIA_SUGAR_ONE_CSV
DEFAULT_LIVIA_PREPARED_DIR: Final[Path] = DEFAULT_PERSONALIZATION_DATA_ROOT / "prepared"
DEFAULT_LIVIA_PREPARED_NAME: Final[str] = "livia_chronological.csv"
DEFAULT_LIVIA_PREPARED_CSV: Final[Path] = (
    DEFAULT_LIVIA_PREPARED_DIR / DEFAULT_LIVIA_PREPARED_NAME
)

# Quality Loop users present in loop.csv but excluded from loop_ai_ready_joined2.csv
# (passed basal/bolus/carb completeness; not selected by row-balance builder).
LOOP_HOLDOUT_QUALITY_USERS: Final[tuple[str, ...]] = (
    "154",
    "556",
    "730",
    "1017",
    "1029",
    "1082",
)

# Extra Loop User IDs from joined2 test. Empty: T1DM is already covered by
# Livia + LOOP_HOLDOUT_QUALITY_USERS, so loop_467 / loop_745 are not run.
LOOP_TEST_EXTRA_USERS: Final[tuple[str, ...]] = ()

# Loop numeric User IDs used in personalization (quality holdouts only).
LOOP_PERSONALIZATION_COHORT_USERS: Final[tuple[str, ...]] = (
    LOOP_HOLDOUT_QUALITY_USERS + LOOP_TEST_EXTRA_USERS
)

# Step 2b pilot: enough users to judge LR transfer; remainder deferred for later report.
HOLDOUT_LR_PILOT_USERS: Final[tuple[str, ...]] = ("154", "556", "730")
HOLDOUT_LR_DEFERRED_USERS: Final[tuple[str, ...]] = ("1017", "1029", "1082")

# Default chronological split fractions (of the person's full timeline).
DEFAULT_TEST_FRACTION: Final[float] = 0.25
DEFAULT_VAL_FRACTION_OF_REMAINDER: Final[float] = 0.15

# Step 3: data-size sweep grid (personal train days). "all" = full train split.
DEFAULT_DATA_SIZE_DAYS: Final[tuple[int | str, ...]] = (1, 3, 7, 14, 30, 60, "all")

# Independent-from-global LwF (teacher = sugar_one_1.0). λ=0 from 30 days
# reuses the existing λ=0 independent data-size runs.
LWF_CURRICULUM_ZERO_FROM_DAYS: Final[int] = 30
LWF_DECAY_START: Final[float] = 0.5
LWF_DECAY_SCHEDULE: Final[dict[int, float]] = {
    1: 0.5,
    3: 0.4,
    7: 0.3,
    14: 0.2,
}
LWF_CONST_LAMBDA: Final[float] = 0.1


def decaying_lwf_lambda(
    day_budget: int | None,
    *,
    schedule: dict[int, float] | None = None,
    zero_from_days: int = LWF_CURRICULUM_ZERO_FROM_DAYS,
) -> float:
    """High LwF on short histories; 0 from ``zero_from_days`` onward (incl. all)."""
    table = LWF_DECAY_SCHEDULE if schedule is None else schedule
    if day_budget is None or day_budget >= zero_from_days:
        return 0.0
    if day_budget in table:
        return float(table[day_budget])
    span = float(zero_from_days)
    return round(max(table.values()) * max(0.0, 1.0 - day_budget / span), 4)


# Step 2: LR grid for holdout transfer check (Livia best was 2e-4).
DEFAULT_HOLDOUT_LR_GRID: Final[tuple[float, ...]] = (0.0001, 0.0002, 0.0004)
LIVIA_REFERENCE_LR: Final[float] = 0.0002

# Step 2: LR multipliers relative to base model ``tuning_meta.json`` lr (sugar_one_1.0: 4e-4).
DEFAULT_LR_MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)

# weight_decay fixed at DEFAULT_WEIGHT_DECAY for personalization; multipliers for legacy sweeps only.
DEFAULT_WEIGHT_DECAY: Final[float] = 3e-5
DEFAULT_WEIGHT_DECAY_MULTIPLIERS: Final[tuple[float, ...]] = (1.0,)

# Personalization fine-tune defaults (Step 2+).
# Production path: plain fine-tune on global checkpoint (no LwF teacher).
DEFAULT_PERSONAL_LWF_LAMBDA: Final[float] = 0.0
DEFAULT_FT_PATIENCE: Final[int] = 3
DEFAULT_VAL_EVERY_N_EPOCHS: Final[int] = 2
DEFAULT_PROGRESS_LOG_INTERVAL_S: Final[float] = 10.0

# Sliding-window stride for train windows (5-min CGM steps).
DENSE_WINDOW_STRIDE: Final[int] = 1
SPARSE_WINDOW_STRIDE: Final[int] = 6  # 6×5 min = 30 min between window starts
DEFAULT_TRAIN_WINDOW_STRIDE: Final[int] = SPARSE_WINDOW_STRIDE

DEFAULT_SEED: Final[int] = 43
DEFAULT_HORIZON: Final[int] = 12
DEFAULT_STUDY_GROUP: Final[str] = "T1DM"

# Loop / SugarOne CSV column names.
COL_SEQ: Final[str] = "sequence_id"
COL_USER: Final[str] = "User ID"
COL_TS: Final[str] = "Timestamp"
COL_SPLIT: Final[str] = "Recommended Split"
COL_GROUP: Final[str] = "Study Group"
COL_EVENT: Final[str] = "Event Type"
COL_GLU: Final[str] = "Glucose (mg/dL)"
COL_BASAL: Final[str] = "Basal Rate (U/h)"
COL_BOLUS: Final[str] = "Bolus Insulin (U)"
COL_CARB: Final[str] = "Carbohydrates (g)"

TS_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"

SUGAR_ONE_VALUE_COLUMNS: Final[dict[str, str]] = {
    "glucose": COL_GLU,
    "basal": COL_BASAL,
    "bolus": COL_BOLUS,
    "carbs": COL_CARB,
}
