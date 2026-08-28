"""Defaults for NeuralForecast personalization."""
from __future__ import annotations

from pathlib import Path
from typing import Final

from common.paths import DEFAULT_RUNS_ROOT
from personalization.constants import DEFAULT_DATA_SIZE_DAYS, DEFAULT_SEED

DEFAULT_NF_HOLDOUT_ROOT: Final[Path] = DEFAULT_RUNS_ROOT / "nf_holdout"
DEFAULT_NF_PERSONALIZATION_ROOT: Final[Path] = DEFAULT_RUNS_ROOT / "personalization_nf"
DEFAULT_REPORT_PATH: Final[Path] = Path("docs") / "PERSONALIZATION_NF_REPORT.md"
DEFAULT_FIGURES_DIR: Final[Path] = Path("docs") / "figures" / "personalization_nf"

# Same day grid as SugarOne personalization ("all" = full personal train split).
DATA_SIZE_DAYS: Final[tuple[int | str, ...]] = DEFAULT_DATA_SIZE_DAYS

# Fine-tune keeps the source run's max_steps / lr; ES uses this patience.
DEFAULT_FT_PATIENCE: Final[int] = 10
# Cap train-tail validation so short day budgets still have input+horizon left.
VAL_TAIL_FRACTION: Final[float] = 0.20
DEFAULT_SEED_NF: Final[int] = DEFAULT_SEED

METRICS_FILENAME: Final[str] = "personalization_metrics.json"
ZERO_SHOT_DIRNAME: Final[str] = "zero_shot"
