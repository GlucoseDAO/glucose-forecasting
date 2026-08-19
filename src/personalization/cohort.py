"""Phase 4 personalization cohort: Livia, Loop holdouts, joined2 test users."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from common.data.loading import STUDY_GROUP_ORDER, normalize_study_group_label
from personalization.constants import LOOP_HOLDOUT_QUALITY_USERS

LIVIA_CSV: Final[Path] = Path("data/input/personalization/prepared/livia_chronological.csv")
HOLDOUT_CSV_DIR: Final[Path] = Path("data/input/personalization/holdouts")
JOINED2_CSV_DIR: Final[Path] = Path("data/input/personalization/joined2_test")
JOINED2_CSV: Final[Path] = Path("data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv")

COHORT_LIVIA: Final[str] = "livia"
COHORT_QUALITY_HOLDOUT: Final[str] = "quality_holdout"
COHORT_JOINED2_TEST: Final[str] = "joined2_test"

# Two users per AI-READY study group from joined2 Recommended Split == test.
# Rule: largest test-split row count, then User ID ascending. Frozen so reruns
# stay comparable if the CSV later gains rows. T1DM is omitted here — Livia plus
# the six Loop quality holdouts already cover that group.
JOINED2_TEST_USERS: Final[tuple[tuple[str, str], ...]] = (
    ("ai_ready_1030", "Healthy"),
    ("ai_ready_1043", "Healthy"),
    ("ai_ready_1034", "Pre-T2DM"),
    ("ai_ready_1049", "Pre-T2DM"),
    ("ai_ready_1019", "Oral-T2DM"),
    ("ai_ready_1127", "Oral-T2DM"),
    ("ai_ready_1413", "Insulin-T2DM"),
    ("ai_ready_1036", "Insulin-T2DM"),
)


@dataclass(frozen=True)
class Phase4Subject:
    user_id: str
    subject: str
    csv: Path
    cohort: str
    study_group: str
    display: str


def _holdout_subject(user_id: str) -> Phase4Subject:
    return Phase4Subject(
        user_id=user_id,
        subject=f"loop_{user_id}",
        csv=HOLDOUT_CSV_DIR / f"loop_{user_id}_chronological.csv",
        cohort=COHORT_QUALITY_HOLDOUT,
        study_group="T1DM",
        display=f"User {user_id}",
    )


def _joined2_subject(user_id: str, study_group: str) -> Phase4Subject:
    short = user_id.removeprefix("ai_ready_").removeprefix("loop_")
    return Phase4Subject(
        user_id=user_id,
        subject=user_id,
        csv=JOINED2_CSV_DIR / f"{user_id}_chronological.csv",
        cohort=COHORT_JOINED2_TEST,
        study_group=study_group,
        display=f"{short} ({study_group})",
    )


PHASE4_SUBJECTS: tuple[Phase4Subject, ...] = (
    Phase4Subject(
        user_id="livia",
        subject="livia",
        csv=LIVIA_CSV,
        cohort=COHORT_LIVIA,
        study_group="T1DM",
        display="Livia",
    ),
    *(_holdout_subject(uid) for uid in LOOP_HOLDOUT_QUALITY_USERS),
    *(_joined2_subject(uid, group) for uid, group in JOINED2_TEST_USERS),
)

SUBJECT_BY_NAME: dict[str, Phase4Subject] = {spec.subject: spec for spec in PHASE4_SUBJECTS}
SUBJECT_BY_USER: dict[str, Phase4Subject] = {spec.user_id: spec for spec in PHASE4_SUBJECTS}


EXTRA_DISPLAY: dict[str, str] = {
    "livia_indep": "Independent (λ=0)",
    "livia_curr_plain": "Plain curriculum (legacy chain)",
    "livia_curr_lwf": "LwF decay curriculum (legacy chain)",
    "livia_lwf_decay": "LwF decay (from global)",
    "livia_lwf_01": "LwF λ=0.1 (from global)",
    "loop_154_indep": "154 independent (λ=0)",
    "loop_154_lwf_decay": "154 LwF decay (from global)",
    "loop_154_lwf_01": "154 LwF λ=0.1 (from global)",
}


def display_name_for(subject: str) -> str:
    spec = SUBJECT_BY_NAME.get(subject) or SUBJECT_BY_USER.get(subject)
    if spec is not None:
        return spec.display
    return EXTRA_DISPLAY.get(subject, subject)


def original_cohort_subjects() -> tuple[Phase4Subject, ...]:
    return tuple(
        s
        for s in PHASE4_SUBJECTS
        if s.cohort in {COHORT_LIVIA, COHORT_QUALITY_HOLDOUT}
    )


def joined2_test_subjects() -> tuple[Phase4Subject, ...]:
    return tuple(s for s in PHASE4_SUBJECTS if s.cohort == COHORT_JOINED2_TEST)


def select_two_test_users_per_group(user_stats: pl.DataFrame) -> list[tuple[str, str]]:
    """Pick two users per study group: most rows, then User ID ascending.

    ``user_stats`` must have ``uid``, ``group``, ``n_rows``.
    """
    picked: list[tuple[str, str]] = []
    for group in STUDY_GROUP_ORDER:
        part = user_stats.filter(pl.col("group") == group).sort(
            ["n_rows", "uid"], descending=[True, False]
        )
        for row in part.head(2).iter_rows(named=True):
            picked.append((str(row["uid"]), group))
    return picked


def joined2_test_user_stats(joined2_csv: Path = JOINED2_CSV) -> pl.DataFrame:
    lf = pl.scan_csv(joined2_csv, infer_schema_length=10_000)
    df = (
        lf.filter(pl.col("Recommended Split").str.to_lowercase() == "test")
        .with_columns(
            pl.col("User ID").cast(pl.Utf8).alias("uid"),
            pl.col("Study Group").cast(pl.Utf8).alias("group_raw"),
        )
        .group_by(["uid", "group_raw"])
        .agg(pl.len().alias("n_rows"))
        .collect()
    )
    return df.with_columns(
        pl.col("group_raw")
        .map_elements(normalize_study_group_label, return_dtype=pl.Utf8)
        .alias("group")
    )
