#!/usr/bin/env python3
"""Prepare personalization CSVs with chronological train/val/test splits.

Subcommands:
  livia         — assign chronological splits (default: Livia SugarOne fixture)
  holdouts      — extract Loop quality holdout users and assign chronological splits
  joined2-test  — extract two joined2 test users per study group
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from personalization.constants import (
    COL_EVENT,
    COL_GLU,
    COL_GROUP,
    COL_SEQ,
    COL_SPLIT,
    COL_TS,
    COL_USER,
    DEFAULT_LIVIA_PREPARED_DIR,
    DEFAULT_LIVIA_PREPARED_NAME,
    DEFAULT_LIVIA_SOURCE_CSV,
    DEFAULT_STUDY_GROUP,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION_OF_REMAINDER,
    LOOP_HOLDOUT_QUALITY_USERS,
    TS_FORMAT,
)
from personalization.cohort import (
    JOINED2_CSV,
    JOINED2_CSV_DIR,
    JOINED2_TEST_USERS,
)
from personalization.splits import (
    chronological_split_labels,
    limit_train_days,
    split_meta,
    write_split_meta,
)

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="Prepare personalization CSVs with chronological splits.",
)

REQUIRED_LOOP_COLS = (
    COL_SEQ,
    COL_TS,
    COL_EVENT,
    COL_USER,
    COL_GLU,
    "Basal Rate (U/h)",
    "Bolus Insulin (U)",
    "Carbohydrates (g)",
)


def _format_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure Timestamp is string in TS_FORMAT for downstream loaders."""
    ts = df[COL_TS]
    if ts.dtype == pl.Utf8 or ts.dtype == pl.String:
        return df
    return df.with_columns(
        pl.col(COL_TS).dt.strftime(TS_FORMAT).alias(COL_TS)
    )


def _ensure_study_group(df: pl.DataFrame, group: str = DEFAULT_STUDY_GROUP) -> pl.DataFrame:
    if COL_GROUP in df.columns:
        return df.with_columns(
            pl.when(pl.col(COL_GROUP).is_null() | (pl.col(COL_GROUP).cast(pl.Utf8) == ""))
            .then(pl.lit(group))
            .otherwise(pl.col(COL_GROUP))
            .alias(COL_GROUP)
        )
    return df.with_columns(pl.lit(group).alias(COL_GROUP))


def prepare_person_frame(
    df: pl.DataFrame,
    *,
    test_fraction: float,
    val_fraction_of_remainder: float,
    personal_days: int | None,
    study_group: str,
) -> tuple[pl.DataFrame, dict]:
    missing = [c for c in (COL_TS, COL_USER, COL_GLU) if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"CSV missing required columns: {missing}")

    work = _ensure_study_group(df, study_group)
    # Drop existing split if present — we always reassign chronologically.
    if COL_SPLIT in work.columns:
        work = work.drop(COL_SPLIT)

    labeled = chronological_split_labels(
        work,
        test_fraction=test_fraction,
        val_fraction_of_remainder=val_fraction_of_remainder,
    )
    if personal_days is not None:
        labeled = limit_train_days(labeled, personal_days)

    labeled = _format_timestamps(labeled)
    meta = split_meta(labeled)
    meta["test_fraction"] = test_fraction
    meta["val_fraction_of_remainder"] = val_fraction_of_remainder
    meta["personal_days"] = personal_days
    return labeled, meta


def ensure_holdout_csv(
    loop_csv: Path,
    user_id: str,
    out_dir: Path,
    test_fraction: float,
    val_fraction_of_remainder: float,
) -> Path:
    """Write one chronological holdout CSV if it does not already exist."""
    out_csv = out_dir / f"loop_{user_id}_chronological.csv"
    if out_csv.exists():
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    person = (
        pl.scan_csv(loop_csv, infer_schema_length=10_000)
        .with_columns(pl.col(COL_USER).cast(pl.Utf8))
        .filter(pl.col(COL_USER) == user_id)
        .collect()
    )
    if person.is_empty():
        raise ValueError(f"User {user_id} not found in {loop_csv}")

    labeled, meta = prepare_person_frame(
        person,
        test_fraction=test_fraction,
        val_fraction_of_remainder=val_fraction_of_remainder,
        personal_days=None,
        study_group="T1DM",
    )
    meta["source"] = str(loop_csv)
    meta["subject"] = f"loop_{user_id}"
    meta["user_ids"] = [user_id]
    labeled.write_csv(out_csv)
    write_split_meta(out_dir / f"loop_{user_id}_split_meta.json", meta)
    return out_csv


@app.command("livia")
def prepare_livia(
    input: Path = typer.Option(
        DEFAULT_LIVIA_SOURCE_CSV,
        "--input",
        help="Livia SugarOne CSV (default: fixtures/livia_data/livia_sugar_one_ready.csv).",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_LIVIA_PREPARED_DIR,
        "--out-dir",
        help="Output directory for prepared CSV + split_meta.json.",
    ),
    out_name: str = typer.Option(
        DEFAULT_LIVIA_PREPARED_NAME,
        "--out-name",
        help="Output CSV filename.",
    ),
    test_fraction: float = typer.Option(DEFAULT_TEST_FRACTION, "--test-fraction"),
    val_fraction_of_remainder: float = typer.Option(
        DEFAULT_VAL_FRACTION_OF_REMAINDER, "--val-fraction-of-remainder"
    ),
    personal_days: Optional[int] = typer.Option(
        None,
        "--personal-days",
        help="If set, keep only the first N days of the train split.",
    ),
    study_group: str = typer.Option(DEFAULT_STUDY_GROUP, "--study-group"),
) -> None:
    """Assign chronological train/val/test on Livia personal data."""
    if not input.exists():
        typer.echo(f"Error: input not found: {input}", err=True)
        raise typer.Exit(1)

    df = pl.read_csv(input, infer_schema_length=10_000)
    labeled, meta = prepare_person_frame(
        df,
        test_fraction=test_fraction,
        val_fraction_of_remainder=val_fraction_of_remainder,
        personal_days=personal_days,
        study_group=study_group,
    )
    meta["source"] = str(input)
    meta["subject"] = "livia"
    users = labeled[COL_USER].unique().to_list()
    meta["user_ids"] = [str(u) for u in users]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / out_name
    labeled.write_csv(out_csv)
    write_split_meta(out_dir / f"{Path(out_name).stem}_split_meta.json", meta)
    typer.echo(f"Wrote {out_csv} ({labeled.height:,} rows)")
    typer.echo(json.dumps(meta, indent=2))


@app.command("holdouts")
def prepare_holdouts(
    loop_csv: Path = typer.Option(
        Path("data/input/loop_and_ai_ready/loop.csv"),
        "--loop-csv",
        help="Full Loop export CSV.",
    ),
    out_dir: Path = typer.Option(
        Path("data/input/personalization/holdouts"),
        "--out-dir",
    ),
    users: Optional[str] = typer.Option(
        None,
        "--users",
        help="Comma-separated User IDs (default: quality holdout list).",
    ),
    test_fraction: float = typer.Option(DEFAULT_TEST_FRACTION, "--test-fraction"),
    val_fraction_of_remainder: float = typer.Option(
        DEFAULT_VAL_FRACTION_OF_REMAINDER, "--val-fraction-of-remainder"
    ),
    personal_days: Optional[int] = typer.Option(None, "--personal-days"),
    study_group: str = typer.Option(DEFAULT_STUDY_GROUP, "--study-group"),
) -> None:
    """Extract Loop holdout users and write one chronological CSV each."""
    if not loop_csv.exists():
        typer.echo(f"Error: loop CSV not found: {loop_csv}", err=True)
        raise typer.Exit(1)

    user_list = (
        [u.strip() for u in users.split(",") if u.strip()]
        if users
        else list(LOOP_HOLDOUT_QUALITY_USERS)
    )
    typer.echo(f"Scanning {loop_csv} for users: {user_list}")

    # Cast User ID to string for reliable filtering.
    lf = pl.scan_csv(loop_csv, infer_schema_length=10_000).with_columns(
        pl.col(COL_USER).cast(pl.Utf8)
    )
    available = set(lf.select(COL_USER).unique().collect()[COL_USER].to_list())
    missing_users = [u for u in user_list if u not in available]
    if missing_users:
        typer.echo(f"Warning: users not found in loop.csv: {missing_users}", err=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []

    for uid in user_list:
        if uid not in available:
            continue
        person = lf.filter(pl.col(COL_USER) == uid).collect()
        for col in REQUIRED_LOOP_COLS:
            if col not in person.columns:
                typer.echo(f"Error: {uid} missing column {col}", err=True)
                raise typer.Exit(1)

        labeled, meta = prepare_person_frame(
            person,
            test_fraction=test_fraction,
            val_fraction_of_remainder=val_fraction_of_remainder,
            personal_days=personal_days,
            study_group=study_group,
        )
        meta["source"] = str(loop_csv)
        meta["subject"] = f"loop_{uid}"
        meta["user_ids"] = [uid]

        out_csv = out_dir / f"loop_{uid}_chronological.csv"
        labeled.write_csv(out_csv)
        write_split_meta(out_dir / f"loop_{uid}_split_meta.json", meta)
        typer.echo(f"Wrote {out_csv} ({labeled.height:,} rows)")
        index.append({"user_id": uid, "csv": str(out_csv), **meta})

    index_path = out_dir / "holdouts_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    typer.echo(f"Wrote index {index_path} ({len(index)} users)")


@app.command("joined2-test")
def prepare_joined2_test(
    joined2_csv: Path = typer.Option(
        JOINED2_CSV,
        "--joined2-csv",
        help="loop_ai_ready_joined2.csv with Recommended Split + Study Group.",
    ),
    out_dir: Path = typer.Option(
        JOINED2_CSV_DIR,
        "--out-dir",
    ),
    test_fraction: float = typer.Option(DEFAULT_TEST_FRACTION, "--test-fraction"),
    val_fraction_of_remainder: float = typer.Option(
        DEFAULT_VAL_FRACTION_OF_REMAINDER, "--val-fraction-of-remainder"
    ),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite"),
) -> None:
    """Extract two joined2 test users per study group; chronological splits."""
    if not joined2_csv.exists():
        typer.echo(f"Error: joined2 CSV not found: {joined2_csv}", err=True)
        raise typer.Exit(1)

    user_ids = [uid for uid, _group in JOINED2_TEST_USERS]
    typer.echo(f"Scanning {joined2_csv} for {len(user_ids)} joined2 test users")

    lf = pl.scan_csv(joined2_csv, infer_schema_length=10_000).with_columns(
        pl.col(COL_USER).cast(pl.Utf8)
    )
    people = lf.filter(pl.col(COL_USER).is_in(user_ids)).collect()
    found = set(people[COL_USER].unique().to_list())
    missing = [uid for uid in user_ids if uid not in found]
    if missing:
        typer.echo(f"Error: users not found in joined2 CSV: {missing}", err=True)
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for uid, study_group in JOINED2_TEST_USERS:
        out_csv = out_dir / f"{uid}_chronological.csv"
        meta_path = out_dir / f"{uid}_split_meta.json"
        if skip_existing and out_csv.exists() and meta_path.exists():
            typer.echo(f"Skip existing {out_csv}")
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            index.append({"user_id": uid, "csv": str(out_csv), **existing})
            continue
        person = people.filter(pl.col(COL_USER) == uid)
        labeled, meta = prepare_person_frame(
            person,
            test_fraction=test_fraction,
            val_fraction_of_remainder=val_fraction_of_remainder,
            personal_days=None,
            study_group=study_group,
        )
        meta["source"] = str(joined2_csv)
        meta["subject"] = uid
        meta["user_ids"] = [uid]
        meta["study_group"] = study_group
        meta["cohort"] = "joined2_test"
        labeled.write_csv(out_csv)
        write_split_meta(meta_path, meta)
        typer.echo(f"Wrote {out_csv} ({labeled.height:,} rows, {study_group})")
        index.append({"user_id": uid, "csv": str(out_csv), **meta})

    index_path = out_dir / "joined2_test_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    typer.echo(f"Wrote index {index_path} ({len(index)} users)")


def main() -> None:
    # Ensure project root imports work when invoked as a script.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()


if __name__ == "__main__":
    main()
