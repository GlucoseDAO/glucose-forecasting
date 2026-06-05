#!/usr/bin/env python3
"""
Join `data/loop_and_ai_ready/loop.csv` with `ai_ready_full4.csv` into one dataset.

- Aligns column names and adds insulin + carb covariates (null on ai_ready rows).
- Prefixes sequence_id and User ID so the two sources never collide.
- Assigns loop `Recommended Split` user-wise with the same train/val/test *row*
  proportions as in ai_ready_full4, using a greedy partition on users so each
  split gets similar fractions of loop rows (helps future balancing vs ai_ready).
- Adds `Data Source` (`loop` | `ai_ready`) for downstream weighting or filtering.

Default output: Parquet (streaming-friendly). Use --format csv if needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
import typer

app = typer.Typer(add_completion=False)

# From ai_ready_full4.csv aggregate split (train+val+test = 6027765 rows).
TRAIN_ROW_FRACTION = 4_195_399 / 6_027_765
VAL_ROW_FRACTION = 910_133 / 6_027_765
TEST_ROW_FRACTION = 922_233 / 6_027_765

COL_TS = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_GLU = "Glucose Value (mg/dL)"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_EVENT = "Event Type"

DATA_SOURCE_LOOP = "loop"
DATA_SOURCE_AI = "ai_ready"

SEQ_PREFIX_LOOP = "L-"
SEQ_PREFIX_AI = "A-"
USER_PREFIX_LOOP = "loop_"
USER_PREFIX_AI = "ai_ready_"

LOOP_STUDY_GROUP = "T1DM"

COL_BASAL = "Basal Rate (U/h)"
COL_BOLUS = "Bolus Insulin (U)"
COL_CARB = "Carbohydrates (g)"
COL_BASAL_OBS = "Basal Observed"
COL_BOLUS_OBS = "Bolus Observed"
COL_CARB_OBS = "Carb Observed"
COL_DATA_SOURCE = "Data Source"


def greedy_user_splits_by_row_mass(
    user_counts: pl.DataFrame,
    user_col: str,
    count_col: str,
) -> pl.DataFrame:
    """
    Assign each user to train/val/test so split row totals approximate
    TRAIN_ROW_FRACTION / VAL_ROW_FRACTION / TEST_ROW_FRACTION of loop rows.
    Users processed in descending row count so heavy users are split first.
    """
    total_rows = int(user_counts[count_col].sum())
    if total_rows <= 0:
        raise ValueError("No rows in loop user count table.")

    targets = {
        "train": TRAIN_ROW_FRACTION * total_rows,
        "val": VAL_ROW_FRACTION * total_rows,
        "test": TEST_ROW_FRACTION * total_rows,
    }
    currents = {"train": 0.0, "val": 0.0, "test": 0.0}
    ordered = user_counts.sort(count_col, descending=True)
    splits: list[str] = []

    for _uid, n in ordered.select([user_col, count_col]).iter_rows():
        n_f = float(n)
        scores = {s: currents[s] / targets[s] for s in currents}
        best = min(scores, key=scores.get)
        splits.append(best)
        currents[best] += n_f

    return ordered.with_columns(pl.Series(COL_SPLIT, splits))


def final_column_order() -> list[str]:
    """Stable column order for both branches before concat."""
    return [
        COL_SEQ,
        COL_TS,
        COL_EVENT,
        COL_USER,
        COL_GLU,
        COL_BASAL,
        COL_BOLUS,
        COL_CARB,
        "Heart Rate",
        "Step Count",
        "Active Calories (kcal)",
        "Stress Level",
        "Respiratory Rate",
        "Oxygen Saturation (%)",
        "Age",
        COL_SPLIT,
        COL_GROUP,
        COL_DATA_SOURCE,
        "Glucose Observed",
        "Steps Observed",
        "HR Observed",
        "kcal_observed",
        "stress_observed",
        "resp_observed",
        "spo2_observed",
        COL_BASAL_OBS,
        COL_BOLUS_OBS,
        COL_CARB_OBS,
    ]


def build_loop_lazy(
    loop_path: Path,
    split_map: pl.DataFrame,
) -> pl.LazyFrame:
    lf = pl.scan_csv(loop_path, infer_schema_length=10_000)
    lf = lf.join(
        split_map.lazy(),
        on=COL_USER,
        how="left",
    )
    lf = lf.with_columns(
        [
            (pl.lit(SEQ_PREFIX_LOOP) + pl.col(COL_SEQ).cast(pl.Utf8)).alias(COL_SEQ),
            (pl.lit(USER_PREFIX_LOOP) + pl.col(COL_USER).cast(pl.Utf8)).alias(COL_USER),
            pl.col("Timestamp").alias(COL_TS),
            pl.col("Glucose (mg/dL)")
            .cast(pl.Float64, strict=False)
            .alias(COL_GLU),
            pl.col(COL_BASAL).cast(pl.Float64, strict=False),
            pl.col(COL_BOLUS).cast(pl.Float64, strict=False),
            pl.col(COL_CARB).cast(pl.Float64, strict=False),
            pl.lit(None).cast(pl.Utf8).alias("Heart Rate"),
            pl.lit(None).cast(pl.Utf8).alias("Step Count"),
            pl.lit(None).cast(pl.Float64).alias("Active Calories (kcal)"),
            pl.lit(None).cast(pl.Float64).alias("Stress Level"),
            pl.lit(None).cast(pl.Float64).alias("Respiratory Rate"),
            pl.lit(None).cast(pl.Float64).alias("Oxygen Saturation (%)"),
            pl.lit(None).cast(pl.Float64).alias("Age"),
            pl.lit(LOOP_STUDY_GROUP).alias(COL_GROUP),
            pl.lit(DATA_SOURCE_LOOP).alias(COL_DATA_SOURCE),
            pl.when(pl.col(COL_EVENT) == "Interpolated")
            .then(pl.lit(0.0))
            .otherwise(pl.lit(1.0))
            .alias("Glucose Observed"),
            pl.lit(0.0).alias("Steps Observed"),
            pl.lit(0.0).alias("HR Observed"),
            pl.lit(0.0).alias("kcal_observed"),
            pl.lit(0.0).alias("stress_observed"),
            pl.lit(0.0).alias("resp_observed"),
            pl.lit(0.0).alias("spo2_observed"),
            pl.col(COL_BASAL).is_not_null().cast(pl.Float64).alias(COL_BASAL_OBS),
            pl.col(COL_BOLUS).is_not_null().cast(pl.Float64).alias(COL_BOLUS_OBS),
            pl.col(COL_CARB).is_not_null().cast(pl.Float64).alias(COL_CARB_OBS),
        ]
    )
    return lf.select(final_column_order())


def build_ai_ready_lazy(ai_path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(
        ai_path,
        infer_schema_length=10_000,
        schema_overrides={
            COL_SEQ: pl.Utf8,
            COL_USER: pl.Utf8,
            "Heart Rate": pl.Utf8,
            "Step Count": pl.Utf8,
        },
    )
    lf = lf.with_columns(
        [
            pl.col("Active Calories (kcal)").cast(pl.Float64, strict=False),
            pl.col("Stress Level").cast(pl.Float64, strict=False),
            pl.col("Respiratory Rate").cast(pl.Float64, strict=False),
            pl.col("Oxygen Saturation (%)").cast(pl.Float64, strict=False),
            pl.col("Age").cast(pl.Float64, strict=False),
        ]
    )
    lf = lf.with_columns(
        [
            (pl.lit(SEQ_PREFIX_AI) + pl.col(COL_SEQ).cast(pl.Utf8)).alias(COL_SEQ),
            (pl.lit(USER_PREFIX_AI) + pl.col(COL_USER).cast(pl.Utf8)).alias(COL_USER),
            pl.col(COL_GLU).cast(pl.Float64, strict=False),
            pl.lit(None).cast(pl.Float64).alias(COL_BASAL),
            pl.lit(None).cast(pl.Float64).alias(COL_BOLUS),
            pl.lit(None).cast(pl.Float64).alias(COL_CARB),
            pl.lit(DATA_SOURCE_AI).alias(COL_DATA_SOURCE),
            pl.lit(0.0).alias(COL_BASAL_OBS),
            pl.lit(0.0).alias(COL_BOLUS_OBS),
            pl.lit(0.0).alias(COL_CARB_OBS),
        ]
    )
    return lf.select(final_column_order())


@app.command()
def main(
    loop_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop.csv"),
        help="Path to loop.csv",
    ),
    ai_ready_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/ai_ready_full4.csv"),
        help="Path to ai_ready_full4.csv",
    ),
    output: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop_ai_ready_joined.parquet"),
        help="Output path (.parquet or .csv by --format)",
    ),
    fmt: Literal["parquet", "csv"] = typer.Option(
        "parquet",
        "--format",
        help="Output format",
    ),
    loop_max_rows: int = typer.Option(
        0,
        help="If >0, only read first N rows from loop (debug).",
    ),
    ai_max_rows: int = typer.Option(
        0,
        help="If >0, only read first N rows from ai_ready (debug).",
    ),
) -> None:
    if not loop_csv.is_file():
        raise typer.BadParameter(f"Missing loop CSV: {loop_csv}")
    if not ai_ready_csv.is_file():
        raise typer.BadParameter(f"Missing ai_ready CSV: {ai_ready_csv}")

    typer.echo("Computing loop user row counts (full loop file)...")
    user_counts = (
        pl.scan_csv(loop_csv, infer_schema_length=10_000)
        .group_by(COL_USER)
        .len()
        .collect()
    )
    user_counts = user_counts.rename({"len": "n_rows"})
    split_map = greedy_user_splits_by_row_mass(user_counts, COL_USER, "n_rows")
    typer.echo(
        f"Loop users: {len(split_map):,} | split row targets "
        f"train={TRAIN_ROW_FRACTION:.4f} val={VAL_ROW_FRACTION:.4f} "
        f"test={TEST_ROW_FRACTION:.4f}"
    )
    for s in ("train", "val", "test"):
        n_u = split_map.filter(pl.col(COL_SPLIT) == s).height
        n_r = split_map.filter(pl.col(COL_SPLIT) == s)["n_rows"].sum()
        typer.echo(f"  {s}: users={n_u:,} rows={int(n_r):,}")

    lf_loop = build_loop_lazy(loop_csv, split_map.drop("n_rows"))
    lf_ai = build_ai_ready_lazy(ai_ready_csv)
    if loop_max_rows > 0:
        lf_loop = lf_loop.head(loop_max_rows)
    if ai_max_rows > 0:
        lf_ai = lf_ai.head(ai_max_rows)

    joined = pl.concat([lf_loop, lf_ai], how="vertical_relaxed")

    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        out_path = output if output.suffix else output.with_suffix(".parquet")
        typer.echo(f"Writing Parquet (streaming): {out_path}")
        joined.sink_parquet(out_path, compression="zstd")
    else:
        out_path = output if output.suffix == ".csv" else output.with_suffix(".csv")
        typer.echo(f"Writing CSV (streaming): {out_path}")
        joined.sink_csv(out_path)

    typer.echo("Done.")


if __name__ == "__main__":
    app()
