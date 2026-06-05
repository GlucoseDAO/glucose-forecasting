#!/usr/bin/env python3
"""
Stream `loop_ai_ready_joined.parquet` to CSV with loop-style glucose/timestamp
column names plus `Recommended Split` and `Study Group` copied from the parquet
(loop: user-wise split from join; ai_ready: original cohort labels).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False)

COL_TS_JOINED = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_TS_LOOP = "Timestamp"
COL_GLU_JOINED = "Glucose Value (mg/dL)"
COL_GLU_LOOP = "Glucose (mg/dL)"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"


@app.command()
def main(
    parquet_path: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop_ai_ready_joined.parquet"),
        help="Input joined parquet.",
    ),
    output_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop_ai_ready_joined_loop_columns.csv"),
        help="Output CSV path.",
    ),
) -> None:
    if not parquet_path.is_file():
        raise typer.BadParameter(f"Missing parquet: {parquet_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    lf = (
        pl.scan_parquet(parquet_path)
        .select(
            [
                "sequence_id",
                COL_TS_JOINED,
                "Event Type",
                "User ID",
                COL_GLU_JOINED,
                "Basal Rate (U/h)",
                "Bolus Insulin (U)",
                "Carbohydrates (g)",
                COL_SPLIT,
                COL_GROUP,
            ]
        )
        .rename(
            {
                COL_TS_JOINED: COL_TS_LOOP,
                COL_GLU_JOINED: COL_GLU_LOOP,
            }
        )
    )

    typer.echo(f"Writing {output_csv} (streaming)...")
    lf.sink_csv(output_csv)
    typer.echo("Done.")


if __name__ == "__main__":
    app()
