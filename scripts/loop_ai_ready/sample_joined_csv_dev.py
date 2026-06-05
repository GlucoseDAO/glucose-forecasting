#!/usr/bin/env python3
"""
Build a ~1/N development CSV from `loop_ai_ready_joined.csv`:

1. Per user, keep ~half the rows (deterministic `sequence_id` hash) — «single user
   data by half».
2. Within each stratum (Recommended Split × loop | ai_ready), keep a subset of
   whole users so that, before the row-half step, retained users account for
   (2/N) of that stratum's rows; combined with the half-row step this yields
   ~1/N of rows while preserving train/val/test and subdataset proportions
   (heavy users handled explicitly rather than a global hash threshold).

Subdataset = rows whose `User ID` is prefixed `loop_` vs `ai_ready_` (no
`Data Source` column in this export).
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False)

COL_USER = "User ID"
COL_SPLIT = "Recommended Split"

USER_PREFIX_LOOP = "loop_"
USER_PREFIX_AI = "ai_ready_"

DATA_SOURCE_LOOP = "loop"
DATA_SOURCE_AI = "ai_ready"
DATA_SOURCE_OTHER = "other"

DEFAULT_SEED = 42
DEFAULT_ROW_HASH_SEED = 0xBEEF


def infer_data_source_expr(user_col: str = COL_USER) -> pl.Expr:
    u = pl.col(user_col)
    return (
        pl.when(u.str.starts_with(USER_PREFIX_LOOP))
        .then(pl.lit(DATA_SOURCE_LOOP))
        .when(u.str.starts_with(USER_PREFIX_AI))
        .then(pl.lit(DATA_SOURCE_AI))
        .otherwise(pl.lit(DATA_SOURCE_OTHER))
    )


def users_to_keep_per_stratum(
    users: pl.DataFrame,
    *,
    shrink_factor: int,
    seed: int,
) -> set[str]:
    """
    For each stratum (split × data source), choose a prefix of shuffled users
    until sum(n_u) >= round(C * 2 / shrink_factor).

    That mass, after per-user row-halving (~1/2), matches ~C / shrink_factor.
    """
    if shrink_factor < 2:
        raise ValueError("shrink_factor must be >= 2.")

    strata: defaultdict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for row in users.iter_rows(named=True):
        uid = row[COL_USER]
        n = row["n"]
        split_v = row[COL_SPLIT]
        ds = row["_data_source"]
        strata[(split_v, ds)].append((uid, int(n)))

    others = {(s, d) for (s, d) in strata if d == DATA_SOURCE_OTHER}
    if others:
        raise ValueError(f"Unexpected User ID prefixes in strata: {others}")

    rng = random.Random(seed)
    kept: set[str] = set()

    for key in sorted(strata.keys()):
        pairs = strata[key]
        total_c = sum(n for _, n in pairs)
        target_pre_half = max(1, round(total_c * 2 / shrink_factor))
        rng.shuffle(pairs)
        cum = 0
        picked: list[tuple[str, int]] = []
        for uid, n in pairs:
            cum += n
            picked.append((uid, n))
            if cum >= target_pre_half:
                break
        while len(picked) > 1:
            last_uid, last_n = picked[-1]
            if cum - last_n >= target_pre_half:
                cum -= last_n
                picked.pop()
                continue
            break
        for uid, _ in picked:
            kept.add(uid)

    return kept


@app.command()
def main(
    input_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop_ai_ready_joined.csv"),
        help="Full joined CSV.",
    ),
    output_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop_ai_ready_joined_dev.csv"),
        help="Output subset CSV (same columns / format).",
    ),
    shrink_factor: int = typer.Option(
        20,
        help="Target overall shrink vs input row count (e.g. 20 → ~1/20 rows).",
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="RNG seed for per-stratum user order."),
    row_hash_seed: int = typer.Option(
        DEFAULT_ROW_HASH_SEED,
        help="Seed for deterministic per-user row half-sample on sequence_id.",
    ),
) -> None:
    if not input_csv.is_file():
        raise typer.BadParameter(f"Missing input: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("Pass 1: aggregating per user + strata...")
    lf = pl.scan_csv(input_csv, infer_schema_length=10000)
    users = (
        lf.with_columns(infer_data_source_expr().alias("_data_source"))
        .group_by([COL_USER, COL_SPLIT])
        .agg(pl.len().alias("n"), pl.col("_data_source").first().alias("_data_source"))
        .collect(engine="streaming")
    )

    typer.echo(f"  unique users: {users.height}")
    keep_ids = users_to_keep_per_stratum(users, shrink_factor=shrink_factor, seed=seed)
    typer.echo(f"  users after stratified selection: {len(keep_ids)}")

    typer.echo(f"Pass 2: streaming rows (half per user row-level + user filter) → {output_csv}...")
    lf2 = pl.scan_csv(input_csv, infer_schema_length=10000)
    # sequence_id repeats for many rows (same session); sample rows via per-user index.
    keyed = (
        lf2.filter(pl.col(COL_USER).is_in(keep_ids))
        .with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).over(COL_USER).alias("_rn"))
        .filter(
            pl.concat_str(
                [pl.col(COL_USER), pl.lit("#"), pl.col("_rn").cast(pl.Utf8)],
                separator="",
            ).hash(seed=row_hash_seed)
            % 2
            == 0
        )
        .drop("_rn")
    )

    keyed.sink_csv(output_csv)
    typer.echo("Done.")


if __name__ == "__main__":
    app()
