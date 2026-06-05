#!/usr/bin/env python3
"""
Build a Study-Group-balanced loop + ai_ready CSV (loop-style columns).

- Keeps all rows from ``ai_ready_full4.csv``.
- Adds quality-filtered loop users (T1DM) until T1DM row count reaches the
  combined non-T1DM row count from ai_ready (whole users only; no row trimming).
- Loop quality: every sequence for the user has basal, bolus, and carb values
  present on at least one row (EGV rows may still have empty bolus/carb).
- Assigns loop ``Recommended Split`` user-wise with the same train/val/test
  row fractions as ai_ready.
- Optionally writes a ~1/N dev subset with split and source proportions preserved.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False)

COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS_AI = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_TS = "Timestamp"
COL_GLU_AI = "Glucose Value (mg/dL)"
COL_GLU = "Glucose (mg/dL)"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"
COL_BASAL = "Basal Rate (U/h)"
COL_BOLUS = "Bolus Insulin (U)"
COL_CARB = "Carbohydrates (g)"

LOOP_STUDY_GROUP = "T1DM"
SEQ_PREFIX_LOOP = "L-"
SEQ_PREFIX_AI = "A-"
USER_PREFIX_LOOP = "loop_"
USER_PREFIX_AI = "ai_ready_"

DATA_SOURCE_LOOP = "loop"
DATA_SOURCE_AI = "ai_ready"

DEFAULT_SEED = 42
DEFAULT_ROW_HASH_SEED = 0xBEEF

LOOP_COLUMNS: list[str] = [
    COL_SEQ,
    COL_TS,
    COL_EVENT,
    COL_USER,
    COL_GLU,
    COL_BASAL,
    COL_BOLUS,
    COL_CARB,
    COL_SPLIT,
    COL_GROUP,
]


def ai_ready_split_fractions(ai_path: Path) -> dict[str, float]:
    counts = (
        pl.scan_csv(ai_path, infer_schema_length=10_000)
        .group_by(COL_SPLIT)
        .len()
        .collect()
    )
    total = int(counts["len"].sum())
    if total <= 0:
        raise ValueError(f"No rows in {ai_path}")
    return {
        str(row[COL_SPLIT]): float(row["len"]) / total
        for row in counts.iter_rows(named=True)
    }


def greedy_user_splits_by_row_mass(
    user_counts: pl.DataFrame,
    user_col: str,
    count_col: str,
    split_fractions: dict[str, float],
) -> pl.DataFrame:
    """Assign users to splits so row totals approximate ``split_fractions``."""
    total_rows = int(user_counts[count_col].sum())
    if total_rows <= 0:
        raise ValueError("No rows in loop user count table.")

    splits = sorted(split_fractions)
    targets = {s: split_fractions[s] * total_rows for s in splits}
    currents = {s: 0.0 for s in splits}
    ordered = user_counts.sort(count_col, descending=True)
    assigned: list[str] = []

    for _uid, n in ordered.select([user_col, count_col]).iter_rows():
        n_f = float(n)
        scores = {s: currents[s] / targets[s] for s in splits}
        best = min(scores, key=scores.get)
        assigned.append(best)
        currents[best] += n_f

    return ordered.with_columns(pl.Series(COL_SPLIT, assigned))


def quality_loop_users(loop_path: Path) -> pl.DataFrame:
    """
    Users whose every sequence has basal, bolus, and carb present somewhere.

    Returns one row per user with ``n_rows``.
    """
    seq_stats = (
        pl.scan_csv(loop_path, infer_schema_length=10_000)
        .group_by(COL_SEQ, COL_USER)
        .agg(
            pl.len().alias("n_rows"),
            pl.col(COL_BASAL).is_not_null().any().alias("has_basal"),
            pl.col(COL_BOLUS).is_not_null().any().alias("has_bolus"),
            pl.col(COL_CARB).is_not_null().any().alias("has_carb"),
        )
        .collect(engine="streaming")
    )
    user_stats = (
        seq_stats.group_by(COL_USER)
        .agg(
            pl.col("has_basal").all().alias("all_seq_basal"),
            pl.col("has_bolus").all().alias("all_seq_bolus"),
            pl.col("has_carb").all().alias("all_seq_carb"),
            pl.col("n_rows").sum().alias("n_rows"),
        )
        .filter(
            pl.col("all_seq_basal")
            & pl.col("all_seq_bolus")
            & pl.col("all_seq_carb")
        )
        .sort("n_rows", descending=True)
    )
    return user_stats


def select_loop_users_for_balance(
    quality_users: pl.DataFrame,
    target_rows: int,
) -> pl.DataFrame:
    """Greedy whole-user selection until cumulative rows reach ``target_rows``."""
    if quality_users.is_empty():
        raise ValueError("No quality loop users found.")

    picked: list[dict[str, object]] = []
    cumulative = 0
    for row in quality_users.iter_rows(named=True):
        picked.append({COL_USER: row[COL_USER], "n_rows": row["n_rows"]})
        cumulative += int(row["n_rows"])
        if cumulative >= target_rows:
            break

    return pl.DataFrame(picked)


def build_ai_ready_lazy(ai_path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(
        ai_path,
        infer_schema_length=10_000,
        schema_overrides={
            COL_SEQ: pl.Utf8,
            COL_USER: pl.Utf8,
        },
    )
    return lf.select(
        [
            (pl.lit(SEQ_PREFIX_AI) + pl.col(COL_SEQ).cast(pl.Utf8)).alias(COL_SEQ),
            pl.col(COL_TS_AI).alias(COL_TS),
            pl.col(COL_EVENT),
            (pl.lit(USER_PREFIX_AI) + pl.col(COL_USER).cast(pl.Utf8)).alias(COL_USER),
            pl.col(COL_GLU_AI).alias(COL_GLU),
            pl.lit("").alias(COL_BASAL),
            pl.lit("").alias(COL_BOLUS),
            pl.lit("").alias(COL_CARB),
            pl.col(COL_SPLIT),
            pl.col(COL_GROUP),
        ]
    )


def build_loop_lazy(
    loop_path: Path,
    selected_users: list[str],
    split_map: pl.DataFrame,
) -> pl.LazyFrame:
    lf = (
        pl.scan_csv(loop_path, infer_schema_length=10_000)
        .filter(pl.col(COL_USER).is_in(selected_users))
        .join(split_map.lazy(), on=COL_USER, how="left")
    )
    return lf.select(
        [
            (pl.lit(SEQ_PREFIX_LOOP) + pl.col(COL_SEQ).cast(pl.Utf8)).alias(COL_SEQ),
            pl.col(COL_TS),
            pl.col(COL_EVENT),
            (pl.lit(USER_PREFIX_LOOP) + pl.col(COL_USER).cast(pl.Utf8)).alias(COL_USER),
            pl.col(COL_GLU),
            pl.col(COL_BASAL),
            pl.col(COL_BOLUS),
            pl.col(COL_CARB),
            pl.col(COL_SPLIT),
            pl.lit(LOOP_STUDY_GROUP).alias(COL_GROUP),
        ]
    )


def infer_data_source_expr(user_col: str = COL_USER) -> pl.Expr:
    u = pl.col(user_col)
    return (
        pl.when(u.str.starts_with(USER_PREFIX_LOOP))
        .then(pl.lit(DATA_SOURCE_LOOP))
        .when(u.str.starts_with(USER_PREFIX_AI))
        .then(pl.lit(DATA_SOURCE_AI))
        .otherwise(pl.lit("other"))
    )


def users_to_keep_per_stratum(
    users: pl.DataFrame,
    *,
    shrink_factor: int,
    seed: int,
) -> set[str]:
    if shrink_factor < 2:
        raise ValueError("shrink_factor must be >= 2.")

    strata: defaultdict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for row in users.iter_rows(named=True):
        uid = row[COL_USER]
        n = row["n"]
        split_v = row[COL_SPLIT]
        ds = row["_data_source"]
        strata[(split_v, ds)].append((uid, int(n)))

    others = {(s, d) for (s, d) in strata if d == "other"}
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


def write_dev_subset(
    input_csv: Path,
    output_csv: Path,
    *,
    shrink_factor: int,
    seed: int,
    row_hash_seed: int,
) -> None:
    typer.echo("Dev pass 1: aggregating per user + strata...")
    lf = pl.scan_csv(input_csv, infer_schema_length=10_000)
    users = (
        lf.with_columns(infer_data_source_expr().alias("_data_source"))
        .group_by([COL_USER, COL_SPLIT])
        .agg(pl.len().alias("n"), pl.col("_data_source").first().alias("_data_source"))
        .collect(engine="streaming")
    )

    keep_ids = users_to_keep_per_stratum(users, shrink_factor=shrink_factor, seed=seed)
    typer.echo(
        f"  unique users: {users.height:,} | kept: {len(keep_ids):,} "
        f"(target ~1/{shrink_factor} rows)"
    )

    typer.echo(f"Dev pass 2: streaming subset -> {output_csv}")
    lf2 = pl.scan_csv(input_csv, infer_schema_length=10_000)
    keyed = (
        lf2.filter(pl.col(COL_USER).is_in(list(keep_ids)))
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
    output_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop_ai_ready_joined2.csv"),
        help="Balanced output CSV (loop-style columns).",
    ),
    dev_output_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv"),
        help="Optional ~1/N dev subset CSV (empty path to skip).",
    ),
    dev_shrink_factor: int = typer.Option(
        20,
        help="Dev subset target shrink vs full joined CSV (e.g. 20 -> ~1/20 rows).",
    ),
    dev_seed: int = typer.Option(DEFAULT_SEED, help="RNG seed for dev user sampling."),
    dev_row_hash_seed: int = typer.Option(
        DEFAULT_ROW_HASH_SEED,
        help="Seed for deterministic per-user row half-sample in dev subset.",
    ),
) -> None:
    if not loop_csv.is_file():
        raise typer.BadParameter(f"Missing loop CSV: {loop_csv}")
    if not ai_ready_csv.is_file():
        raise typer.BadParameter(f"Missing ai_ready CSV: {ai_ready_csv}")

    split_fractions = ai_ready_split_fractions(ai_ready_csv)
    ai_row_count = int(
        pl.scan_csv(ai_ready_csv, infer_schema_length=10_000)
        .select(pl.len())
        .collect()
        .item()
    )
    typer.echo(f"ai_ready rows (non-T1DM target for loop): {ai_row_count:,}")
    typer.echo(
        "ai_ready split fractions: "
        + ", ".join(f"{k}={v:.4f}" for k, v in sorted(split_fractions.items()))
    )

    typer.echo("Scanning loop quality users (all sequences with basal/bolus/carb)...")
    quality_users = quality_loop_users(loop_csv)
    typer.echo(
        f"  quality users: {quality_users.height:,} | rows: "
        f"{int(quality_users['n_rows'].sum()):,}"
    )

    selected = select_loop_users_for_balance(quality_users, ai_row_count)
    selected_users = selected[COL_USER].to_list()
    loop_row_count = int(selected["n_rows"].sum())
    typer.echo(
        f"Selected loop users: {selected.height:,} | loop rows: {loop_row_count:,} "
        f"(target {ai_row_count:,})"
    )

    split_map = greedy_user_splits_by_row_mass(
        selected,
        COL_USER,
        "n_rows",
        split_fractions,
    )
    for split_name in sorted(split_fractions):
        n_u = split_map.filter(pl.col(COL_SPLIT) == split_name).height
        n_r = int(split_map.filter(pl.col(COL_SPLIT) == split_name)["n_rows"].sum())
        typer.echo(f"  loop {split_name}: users={n_u:,} rows={n_r:,}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Writing joined CSV (streaming): {output_csv}")
    joined = pl.concat(
        [
            build_loop_lazy(loop_csv, selected_users, split_map.drop("n_rows")),
            build_ai_ready_lazy(ai_ready_csv),
        ],
        how="vertical_relaxed",
    )
    joined.sink_csv(output_csv)

    typer.echo(
        f"Done. Total rows ~ {loop_row_count + ai_row_count:,} "
        f"(T1DM {loop_row_count:,} vs non-T1DM {ai_row_count:,})"
    )

    if dev_output_csv.name:
        write_dev_subset(
            output_csv,
            dev_output_csv,
            shrink_factor=dev_shrink_factor,
            seed=dev_seed,
            row_hash_seed=dev_row_hash_seed,
        )
        typer.echo(f"Dev subset written: {dev_output_csv}")


if __name__ == "__main__":
    app()
