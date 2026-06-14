#!/usr/bin/env python3
"""
evaluate_model.py — Evaluate GluMind or GluMindIC checkpoints on arbitrary CSV data.

Supports cross-model / cross-dataset comparison:
  - Models: scripts/glumind/glumind_model.py (HR + steps) or
            scripts/glumind_ic/glumind_ic_model.py (basal + bolus + carbs)
  - Datasets: ai_ready (Glucose Value, HR, Step Count) or
              loop_ai_ready (Glucose, Basal Rate, Bolus, Carbs)

Missing covariates are filled with 0.0 before imputation.
Use --zero-cov to force all non-glucose covariates to 0.0 after imputation
(for fair comparison with models trained without covariates on the same dataset).
Use --include-cov / --exclude-cov for finer-grained covariate ablation at inference.
Use --covariates to inspect which covariate columns exist in a CSV (no model run).
If the CSV has a 'Recommended Split' column, rows with split == --test-split
(default: test) are evaluated; otherwise all rows are used.

Model resolution (same as evaluate_glumind.py):
  --run-dir       directory with tuning_meta.json and best_model.pt
  --registry-dir  auto-picks lowest val_mae from _analysis_registry.csv
  --checkpoint    explicit .pt file (requires --run-dir for architecture meta)

Examples:
  uv run evaluate-model \\
      --run-dir test_model \\
      --test-csv data/actual/with_complex_steps_processing/ai_ready_plus_type1_v1_val_in_val_and_test.csv

  uv run evaluate-model \\
      --run-dir runs/glumind_ic_tune/production/trial_0 \\
      --model-type glumind_ic \\
      --test-csv data/loop_and_ai_ready/loop_ai_ready_joined2.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from scripts.glumind.glumind_model import GluMindModel
from scripts.glumind.train_glumind import (
    COL_GROUP,
    COL_SEQ,
    COL_SPLIT,
    COL_TS as GLUMIND_COL_TS,
    COL_USER,
    TS_FORMAT,
    GlucoseWindowDataset,
    apply_split_scheme as apply_split_scheme_glumind,
    impute_and_sort as impute_and_sort_glumind,
    load_splits_streaming as load_splits_glumind,
    mae_rmse_mard,
)
from scripts.glumind_ic.glumind_ic_model import GluMindICModel
from scripts.glumind_ic.train_glumind_ic import (
    COL_TS as IC_COL_TS,
    GlucoseICWindowDataset,
    apply_split_scheme as apply_split_scheme_ic,
    impute_and_sort as impute_and_sort_ic,
    load_splits_streaming as load_splits_ic,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

ModelKind = Literal["glumind", "glumind_ic"]

COL_EVENT = "Event Type"

GLUMIND_COVARIATES: dict[str, list[str]] = {
    "glucose": ["Glucose Value (mg/dL)", "Glucose (mg/dL)"],
    "hr": ["Heart Rate"],
    "steps": ["Step Count"],
}

IC_COVARIATES: dict[str, list[str]] = {
    "glucose": ["Glucose Value (mg/dL)", "Glucose (mg/dL)"],
    "basal": ["Basal Rate (U/h)"],
    "bolus": ["Bolus Insulin (U)"],
    "carbs": ["Carbohydrates (g)"],
}

# User-facing names accepted by --include-cov / --exclude-cov (case-insensitive).
COVARIATE_NAME_ALIASES: dict[str, list[str]] = {
    "hr": ["hr", "heart_rate", "heart rate", "heartrate"],
    "steps": ["steps", "step", "step_count", "step count", "stepcount"],
    "basal": ["basal", "basal_rate", "basal rate", "basalrate"],
    "bolus": ["bolus", "bolus_insulin", "bolus insulin", "insulin", "bolusinsulin"],
    "carbs": ["carbs", "carb", "carbohydrates", "carbohydrate", "carbohydrate_g"],
}

TS_ALIASES = [GLUMIND_COL_TS, IC_COL_TS]


class ModelTypeChoice(str, Enum):
    auto = "auto"
    glumind = "glumind"
    glumind_ic = "glumind_ic"


# ---------------------------------------------------------------------------
# Registry / checkpoint helpers (shared with evaluate_glumind.py)
# ---------------------------------------------------------------------------

def _find_best_run_dir(registry_dir: Path) -> tuple[Path, dict]:
    registry_csv = registry_dir / "_analysis_registry.csv"
    if not registry_csv.exists():
        typer.echo(f"Error: _analysis_registry.csv not found in {registry_dir}", err=True)
        raise typer.Exit(1)

    best_row: dict | None = None
    best_mae: float = float("inf")

    with open(registry_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_mae_str = row.get("val_mae", "").strip()
            if not val_mae_str:
                continue
            val_mae = float(val_mae_str)
            if val_mae < best_mae:
                best_mae = val_mae
                best_row = row

    if best_row is None:
        typer.echo("Error: No valid rows with val_mae found in the registry.", err=True)
        raise typer.Exit(1)

    run_dir_rel = Path(best_row["run_dir"])
    run_dir_abs = project_root / run_dir_rel
    final_step = best_row.get("final_step", "").strip()
    step_dir = run_dir_abs / final_step if final_step else run_dir_abs

    typer.echo(
        f"Best run (val_mae={best_mae:.6f}): {run_dir_rel}"
        + (f"  step={final_step}" if final_step else "")
    )
    return step_dir, best_row


def _load_meta(run_dir: Path) -> dict:
    for name in ("tuning_meta.json", "config.json"):
        p = run_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    typer.echo(f"Error: No metadata file (tuning_meta.json / config.json) in {run_dir}", err=True)
    raise typer.Exit(1)


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            typer.echo(f"Error: Checkpoint not found: {checkpoint}", err=True)
            raise typer.Exit(1)
        return checkpoint

    for name in ("best_model.pt", "last_model.pt"):
        p = run_dir / name
        if p.exists():
            return p

    typer.echo(f"Error: No model weights (best_model.pt / last_model.pt) found in {run_dir}", err=True)
    raise typer.Exit(1)


def _resolve_csv_path(csv_value: str | Path) -> Path:
    csv_path = Path(csv_value)
    if csv_path.exists():
        return csv_path
    alt = project_root / csv_value
    if alt.exists():
        return alt
    typer.echo(f"Error: CSV not found: {csv_path}", err=True)
    raise typer.Exit(1)


def _detect_model_kind(meta: dict, state: dict[str, torch.Tensor]) -> ModelKind:
    explicit = meta.get("model_type") or meta.get("model")
    if explicit in ("glumind", "glumind_ic", "GluMind", "GluMindIC"):
        return "glumind_ic" if "ic" in str(explicit).lower() else "glumind"

    normalized_keys = {k.removeprefix("_orig_mod.") for k in state}
    ic_keys = ("embed_basal.weight", "embed_bolus.weight", "embed_carbs.weight")
    glumind_keys = ("embed_hr.weight", "embed_steps.weight")
    if any(k in normalized_keys for k in ic_keys):
        return "glumind_ic"
    if any(k in normalized_keys for k in glumind_keys):
        return "glumind"

    typer.echo(
        "Error: Could not auto-detect model type from checkpoint. "
        "Pass --model-type glumind or glumind_ic.",
        err=True,
    )
    raise typer.Exit(1)


def _pick_header_column(header: list[str], aliases: list[str]) -> str | None:
    header_set = set(header)
    for alias in aliases:
        if alias in header_set:
            return alias
    return None


def _covariate_map(model_kind: ModelKind) -> dict[str, list[str]]:
    return GLUMIND_COVARIATES if model_kind == "glumind" else IC_COVARIATES


def _canonical_feature_cols(model_kind: ModelKind) -> list[str]:
    if model_kind == "glumind":
        return ["glucose", "hr", "steps"]
    return ["glucose", "basal", "bolus", "carbs"]


def _non_glucose_covariate_cols(model_kind: ModelKind) -> list[str]:
    return [c for c in _canonical_feature_cols(model_kind) if c != "glucose"]


def _normalize_covariate_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


def _alias_to_canonical(name: str, model_kind: ModelKind) -> str:
    """Map a user token to a canonical covariate name for the model kind."""
    normalized = _normalize_covariate_token(name)
    valid = set(_non_glucose_covariate_cols(model_kind))
    if normalized in valid:
        return normalized
    for canonical, aliases in COVARIATE_NAME_ALIASES.items():
        alias_norms = {_normalize_covariate_token(a) for a in aliases}
        if normalized in alias_norms or normalized == canonical:
            if canonical in valid:
                return canonical
    valid_list = ", ".join(sorted(valid))
    raise ValueError(
        f"Unknown covariate {name!r} for model {model_kind!r}. "
        f"Valid names: {valid_list}"
    )


def _split_cov_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_covariate_names(raw: str | None, model_kind: ModelKind) -> list[str]:
    tokens = _split_cov_arg(raw)
    if not tokens:
        return []
    canonical: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        name = _alias_to_canonical(token, model_kind)
        if name not in seen:
            canonical.append(name)
            seen.add(name)
    return canonical


def _resolve_covariate_zeroing(
    model_kind: ModelKind,
    *,
    zero_cov: bool,
    include_cov: str | None,
    exclude_cov: str | None,
) -> tuple[list[str], list[str]]:
    """Return (active_non_glucose_covariates, zeroed_non_glucose_covariates)."""
    all_cov = _non_glucose_covariate_cols(model_kind)
    include_tokens = _split_cov_arg(include_cov)
    exclude_tokens = _split_cov_arg(exclude_cov)

    if zero_cov and (include_tokens or exclude_tokens):
        raise ValueError(
            "Use either --zero-cov or --include-cov / --exclude-cov, not both."
        )
    if include_tokens and exclude_tokens:
        raise ValueError("Use either --include-cov or --exclude-cov, not both.")

    if zero_cov:
        return [], list(all_cov)

    if include_tokens:
        included = _parse_covariate_names(include_cov, model_kind)
        if not included:
            raise ValueError("--include-cov requires at least one covariate name.")
        included_set = set(included)
        zeroed = [c for c in all_cov if c not in included_set]
        return included, zeroed

    if exclude_tokens:
        excluded = _parse_covariate_names(exclude_cov, model_kind)
        if not excluded:
            raise ValueError("--exclude-cov requires at least one covariate name.")
        excluded_set = set(excluded)
        active = [c for c in all_cov if c not in excluded_set]
        return active, excluded

    return list(all_cov), []


def _zero_covariates(df: pl.DataFrame, covariates: list[str]) -> pl.DataFrame:
    """Replace named covariates with 0.0 (applied after imputation)."""
    if not covariates:
        return df
    present = [c for c in covariates if c in df.columns]
    if not present:
        return df
    return df.with_columns([pl.lit(0.0).cast(pl.Float32).alias(c) for c in present])


def _zero_non_glucose_covariates(df: pl.DataFrame, model_kind: ModelKind) -> pl.DataFrame:
    """Replace all non-glucose covariates with 0.0 (applied after imputation)."""
    return _zero_covariates(df, _non_glucose_covariate_cols(model_kind))


def _is_filled_expr(source_col: str) -> pl.Expr:
    return pl.col(source_col).is_not_null() & (
        pl.col(source_col).cast(pl.Utf8).str.strip_chars() != ""
    )


def _read_csv_header(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        return next(csv.reader(f))


def _covariate_column_stats(
    csv_path: Path,
    source_col: str,
    eval_split: str | None,
) -> tuple[int, int]:
    """Return (total_rows, filled_rows) optionally filtered by split."""
    header = _read_csv_header(csv_path)
    has_split = COL_SPLIT in header
    lf = pl.scan_csv(
        csv_path,
        infer_schema_length=10_000,
        schema_overrides={source_col: pl.Utf8},
    )
    if eval_split and has_split:
        lf = lf.filter(pl.col(COL_SPLIT) == eval_split)
    stats = lf.select([
        pl.len().alias("total"),
        _is_filled_expr(source_col).sum().alias("filled"),
    ]).collect()
    return int(stats["total"][0]), int(stats["filled"][0])


def _print_dataset_covariates(
    csv_path: Path,
    model_kind: ModelKind | None,
    eval_split: str | None,
) -> None:
    """Print covariate column mapping and fill stats for the target CSV."""
    header = _read_csv_header(csv_path)
    kinds: list[ModelKind] = (
        [model_kind] if model_kind is not None else ["glumind", "glumind_ic"]
    )
    split_label = eval_split if eval_split else "all rows"
    typer.echo(f"Dataset : {csv_path}")
    typer.echo(f"Split   : {split_label}")
    typer.echo("")

    for kind in kinds:
        cov_map = _covariate_map(kind)
        typer.echo(f"Model type: {kind}")
        typer.echo(f"  Feature channels: {', '.join(_canonical_feature_cols(kind))}")
        typer.echo(
            f"  Non-glucose covariates (--include-cov / --exclude-cov): "
            f"{', '.join(_non_glucose_covariate_cols(kind))}"
        )
        typer.echo("  Columns:")
        for canonical, aliases in cov_map.items():
            source_col = _pick_header_column(header, aliases)
            if source_col is None:
                typer.echo(f"    {canonical:8s}  missing  (loaded as 0.0)")
                continue
            total, filled = _covariate_column_stats(csv_path, source_col, eval_split)
            pct = 100.0 * filled / total if total else 0.0
            typer.echo(
                f"    {canonical:8s}  {source_col!r}  "
                f"filled {filled:,}/{total:,} ({pct:.1f}%)"
            )
        typer.echo("  Accepted aliases:")
        for canonical in _non_glucose_covariate_cols(kind):
            aliases = COVARIATE_NAME_ALIASES.get(canonical, [canonical])
            typer.echo(f"    {canonical}: {', '.join(aliases)}")
        typer.echo("")


def _load_csv_flexible(
    csv_path: Path,
    model_kind: ModelKind,
    unique_id_choice: str,
    drop_interpolated: bool,
    eval_split: str | None,
    train_only: bool,
) -> pl.DataFrame:
    """Load CSV with canonical columns; missing covariates become 0.0."""
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))

    uid_aliases = [COL_SEQ] if unique_id_choice == "sequence_id" else [COL_USER]
    uid_col = _pick_header_column(header, uid_aliases)
    if uid_col is None:
        typer.echo(
            f"Error: Could not find unique id column ({uid_aliases}) in {csv_path.name}.",
            err=True,
        )
        raise typer.Exit(1)

    ts_col = _pick_header_column(header, TS_ALIASES)
    if ts_col is None:
        typer.echo(f"Error: Could not find timestamp column in {csv_path.name}.", err=True)
        raise typer.Exit(1)

    cov_map = _covariate_map(model_kind)
    glucose_col = _pick_header_column(header, cov_map["glucose"])
    if glucose_col is None:
        typer.echo(f"Error: Could not find glucose column in {csv_path.name}.", err=True)
        raise typer.Exit(1)

    has_split_col = COL_SPLIT in header
    has_group_col = COL_GROUP in header
    has_event_col = COL_EVENT in header

    schema_overrides: dict[str, pl.DataType] = {COL_SEQ: pl.Utf8, COL_USER: pl.Utf8}
    for aliases in cov_map.values():
        for alias in aliases:
            schema_overrides[alias] = pl.Utf8

    select_exprs: list[pl.Expr] = [
        pl.col(uid_col).alias("unique_id"),
        pl.col(ts_col).alias("ds"),
        pl.col(glucose_col).alias("glucose"),
    ]

    missing_covariates: list[str] = []
    for canonical, aliases in cov_map.items():
        if canonical == "glucose":
            continue
        source_col = _pick_header_column(header, aliases)
        if source_col is None:
            select_exprs.append(pl.lit(0.0).alias(canonical))
            missing_covariates.append(canonical)
        else:
            select_exprs.append(pl.col(source_col).alias(canonical))

    if has_group_col:
        select_exprs.append(pl.col(COL_GROUP).alias("study_group"))
    if has_split_col:
        select_exprs.append(pl.col(COL_SPLIT).alias("split"))
    if has_event_col:
        select_exprs.append(pl.col(COL_EVENT).alias("event_type"))

    lf = (
        pl.scan_csv(csv_path, infer_schema_length=10_000, schema_overrides=schema_overrides)
        .select(select_exprs)
        .with_columns([
            pl.col("ds").str.strptime(pl.Datetime, TS_FORMAT, strict=False),
            pl.col("glucose").cast(pl.Float32, strict=False),
            *[
                pl.col(c).cast(pl.Float32, strict=False)
                for c in _canonical_feature_cols(model_kind)
                if c != "glucose"
            ],
        ])
        .drop_nulls(subset=["unique_id", "ds"])
    )

    if drop_interpolated and has_event_col:
        lf = lf.filter(pl.col("event_type") != "Interpolated")

    if train_only and has_split_col:
        lf = lf.filter(pl.col("split") == "train")
    elif eval_split is not None and has_split_col:
        lf = lf.filter(pl.col("split") == eval_split)
    elif eval_split is not None and not has_split_col:
        typer.echo(
            f"Warning: --test-split='{eval_split}' requested but CSV has no "
            f"'{COL_SPLIT}' column — using all rows.",
            err=True,
        )

    df = lf.collect()

    if "study_group" not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias("study_group"))

    if missing_covariates:
        typer.echo(
            f"  Missing covariates filled with 0.0: {', '.join(missing_covariates)}"
        )

    split_label = "train" if train_only else (eval_split or "all")
    typer.echo(f"  Loaded {len(df):,} rows ({split_label}) from {csv_path.name}")
    return df


def _load_train_for_scalers(
    test_csv_path: Path,
    model_kind: ModelKind,
    meta: dict,
    train_csv_override: Path | None,
) -> pl.DataFrame:
    """Load training rows for scaler fitting."""
    scaler_csv = train_csv_override or _resolve_csv_path(meta["csv"])
    typer.echo(f"Fitting scalers from: {scaler_csv}")

    split_scheme = meta.get("split_scheme", "classic")
    unique_id = meta.get("unique_id", "sequence_id")
    drop_interpolated = meta.get("drop_interpolated", False)
    impute = impute_and_sort_glumind if model_kind == "glumind" else impute_and_sort_ic

    if scaler_csv == test_csv_path:
        train_df = _load_csv_flexible(
            scaler_csv,
            model_kind=model_kind,
            unique_id_choice=unique_id,
            drop_interpolated=drop_interpolated,
            eval_split=None,
            train_only=True,
        )
        if train_df.is_empty():
            typer.echo(
                "Warning: No train split rows in CSV — fitting scalers on all rows.",
                err=True,
            )
            train_df = _load_csv_flexible(
                scaler_csv,
                model_kind=model_kind,
                unique_id_choice=unique_id,
                drop_interpolated=drop_interpolated,
                eval_split=None,
                train_only=False,
            )
        return impute(train_df)

    load_splits = load_splits_glumind if model_kind == "glumind" else load_splits_ic
    apply_split = apply_split_scheme_glumind if model_kind == "glumind" else apply_split_scheme_ic

    train_df_raw, val_df_raw, test_df_raw = load_splits(
        scaler_csv,
        unique_id_choice=unique_id,
        drop_interpolated=drop_interpolated,
    )
    train_df_raw, _, _ = apply_split(train_df_raw, val_df_raw, test_df_raw, split_scheme)

    if train_df_raw.is_empty():
        typer.echo(
            "Warning: Training split empty after load_splits_streaming — "
            "using flexible loader on all rows.",
            err=True,
        )
        train_df = _load_csv_flexible(
            scaler_csv,
            model_kind=model_kind,
            unique_id_choice=unique_id,
            drop_interpolated=drop_interpolated,
            eval_split=None,
            train_only=False,
        )
        return impute(train_df)

    return impute(train_df_raw)


def _build_train_dataset(
    train_df: pl.DataFrame,
    model_kind: ModelKind,
    meta: dict,
) -> GlucoseWindowDataset | GlucoseICWindowDataset:
    if model_kind == "glumind":
        return GlucoseWindowDataset(
            train_df,
            input_steps=meta["input_steps"],
            horizon=meta["horizon"],
            fit_scalers=True,
        )
    return GlucoseICWindowDataset(
        train_df,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        fit_scalers=True,
    )


def _build_eval_dataset(
    eval_df: pl.DataFrame,
    train_ds: GlucoseWindowDataset | GlucoseICWindowDataset,
    model_kind: ModelKind,
    meta: dict,
) -> GlucoseWindowDataset | GlucoseICWindowDataset:
    if model_kind == "glumind":
        assert isinstance(train_ds, GlucoseWindowDataset)
        return GlucoseWindowDataset(
            eval_df,
            input_steps=meta["input_steps"],
            horizon=meta["horizon"],
            scaler_glucose=train_ds.scaler_glucose,
            scaler_hr=train_ds.scaler_hr,
            scaler_steps=train_ds.scaler_steps,
            fit_scalers=False,
        )
    assert isinstance(train_ds, GlucoseICWindowDataset)
    return GlucoseICWindowDataset(
        eval_df,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        scaler_glucose=train_ds.scaler_glucose,
        scaler_basal=train_ds.scaler_basal,
        scaler_bolus=train_ds.scaler_bolus,
        scaler_carbs=train_ds.scaler_carbs,
        fit_scalers=False,
    )


def _build_model(model_kind: ModelKind, meta: dict) -> nn.Module:
    common = dict(
        n_time_steps=meta["input_steps"],
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        ff_units=meta["ff_units"],
        n_blocks=meta["n_blocks"],
        prediction_horizon=meta["horizon"],
        dropout=meta.get("dropout", 0.1),
    )
    if model_kind == "glumind":
        return GluMindModel(n_features=3, **common)
    return GluMindICModel(n_features=4, **common)


def _load_model_weights(
    model: nn.Module,
    ckpt_path: Path,
    device: str,
) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()


DEFAULT_INFERENCE_LOG_INTERVAL_S = 10.0


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


@torch.no_grad()
def _run_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_windows: int,
    log_interval_s: float = DEFAULT_INFERENCE_LOG_INTERVAL_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with periodic progress logs and ETA."""
    model.eval()
    device_t = torch.device(device)
    n_batches_total = len(loader)
    batch_size = loader.batch_size or 1

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    t_start = time.perf_counter()
    t_last_log = 0.0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device_t), y.to(device_t)
        pred = model(x)
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())

        now = time.perf_counter()
        elapsed = now - t_start
        should_log = (
            batch_idx == 1
            or batch_idx == n_batches_total
            or (elapsed - t_last_log) >= log_interval_s
        )
        if should_log:
            pct = 100.0 * batch_idx / n_batches_total
            batches_per_s = batch_idx / elapsed if elapsed > 0 else 0.0
            remaining_batches = n_batches_total - batch_idx
            eta_s = remaining_batches / batches_per_s if batches_per_s > 0 else 0.0
            windows_done = min(batch_idx * batch_size, n_windows)
            typer.echo(
                f"  inference {batch_idx:,}/{n_batches_total:,} batches "
                f"({pct:.1f}%) | ~{windows_done:,}/{n_windows:,} windows | "
                f"elapsed {_format_duration(elapsed)} | "
                f"ETA {_format_duration(eta_s)}"
            )
            t_last_log = elapsed

    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([])
    pred_arr = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    return true_arr, pred_arr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    test_csv: Path = typer.Option(
        ...,
        "--test-csv",
        help="CSV file to evaluate on.",
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        help="Run directory with tuning_meta.json and best_model.pt.",
    ),
    registry_dir: Path | None = typer.Option(
        None,
        "--registry-dir",
        help="Directory with _analysis_registry.csv; picks lowest val_mae run.",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Explicit .pt weights file (requires --run-dir for architecture meta).",
    ),
    train_csv: Path | None = typer.Option(
        None,
        "--train-csv",
        help="CSV for scaler fitting (default: csv from tuning_meta.json).",
    ),
    model_type: ModelTypeChoice = typer.Option(
        ModelTypeChoice.auto,
        "--model-type",
        help="Model architecture: glumind (HR+steps) or glumind_ic (insulin+carbs).",
    ),
    test_split: str | None = typer.Option(
        "test",
        "--test-split",
        help=(
            "Evaluate rows where 'Recommended Split' equals this value. "
            "Use --test-split='' to disable split filtering."
        ),
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="DataLoader batch size (default: from tuning_meta.json).",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional path to write metrics as JSON for batch comparisons.",
    ),
    log_interval: float = typer.Option(
        DEFAULT_INFERENCE_LOG_INTERVAL_S,
        "--log-interval",
        help="Seconds between inference progress log lines (0 = log first and last only).",
    ),
    zero_cov: bool = typer.Option(
        False,
        "--zero-cov",
        help=(
            "Zero all non-glucose covariates at evaluation time (after imputation). "
            "Equivalent to excluding every non-glucose covariate. "
            "Mutually exclusive with --include-cov / --exclude-cov."
        ),
    ),
    include_cov: str | None = typer.Option(
        None,
        "--include-cov",
        help=(
            "Comma-separated covariates to keep at inference; all other "
            "non-glucose covariates are zeroed after imputation. "
            "Example: --include-cov basal,bolus"
        ),
    ),
    exclude_cov: str | None = typer.Option(
        None,
        "--exclude-cov",
        help=(
            "Comma-separated covariates to zero at inference; remaining "
            "non-glucose covariates are kept. Example: --exclude-cov carbs"
        ),
    ),
    covariates: bool = typer.Option(
        False,
        "--covariates",
        help=(
            "Print covariate columns available in --test-csv and exit "
            "(no checkpoint or inference required)."
        ),
    ),
) -> None:
    test_path = _resolve_csv_path(test_csv)
    eval_split = test_split if test_split else None

    if covariates:
        resolved_kind: ModelKind | None
        if model_type == ModelTypeChoice.auto:
            resolved_kind = None
        else:
            resolved_kind = model_type.value  # type: ignore[assignment]
        _print_dataset_covariates(test_path, resolved_kind, eval_split)
        raise typer.Exit(0)

    if run_dir is None and registry_dir is None:
        typer.echo("Error: Provide at least one of --run-dir or --registry-dir.", err=True)
        raise typer.Exit(1)

    if run_dir is not None:
        resolved_run_dir = run_dir
    else:
        resolved_run_dir, _ = _find_best_run_dir(registry_dir)  # type: ignore[arg-type]

    if not resolved_run_dir.exists():
        typer.echo(f"Error: Run directory does not exist: {resolved_run_dir}", err=True)
        raise typer.Exit(1)

    meta = _load_meta(resolved_run_dir)
    ckpt_path = _resolve_checkpoint(resolved_run_dir, checkpoint)

    typer.echo(f"Run directory: {resolved_run_dir}")
    typer.echo(f"Checkpoint   : {ckpt_path}")
    typer.echo(f"Test CSV     : {test_path}")

    state_probe = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if model_type == ModelTypeChoice.auto:
        resolved_kind = _detect_model_kind(meta, state_probe)
    else:
        resolved_kind = model_type.value  # type: ignore[assignment]

    typer.echo(f"Model type   : {resolved_kind}")

    try:
        active_cov, zeroed_cov = _resolve_covariate_zeroing(
            resolved_kind,
            zero_cov=zero_cov,
            include_cov=include_cov,
            exclude_cov=exclude_cov,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if zeroed_cov:
        if zero_cov:
            typer.echo(f"  --zero-cov: covariates set to 0.0: {', '.join(zeroed_cov)}")
        elif include_cov:
            typer.echo(
                f"  --include-cov {include_cov}: active={', '.join(active_cov)}; "
                f"zeroed={', '.join(zeroed_cov)}"
            )
        else:
            typer.echo(
                f"  --exclude-cov {exclude_cov}: active={', '.join(active_cov)}; "
                f"zeroed={', '.join(zeroed_cov)}"
            )

    train_df = _load_train_for_scalers(
        test_path,
        model_kind=resolved_kind,
        meta=meta,
        train_csv_override=train_csv,
    )
    train_ds = _build_train_dataset(train_df, resolved_kind, meta)
    typer.echo(
        f"Scalers fitted on {len(train_df):,} training rows, {len(train_ds):,} windows."
    )

    impute = impute_and_sort_glumind if resolved_kind == "glumind" else impute_and_sort_ic
    typer.echo(f"Loading evaluation data from: {test_path}")
    eval_df = _load_csv_flexible(
        test_path,
        model_kind=resolved_kind,
        unique_id_choice=meta.get("unique_id", "sequence_id"),
        drop_interpolated=meta.get("drop_interpolated", False),
        eval_split=eval_split,
        train_only=False,
    )
    eval_df = impute(eval_df)
    if zeroed_cov:
        eval_df = _zero_covariates(eval_df, zeroed_cov)

    if eval_df.is_empty():
        typer.echo("Error: Evaluation dataframe is empty after loading/filtering.", err=True)
        raise typer.Exit(1)

    eval_ds = _build_eval_dataset(eval_df, train_ds, resolved_kind, meta)
    if len(eval_ds) == 0:
        typer.echo(
            f"Error: No windows could be built. Each series needs at least "
            f"{meta['input_steps'] + meta['horizon']} rows.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Evaluation windows: {len(eval_ds):,}")
    resolved_batch_size = batch_size or meta.get("batch_size", 4096)
    eval_loader = DataLoader(eval_ds, batch_size=resolved_batch_size, shuffle=False)

    model = _build_model(resolved_kind, meta)
    _load_model_weights(model, ckpt_path, device)

    typer.echo("Running inference...")
    log_interval_s = max(0.0, log_interval)
    y_true_scaled, y_pred_scaled = _run_evaluate(
        model,
        eval_loader,
        device,
        n_windows=len(eval_ds),
        log_interval_s=log_interval_s,
    )

    y_true = train_ds.scaler_glucose.inverse_transform(
        y_true_scaled.ravel().reshape(-1, 1)
    ).ravel()
    y_pred = train_ds.scaler_glucose.inverse_transform(
        y_pred_scaled.ravel().reshape(-1, 1)
    ).ravel()

    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)

    split_used = eval_split if eval_split else "all"
    typer.echo("\n" + "=" * 50)
    typer.echo("EVALUATION RESULTS")
    typer.echo(f"  Model type : {resolved_kind}")
    typer.echo(f"  Test CSV   : {test_path}")
    typer.echo(f"  Split used : {split_used}")
    typer.echo(f"  Zero cov   : {zero_cov}")
    if active_cov:
        typer.echo(f"  Active cov : {', '.join(active_cov)}")
    if zeroed_cov:
        typer.echo(f"  Zeroed cov : {', '.join(zeroed_cov)}")
    typer.echo(f"  Checkpoint : {ckpt_path}")
    typer.echo(f"  Windows    : {len(eval_ds):,}")
    typer.echo("-" * 50)
    typer.echo(f"  MAE : {mae:.4f}")
    typer.echo(f"  RMSE: {rmse:.4f}")
    typer.echo(f"  MARD: {mard:.4f}%")
    typer.echo("=" * 50)

    if output_json is not None:
        payload = {
            "model_type": resolved_kind,
            "test_csv": str(test_path),
            "run_dir": str(resolved_run_dir),
            "checkpoint": str(ckpt_path),
            "split_used": split_used,
            "zero_cov": zero_cov,
            "include_cov": _split_cov_arg(include_cov) or None,
            "exclude_cov": _split_cov_arg(exclude_cov) or None,
            "active_covariates": active_cov,
            "zeroed_covariates": zeroed_cov,
            "windows": len(eval_ds),
            "mae": mae,
            "rmse": rmse,
            "mard": mard,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Metrics written to {output_json}")


if __name__ == "__main__":
    app()
