#!/usr/bin/env python3
"""
Shared PyTorch checkpoint evaluation used by ``glucose evaluate``.

Supports registered model families (via ``common.model_spec``):
  glumind / glumind_uni / sugar_one / sugar_jepa

Prefer::

  uv run glucose evaluate --run-dir <run> --data <csv> ...
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Literal

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from common.checkpoint import strip_compile_prefix
from common.data.columns import (
    COL_EVENT,
    COL_GROUP,
    COL_SEQ,
    COL_SPLIT,
    COL_TS,
    COL_TS_SHORT,
    COL_USER,
    TS_FORMAT,
)
from common.data.loading import impute_and_sort as common_impute_and_sort
from common.evaluation.config import SUPPORTED_MODEL_TYPES
from common.model_spec import detect_family_kind, get_family_spec
from common.paths import resolve_project_path
from common.registry import (
    find_best_run_dir as _common_find_best_run_dir,
    load_run_meta as _load_meta,
    resolve_checkpoint as _common_resolve_checkpoint,
    resolve_csv_path as _common_resolve_csv_path,
    try_resolve_csv_path as _common_try_resolve_csv_path,
)
from common.scalers import (
    SCALERS_FILENAME,
    load_scalers,
    resolve_scalers_path,
    save_scalers_for_run,
)
from common.evaluation.core import (
    COVARIATE_NAME_ALIASES,
    _alias_to_canonical,
    _load_csv_flexible as _common_load_csv_flexible,
    _parse_covariate_names,
    _pick_header_column,
    _resolve_covariate_zeroing,
    _split_cov_arg,
    _zero_covariates,
    DEFAULT_INFERENCE_LOG_INTERVAL_S,
)
from glumind.train_glumind import (
    apply_split_scheme as apply_split_scheme_glumind,
    load_splits_streaming as load_splits_glumind,
    mae_rmse_mard,
)
from sugar_one.train_sugar_one import (
    apply_split_scheme as apply_split_scheme_ic,
    load_splits_streaming as load_splits_ic,
)

ModelKind = Literal["glumind", "sugar_one", "glumind_uni", "sugar_jepa"]
SUPPORTED_KINDS: tuple[str, ...] = tuple(
    t for t in SUPPORTED_MODEL_TYPES if t != "auto"
)

TS_ALIASES = [COL_TS, COL_TS_SHORT]


# ---------------------------------------------------------------------------
# Registry / checkpoint helpers (via common.registry; thin wrappers bind project_root).
# ---------------------------------------------------------------------------

def _find_best_run_dir(registry_dir: Path) -> tuple[Path, dict]:
    return _common_find_best_run_dir(registry_dir, project_root)


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    return _common_resolve_checkpoint(run_dir, checkpoint)


def _resolve_csv_path(csv_value: str | Path) -> Path:
    return _common_resolve_csv_path(csv_value, project_root)


def _detect_model_kind(meta: dict, state: dict[str, torch.Tensor]) -> ModelKind:
    try:
        detected = detect_family_kind(meta, state)
    except ValueError as exc:
        typer.echo(
            f"Error: Could not auto-detect model type from checkpoint. "
            f"Pass --model-type explicitly ({'|'.join(SUPPORTED_KINDS)}). ({exc})",
            err=True,
        )
        raise typer.Exit(1) from exc
    if detected not in SUPPORTED_KINDS:
        typer.echo(
            f"Error: unsupported model family {detected!r}; "
            f"expected one of {', '.join(SUPPORTED_KINDS)}.",
            err=True,
        )
        raise typer.Exit(1)
    return detected  # type: ignore[return-value]


def _covariate_map(model_kind: ModelKind) -> dict[str, list[str]]:
    spec = get_family_spec(model_kind)
    return {name: list(aliases) for name, aliases in spec.csv_column_aliases.items()}


def _canonical_feature_cols(model_kind: ModelKind) -> list[str]:
    return list(get_family_spec(model_kind).feature_names)


def _non_glucose_covariate_cols(model_kind: ModelKind) -> list[str]:
    aliases = get_family_spec(model_kind).covariate_aliases
    if aliases:
        return list(aliases.keys())
    return [c for c in _canonical_feature_cols(model_kind) if c != "glucose"]


def _impute_for_kind(df: pl.DataFrame, model_kind: ModelKind) -> pl.DataFrame:
    spec = get_family_spec(model_kind)
    return common_impute_and_sort(
        df,
        ffill_bfill_columns=list(spec.ffill_bfill_columns),
        zero_fill_columns=list(spec.zero_fill_columns),
    )


def _uses_glumind_splits(model_kind: ModelKind) -> bool:
    return model_kind in ("glumind", "glumind_uni")


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
        [model_kind] if model_kind is not None else list(SUPPORTED_KINDS)  # type: ignore[arg-type]
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
    return _common_load_csv_flexible(
        csv_path,
        model_kind,
        unique_id_choice,
        drop_interpolated,
        eval_split,
        train_only,
        col_seq=COL_SEQ,
        col_user=COL_USER,
        col_split=COL_SPLIT,
        col_group=COL_GROUP,
        ts_aliases=TS_ALIASES,
        ts_format=TS_FORMAT,
    )


def _load_train_for_scalers(
    test_csv_path: Path,
    model_kind: ModelKind,
    meta: dict,
    train_csv_override: Path | None,
    *,
    allow_fit_on_eval: bool = False,
) -> pl.DataFrame:
    """Load training rows for scaler fitting (legacy path when no scalers.json)."""
    if train_csv_override is not None:
        scaler_csv = train_csv_override
    else:
        csv_meta = meta.get("csv")
        if not csv_meta:
            typer.echo("Error: tuning_meta.json has no 'csv' field for scaler fitting.", err=True)
            raise typer.Exit(1)
        resolved = _common_try_resolve_csv_path(csv_meta, project_root)
        if resolved is None:
            typer.echo(
                f"Error: Training CSV from metadata not found: {csv_meta}\n"
                "Provide --train-csv, or place scalers.json in the run directory, "
                "or pass --allow-fit-on-eval to fit scalers on the evaluation CSV "
                "(not recommended for small personal datasets).",
                err=True,
            )
            raise typer.Exit(1)
        scaler_csv = resolved

    typer.echo(f"Fitting scalers from: {scaler_csv}")

    split_scheme = meta.get("split_scheme", "classic")
    unique_id = meta.get("unique_id", "sequence_id")
    drop_interpolated = meta.get("drop_interpolated", False)
    impute = lambda frame: _impute_for_kind(frame, model_kind)

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
            if not allow_fit_on_eval:
                typer.echo(
                    "Error: No train split rows in scaler CSV. Refusing to fit scalers "
                    "on evaluation/all rows (corrupts metrics on small datasets). "
                    "Pass --allow-fit-on-eval to override, or provide scalers.json / "
                    "a proper --train-csv.",
                    err=True,
                )
                raise typer.Exit(1)
            typer.echo(
                "Warning: No train split rows — fitting scalers on all rows "
                "(--allow-fit-on-eval).",
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

    if _uses_glumind_splits(model_kind):
        load_splits = load_splits_glumind
        apply_split = apply_split_scheme_glumind
    else:
        load_splits = load_splits_ic
        apply_split = apply_split_scheme_ic

    train_df_raw, val_df_raw, test_df_raw = load_splits(
        scaler_csv,
        unique_id_choice=unique_id,
        drop_interpolated=drop_interpolated,
    )
    train_df_raw, _, _ = apply_split(train_df_raw, val_df_raw, test_df_raw, split_scheme)

    if train_df_raw.is_empty():
        if not allow_fit_on_eval:
            typer.echo(
                "Error: Training split empty after load_splits. Refusing to fit on "
                "all rows without --allow-fit-on-eval.",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(
            "Warning: Training split empty — fitting scalers on all rows "
            "(--allow-fit-on-eval).",
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
) -> Any:
    return get_family_spec(model_kind).build_window_dataset(
        train_df,
        input_steps=int(meta["input_steps"]),
        horizon=int(meta["horizon"]),
        fit_scalers=True,
        meta=meta,
    )


def _build_eval_dataset_from_scalers(
    eval_df: pl.DataFrame,
    scalers: dict,
    model_kind: ModelKind,
    meta: dict,
) -> Any:
    return get_family_spec(model_kind).build_window_dataset(
        eval_df,
        input_steps=int(meta["input_steps"]),
        horizon=int(meta["horizon"]),
        scalers=scalers,
        fit_scalers=False,
        meta=meta,
    )


def _build_eval_dataset(
    eval_df: pl.DataFrame,
    train_ds: Any,
    model_kind: ModelKind,
    meta: dict,
) -> Any:
    scalers = get_family_spec(model_kind).extract_scalers(train_ds)
    return _build_eval_dataset_from_scalers(eval_df, scalers, model_kind, meta)


def _resolve_feature_scalers(
    resolved_run_dir: Path,
    meta: dict,
    model_kind: ModelKind,
    test_path: Path,
    train_csv: Path | None,
    *,
    refit_scalers: bool,
    allow_fit_on_eval: bool,
) -> tuple[dict, str]:
    """Return ``({feature: scaler}, source_description)``."""
    expected_features = list(get_family_spec(model_kind).feature_names)
    sidecar = None if refit_scalers else resolve_scalers_path(resolved_run_dir, meta)
    if sidecar is not None:
        kind, scalers, _ = load_scalers(sidecar)
        if kind is not None and kind != model_kind:
            typer.echo(
                f"Error: scalers.json kind={kind!r} does not match "
                f"model type={model_kind!r}.",
                err=True,
            )
            raise typer.Exit(1)
        missing = [f for f in expected_features if f not in scalers]
        if missing:
            typer.echo(
                f"Error: scalers.json missing features {missing} "
                f"(expected {expected_features}).",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Loaded scalers from: {sidecar}")
        return {f: scalers[f] for f in expected_features}, str(sidecar)

    if allow_fit_on_eval and train_csv is None and not meta.get("csv"):
        train_csv = test_path

    train_df = _load_train_for_scalers(
        test_path,
        model_kind=model_kind,
        meta=meta,
        train_csv_override=train_csv,
        allow_fit_on_eval=allow_fit_on_eval,
    )
    train_ds = _build_train_dataset(train_df, model_kind, meta)
    scalers = get_family_spec(model_kind).extract_scalers(train_ds)
    # Backfill sidecar for next eval when we had to re-fit from CSV.
    try:
        save_scalers_for_run(
            resolved_run_dir,
            kind=model_kind,
            scalers=scalers,
            provenance={
                "csv": str(train_csv or meta.get("csv", "")),
                "source": "legacy_refit",
                "n_rows": len(train_df),
                "train_windows": len(train_ds),
            },
        )
        typer.echo(f"Wrote {SCALERS_FILENAME} to {resolved_run_dir} (legacy re-fit).")
    except OSError as exc:
        typer.echo(f"Warning: could not write {SCALERS_FILENAME}: {exc}", err=True)
    return scalers, f"refit:{len(train_df)} rows"


def _build_model(model_kind: ModelKind, meta: dict) -> nn.Module:
    return get_family_spec(model_kind).build_model(meta, torch.device("cpu"))


def _load_model_weights(
    model: nn.Module,
    ckpt_path: Path,
    device: str,
) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compile_prefix(state)
    model.load_state_dict(state)
    model.to(device)
    model.eval()


def _run_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_windows: int,
    log_interval_s: float = DEFAULT_INFERENCE_LOG_INTERVAL_S,
    *,
    model_kind: ModelKind = "glumind",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with periodic progress logs and ETA (family-aware via Spec)."""
    import time
    from datetime import timedelta

    spec = get_family_spec(model_kind)
    model.eval()
    device_t = torch.device(device)
    n_batches_total = len(loader)
    batch_size = loader.batch_size or 1
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    t_start = time.perf_counter()
    t_last_log = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            y, pred = spec.infer_batch(model, batch, device_t)
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
                    f"elapsed {timedelta(seconds=int(elapsed))} | "
                    f"ETA {timedelta(seconds=int(eta_s))}"
                )
                t_last_log = elapsed

    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([])
    pred_arr = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    return true_arr, pred_arr


def evaluate_checkpoint(
    *,
    test_csv: Path,
    run_dir: Path | None = None,
    registry_dir: Path | None = None,
    checkpoint: Path | None = None,
    train_csv: Path | None = None,
    refit_scalers: bool = False,
    allow_fit_on_eval: bool = False,
    model_type: str = "auto",
    test_split: str | None = "test",
    batch_size: int | None = None,
    device: str = "auto",
    log_interval: float = DEFAULT_INFERENCE_LOG_INTERVAL_S,
    zero_cov: bool = False,
    include_cov: str | None = None,
    exclude_cov: str | None = None,
    project_root: Path | None = None,
    echo: bool = True,
) -> dict:
    """Evaluate a custom PyTorch checkpoint; return metrics payload dict.

    Shared by ``glucose evaluate``. Raises ``typer.Exit``
    on user-facing errors (same behavior as the CLI).
    """
    from common.evaluation.device import resolve_torch_device

    root = project_root if project_root is not None else globals()["project_root"]
    device = resolve_torch_device(device)
    test_path = _common_resolve_csv_path(test_csv, root)
    eval_split = test_split if test_split else None

    if run_dir is None and registry_dir is None:
        typer.echo("Error: Provide at least one of --run-dir or --registry-dir.", err=True)
        raise typer.Exit(1)

    if run_dir is not None:
        resolved_run_dir = resolve_project_path(run_dir, root)
    else:
        resolved_registry = resolve_project_path(registry_dir, root)  # type: ignore[arg-type]
        resolved_run_dir, _ = _common_find_best_run_dir(resolved_registry, root)

    if not resolved_run_dir.exists():
        typer.echo(f"Error: Run directory does not exist: {resolved_run_dir}", err=True)
        raise typer.Exit(1)

    meta = _load_meta(resolved_run_dir)
    ckpt_path = _common_resolve_checkpoint(resolved_run_dir, checkpoint)

    if echo:
        typer.echo(f"Run directory: {resolved_run_dir}")
        typer.echo(f"Checkpoint   : {ckpt_path}")
        typer.echo(f"Test CSV     : {test_path}")

    state_probe = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    normalized_type = str(model_type or "auto").strip().lower().replace("-", "_")
    if normalized_type in ("", "auto"):
        resolved_kind = _detect_model_kind(meta, state_probe)
    else:
        if normalized_type not in SUPPORTED_KINDS:
            typer.echo(
                f"Error: model_type must be auto|{'|'.join(SUPPORTED_KINDS)}, "
                f"got {model_type!r}.",
                err=True,
            )
            raise typer.Exit(1)
        resolved_kind = normalized_type  # type: ignore[assignment]

    if echo:
        typer.echo(f"Model type   : {resolved_kind}")
        typer.echo(f"Device       : {device}")

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

    if echo and zeroed_cov:
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

    # _resolve_feature_scalers / _load_csv_flexible use module-level helpers that
    # bind to the package project_root; temporarily swap if a custom root is used.
    global_root = globals()["project_root"]
    globals()["project_root"] = root
    try:
        scalers, scaler_source = _resolve_feature_scalers(
            resolved_run_dir,
            meta,
            resolved_kind,
            test_path,
            train_csv,
            refit_scalers=refit_scalers,
            allow_fit_on_eval=allow_fit_on_eval,
        )
        if echo:
            typer.echo(f"Scaler source: {scaler_source}")

        if echo:
            typer.echo(f"Loading evaluation data from: {test_path}")
        eval_df = _load_csv_flexible(
            test_path,
            model_kind=resolved_kind,
            unique_id_choice=meta.get("unique_id", "sequence_id"),
            drop_interpolated=meta.get("drop_interpolated", False),
            eval_split=eval_split,
            train_only=False,
        )
        eval_df = _impute_for_kind(eval_df, resolved_kind)
        if zeroed_cov:
            eval_df = _zero_covariates(eval_df, zeroed_cov)
    finally:
        globals()["project_root"] = global_root

    if eval_df.is_empty():
        typer.echo("Error: Evaluation dataframe is empty after loading/filtering.", err=True)
        raise typer.Exit(1)

    eval_ds = _build_eval_dataset_from_scalers(eval_df, scalers, resolved_kind, meta)
    if len(eval_ds) == 0:
        typer.echo(
            f"Error: No windows could be built. Each series needs at least "
            f"{meta['input_steps'] + meta['horizon']} rows.",
            err=True,
        )
        raise typer.Exit(1)

    if echo:
        typer.echo(f"Evaluation windows: {len(eval_ds):,}")
    resolved_batch_size = batch_size or meta.get("batch_size", 4096)
    eval_loader = DataLoader(eval_ds, batch_size=resolved_batch_size, shuffle=False)

    model = _build_model(resolved_kind, meta)
    _load_model_weights(model, ckpt_path, device)

    if echo:
        typer.echo("Running inference...")
    log_interval_s = max(0.0, log_interval)
    y_true_scaled, y_pred_scaled = _run_evaluate(
        model,
        eval_loader,
        device,
        n_windows=len(eval_ds),
        log_interval_s=log_interval_s,
        model_kind=resolved_kind,
    )

    scaler_glucose = scalers["glucose"]
    y_true = scaler_glucose.inverse_transform(
        y_true_scaled.ravel().reshape(-1, 1)
    ).ravel()
    y_pred = scaler_glucose.inverse_transform(
        y_pred_scaled.ravel().reshape(-1, 1)
    ).ravel()

    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)
    split_used = eval_split if eval_split else "all"
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
        "scaler_source": scaler_source,
        "device": device,
        "mae": mae,
        "rmse": rmse,
        "mard": mard,
    }

    if echo:
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

    return payload
