#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NBEATSx, NHITS, TFT

from common.paths import DEFAULT_RUNS_ROOT

# ---- Source CSV columns ----
COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"

COL_GLU = "Glucose Value (mg/dL)"
COL_HR = "Heart Rate"
COL_STEPS = "Step Count"

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"

MODEL_MAP = {
    "tft": TFT,
    "nhits": NHITS,
    "nbeatsx": NBEATSx,
}


@dataclass(frozen=True)
class RunConfig:
    model: str
    lr: float
    max_steps: int
    val_check_steps: int
    batch_size: int
    valid_batch_size: int
    windows_batch_size: int
    inference_windows_batch_size: int
    step_size: int


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument(
        "--split_scheme",
        choices=["classic", "trainval_test_as_val"],
        default="classic",
        help=(
            "classic: use CSV train/val/test as-is. "
            "trainval_test_as_val: train <- train+val, val <- test, test disabled."
        ),
    )
    ap.add_argument("--unique_id", choices=["sequence_id", "user_id"], default="sequence_id")

    ap.add_argument("--model", choices=["tft", "nhits", "nbeatsx", "all"], default="tft")
    ap.add_argument("--grid", type=Path, default=None,
                    help="Optional JSON grid. Example: {'tft':[{'lr':1e-3,'max_steps':2000}], 'nhits':[...]} "
                         "or {'*':[...]} for all models.")

    # Forecast horizon
    ap.add_argument("--h_min", type=int, default=60)
    ap.add_argument("--freq", type=str, default="5min")

    # Context and internal train-tail validation (for valid_loss / early stopping)
    ap.add_argument("--input_hours", type=float, default=6.0)
    ap.add_argument("--train_tail_val_hours", type=float, default=24.0)

    # Training defaults (used when grid does not override)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--val_check_steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--valid_batch_size", type=int, default=8)
    ap.add_argument("--windows_batch_size", type=int, default=256)
    ap.add_argument("--inference_windows_batch_size", type=int, default=256)
    ap.add_argument("--step_size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    ap.add_argument("--seed", type=int, default=1)

    # IO / streaming
    ap.add_argument("--chunk_size", type=int, default=1_000_000)
    ap.add_argument("--max_train_series", type=int, default=0)
    ap.add_argument("--max_eval_series", type=int, default=0)
    ap.add_argument("--max_points_per_series", type=int, default=0)

    ap.add_argument("--study_groups", type=str, default="",
                    help="Comma-separated list of Study Group values. Empty = all groups.")
    ap.add_argument("--global_model", action="store_true",
                    help="Train a single model on all study groups (no per-group split).")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--save_predictions", action="store_true")

    ap.add_argument("--ckpt_every_n_steps", type=int, default=400)
    ap.add_argument("--early_stop_patience", type=int, default=10)
    ap.add_argument("--save_all_checkpoints", action="store_true",
                    help="Save every checkpoint (do not keep only best by valid_loss).")
    ap.add_argument("--eval_checkpoints", action="store_true",
                    help="Evaluate every saved checkpoint on val/test after training.")

    ap.add_argument("--train_event_type", type=str, default="",
                    help="Optional: filter TRAIN by Event Type. Empty string disables.")
    ap.add_argument("--drop_interpolated", action="store_true",
                    help="Drop rows where Event Type == 'Interpolated' for all splits.")
    ap.add_argument("--mask_interpolated_targets", action="store_true",
                    help="Keep interpolated rows in history, but exclude them from eval metrics.")
    return ap.parse_args()


def freq_to_minutes(freq: str) -> int:
    td = pd.to_timedelta(freq)
    return int(td.total_seconds() // 60)


def steps_from_minutes(minutes: int, freq: str) -> int:
    step_min = freq_to_minutes(freq)
    if minutes % step_min != 0:
        raise ValueError(f"h_min={minutes} not divisible by freq={freq} ({step_min} minutes)")
    return minutes // step_min


def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    chunk_size: int,
    train_event_type: str,
    drop_interpolated: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uid_col = COL_SEQ if unique_id_choice == "sequence_id" else COL_USER
    usecols = [uid_col, COL_TS, COL_SPLIT, COL_GROUP, COL_EVENT, COL_GLU, COL_HR, COL_STEPS]

    train_parts, val_parts, test_parts = [], [], []
    print("Loading train/val/test splits (streaming)...")

    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        seen += len(chunk)
        if seen % 6_000_000 == 0:
            print(f"... loaded {seen:,} rows")

        chunk[COL_TS] = pd.to_datetime(chunk[COL_TS], format=TS_FORMAT, errors="coerce")
        chunk = chunk.rename(
            columns={
                uid_col: "unique_id",
                COL_TS: "ds",
                COL_GLU: "y",
                COL_HR: "hr",
                COL_STEPS: "steps",
                COL_GROUP: "study_group",
                COL_SPLIT: "split",
                COL_EVENT: "event_type",
            }
        )
        chunk = chunk.dropna(subset=["unique_id", "ds", "split", "study_group"])

        for c in ["y", "hr", "steps"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        tr = chunk[chunk["split"] == "train"]
        if train_event_type:
            tr = tr[tr["event_type"] == train_event_type]
        if drop_interpolated:
            tr = tr[tr["event_type"] != "Interpolated"]
        va = chunk[chunk["split"] == "val"]
        te = chunk[chunk["split"] == "test"]
        if drop_interpolated:
            va = va[va["event_type"] != "Interpolated"]
            te = te[te["event_type"] != "Interpolated"]

        if not tr.empty:
            train_parts.append(tr)
        if not va.empty:
            val_parts.append(va)
        if not te.empty:
            test_parts.append(te)

    expected_cols = ["unique_id", "ds", "y", "hr", "steps", "study_group", "split", "event_type"]
    train_df = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pd.DataFrame(columns=expected_cols)
    )
    val_df = (
        pd.concat(val_parts, ignore_index=True)
        if val_parts
        else pd.DataFrame(columns=expected_cols)
    )
    test_df = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=expected_cols)
    )
    return train_df, val_df, test_df


def apply_split_scheme(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_scheme == "classic":
        return train_df, val_df, test_df

    if split_scheme == "trainval_test_as_val":
        if test_df.empty:
            raise ValueError(
                "split_scheme=trainval_test_as_val requires a non-empty test split."
            )

        merged_train = (
            pd.concat([train_df, val_df], ignore_index=True)
            if not val_df.empty
            else train_df.copy()
        )
        remapped_val = test_df.copy()
        remapped_test = pd.DataFrame(columns=test_df.columns)

        print(
            "Applied split scheme: train <- train+val | val <- test | test disabled."
        )
        print(
            "Note: this mode is for tuning only and does not produce held-out test metrics."
        )
        return merged_train, remapped_val, remapped_test

    raise ValueError(f"Unknown split_scheme: {split_scheme}")


def basic_impute_and_types(df: pd.DataFrame, max_points_per_series: int = 0) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    if max_points_per_series and max_points_per_series > 0:
        df = df.groupby("unique_id", sort=False, observed=True).tail(max_points_per_series).copy()
        df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    for c in ["y", "hr", "steps"]:
        df[c] = df.groupby("unique_id", sort=False)[c].ffill()
        df[c] = df.groupby("unique_id", sort=False)[c].bfill()

    df[["y", "hr", "steps"]] = df[["y", "hr", "steps"]].fillna(0.0)

    uid_num = pd.to_numeric(df["unique_id"], errors="coerce")
    uid_num = uid_num.where(np.isfinite(uid_num))
    if uid_num.notna().all():
        df["unique_id"] = uid_num.astype(np.int64)
    else:
        n_bad = int(uid_num.isna().sum())
        print(
            f"  Note: Found {n_bad} non-numeric/invalid unique_id values; keeping unique_id as string."
        )
        uid_str = df["unique_id"].astype("string").str.strip()
        bad_empty = uid_str.isna() | (uid_str == "")
        if bad_empty.any():
            dropped = int(bad_empty.sum())
            print(f"  Note: Dropped {dropped} rows with empty unique_id after normalization.")
            df = df.loc[~bad_empty].copy()
            uid_str = uid_str.loc[~bad_empty]
        df["unique_id"] = uid_str.astype("string")

    df[["y", "hr", "steps"]] = df[["y", "hr", "steps"]].astype(np.float32)
    df["study_group"] = df["study_group"].astype("category")
    return df


def filter_min_length(df: pd.DataFrame, min_len: int, log_label: str = "") -> pd.DataFrame:
    if df.empty:
        return df
    sizes = df.groupby("unique_id", sort=False, observed=True).size()
    keep = sizes[sizes >= min_len].index
    n_skipped = int(sizes.shape[0] - len(keep))
    if n_skipped > 0:
        if log_label:
            print(
                f"  Note: [{log_label}] Skipped {n_skipped} series/segments shorter than {min_len} steps."
            )
        else:
            print(f"  Note: Skipped {n_skipped} series/segments shorter than {min_len} steps.")
    return df[df["unique_id"].isin(keep)].copy()


def maybe_limit_series(df: pd.DataFrame, max_series: int) -> pd.DataFrame:
    if df.empty or not max_series or max_series <= 0:
        return df
    keep = df["unique_id"].drop_duplicates().head(max_series).to_numpy()
    return df[df["unique_id"].isin(keep)].copy()


def n_series(df: pd.DataFrame) -> int:
    if "unique_id" not in df.columns or df.empty:
        return 0
    return int(df["unique_id"].nunique())


def mae_rmse_mard(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    nonzero = y_true != 0
    if nonzero.any():
        mard = float(np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero])) * 100)
    else:
        mard = float("nan")
    return mae, rmse, mard


def eval_split(
    nf: NeuralForecast,
    split_name: str,
    df: pd.DataFrame,
    h: int,
    input_size: int,
    hist_exogs: list[str],
    run_dir: Path,
    save_predictions: bool,
    max_eval_series: int = 0,
    mask_interpolated_targets: bool = False,
):
    if df.empty:
        print(f"[{split_name}] empty split, skipping.")
        return

    df = maybe_limit_series(df, max_eval_series)
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    df = filter_min_length(df, input_size + h, log_label=split_name)

    n_series = df["unique_id"].nunique()
    print(f"Evaluating {split_name}: rows={len(df):,} | series={n_series:,}")

    true_df = (
        df.groupby("unique_id", sort=False, observed=True)
        .tail(h)[["unique_id", "ds", "y", "study_group", "event_type"]]
        .copy()
    )
    hist_cols = ["unique_id", "ds", "y"] + list(hist_exogs)
    hist_df = df.drop(true_df.index)[hist_cols].copy()

    preds = nf.predict(df=hist_df)
    pred_col = "TFT" if "TFT" in preds.columns else [c for c in preds.columns if c not in ("unique_id", "ds")][0]

    eval_df = true_df.merge(preds[["unique_id", "ds", pred_col]], on=["unique_id", "ds"], how="inner")
    eval_df = eval_df.rename(columns={pred_col: "yhat"})

    if mask_interpolated_targets:
        before = len(eval_df)
        eval_df = eval_df[eval_df["event_type"] != "Interpolated"].copy()
        dropped = before - len(eval_df)
        if dropped:
            print(f"[{split_name}] dropped {dropped:,} interpolated target rows from metrics.")

    if eval_df.empty:
        print(f"[{split_name}] no evaluation rows after filtering.")
        return

    overall_mae, overall_rmse, overall_mard = mae_rmse_mard(
        eval_df["y"].to_numpy(np.float32),
        eval_df["yhat"].to_numpy(np.float32),
    )

    print(f"\n=== {split_name.upper()} METRICS (overall) ===")
    print(f"MAE : {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MARD: {overall_mard:.4f}%")

    rows = []
    for g, part in eval_df.groupby("study_group", observed=True):
        m, r, md = mae_rmse_mard(part["y"].to_numpy(np.float32), part["yhat"].to_numpy(np.float32))
        rows.append((str(g), len(part), m, r, md))

    by_group = (
        pd.DataFrame(rows, columns=["study_group", "n_points", "mae", "rmse", "mard"])
        .sort_values("mae")
        .reset_index(drop=True)
    )

    print(f"\n=== {split_name.upper()} METRICS (by Study Group) ===")
    print(by_group.to_string(index=False))

    by_group.to_csv(run_dir / f"{split_name}_metrics_by_study_group.csv", index=False)
    pd.DataFrame([{"mae": overall_mae, "rmse": overall_rmse, "mard": overall_mard}]).to_csv(
        run_dir / f"{split_name}_metrics_overall.csv", index=False
    )

    if save_predictions:
        eval_df.to_csv(run_dir / f"{split_name}_predictions.csv", index=False)


def parse_step(ckpt_path: Path) -> int:
    m = re.search(r"step=(\d+)", ckpt_path.name)
    if m:
        return int(m.group(1))
    if ckpt_path.name == "last.ckpt":
        return 10**18
    return -1


def load_model_from_ckpt(model_name: str, ckpt_path: Path, device: str):
    cls = MODEL_MAP[model_name]

    try:
        model = cls.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "hyper_parameters" not in ckpt:
            raise RuntimeError("Checkpoint missing hyper_parameters; cannot reconstruct model safely.")
        hparams = ckpt["hyper_parameters"]
        model = cls(**hparams)
        model.load_state_dict(ckpt["state_dict"], strict=True)

    model.trainer_kwargs = {
        "accelerator": device,
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "callbacks": [],
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    model.eval()
    return model


def prime_neuralforecast(nf: NeuralForecast, df_small: pd.DataFrame, device: str):
    for m in nf.models:
        m.trainer_kwargs = {
            "accelerator": device,
            "devices": 1,
            "max_steps": 1,
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
            "logger": False,
            "enable_checkpointing": True,
            "callbacks": [],
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        if hasattr(m, "val_check_steps"):
            m.val_check_steps = 10**9

    nf.fit(df=df_small, val_size=0)


def load_grid(path: Path | None) -> dict:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_configs(args, model_name: str, grid: dict) -> list[RunConfig]:
    base = {
        "lr": args.lr,
        "max_steps": args.max_steps,
        "val_check_steps": args.val_check_steps,
        "batch_size": args.batch_size,
        "valid_batch_size": args.valid_batch_size,
        "windows_batch_size": args.windows_batch_size,
        "inference_windows_batch_size": args.inference_windows_batch_size,
        "step_size": args.step_size,
    }

    raw_list = []
    if "*" in grid:
        raw_list = grid["*"]
    elif model_name in grid:
        raw_list = grid[model_name]

    if not raw_list:
        return [RunConfig(model=model_name, **base)]

    configs = []
    for item in raw_list:
        merged = dict(base)
        merged.update(item or {})
        configs.append(RunConfig(model=model_name, **merged))
    return configs


def run_id_from_config(cfg: RunConfig) -> str:
    return (
        f"{cfg.model}_lr{cfg.lr:g}_ms{cfg.max_steps}_"
        f"bs{cfg.batch_size}_ws{cfg.windows_batch_size}_ss{cfg.step_size}"
    )


def build_model(
    cfg: RunConfig,
    h: int,
    input_size: int,
    hist_exogs: list[str],
    trainer_kwargs: dict,
):
    model_cls = MODEL_MAP[cfg.model]
    kwargs = dict(
        h=h,
        input_size=input_size,
        loss=MAE(),
        valid_loss=MAE(),
        max_steps=cfg.max_steps,
        val_check_steps=min(cfg.val_check_steps, cfg.max_steps),
        learning_rate=cfg.lr,
        batch_size=cfg.batch_size,
        valid_batch_size=cfg.valid_batch_size,
        windows_batch_size=cfg.windows_batch_size,
        inference_windows_batch_size=cfg.inference_windows_batch_size,
        step_size=cfg.step_size,
        **trainer_kwargs,
    )
    kwargs["hist_exog_list"] = hist_exogs
    return model_cls(**kwargs)


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    step_min = freq_to_minutes(args.freq)
    h = steps_from_minutes(args.h_min, args.freq)
    input_size = int(round((args.input_hours * 60) / step_min))
    train_tail_val_size = int(round((args.train_tail_val_hours * 60) / step_min))

    print(f"pytorch_lightning version: {pl.__version__}")
    print(f"h (steps): {h} | input_size (steps): {input_size} | freq: {args.freq} ({step_min} min/step)")
    print(f"train-tail internal val_size (steps): {train_tail_val_size}")
    print(f"step_size: {args.step_size}")

    train_event_type = args.train_event_type.strip()
    if train_event_type == "":
        train_event_type = ""

    train_df, val_df, test_df = load_splits_streaming(
        args.csv, args.unique_id, args.chunk_size, train_event_type, args.drop_interpolated
    )
    train_df, val_df, test_df = apply_split_scheme(
        train_df, val_df, test_df, args.split_scheme
    )

    print(f"Train rows: {len(train_df):,} | series: {n_series(train_df):,}")
    print(f"Val   rows: {len(val_df):,} | series: {n_series(val_df):,}")
    print(f"Test  rows: {len(test_df):,} | series: {n_series(test_df):,}")

    train_df = basic_impute_and_types(train_df, args.max_points_per_series)
    val_df = basic_impute_and_types(val_df, args.max_points_per_series)
    test_df = basic_impute_and_types(test_df, args.max_points_per_series)

    if args.max_train_series and args.max_train_series > 0:
        train_df = maybe_limit_series(train_df, args.max_train_series)
        print(f"SMOKE TEST: limited train series to {args.max_train_series}")

    min_train_len = input_size + train_tail_val_size + h
    min_eval_len = input_size + h

    train_df = filter_min_length(train_df, min_train_len, log_label="train")
    val_df = filter_min_length(val_df, min_eval_len, log_label="val")
    test_df = filter_min_length(test_df, min_eval_len, log_label="test")

    print("After length filter:")
    print(f"  Train rows: {len(train_df):,} | series: {n_series(train_df):,}")
    print(f"  Val   rows: {len(val_df):,} | series: {n_series(val_df):,}")
    print(f"  Test  rows: {len(test_df):,} | series: {n_series(test_df):,}")

    all_groups = sorted(train_df["study_group"].cat.categories.to_list())
    if args.global_model:
        groups = ["__ALL__"]
    else:
        if args.study_groups.strip():
            want = [g.strip() for g in args.study_groups.split(",") if g.strip()]
            groups = [g for g in all_groups if g in want]
        else:
            groups = all_groups

        if not groups:
            raise SystemExit("No study groups selected after filtering.")

    default_hist_exogs = ["hr", "steps"]

    grid = load_grid(args.grid)
    models = [args.model] if args.model != "all" else ["tft", "nhits", "nbeatsx"]

    for group in groups:
        print(f"\n=== STUDY GROUP: {group} ===")
        if group == "__ALL__":
            g_train = train_df.copy()
            g_val = val_df.copy()
            g_test = test_df.copy()
        else:
            g_train = train_df[train_df["study_group"] == group].copy()
            g_val = val_df[val_df["study_group"] == group].copy()
            g_test = test_df[test_df["study_group"] == group].copy()

        if g_train.empty:
            print(f"[{group}] train split empty, skipping.")
            continue

        for model_name in models:
            configs = iter_configs(args, model_name, grid)
            for cfg in configs:
                run_id = run_id_from_config(cfg)
                run_dir = args.out_dir / f"{group}/{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                run_dir.mkdir(parents=True, exist_ok=True)

                ckpt_steps = ModelCheckpoint(
                    dirpath=str(run_dir / "checkpoints"),
                    filename="step-{step}",
                    save_top_k=-1 if args.save_all_checkpoints else 1,
                    monitor="valid_loss",
                    mode="min",
                    every_n_train_steps=args.ckpt_every_n_steps,
                    save_last=True,
                )
                es_cb = EarlyStopping(monitor="valid_loss", mode="min", patience=args.early_stop_patience)

                trainer_kwargs = dict(
                    accelerator=args.device,
                    devices=1,
                    default_root_dir=str(run_dir),
                    enable_checkpointing=True,
                    callbacks=[ckpt_steps, es_cb],
                    log_every_n_steps=10,
                )

                model_hist_exogs = default_hist_exogs
                model = build_model(cfg, h, input_size, model_hist_exogs, trainer_kwargs)
                nf = NeuralForecast(models=[model], freq=args.freq)

                train_cols = ["unique_id", "ds", "y"] + model_hist_exogs
                train_nf = g_train[train_cols]

                print(f"\nFitting {cfg.model.upper()} | {group} | {run_id}")
                print(f"Internal validation: last {train_tail_val_size} steps from each TRAIN series.")
                nf.fit(df=train_nf, val_size=train_tail_val_size)

                eval_split(
                    nf=nf,
                    split_name="val",
                    df=g_val,
                    h=h,
                    input_size=input_size,
                    hist_exogs=model_hist_exogs,
                    run_dir=run_dir,
                    save_predictions=args.save_predictions,
                    max_eval_series=args.max_eval_series,
                    mask_interpolated_targets=args.mask_interpolated_targets,
                )

                eval_split(
                    nf=nf,
                    split_name="test",
                    df=g_test,
                    h=h,
                    input_size=input_size,
                    hist_exogs=model_hist_exogs,
                    run_dir=run_dir,
                    save_predictions=args.save_predictions,
                    max_eval_series=args.max_eval_series,
                    mask_interpolated_targets=args.mask_interpolated_targets,
                )

                if args.eval_checkpoints:
                    ckpt_dir = run_dir / "checkpoints"
                    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=parse_step)
                    if not ckpts:
                        print(f"No checkpoints found in {ckpt_dir}, skipping eval_checkpoints.")
                    else:
                        # Prime once with the first checkpoint to satisfy NeuralForecast internals.
                        prime_model = load_model_from_ckpt(model_name, ckpts[0], args.device)
                        nf_ckpt = NeuralForecast(models=[prime_model], freq=args.freq)
                        prime_df = filter_min_length(g_val, input_size + h)
                        prime_df = maybe_limit_series(prime_df, 2)
                        prime_df = prime_df.groupby("unique_id", sort=False, observed=True).tail(input_size + h).copy()
                        prime_df = prime_df.sort_values(["unique_id", "ds"], kind="mergesort")
                        prime_cols = ["unique_id", "ds", "y"] + model_hist_exogs
                        if not prime_df.empty:
                            try:
                                _ = nf_ckpt.predict(df=prime_df[prime_cols])
                            except Exception as e:
                                print(f"[priming] predict() pre-fit failed ({type(e).__name__}: {e}). Doing 1-step fit priming...")
                                prime_neuralforecast(
                                    nf_ckpt,
                                    prime_df[prime_cols],
                                    args.device,
                                )
                        for ckpt_path in ckpts:
                            eval_dir = run_dir / "eval_checkpoints" / ckpt_path.stem
                            eval_dir.mkdir(parents=True, exist_ok=True)
                            model = load_model_from_ckpt(model_name, ckpt_path, args.device)
                            nf_ckpt.models = [model]

                            print(f"\nEvaluating checkpoint: {ckpt_path.name}")
                            eval_split(
                                nf=nf_ckpt,
                                split_name="val",
                                df=g_val,
                                h=h,
                                input_size=input_size,
                                hist_exogs=model_hist_exogs,
                                run_dir=eval_dir,
                                save_predictions=args.save_predictions,
                                max_eval_series=args.max_eval_series,
                                mask_interpolated_targets=args.mask_interpolated_targets,
                            )
                            eval_split(
                                nf=nf_ckpt,
                                split_name="test",
                                df=g_test,
                                h=h,
                                input_size=input_size,
                                hist_exogs=model_hist_exogs,
                                run_dir=eval_dir,
                                save_predictions=args.save_predictions,
                                max_eval_series=args.max_eval_series,
                                mask_interpolated_targets=args.mask_interpolated_targets,
                            )

                cfg_path = run_dir / "run_config.json"
                with cfg_path.open("w", encoding="utf-8") as f:
                    json.dump(cfg.__dict__, f, indent=2, sort_keys=True)

                print(f"Saved run to: {run_dir}")
                print(f"Checkpoints: {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()

# ["tft", "nhits", "nbeatsx"]

# uv run python src/neuralforecast/tune_nf_baselines_by_group.py --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv --model tft --global_model --save_all_checkpoints --eval_checkpoints --mask_interpolated_targets
# uv run python src/neuralforecast/tune_nf_baselines_by_group.py --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv --model tft --mask_interpolated_targets

# uv run python src/neuralforecast/tune_nf_baselines_by_group.py --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv --model nhits --global_model --save_all_checkpoints --eval_checkpoints --mask_interpolated_targets

# uv run python src/neuralforecast/tune_nf_baselines_by_group.py --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv --model nbeatsx --global_model --save_all_checkpoints --eval_checkpoints --mask_interpolated_targets
