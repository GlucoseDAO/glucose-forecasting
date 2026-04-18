#!/usr/bin/env python3
"""
GluMind — Multimodal Parallel-Attention Transformer for Blood Glucose Forecasting.

Architecture (Farahmand et al., 2025b, arXiv:2509.18457):
  Parallel cross-attention (multimodal fusion) + multi-scale self-attention
  with optional LwF knowledge retention for continual cross-cohort learning.

Works with the same CSV format as tune_nf_baselines_by_group.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from scripts.glumind.glumind_model import GluMindModel

# ---------------------------------------------------------------------------
# Source CSV columns (same conventions as tune_nf_baselines_by_group.py)
# ---------------------------------------------------------------------------
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

STUDY_GROUP_ORDER = ["Healthy", "Pre-T2DM", "Oral-T2DM", "Insulin-T2DM", "T1DM"]

STUDY_GROUP_ALIASES = {
    "healthy": "Healthy",
    "pre_t2dm": "Pre-T2DM",
    "prediabetes": "Pre-T2DM",
    "pre_diabetes": "Pre-T2DM",
    "pre_diabetes_lifestyle_controlled": "Pre-T2DM",
    "oral_t2dm": "Oral-T2DM",
    "oral_medication": "Oral-T2DM",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled": "Oral-T2DM",
    "insulin_t2dm": "Insulin-T2DM",
    "insulin_dependent": "Insulin-T2DM",
}


def normalize_study_group_label(value: str) -> str:
    """Map raw dataset cohort labels to canonical study-group names."""
    raw = str(value).strip()
    key = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return STUDY_GROUP_ALIASES.get(key, raw)


def normalize_study_groups_column(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize study_group labels in-place-safe form."""
    if df.is_empty():
        return df
    return df.with_columns(
        pl.col("study_group")
        .cast(pl.Utf8)
        .map_elements(normalize_study_group_label, return_dtype=pl.Utf8)
    )


def resolve_num_workers(num_workers: int, device: torch.device) -> int:
    """Resolve DataLoader workers with an auto mode tuned for GPU training."""
    if num_workers >= 0:
        return num_workers
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    return min(8, max(2, cpu_count // 2))


# ============================================================================
#  DATA LOADING
# ============================================================================

def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    drop_interpolated: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Lazy CSV scan via Polars — returns (train, val, test) DataFrames."""
    uid_col = COL_SEQ if unique_id_choice == "sequence_id" else COL_USER
    print("Loading train/val/test splits (streaming)...")

    lf = (
        pl.scan_csv(
            csv_path,
            infer_schema_length=10_000,
            schema_overrides={COL_SEQ: pl.Utf8, COL_USER: pl.Utf8},
        )
        .select([uid_col, COL_TS, COL_SPLIT, COL_GROUP, COL_EVENT,
                 COL_GLU, COL_HR, COL_STEPS])
        .rename({
            uid_col: "unique_id",
            COL_TS: "ds",
            COL_GLU: "glucose",
            COL_HR: "hr",
            COL_STEPS: "steps",
            COL_GROUP: "study_group",
            COL_SPLIT: "split",
            COL_EVENT: "event_type",
        })
        .with_columns([
            pl.col("ds").str.strptime(pl.Datetime, TS_FORMAT, strict=False),
            pl.col("glucose").cast(pl.Float32, strict=False),
            pl.col("hr").cast(pl.Float32, strict=False),
            pl.col("steps").cast(pl.Float32, strict=False),
        ])
        .drop_nulls(subset=["unique_id", "ds", "split", "study_group"])
    )

    if drop_interpolated:
        lf = lf.filter(pl.col("event_type") != "Interpolated")

    df = lf.collect()
    print(f"  ... loaded {len(df):,} rows total")

    train_df = df.filter(pl.col("split") == "train")
    val_df   = df.filter(pl.col("split") == "val")
    test_df  = df.filter(pl.col("split") == "test")
    return train_df, val_df, test_df


def apply_split_scheme(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    split_scheme: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply optional split remapping while preserving classic defaults."""
    if split_scheme == "classic":
        return train_df, val_df, test_df

    if split_scheme == "trainval_test_as_val":
        if test_df.is_empty():
            raise ValueError(
                "split_scheme=trainval_test_as_val requires a non-empty test split."
            )
        merged_train = pl.concat([train_df, val_df]) if not val_df.is_empty() else train_df
        remapped_val = test_df
        remapped_test = test_df.clear()
        print(
            "Applied split scheme: train <- train+val | val <- test | "
            "test disabled."
        )
        print(
            "Note: this mode is for tuning only and does not produce held-out "
            "test metrics."
        )
        return merged_train, remapped_val, remapped_test

    raise ValueError(f"Unknown split_scheme: {split_scheme}")


def impute_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by (unique_id, ds), forward-fill then back-fill numeric columns per series."""
    if df.is_empty():
        return df
    return (
        df.sort(["unique_id", "ds"])
        .with_columns([
            pl.col(c)
            .forward_fill()
            .backward_fill()
            .fill_null(0.0)
            .cast(pl.Float32)
            .over("unique_id")
            for c in ["glucose", "hr", "steps"]
        ])
    )


def limit_series(df: pl.DataFrame, max_series: int) -> pl.DataFrame:
    if df.is_empty() or max_series <= 0:
        return df
    keep = df["unique_id"].unique(maintain_order=True).head(max_series)
    return df.filter(pl.col("unique_id").is_in(keep))


# ============================================================================
#  SLIDING-WINDOW DATASET
# ============================================================================

class GlucoseWindowDataset(Dataset):
    """Lazy sliding-window dataset for multimodal glucose forecasting.

    Stores only the scaled per-series arrays; windows are sliced on-the-fly in
    __getitem__ so peak RAM is O(n_rows) instead of O(n_windows × input_steps).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        scaler_glucose: MinMaxScaler | None = None,
        scaler_hr: MinMaxScaler | None = None,
        scaler_steps: MinMaxScaler | None = None,
        fit_scalers: bool = False,
    ):
        self.input_steps = input_steps
        self.horizon = horizon
        window_len = input_steps + horizon

        # Gather raw arrays per series
        raw_glucose: list[np.ndarray] = []
        raw_hr: list[np.ndarray] = []
        raw_steps: list[np.ndarray] = []
        uids: list = []
        sgroups: list[str] = []
        for (uid_val,), grp in df.sort(["unique_id", "ds"]).group_by(["unique_id"], maintain_order=True):
            uids.append(uid_val)
            sgroups.append(grp["study_group"][0])
            raw_glucose.append(grp["glucose"].to_numpy())
            raw_hr.append(grp["hr"].to_numpy())
            raw_steps.append(grp["steps"].to_numpy())

        # Fit or reuse scalers
        if fit_scalers or scaler_glucose is None:
            all_g = np.concatenate(raw_glucose).reshape(-1, 1)
            all_h = np.concatenate(raw_hr).reshape(-1, 1)
            all_s = np.concatenate(raw_steps).reshape(-1, 1)
            self.scaler_glucose = MinMaxScaler().fit(all_g)
            self.scaler_hr = MinMaxScaler().fit(all_h)
            self.scaler_steps = MinMaxScaler().fit(all_s)
        else:
            self.scaler_glucose = scaler_glucose
            self.scaler_hr = scaler_hr
            self.scaler_steps = scaler_steps

        # Scale each series and build an index: (series_idx, window_start_offset)
        self._series_g: list[np.ndarray] = []
        self._series_h: list[np.ndarray] = []
        self._series_s: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []   # (series_idx, start)
        self.series_ids: list = []
        self.study_groups: list[str] = []

        n_skipped = 0
        for i, (uid, sg, rg, rh, rs) in enumerate(
            zip(uids, sgroups, raw_glucose, raw_hr, raw_steps)
        ):
            g = self.scaler_glucose.transform(rg.reshape(-1, 1)).ravel().astype(np.float32)
            h = self.scaler_hr.transform(rh.reshape(-1, 1)).ravel().astype(np.float32)
            s = self.scaler_steps.transform(rs.reshape(-1, 1)).ravel().astype(np.float32)
            self._series_g.append(g)
            self._series_h.append(h)
            self._series_s.append(s)
            n_windows = len(g) - window_len + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            for start in range(n_windows):
                self._index.append((i, start))
                self.series_ids.append(uid)
                self.study_groups.append(sg)

        if n_skipped > 0:
            print(f"  Note: Skipped {n_skipped} series/segments shorter than "
                  f"{window_len} steps.")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        series_idx, start = self._index[idx]
        g = self._series_g[series_idx]
        h = self._series_h[series_idx]
        s = self._series_s[series_idx]
        x = np.stack([
            g[start : start + self.input_steps],
            h[start : start + self.input_steps],
            s[start : start + self.input_steps],
        ], axis=-1)  # (input_steps, 3)
        y = g[start + self.input_steps : start + self.input_steps + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


# ============================================================================
#  MODEL ARCHITECTURE
# ============================================================================
# Kept in separate module for cleaner inference/evaluation reuse.


# ============================================================================
#  METRICS
# ============================================================================

def mae_rmse_mard(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Compute MAE, RMSE, MARD (same formula as tune_nf_baselines_by_group)."""
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    nonzero = y_true != 0
    if nonzero.any():
        mard = float(np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero])) * 100)
    else:
        mard = float("nan")
    return mae, rmse, mard


# ============================================================================
#  TRAINING & EVALUATION
# ============================================================================

def train_one_epoch(
    model: GluMindModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    teacher: GluMindModel | None = None,
    lwf_lambda: float = 0.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            task_loss = loss_fn(pred, y)

            if teacher is not None and lwf_lambda > 0:
                with torch.no_grad():
                    teacher_pred = teacher(x)
                distill_loss = loss_fn(pred, teacher_pred)
                loss = (1 - lwf_lambda) * task_loss + lwf_lambda * distill_loss
            else:
                loss = task_loss

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: GluMindModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model. Returns (avg_loss, all_true, all_pred)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
        n_batches += 1
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([])
    pred_arr = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    return avg_loss, true_arr, pred_arr


def compute_and_print_metrics(
    true_arr: np.ndarray,
    pred_arr: np.ndarray,
    scaler_glucose: MinMaxScaler,
    split_name: str,
    run_dir: Path,
    dataset: GlucoseWindowDataset | None = None,
):
    """Inverse-transform, compute MAE/RMSE/MARD, save CSVs."""
    # Flatten for overall metrics
    t_flat = true_arr.ravel().reshape(-1, 1)
    p_flat = pred_arr.ravel().reshape(-1, 1)
    t_inv = scaler_glucose.inverse_transform(t_flat).ravel()
    p_inv = scaler_glucose.inverse_transform(p_flat).ravel()

    mae, rmse, mard = mae_rmse_mard(t_inv, p_inv)
    print(f"\n=== {split_name.upper()} METRICS (overall, mg/dL) ===")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MARD: {mard:.2f}%")

    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        run_dir / f"{split_name}_metrics_overall.csv"
    )

    # Per-study-group breakdown
    if dataset is not None and len(dataset.study_groups) == len(true_arr):
        groups = np.array(dataset.study_groups)
        rows = []
        for g in sorted(set(groups)):
            mask = groups == g
            if not mask.any():
                continue
            tg_inv = scaler_glucose.inverse_transform(true_arr[mask].ravel().reshape(-1, 1)).ravel()
            pg_inv = scaler_glucose.inverse_transform(pred_arr[mask].ravel().reshape(-1, 1)).ravel()
            m, r, md = mae_rmse_mard(tg_inv, pg_inv)
            rows.append({"study_group": g, "n_windows": int(mask.sum()),
                          "mae": m, "rmse": r, "mard": md})

        by_group = pl.DataFrame(rows).sort("mae")
        print(f"\n=== {split_name.upper()} METRICS (by Study Group) ===")
        print(by_group)
        by_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")

    return mae, rmse, mard


def save_full_checkpoint(
    path: Path,
    model: GluMindModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    args,
):
    """Save a full checkpoint: model + optimizer + scheduler + metadata."""
    ckpt = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
    }
    torch.save(ckpt, path)


def load_full_checkpoint(
    path: Path,
    model: GluMindModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | None = None,
) -> int:
    """Load a full checkpoint. Returns the epoch number to resume from."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    resume_epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed from checkpoint: epoch={resume_epoch}, "
          f"best_val_loss={best_val:.6f}")
    return resume_epoch, best_val


def train_loop(
    model: GluMindModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    run_dir: Path,
    teacher: GluMindModel | None = None,
    lwf_lambda: float = 0.0,
    verbose_every: int = 10,
    ckpt_every_n_epochs: int = 0,
    ckpt_eval_callback=None,
    start_epoch: int = 1,
    best_val_loss: float = float("inf"),
    args=None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
    val_every_n_epochs: int = 1,
) -> GluMindModel:
    """Full training loop with early stopping on validation loss.

    Args:
        ckpt_every_n_epochs: Save checkpoint + run eval every N epochs (0=off).
        ckpt_eval_callback: Callable(model, epoch, ckpt_dir) for full eval at
            checkpoint epochs.
        start_epoch: Epoch to start from (for resume).
        best_val_loss: Best validation loss so far (for resume).
        args: Parsed CLI args (needed for full checkpoint saving).
    """
    wait = 0
    best_epoch = start_epoch - 1
    start_time = time.time()  # Start time for ETA calculation
    total_time = 0.0

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            teacher=teacher, lwf_lambda=lwf_lambda,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
        )

        val_loss_str = "SKIP"
        should_eval_val = (
            val_loader is not None
            and (epoch == start_epoch or epoch % val_every_n_epochs == 0 or epoch == epochs)
        )
        if should_eval_val:
            val_loss, _, _ = evaluate(
                model, val_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            val_loss_str = f"{val_loss:.6f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                # Save metadata so you always know which epoch is in best_model.pt
                with open(run_dir / "best_info.json", "w") as f:
                    json.dump({"epoch": epoch, "val_loss": best_val_loss}, f)
                print(f"  ★ New best at epoch {epoch} (val_loss={val_loss:.6f})")
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(patience={patience})")
                    break
        elif val_loader is None:
            val_loss_str = "N/A"
            # No validation — save every improvement on train loss
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with open(run_dir / "best_info.json", "w") as f:
                    json.dump({"epoch": epoch, "train_loss": float(train_loss)}, f)

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        total_time += dt
        avg_time = total_time / (epoch - start_epoch + 1)
        remaining_epochs = epochs - epoch
        eta_seconds = avg_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss_str} | "
                  f"{dt:.1f}s/epoch | ETA: {eta_str}")

        # Periodic checkpoint with full eval
        if ckpt_every_n_epochs > 0 and epoch % ckpt_every_n_epochs == 0:
            ckpt_dir = run_dir / "checkpoints" / f"epoch_{epoch:04d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Save full checkpoint (model + optimizer + scheduler + epoch)
            save_full_checkpoint(
                ckpt_dir / "checkpoint.pt",
                model, optimizer, scheduler, epoch, best_val_loss, args,
            )
            print(f"  [Checkpoint] Saved at epoch {epoch} → {ckpt_dir}")
            # Run full eval if callback provided
            if ckpt_eval_callback is not None:
                ckpt_eval_callback(model, epoch, ckpt_dir)

    # Save last full checkpoint
    save_full_checkpoint(
        run_dir / "last_checkpoint.pt",
        model, optimizer, scheduler, epoch, best_val_loss, args,
    )
    # Also save plain weights for easy loading
    torch.save(model.state_dict(), run_dir / "last_model.pt")

    print(f"\n  Summary: best_model.pt = epoch {best_epoch} | "
          f"last_model.pt = epoch {epoch}")

    # Load best
    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=True))
    return model


# ============================================================================
#  TRAINING MODES
# ============================================================================

def make_model(args, device: torch.device) -> GluMindModel:
    model = GluMindModel(
        n_time_steps=args.input_steps,
        n_features=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_units=args.ff_units,
        n_blocks=args.n_blocks,
        prediction_horizon=args.horizon,
        dropout=args.dropout,
    ).to(device)
    if device.type == "cuda" and args.compile_mode != "none":
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"torch.compile enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"Warning: torch.compile failed, using eager mode ({e})")
    return model


def update_latest_symlink(run_dir: Path, out_dir: Path):
    """Write a 'latest.txt' pointer to the most recent run directory.

    Using a plain text file instead of a symlink avoids the Windows privilege
    requirement (WinError 1314) that blocks symlink creation for non-admin users
    without Developer Mode enabled.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_txt = out_dir / "latest.txt"
    latest_txt.write_text(str(run_dir) + "\n", encoding="utf-8")
    print(f"Latest run pointer: {latest_txt} -> {run_dir}")


def make_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    return optimizer, scheduler


def build_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    args,
) -> tuple[GlucoseWindowDataset, GlucoseWindowDataset | None, GlucoseWindowDataset | None]:
    """Build window datasets, fitting scalers on train."""
    train_ds = GlucoseWindowDataset(
        train_df, args.input_steps, args.horizon, fit_scalers=True,
    )
    val_ds = None
    if not val_df.is_empty():
        val_ds = GlucoseWindowDataset(
            val_df, args.input_steps, args.horizon,
            scaler_glucose=train_ds.scaler_glucose,
            scaler_hr=train_ds.scaler_hr,
            scaler_steps=train_ds.scaler_steps,
        )
    test_ds = None
    if not test_df.is_empty():
        test_ds = GlucoseWindowDataset(
            test_df, args.input_steps, args.horizon,
            scaler_glucose=train_ds.scaler_glucose,
            scaler_hr=train_ds.scaler_hr,
            scaler_steps=train_ds.scaler_steps,
        )
    return train_ds, val_ds, test_ds


def run_train_and_eval(
    model: GluMindModel,
    train_ds: GlucoseWindowDataset,
    val_ds: GlucoseWindowDataset | None,
    test_ds: GlucoseWindowDataset | None,
    args,
    device: torch.device,
    run_name: str,
    teacher: GluMindModel | None = None,
    lwf_lambda: float = 0.0,
):
    """Train, evaluate, and save results."""
    # Create run dir
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save initial metadata (args + dataset sizes)
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update({
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "test_samples": len(test_ds) if test_ds else 0,
        "start_time": datetime.now().isoformat()
    })
    with open(run_dir / "tuning_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Create/Update 'latest' symlink
    update_latest_symlink(run_dir, args.out_dir)

    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        **loader_kwargs,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            **loader_kwargs,
        )
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            **loader_kwargs,
        )

    optimizer, scheduler = make_optimizer_and_scheduler(model, args)
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda" and args.precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.precision == "fp16"),
    )

    # Handle resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, best_val_loss = load_full_checkpoint(
            resume_path, model, optimizer, scheduler, device,
        )
        start_epoch += 1  # resume from NEXT epoch

    print(f"\n{'=' * 60}")
    print(f"Training: {len(train_ds):,} windows | "
          f"Val: {len(val_ds) if val_ds else 0:,} | "
          f"Test: {len(test_ds) if test_ds else 0:,} | "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")
    if start_epoch > 1:
        print(f"Resuming from epoch {start_epoch}")
    print(f"{'=' * 60}")

    # Build checkpoint eval callback for periodic full evaluation
    def ckpt_eval_callback(mdl, epoch, ckpt_dir):
        if val_loader is not None:
            _, vt, vp = evaluate(
                mdl, val_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            compute_and_print_metrics(
                vt, vp, train_ds.scaler_glucose,
                f"val_epoch{epoch:04d}", ckpt_dir, val_ds,
            )
        if test_loader is not None:
            _, tt, tp = evaluate(
                mdl, test_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            compute_and_print_metrics(
                tt, tp, train_ds.scaler_glucose,
                f"test_epoch{epoch:04d}", ckpt_dir, test_ds,
            )

    model = train_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        loss_fn, device, args.epochs, args.patience, run_dir,
        teacher=teacher, lwf_lambda=lwf_lambda,
        verbose_every=args.log_every,
        ckpt_every_n_epochs=args.ckpt_every_n_epochs,
        ckpt_eval_callback=ckpt_eval_callback,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        args=args,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=scaler,
        val_every_n_epochs=args.val_every_n_epochs,
    )

    # Final evaluation on val (using best model)
    if val_loader is not None:
        _, val_true, val_pred = evaluate(
            model, val_loader, loss_fn, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        compute_and_print_metrics(
            val_true, val_pred, train_ds.scaler_glucose,
            "val", run_dir, val_ds,
        )

    # Final evaluation on test (using best model)
    if test_loader is not None:
        _, test_true, test_pred = evaluate(
            model, test_loader, loss_fn, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        compute_and_print_metrics(
            test_true, test_pred, train_ds.scaler_glucose,
            "test", run_dir, test_ds,
        )

    # Save config
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model


def mode_global(train_df, val_df, test_df, args, device):
    """Train one model on all data."""
    print("\n=== MODE: GLOBAL ===")
    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df, args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"glumind_global_h{args.horizon}_{ts}"
    run_dir = args.out_dir / run_name
    update_latest_symlink(run_dir, args.out_dir)

    print(f"--> Training global model (dir={run_dir})")

    model = make_model(args, device)
    run_train_and_eval(model, train_ds, val_ds, test_ds, args, device, run_name)


def mode_per_group(train_df, val_df, test_df, args, device):
    """Train separate model per study group."""
    print("\n=== MODE: PER_GROUP ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]

    for group in groups:
        print(f"\n--- Group: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        va = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        te = test_df.filter(pl.col("study_group") == group) if not test_df.is_empty() else test_df

        if tr.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue

        train_ds, val_ds, test_ds = build_datasets(tr, va, te, args)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"glumind_group_{safe_name}_h{args.horizon}_{ts}"

        model = make_model(args, device)
        run_train_and_eval(model, train_ds, val_ds, test_ds,
                           args, device, run_name)


def mode_cohort_wise(train_df, val_df, test_df, args, device):
    """Sequential fine-tuning across groups (reset between groups)."""
    print("\n=== MODE: COHORT_WISE ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]

    for group in groups:
        print(f"\n--- Cohort: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        va = val_df.filter(pl.col("study_group") == group) if not val_df.is_empty() else val_df
        te = test_df  # evaluate on ALL test groups

        if tr.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue

        train_ds, val_ds, test_ds = build_datasets(tr, va, te, args)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"glumind_cohort_{safe_name}_h{args.horizon}_{ts}"

        model = make_model(args, device)
        run_train_and_eval(model, train_ds, val_ds, test_ds,
                           args, device, run_name)


def mode_continual(train_df, val_df, test_df, args, device):
    """Continual learning across cohorts with LwF knowledge retention."""
    print("\n=== MODE: CONTINUAL (LwF) ===")
    present = set(train_df["study_group"].unique().to_list())
    groups = [g for g in STUDY_GROUP_ORDER if g in present]
    if args.continual_order == "reverse":
        groups = list(reversed(groups))
    print(f"Continual group order: {groups}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_parent = f"glumind_continual_h{args.horizon}_{run_ts}"
    print(f"Continual parent run dir: {args.out_dir / run_parent}")

    model = make_model(args, device)
    teacher: GluMindModel | None = None

    all_train_ds = GlucoseWindowDataset(
        train_df, args.input_steps, args.horizon, fit_scalers=True,
    )
    global_scaler_g = all_train_ds.scaler_glucose
    global_scaler_h = all_train_ds.scaler_hr
    global_scaler_s = all_train_ds.scaler_steps

    for i, group in enumerate(groups):
        print(f"\n--- Continual step {i + 1}/{len(groups)}: {group} ---")
        tr = train_df.filter(pl.col("study_group") == group)
        if not val_df.is_empty():
            if args.continual_val_scope == "all_groups":
                va = val_df
            else:
                va = val_df.filter(pl.col("study_group") == group)
        else:
            va = val_df

        if tr.is_empty():
            print(f"  No training data for {group}, skipping.")
            continue

        train_ds = GlucoseWindowDataset(
            tr, args.input_steps, args.horizon,
            scaler_glucose=global_scaler_g,
            scaler_hr=global_scaler_h,
            scaler_steps=global_scaler_s,
        )
        val_ds = None
        if not va.is_empty():
            val_ds = GlucoseWindowDataset(
                va, args.input_steps, args.horizon,
                scaler_glucose=global_scaler_g,
                scaler_hr=global_scaler_h,
                scaler_steps=global_scaler_s,
            )
        test_ds = None
        if not test_df.is_empty():
            test_ds = GlucoseWindowDataset(
                test_df, args.input_steps, args.horizon,
                scaler_glucose=global_scaler_g,
                scaler_hr=global_scaler_h,
                scaler_steps=global_scaler_s,
            )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = group.replace(" ", "_").replace("-", "_")
        run_name = f"{run_parent}/step_{i + 1:02d}_{safe_name}_{ts}"

        lwf = args.lwf_lambda if teacher is not None else 0.0
        model = run_train_and_eval(
            model, train_ds, val_ds, test_ds, args, device, run_name,
            teacher=teacher, lwf_lambda=lwf,
        )

        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        print(f"  Saved teacher snapshot after {group}")


# ============================================================================
#  CLI
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="GluMind: Multimodal Parallel-Attention Transformer "
                    "for Blood Glucose Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    ap.add_argument("--csv", type=Path, required=True,
                    help="Path to processed dataset CSV.")
    ap.add_argument("--unique_id", choices=["sequence_id", "user_id"],
                    default="sequence_id")
    ap.add_argument("--chunk_size", type=int, default=1_000_000)
    ap.add_argument("--max_train_series", type=int, default=0,
                    help="Limit training series (0 = all).")
    ap.add_argument("--max_eval_series", type=int, default=0,
                    help="Limit evaluation series (0 = all).")
    ap.add_argument("--drop_interpolated", action="store_true")
    ap.add_argument("--mask_interpolated_targets", action="store_true")
    ap.add_argument("--study_groups", type=str, default="",
                    help="Comma-separated list of Study Group values. "
                         "Empty = all groups.")
    ap.add_argument(
        "--split_scheme",
        choices=["classic", "trainval_test_as_val"],
        default="classic",
        help="Data split policy. 'classic' uses Recommended Split as-is. "
             "'trainval_test_as_val' merges train+val for training and uses "
             "test as validation (test eval is disabled).",
    )

    # Training mode
    ap.add_argument("--mode",
                    choices=["global", "per_group", "cohort_wise", "continual"],
                    default="global")

    # Forecast
    ap.add_argument("--horizon", type=int, default=12,
                    help="Prediction horizon in steps (12=60min, 6=30min, 1=5min).")
    ap.add_argument("--input_steps", type=int, default=80,
                    help="Input window in steps (80 = 400 min at 5-min freq).")

    # Architecture
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--ff_units", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16",
                    help="Mixed precision mode on CUDA.")
    ap.add_argument("--compile_mode",
                    choices=["none", "default", "reduce-overhead", "max-autotune"],
                    default="none",
                    help="Enable torch.compile for model graph optimization.")
    ap.add_argument("--disable_tf32", action="store_true",
                    help="Disable TF32 on CUDA matmul/cuDNN.")
    ap.add_argument("--num_workers", type=int, default=-1,
                    help="DataLoader workers (-1 = auto; cuda->cpu_count/2 capped at 8).")
    ap.add_argument("--prefetch_factor", type=int, default=4,
                    help="DataLoader prefetch factor (only when num_workers>0).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience (0 = disabled).")
    ap.add_argument("--log_every", type=int, default=10,
                    help="Print loss every N epochs.")
    ap.add_argument("--ckpt_every_n_epochs", type=int, default=0,
                    help="Save checkpoint + run full eval every N epochs "
                         "(0 = disabled). Results saved to "
                         "checkpoints/epoch_NNNN/ subdirs.")
    ap.add_argument("--val_every_n_epochs", type=int, default=1,
                    help="Run validation every N epochs (1 = every epoch).")
    ap.add_argument("--resume_from", type=str, default="",
                    help="Path to a checkpoint.pt file to resume training "
                         "from. Restores model, optimizer, scheduler, and "
                         "epoch number.")

    # Continual learning
    ap.add_argument("--lwf_lambda", type=float, default=0.5,
                    help="LwF distillation weight for continual mode.")
    ap.add_argument(
        "--continual_order",
        type=str,
        choices=["default", "reverse"],
        default="default",
        help=(
            "Group order in continual mode: 'default' follows "
            "Healthy->Pre-T2DM->Oral-T2DM->Insulin-T2DM->T1DM; "
            "'reverse' runs the opposite order."
        ),
    )
    ap.add_argument(
        "--continual_val_scope",
        type=str,
        choices=["current_group", "all_groups"],
        default="current_group",
        help=(
            "Validation set scope in continual mode: 'current_group' validates "
            "only on the active cohort; 'all_groups' validates on the full "
            "validation split each step."
        ),
    )

    # System
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("runs/glumind"))
    ap.add_argument("--save_predictions", action="store_true")

    return ap.parse_args()


# ============================================================================
#  MAIN
# ============================================================================

def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        if not args.disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled.")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    # Load data
    train_df, val_df, test_df = load_splits_streaming(
        args.csv, args.unique_id, args.drop_interpolated,
    )
    print(f"Loaded: train={len(train_df):,} | val={len(val_df):,} | "
          f"test={len(test_df):,}")

    # Normalize cohort labels so per-group/cohort/continual modes work across
    # dataset naming variants (e.g., lowercase and long-form labels).
    train_df = normalize_study_groups_column(train_df)
    val_df = normalize_study_groups_column(val_df)
    test_df = normalize_study_groups_column(test_df)

    # Filter study groups if requested
    if args.study_groups:
        groups = [normalize_study_group_label(g.strip())
                  for g in args.study_groups.split(",") if g.strip()]
        train_df = train_df.filter(pl.col("study_group").is_in(groups))
        val_df = val_df.filter(pl.col("study_group").is_in(groups))
        test_df = test_df.filter(pl.col("study_group").is_in(groups))
        print(f"Filtered to groups {groups}: train={len(train_df):,} | "
              f"val={len(val_df):,} | test={len(test_df):,}")

    # Optional split remapping for tuning variants.
    train_df, val_df, test_df = apply_split_scheme(
        train_df, val_df, test_df, args.split_scheme
    )

    # Impute and sort
    train_df = impute_and_sort(train_df)
    val_df = impute_and_sort(val_df)
    test_df = impute_and_sort(test_df)

    # Limit series
    if args.max_train_series > 0:
        train_df = limit_series(train_df, args.max_train_series)
    if args.max_eval_series > 0:
        val_df = limit_series(val_df, args.max_eval_series)
        test_df = limit_series(test_df, args.max_eval_series)

    print(f"After limits: train={len(train_df):,} | val={len(val_df):,} | "
          f"test={len(test_df):,}")
    print(f"Study groups in train: "
          f"{sorted(train_df['study_group'].unique().to_list())}")

    # Dispatch to training mode
    mode_fn = {
        "global": mode_global,
        "per_group": mode_per_group,
        "cohort_wise": mode_cohort_wise,
        "continual": mode_continual,
    }[args.mode]

    mode_fn(train_df, val_df, test_df, args, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
