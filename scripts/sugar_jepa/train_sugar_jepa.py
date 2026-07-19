#!/usr/bin/env python3
"""
SugarJepa — SugarOne + a pretrained CGM-JEPA glucose embedding as a 4th
cross-attention auxiliary stream. See scripts/sugar_jepa/sugar_jepa_model.py
and scripts/sugar_jepa/README.md.

Dataset: data/input/loop_ai_ready_joined2_dev.csv (or the full
loop_ai_ready_joined2.csv).

Proof-of-concept scope: `global` mode only (one model, all study groups) —
per_group / cohort_wise / continual (LwF) from train_sugar_one.py are not
implemented here.

Imputation policy (identical to SugarOne):
  - Basal Rate: forward-fill then back-fill, then fill_null(0.0).
  - Bolus Insulin / Carbohydrates: fill_null(0.0) directly (discrete events).
  - Glucose: forward-fill then back-fill then fill_null(0.0).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from scripts.sugar_one.console_log import echo_plain
from scripts.sugar_jepa.sugar_jepa_model import SugarJepaModel

from scripts.common.data_loading import (
    limit_series,
    normalize_study_group_label,
    normalize_study_groups_column,
    resolve_num_workers,
)
from scripts.common.data_loading import apply_split_scheme as _common_apply_split_scheme
from scripts.common.data_loading import impute_and_sort as _common_impute_and_sort
from scripts.common.data_loading import load_splits_streaming as _common_load_splits_streaming
from scripts.common.metrics import mae_rmse_mard
from scripts.common.checkpoint import load_full_checkpoint as _common_load_full_checkpoint
from scripts.common.checkpoint import save_full_checkpoint as _common_save_full_checkpoint
from scripts.common.checkpoint import update_latest_symlink as _common_update_latest_symlink

app = typer.Typer(
    name="train_sugar_jepa",
    add_completion=False,
    help="SugarJepa: SugarOne + a pretrained CGM-JEPA glucose embedding auxiliary.",
)

# ---------------------------------------------------------------------------
# CSV column names — same as loop_ai_ready_joined2*.csv used by SugarOne.
# ---------------------------------------------------------------------------
COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS = "Timestamp"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"

COL_GLU = "Glucose (mg/dL)"
COL_BASAL = "Basal Rate (U/h)"
COL_BOLUS = "Bolus Insulin (U)"
COL_CARB = "Carbohydrates (g)"

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"

N_FEATURES = 4  # glucose, basal, bolus, carbs


# ============================================================================
#  DATA LOADING
# ============================================================================

def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    drop_interpolated: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return _common_load_splits_streaming(
        csv_path,
        unique_id_choice,
        drop_interpolated,
        col_seq=COL_SEQ,
        col_user=COL_USER,
        col_ts=COL_TS,
        col_split=COL_SPLIT,
        col_group=COL_GROUP,
        col_event=COL_EVENT,
        value_columns={
            "glucose": COL_GLU, "basal": COL_BASAL,
            "bolus": COL_BOLUS, "carbs": COL_CARB,
        },
        ts_format=TS_FORMAT,
        utf8_value_columns=("basal", "bolus", "carbs"),
        log_fn=typer.echo,
    )


def apply_split_scheme(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    split_scheme: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return _common_apply_split_scheme(
        train_df, val_df, test_df, split_scheme,
        log_fn=typer.echo,
        applied_message="Applied split scheme: train <- train+val | val <- test | test disabled.",
        note_message="Note: tuning-only mode; no held-out test metrics.",
        error_repr=True,
    )


def impute_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    return _common_impute_and_sort(
        df,
        ffill_bfill_columns=["glucose", "basal"],
        zero_fill_columns=["bolus", "carbs"],
    )


# ============================================================================
#  SLIDING-WINDOW DATASET
# ============================================================================

class SugarJepaWindowDataset(Dataset):
    """Lazy sliding-window dataset for SugarJepa.

    Each sample provides two views ending at the same point in time ("now"):
      x:            (input_steps, 4) — [glucose, basal, bolus, carbs], MinMax-scaled.
      glucose_jepa: (jepa_window,)   — glucose only, z-score normalized, its
                                        OWN (typically longer) lookback.
      y:            (horizon,)       — future glucose, MinMax-scaled (same
                                        scale as x's glucose channel).

    `lookback = max(input_steps, jepa_window)` so every window has enough
    history for both views; whichever view is shorter is the trailing suffix
    of the full lookback region.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        input_steps: int,
        horizon: int,
        jepa_window: int,
        scaler_glucose: MinMaxScaler | None = None,
        scaler_basal: MinMaxScaler | None = None,
        scaler_bolus: MinMaxScaler | None = None,
        scaler_carbs: MinMaxScaler | None = None,
        scaler_glucose_jepa: StandardScaler | None = None,
        fit_scalers: bool = False,
    ):
        self.input_steps = input_steps
        self.horizon = horizon
        self.jepa_window = jepa_window
        self.lookback = max(input_steps, jepa_window)
        window_len = self.lookback + horizon

        raw_glucose: list[np.ndarray] = []
        raw_basal: list[np.ndarray] = []
        raw_bolus: list[np.ndarray] = []
        raw_carbs: list[np.ndarray] = []
        uids: list = []
        sgroups: list[str] = []

        for (uid_val,), grp in (
            df.sort(["unique_id", "ds"])
            .group_by(["unique_id"], maintain_order=True)
        ):
            uids.append(uid_val)
            sgroups.append(grp["study_group"][0])
            raw_glucose.append(grp["glucose"].to_numpy())
            raw_basal.append(grp["basal"].to_numpy())
            raw_bolus.append(grp["bolus"].to_numpy())
            raw_carbs.append(grp["carbs"].to_numpy())

        if fit_scalers or scaler_glucose is None:
            all_g = np.concatenate(raw_glucose).reshape(-1, 1)
            all_b = np.concatenate(raw_basal).reshape(-1, 1)
            all_bo = np.concatenate(raw_bolus).reshape(-1, 1)
            all_c = np.concatenate(raw_carbs).reshape(-1, 1)
            self.scaler_glucose = MinMaxScaler().fit(all_g)
            self.scaler_basal = MinMaxScaler().fit(all_b)
            self.scaler_bolus = MinMaxScaler().fit(all_bo)
            self.scaler_carbs = MinMaxScaler().fit(all_c)
            # CGM-JEPA was pretrained on z-score normalized glucose, not
            # MinMax [0,1] — a separate scaler keeps the JEPA branch's input
            # distribution close to what the pretrained encoder expects.
            self.scaler_glucose_jepa = StandardScaler().fit(all_g)
        else:
            self.scaler_glucose = scaler_glucose
            self.scaler_basal = scaler_basal
            self.scaler_bolus = scaler_bolus
            self.scaler_carbs = scaler_carbs
            self.scaler_glucose_jepa = scaler_glucose_jepa

        self._series_g: list[np.ndarray] = []
        self._series_g_jepa: list[np.ndarray] = []
        self._series_b: list[np.ndarray] = []
        self._series_bo: list[np.ndarray] = []
        self._series_c: list[np.ndarray] = []
        self._index: list[tuple[int, int]] = []
        self.series_ids: list = []
        self.study_groups: list[str] = []

        n_skipped = 0
        for i, (uid, sg, rg, rb, rbo, rc) in enumerate(
            zip(uids, sgroups, raw_glucose, raw_basal, raw_bolus, raw_carbs)
        ):
            g = self.scaler_glucose.transform(rg.reshape(-1, 1)).ravel().astype(np.float32)
            g_jepa = self.scaler_glucose_jepa.transform(rg.reshape(-1, 1)).ravel().astype(np.float32)
            b = self.scaler_basal.transform(rb.reshape(-1, 1)).ravel().astype(np.float32)
            bo = self.scaler_bolus.transform(rbo.reshape(-1, 1)).ravel().astype(np.float32)
            c = self.scaler_carbs.transform(rc.reshape(-1, 1)).ravel().astype(np.float32)
            self._series_g.append(g)
            self._series_g_jepa.append(g_jepa)
            self._series_b.append(b)
            self._series_bo.append(bo)
            self._series_c.append(c)
            n_windows = len(g) - window_len + 1
            if n_windows <= 0:
                n_skipped += 1
                continue
            for start in range(n_windows):
                self._index.append((i, start))
                self.series_ids.append(uid)
                self.study_groups.append(sg)

        if n_skipped > 0:
            typer.echo(
                f"  Note: Skipped {n_skipped} series shorter than {window_len} steps "
                f"(lookback={self.lookback} = max(input_steps={input_steps}, jepa_window={jepa_window})."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        si, start = self._index[idx]
        g = self._series_g[si]
        g_jepa = self._series_g_jepa[si]
        b = self._series_b[si]
        bo = self._series_bo[si]
        c = self._series_c[si]

        now = start + self.lookback  # index of the first forecast step
        x_start = now - self.input_steps
        jepa_start = now - self.jepa_window

        x = np.stack(
            [g[x_start:now], b[x_start:now], bo[x_start:now], c[x_start:now]], axis=-1
        )  # (input_steps, 4)
        jepa = g_jepa[jepa_start:now]  # (jepa_window,)
        y = g[now : now + self.horizon]  # (horizon,)
        return torch.from_numpy(x), torch.from_numpy(jepa), torch.from_numpy(y)


# ============================================================================
#  TRAINING & EVALUATION
# ============================================================================

def train_one_epoch(
    model: SugarJepaModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
    batch_log_every: int = 0,
    epoch: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_batches_total = len(loader)
    t_epoch = time.perf_counter()

    for x, jepa, y in loader:
        x, jepa, y = x.to(device), jepa.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x, jepa)
            loss = loss_fn(pred, y)

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

        n_batches += 1
        total_loss += loss.item()

        if batch_log_every > 0 and (
            n_batches == 1
            or n_batches % batch_log_every == 0
            or n_batches == n_batches_total
        ):
            elapsed = time.perf_counter() - t_epoch
            batches_per_sec = n_batches / elapsed if elapsed > 0 else 0.0
            remaining = n_batches_total - n_batches
            eta_s = remaining / batches_per_sec if batches_per_sec > 0 else 0.0
            avg_loss = total_loss / n_batches
            echo_plain(
                f"  train epoch {epoch:3d} | batch {n_batches:,}/{n_batches_total:,} | "
                f"{batches_per_sec:.2f} batch/s | loss={avg_loss:.6f} | "
                f"epoch ETA {timedelta(seconds=int(eta_s))}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: SugarJepaModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    batch_log_every: int = 0,
    split_label: str = "eval",
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred = [], []
    n_batches_total = len(loader)
    t_eval = time.perf_counter()

    for x, jepa, y in loader:
        x, jepa, y = x.to(device), jepa.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(x, jepa)
            total_loss += loss_fn(pred, y).item()
        n_batches += 1
        all_true.append(y.float().cpu().numpy())
        all_pred.append(pred.float().cpu().numpy())

        if batch_log_every > 0 and (
            n_batches == 1
            or n_batches % batch_log_every == 0
            or n_batches == n_batches_total
        ):
            elapsed = time.perf_counter() - t_eval
            batches_per_sec = n_batches / elapsed if elapsed > 0 else 0.0
            remaining = n_batches_total - n_batches
            eta_s = remaining / batches_per_sec if batches_per_sec > 0 else 0.0
            echo_plain(
                f"  {split_label} | batch {n_batches:,}/{n_batches_total:,} | "
                f"{batches_per_sec:.2f} batch/s | ETA {timedelta(seconds=int(eta_s))}"
            )

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
    dataset: SugarJepaWindowDataset | None = None,
) -> tuple[float, float, float]:
    t_inv = scaler_glucose.inverse_transform(true_arr.ravel().reshape(-1, 1)).ravel()
    p_inv = scaler_glucose.inverse_transform(pred_arr.ravel().reshape(-1, 1)).ravel()
    mae, rmse, mard = mae_rmse_mard(t_inv, p_inv)

    typer.echo(f"\n=== {split_name.upper()} METRICS (overall, mg/dL) ===")
    typer.echo(f"  MAE : {mae:.4f}")
    typer.echo(f"  RMSE: {rmse:.4f}")
    typer.echo(f"  MARD: {mard:.2f}%")

    pl.DataFrame({"mae": [mae], "rmse": [rmse], "mard": [mard]}).write_csv(
        run_dir / f"{split_name}_metrics_overall.csv"
    )

    if dataset is not None and len(dataset.study_groups) == len(true_arr):
        groups_arr = np.array(dataset.study_groups)
        rows = []
        for g in sorted(set(groups_arr)):
            mask = groups_arr == g
            if not mask.any():
                continue
            tg = scaler_glucose.inverse_transform(true_arr[mask].ravel().reshape(-1, 1)).ravel()
            pg = scaler_glucose.inverse_transform(pred_arr[mask].ravel().reshape(-1, 1)).ravel()
            m, r, md = mae_rmse_mard(tg, pg)
            rows.append({"study_group": g, "n_windows": int(mask.sum()),
                         "mae": m, "rmse": r, "mard": md})
        by_group = pl.DataFrame(rows).sort("mae")
        typer.echo(f"\n=== {split_name.upper()} METRICS (by Study Group) ===")
        for row in by_group.iter_rows(named=True):
            echo_plain(
                f"  {row['study_group']}: n={row['n_windows']} "
                f"mae={row['mae']:.4f} rmse={row['rmse']:.4f} mard={row['mard']:.2f}%"
            )
        by_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")

    return mae, rmse, mard


def save_full_checkpoint(
    path: Path,
    model: SugarJepaModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    cfg: dict,
    *,
    wait: int = 0,
    best_epoch: int = 0,
) -> None:
    _common_save_full_checkpoint(
        path, model, optimizer, scheduler, epoch, best_val_loss, cfg,
        config_key="config", wait=wait, best_epoch=best_epoch, atomic=True,
    )


def load_full_checkpoint(
    path: Path,
    model: SugarJepaModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> tuple[int, float, int, int]:
    return _common_load_full_checkpoint(
        path, model, optimizer, scheduler, device,
        return_wait_and_best_epoch=True, log_fn=echo_plain,
    )


def update_latest_symlink(run_dir: Path, out_dir: Path) -> None:
    _common_update_latest_symlink(run_dir, out_dir, log_fn=typer.echo)


def make_model(cfg: dict, device: torch.device) -> SugarJepaModel:
    model = SugarJepaModel(
        n_time_steps=cfg["input_steps"],
        n_features=N_FEATURES,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_units=cfg["ff_units"],
        n_blocks=cfg["n_blocks"],
        prediction_horizon=cfg["horizon"],
        dropout=cfg["dropout"],
        jepa_weights_dir=cfg["jepa_weights_dir"],
        jepa_patch_size=cfg["jepa_patch_size"],
        jepa_freeze=not cfg["finetune_jepa"],
    ).to(device)
    if device.type == "cuda" and cfg["compile_mode"] != "none":
        model = torch.compile(model, mode=cfg["compile_mode"])
        typer.echo(f"torch.compile enabled (mode={cfg['compile_mode']})")
    return model


def make_optimizer_and_scheduler(
    model: SugarJepaModel,
    cfg: dict,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    if cfg["finetune_jepa"]:
        jepa_param_ids = {id(p) for p in model.jepa_encoder.encoder.parameters()}
        jepa_params = [p for p in model.parameters() if id(p) in jepa_param_ids]
        other_params = [p for p in model.parameters() if id(p) not in jepa_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": cfg["lr"]},
                {"params": jepa_params, "lr": cfg["jepa_lr"]},
            ],
            weight_decay=cfg["weight_decay"],
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )
    return optimizer, scheduler


def build_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    input_steps: int,
    horizon: int,
    jepa_window: int,
) -> tuple[
    SugarJepaWindowDataset,
    SugarJepaWindowDataset | None,
    SugarJepaWindowDataset | None,
]:
    train_ds = SugarJepaWindowDataset(
        train_df, input_steps, horizon, jepa_window, fit_scalers=True,
    )
    common_scalers = dict(
        scaler_glucose=train_ds.scaler_glucose,
        scaler_basal=train_ds.scaler_basal,
        scaler_bolus=train_ds.scaler_bolus,
        scaler_carbs=train_ds.scaler_carbs,
        scaler_glucose_jepa=train_ds.scaler_glucose_jepa,
    )
    val_ds = (
        SugarJepaWindowDataset(val_df, input_steps, horizon, jepa_window, **common_scalers)
        if not val_df.is_empty()
        else None
    )
    test_ds = (
        SugarJepaWindowDataset(test_df, input_steps, horizon, jepa_window, **common_scalers)
        if not test_df.is_empty()
        else None
    )
    return train_ds, val_ds, test_ds


def train_loop(
    model: SugarJepaModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    run_dir: Path,
    cfg: dict,
    verbose_every: int = 10,
    start_epoch: int = 1,
    best_val_loss: float = float("inf"),
    start_wait: int = 0,
    start_best_epoch: int = 0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    scaler: torch.amp.GradScaler | None = None,
    val_every_n_epochs: int = 1,
    batch_log_every: int = 0,
    eval_batch_log_every: int = 0,
) -> SugarJepaModel:
    wait = start_wait
    best_epoch = start_best_epoch if start_best_epoch > 0 else max(0, start_epoch - 1)
    last_completed_epoch = max(0, start_epoch - 1)
    total_time = 0.0
    train_batches = len(train_loader)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        if epoch == start_epoch and batch_log_every > 0:
            echo_plain(
                f"  Epoch {epoch}/{epochs}: {train_batches:,} train batches "
                f"(batch_size={train_loader.batch_size})"
            )
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
            batch_log_every=batch_log_every, epoch=epoch,
        )

        val_loss_str = "SKIP"
        should_eval = (
            val_loader is not None
            and (epoch == start_epoch or epoch % val_every_n_epochs == 0 or epoch == epochs)
        )
        if should_eval:
            val_loss, _, _ = evaluate(
                model, val_loader, loss_fn, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
                batch_log_every=eval_batch_log_every, split_label="val",
            )
            val_loss_str = f"{val_loss:.6f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                with open(run_dir / "best_info.json", "w") as f:
                    json.dump({"epoch": epoch, "val_loss": best_val_loss}, f)
                echo_plain(f"  New best at epoch {epoch} (val_loss={val_loss:.6f})")
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    typer.echo(f"  Early stopping at epoch {epoch} (patience={patience})")
                    break
        elif val_loader is None:
            val_loss_str = "N/A"
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
        eta_str = str(timedelta(seconds=int(avg_time * (epochs - epoch))))

        if epoch == 1 or epoch % verbose_every == 0 or epoch == epochs:
            typer.echo(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss_str} | "
                f"{dt:.1f}s/epoch | ETA: {eta_str}"
            )

        save_full_checkpoint(
            run_dir / "last_checkpoint.pt", model, optimizer, scheduler,
            epoch, best_val_loss, cfg, wait=wait, best_epoch=best_epoch,
        )
        last_completed_epoch = epoch

    save_full_checkpoint(
        run_dir / "last_checkpoint.pt", model, optimizer, scheduler,
        last_completed_epoch, best_val_loss, cfg, wait=wait, best_epoch=best_epoch,
    )
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    echo_plain(
        f"\n  Summary: best_model.pt = epoch {best_epoch} | "
        f"last_checkpoint.pt = epoch {last_completed_epoch}"
    )

    best_path = run_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model


def run_train_and_eval(
    model: SugarJepaModel,
    train_ds: SugarJepaWindowDataset,
    val_ds: SugarJepaWindowDataset | None,
    test_ds: SugarJepaWindowDataset | None,
    cfg: dict,
    device: torch.device,
    run_name: str,
    out_dir: Path,
) -> SugarJepaModel:
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Run directory: {run_dir}")

    meta = dict(cfg)
    meta.update({
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "test_samples": len(test_ds) if test_ds else 0,
        "start_time": datetime.now().isoformat(),
    })
    with open(run_dir / "tuning_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    update_latest_symlink(run_dir, out_dir)

    num_workers = resolve_num_workers(cfg["num_workers"], device)
    pin_memory = device.type == "cuda"
    loader_kwargs: dict = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg["prefetch_factor"]

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    echo_plain(
        f"  DataLoader: train_batches/epoch={len(train_loader):,} | "
        f"val_batches={len(val_loader) if val_loader else 0:,} | "
        f"test_batches={len(test_loader) if test_loader else 0:,} | "
        f"batch_size={cfg['batch_size']} | num_workers={num_workers}"
    )

    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg)
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda" and cfg["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and cfg["precision"] == "fp16")
    )

    start_epoch = 1
    best_val_loss = float("inf")
    start_wait = 0
    start_best_epoch = 0
    if cfg.get("resume_from"):
        resume_path = Path(cfg["resume_from"])
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        last_done, best_val_loss, start_wait, start_best_epoch = load_full_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch = last_done + 1
        if start_epoch > cfg["epochs"]:
            echo_plain(
                f"  Checkpoint already at epoch {last_done} >= max {cfg['epochs']}; "
                "skipping training loop."
            )
            start_epoch = cfg["epochs"] + 1

    typer.echo(f"\n{'=' * 60}")
    typer.echo(
        f"Training: {len(train_ds):,} windows | "
        f"Val: {len(val_ds) if val_ds else 0:,} | "
        f"Test: {len(test_ds) if test_ds else 0:,} | "
        f"Params: {sum(p.numel() for p in model.parameters()):,} "
        f"(trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,})"
    )
    typer.echo(f"{'=' * 60}")

    batch_log_every = int(cfg.get("batch_log_every", 0))
    eval_batch_log_every = int(cfg.get("eval_batch_log_every", 0))

    model = train_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        loss_fn, device, cfg["epochs"], cfg["patience"], run_dir, cfg,
        verbose_every=cfg["log_every"],
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        start_wait=start_wait,
        start_best_epoch=start_best_epoch,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=grad_scaler,
        val_every_n_epochs=cfg["val_every_n_epochs"],
        batch_log_every=batch_log_every,
        eval_batch_log_every=eval_batch_log_every,
    )

    if val_loader is not None:
        _, vt, vp = evaluate(
            model, val_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
            batch_log_every=eval_batch_log_every, split_label="val",
        )
        compute_and_print_metrics(vt, vp, train_ds.scaler_glucose, "val", run_dir, val_ds)

    if test_loader is not None:
        _, tt, tp = evaluate(
            model, test_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
            batch_log_every=eval_batch_log_every, split_label="test",
        )
        compute_and_print_metrics(tt, tp, train_ds.scaler_glucose, "test", run_dir, test_ds)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return model


def _mode_global(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> None:
    typer.echo("\n=== MODE: GLOBAL ===")
    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df, cfg["input_steps"], cfg["horizon"], cfg["jepa_window"]
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sugar_jepa_global_h{cfg['horizon']}_{ts}"

    model = make_model(cfg, device)
    run_train_and_eval(model, train_ds, val_ds, test_ds, cfg, device, run_name, out_dir)


# ============================================================================
#  CLI  (single root command — run: python train_sugar_jepa.py --csv ...)
# ============================================================================

@app.command()
def main(
    csv: Path = typer.Option(..., help="Path to loop_ai_ready_joined2*.csv."),
    unique_id: str = typer.Option("sequence_id", help="sequence_id or user_id."),
    max_train_series: int = typer.Option(0, help="Limit training series (0 = all)."),
    max_eval_series: int = typer.Option(0, help="Limit evaluation series (0 = all)."),
    drop_interpolated: bool = typer.Option(False, help="Drop Interpolated rows."),
    study_groups: str = typer.Option("", help="Comma-separated Study Group filter (empty = all)."),
    split_scheme: str = typer.Option("classic", help="classic | trainval_test_as_val."),
    horizon: int = typer.Option(12, help="Prediction horizon steps (12 = 60 min at 5-min freq)."),
    input_steps: int = typer.Option(128, help="SugarOne backbone input window steps."),
    d_model: int = typer.Option(32, help="Embedding dimension."),
    n_heads: int = typer.Option(8, help="Attention heads."),
    n_blocks: int = typer.Option(5, help="Parallel transformer blocks."),
    ff_units: int = typer.Option(128, help="FFN hidden units."),
    dropout: float = typer.Option(0.1, help="Dropout rate."),
    epochs: int = typer.Option(30, help="Training epochs."),
    batch_size: int = typer.Option(256, help="Batch size."),
    precision: str = typer.Option("bf16", help="fp32 | bf16 | fp16."),
    compile_mode: str = typer.Option("none", help="none | default | reduce-overhead | max-autotune."),
    disable_tf32: bool = typer.Option(False, help="Disable TF32 on CUDA."),
    num_workers: int = typer.Option(0, help="DataLoader workers (-1 = auto; 0 avoids Windows worker-spawn stalls)."),
    prefetch_factor: int = typer.Option(4, help="DataLoader prefetch factor."),
    lr: float = typer.Option(4e-4, help="Learning rate."),
    weight_decay: float = typer.Option(3e-5, help="Weight decay."),
    patience: int = typer.Option(3, help="Early stopping patience (0 = disabled)."),
    log_every: int = typer.Option(1, help="Print every N epochs."),
    val_every_n_epochs: int = typer.Option(5, help="Run validation every N epochs."),
    resume_from: str = typer.Option("", help="Path to checkpoint.pt to resume from."),
    batch_log_every: int = typer.Option(200, help="Log train progress every N batches (0 = off)."),
    eval_batch_log_every: int = typer.Option(300, help="Log eval progress every N batches (0 = off)."),
    jepa_weights_dir: str = typer.Option(
        "scripts/sugar_jepa/pretrained/cgm_jepa",
        help="Local dir with the pretrained CGM-JEPA encoder's config.json + model.safetensors. "
        "See vendor/cgm_jepa/NOTICE.md for why this is a local path rather than a HF Hub repo id "
        "on this machine, and how to fetch from the Hub instead on a machine without that issue.",
    ),
    jepa_window: int = typer.Option(288, help="Glucose-only lookback fed to the JEPA encoder (288 = 24h)."),
    jepa_patch_size: int = typer.Option(12, help="Raw steps per JEPA patch (must match the checkpoint, 12)."),
    finetune_jepa: bool = typer.Option(False, help="Unfreeze the JEPA encoder (else frozen, feature-extractor mode)."),
    jepa_lr: float = typer.Option(4e-5, help="LR for JEPA encoder params, only used if --finetune-jepa."),
    device_name: str = typer.Option("cuda", "--device", help="cpu | mps | cuda."),
    seed: int = typer.Option(42, help="Random seed."),
    out_dir: Path = typer.Option(Path("data/output/runs/sugar_jepa"), help="Output directory."),
) -> None:
    """Train SugarJepa (global mode only) on insulin + carb + JEPA-embedding covariate data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device_name == "mps" and not torch.backends.mps.is_available():
        typer.echo("MPS not available, falling back to CPU.")
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    if device.type == "cuda":
        if not disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            typer.echo("TF32 enabled.")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    typer.echo(f"Device: {device}")

    train_df, val_df, test_df = load_splits_streaming(csv, unique_id, drop_interpolated)
    typer.echo(f"Loaded: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    train_df = normalize_study_groups_column(train_df)
    val_df = normalize_study_groups_column(val_df)
    test_df = normalize_study_groups_column(test_df)

    if study_groups:
        group_list = [
            normalize_study_group_label(g.strip())
            for g in study_groups.split(",") if g.strip()
        ]
        train_df = train_df.filter(pl.col("study_group").is_in(group_list))
        val_df = val_df.filter(pl.col("study_group").is_in(group_list))
        test_df = test_df.filter(pl.col("study_group").is_in(group_list))
        typer.echo(f"Filtered to {group_list}: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    train_df, val_df, test_df = apply_split_scheme(train_df, val_df, test_df, split_scheme)

    if max_train_series > 0:
        train_df = limit_series(train_df, max_train_series)
    if max_eval_series > 0:
        val_df = limit_series(val_df, max_eval_series)
        test_df = limit_series(test_df, max_eval_series)

    train_df = impute_and_sort(train_df)
    val_df = impute_and_sort(val_df)
    test_df = impute_and_sort(test_df)

    typer.echo(f"After limits: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    typer.echo(f"Study groups in train: {sorted(train_df['study_group'].unique().to_list())}")

    cfg = {
        "csv": str(csv), "unique_id": unique_id, "drop_interpolated": drop_interpolated,
        "study_groups": study_groups, "split_scheme": split_scheme, "mode": "global",
        "horizon": horizon, "input_steps": input_steps,
        "d_model": d_model, "n_heads": n_heads, "n_blocks": n_blocks,
        "ff_units": ff_units, "dropout": dropout,
        "epochs": epochs, "batch_size": batch_size, "precision": precision,
        "compile_mode": compile_mode, "disable_tf32": disable_tf32,
        "num_workers": num_workers, "prefetch_factor": prefetch_factor,
        "lr": lr, "weight_decay": weight_decay, "patience": patience,
        "log_every": log_every, "val_every_n_epochs": val_every_n_epochs,
        "resume_from": resume_from,
        "batch_log_every": batch_log_every, "eval_batch_log_every": eval_batch_log_every,
        "jepa_weights_dir": jepa_weights_dir, "jepa_window": jepa_window,
        "jepa_patch_size": jepa_patch_size, "finetune_jepa": finetune_jepa, "jepa_lr": jepa_lr,
        "device": device_name, "seed": seed, "out_dir": str(out_dir),
    }

    _mode_global(train_df, val_df, test_df, cfg, device, out_dir)
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
