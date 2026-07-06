"""Smoke tests for the SugarJepa proof-of-concept (scripts/sugar_jepa/).

Not full coverage (time-boxed POC per CLAUDE.md) — just enough to catch
shape regressions in the model, the sliding-window dataset, and the
checkpoint round-trip. Uses the vendored, locally-cached pretrained CGM-JEPA
encoder (scripts/sugar_jepa/pretrained/cgm_jepa/) — no network access needed.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from scripts.common.checkpoint import load_full_checkpoint, save_full_checkpoint
from scripts.sugar_jepa.sugar_jepa_model import SugarJepaModel
from scripts.sugar_jepa.train_sugar_jepa import SugarJepaWindowDataset

JEPA_WEIGHTS_DIR = "scripts/sugar_jepa/pretrained/cgm_jepa"
JEPA_PATCH_SIZE = 12

BATCH = 3
INPUT_STEPS = 8
JEPA_WINDOW = 24  # 2 patches — smallest sensible multiple of JEPA_PATCH_SIZE for a fast test
HORIZON = 2
D_MODEL = 8
N_HEADS = 2
N_BLOCKS = 1
FF_UNITS = 16


def _tiny_model(freeze: bool = True) -> SugarJepaModel:
    return SugarJepaModel(
        n_time_steps=INPUT_STEPS,
        n_features=4,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        ff_units=FF_UNITS,
        n_blocks=N_BLOCKS,
        prediction_horizon=HORIZON,
        dropout=0.0,
        jepa_weights_dir=JEPA_WEIGHTS_DIR,
        jepa_patch_size=JEPA_PATCH_SIZE,
        jepa_freeze=freeze,
    )


# ---------------------------------------------------------------------------
# Model forward / shapes
# ---------------------------------------------------------------------------


def test_sugar_jepa_forward_shape_and_dtype() -> None:
    model = _tiny_model()
    x = torch.randn(BATCH, INPUT_STEPS, 4)
    jepa = torch.randn(BATCH, JEPA_WINDOW)
    out = model(x, jepa)
    assert out.shape == (BATCH, HORIZON)
    assert out.dtype == torch.float32


def test_sugar_jepa_state_dict_key_patterns_stable() -> None:
    model = _tiny_model()
    keys = set(model.state_dict().keys())
    expected_prefixes = [
        "embed_glucose.",
        "embed_basal.",
        "embed_bolus.",
        "embed_carbs.",
        "jepa_encoder.encoder.",
        "jepa_encoder.proj.",
        "blocks.0.cross_attn.mix_logits",
        "blocks.0.cross_attn.attn_jepa.",
        "blocks.0.multiscale.",
        "flatten_fc.",
        "out_fc.",
    ]
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in keys), f"missing expected key prefix {prefix!r}"
    # 4-way mix (basal, bolus, carbs, jepa) — the one deliberate architectural
    # difference from SugarOneModel's 3-way mix_logits.
    assert model.blocks[0].cross_attn.mix_logits.shape == (4,)


def test_sugar_jepa_frozen_encoder_has_no_gradient() -> None:
    model = _tiny_model(freeze=True)
    x = torch.randn(BATCH, INPUT_STEPS, 4)
    jepa = torch.randn(BATCH, JEPA_WINDOW)
    model(x, jepa).sum().backward()
    for p in model.jepa_encoder.encoder.parameters():
        assert not p.requires_grad
        assert p.grad is None
    # Everything else (incl. the new jepa_encoder.proj) should have gradients.
    assert model.jepa_encoder.proj.weight.grad is not None
    assert model.embed_glucose.weight.grad is not None


def test_sugar_jepa_finetune_encoder_has_gradient() -> None:
    model = _tiny_model(freeze=False)
    x = torch.randn(BATCH, INPUT_STEPS, 4)
    jepa = torch.randn(BATCH, JEPA_WINDOW)
    model(x, jepa).sum().backward()
    assert any(p.grad is not None for p in model.jepa_encoder.encoder.parameters())


# ---------------------------------------------------------------------------
# SugarJepaWindowDataset
# ---------------------------------------------------------------------------


def _sugar_jepa_df(n_rows_per_series: dict[str, int]) -> pl.DataFrame:
    rows = []
    for uid, n in n_rows_per_series.items():
        for i in range(n):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": i,
                    "glucose": 100.0 + i,
                    "basal": 1.0,
                    "bolus": 2.0 if i % 5 == 0 else 0.0,
                    "carbs": 10.0 if i % 7 == 0 else 0.0,
                    "study_group": "T1DM",
                }
            )
    return pl.DataFrame(rows)


def test_sugar_jepa_window_dataset_window_count() -> None:
    input_steps, horizon, jepa_window = 4, 2, 8
    lookback = max(input_steps, jepa_window)
    window_len = lookback + horizon
    n_rows = 20
    df = _sugar_jepa_df({"a": n_rows})
    ds = SugarJepaWindowDataset(df, input_steps, horizon, jepa_window, fit_scalers=True)
    assert len(ds) == n_rows - window_len + 1


def test_sugar_jepa_window_dataset_skips_short_series() -> None:
    input_steps, horizon, jepa_window = 4, 2, 16
    lookback = max(input_steps, jepa_window)
    window_len = lookback + horizon
    df = _sugar_jepa_df({"short": 5, "long": 30})
    ds = SugarJepaWindowDataset(df, input_steps, horizon, jepa_window, fit_scalers=True)
    assert len(ds) == 30 - window_len + 1
    assert "short" not in ds.series_ids


def test_sugar_jepa_window_dataset_getitem_shapes() -> None:
    input_steps, horizon, jepa_window = 4, 2, 12
    df = _sugar_jepa_df({"a": 20})
    ds = SugarJepaWindowDataset(df, input_steps, horizon, jepa_window, fit_scalers=True)
    x, jepa, y = ds[0]
    assert x.shape == (input_steps, 4)
    assert jepa.shape == (jepa_window,)
    assert y.shape == (horizon,)
    # Main branch is MinMax-scaled into [0, 1]; JEPA branch is z-scored (unbounded).
    assert x.min() >= 0.0 - 1e-6
    assert x.max() <= 1.0 + 1e-6


def test_sugar_jepa_window_dataset_jepa_branch_is_zscored_not_minmax() -> None:
    """The JEPA branch must NOT share the main branch's [0,1] MinMax scaling —
    it needs its own z-score normalization to match what the pretrained
    encoder was trained on (see NOTICE.md)."""
    input_steps, horizon, jepa_window = 4, 2, 12
    df = _sugar_jepa_df({"a": 20})
    ds = SugarJepaWindowDataset(df, input_steps, horizon, jepa_window, fit_scalers=True)
    assert ds.scaler_glucose is not ds.scaler_glucose_jepa
    _, jepa, _ = ds[0]
    # z-scored glucose (monotonically increasing raw values) should have
    # both negative and non-negative entries, unlike the [0,1] MinMax branch.
    assert jepa.min() < 0.0


def test_sugar_jepa_window_dataset_reuses_scalers_not_refit() -> None:
    input_steps, horizon, jepa_window = 4, 2, 8
    train_df = _sugar_jepa_df({"train": 20})
    val_df = _sugar_jepa_df({"val": 16})

    train_ds = SugarJepaWindowDataset(train_df, input_steps, horizon, jepa_window, fit_scalers=True)
    val_ds = SugarJepaWindowDataset(
        val_df, input_steps, horizon, jepa_window,
        scaler_glucose=train_ds.scaler_glucose,
        scaler_basal=train_ds.scaler_basal,
        scaler_bolus=train_ds.scaler_bolus,
        scaler_carbs=train_ds.scaler_carbs,
        scaler_glucose_jepa=train_ds.scaler_glucose_jepa,
        fit_scalers=False,
    )
    assert val_ds.scaler_glucose is train_ds.scaler_glucose
    assert val_ds.scaler_glucose_jepa is train_ds.scaler_glucose_jepa


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_sugar_jepa_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    ckpt_path = tmp_path / "checkpoint.pt"
    save_full_checkpoint(
        ckpt_path, model, optimizer, scheduler,
        epoch=3, best_val_loss=0.5, config_dict={"d_model": D_MODEL},
        config_key="config", wait=1, best_epoch=2, atomic=True,
    )
    assert ckpt_path.exists()

    reloaded = _tiny_model()
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    epoch, best_val_loss, wait, best_epoch = load_full_checkpoint(
        ckpt_path, reloaded, reloaded_optimizer, return_wait_and_best_epoch=True,
    )
    assert epoch == 3
    assert best_val_loss == pytest.approx(0.5)
    assert wait == 1
    assert best_epoch == 2
    for k, v in model.state_dict().items():
        assert torch.equal(v, reloaded.state_dict()[k])
