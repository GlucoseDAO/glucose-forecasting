"""Unit tests for src/common/checkpoint.py gaps not covered by
tests/test_train_checkpoint_resume.py (which only exercises the SugarOne
config_key="config" shape).

Covers: strip_compile_prefix, update_latest_symlink, the GluMind shape
(config_key="args", stringify_paths=True), the GluMind-Uni shape
(config_key="cfg"), and atomic=True.
"""
from __future__ import annotations

from pathlib import Path

import torch

from common.checkpoint import (
    load_full_checkpoint,
    read_checkpoint_meta,
    save_full_checkpoint,
    strip_compile_prefix,
    update_latest_symlink,
)
from glumind_uni.glumind_uni_model import GluMindUniModel
from sugar_one.sugar_one_model import SugarOneModel


def _tiny_uni_model() -> GluMindUniModel:
    return GluMindUniModel(
        n_time_steps=8, d_model=8, n_heads=2, ff_units=16, n_blocks=1,
        prediction_horizon=2, dropout=0.0,
    )


# ---------------------------------------------------------------------------
# strip_compile_prefix
# ---------------------------------------------------------------------------


def test_strip_compile_prefix_strips_all_keys() -> None:
    state = {"_orig_mod.embed_glucose.weight": 1, "_orig_mod.out_fc.bias": 2}
    out = strip_compile_prefix(state)
    assert out == {"embed_glucose.weight": 1, "out_fc.bias": 2}


def test_strip_compile_prefix_noop_without_prefix() -> None:
    state = {"embed_glucose.weight": 1, "out_fc.bias": 2}
    out = strip_compile_prefix(state)
    assert out == state


# ---------------------------------------------------------------------------
# update_latest_symlink
# ---------------------------------------------------------------------------


def test_update_latest_symlink_writes_pointer_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    run_dir = out_dir / "run_0001"
    update_latest_symlink(run_dir, out_dir)
    pointer = out_dir / "latest.txt"
    assert pointer.exists()
    assert pointer.read_text(encoding="utf-8").strip() == str(run_dir)


# ---------------------------------------------------------------------------
# save/load_full_checkpoint — GluMind shape (config_key="args", stringify_paths)
# ---------------------------------------------------------------------------


def test_checkpoint_glumind_shape_args_key_stringify_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = _tiny_uni_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    cfg = {"out_dir": Path("data/output/runs/glumind"), "lr": 0.001}
    ckpt_path = run_dir / "checkpoint.pt"
    save_full_checkpoint(
        ckpt_path, model, optimizer, scheduler, epoch=3, best_val_loss=0.25,
        config_dict=cfg, config_key="args", stringify_paths=True,
    )

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "args" in raw
    assert "config" not in raw
    assert isinstance(raw["args"]["out_dir"], str)  # Path was stringified
    assert raw["args"]["out_dir"] == str(Path("data/output/runs/glumind"))

    model2 = _tiny_uni_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.001)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    epoch, best_val = load_full_checkpoint(ckpt_path, model2, opt2, sched2, torch.device("cpu"))
    assert epoch == 3
    assert best_val == 0.25


# ---------------------------------------------------------------------------
# save/load_full_checkpoint — GluMind-Uni shape (config_key="cfg")
# ---------------------------------------------------------------------------


def test_checkpoint_glumind_uni_shape_cfg_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = _tiny_uni_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    cfg = {"horizon": 2, "input_steps": 8}
    ckpt_path = run_dir / "checkpoint.pt"
    save_full_checkpoint(
        ckpt_path, model, optimizer, None, epoch=1, best_val_loss=1.0,
        config_dict=cfg, config_key="cfg",
    )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert raw["cfg"] == cfg
    assert raw["scheduler_state_dict"] is None
    assert "args" not in raw and "config" not in raw


# ---------------------------------------------------------------------------
# atomic=True
# ---------------------------------------------------------------------------


def test_checkpoint_atomic_write_produces_final_file_no_tmp_leftover(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = SugarOneModel(
        n_time_steps=8, d_model=8, n_heads=2, ff_units=16, n_blocks=1,
        prediction_horizon=2, dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ckpt_path = run_dir / "atomic_checkpoint.pt"

    save_full_checkpoint(
        ckpt_path, model, optimizer, None, epoch=5, best_val_loss=0.1,
        config_dict={"lr": 0.001}, config_key="config", wait=1, best_epoch=4,
        atomic=True,
    )
    assert ckpt_path.exists()
    assert not ckpt_path.with_suffix(ckpt_path.suffix + ".tmp").exists()

    meta = read_checkpoint_meta(ckpt_path)
    assert meta is not None
    assert meta["epoch"] == 5
    assert meta["best_epoch"] == 4
    assert meta["wait"] == 1


def test_read_checkpoint_meta_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_checkpoint_meta(tmp_path / "nope.pt") is None
