"""Checkpoint resume continues from last completed epoch, not epoch 1 or best epoch."""
from __future__ import annotations

from pathlib import Path

import torch

from sugar_one.sugar_one_model import SugarOneModel
from sugar_one.train_sugar_one import (
    load_full_checkpoint,
    read_checkpoint_meta,
    save_full_checkpoint,
)


def test_checkpoint_stores_wait_and_resumes_next_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = {"lr": 0.001, "epochs": 10}
    model = SugarOneModel(
        n_time_steps=8,
        d_model=16,
        n_heads=4,
        ff_units=32,
        n_blocks=1,
        prediction_horizon=2,
        dropout=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    ckpt_path = run_dir / "last_checkpoint.pt"
    save_full_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scheduler,
        epoch=7,
        best_val_loss=0.5,
        cfg=cfg,
        wait=2,
        best_epoch=5,
    )

    meta = read_checkpoint_meta(ckpt_path)
    assert meta is not None
    assert meta["epoch"] == 7
    assert meta["best_epoch"] == 5
    assert meta["wait"] == 2

    model2 = SugarOneModel(
        n_time_steps=8,
        d_model=16,
        n_heads=4,
        ff_units=32,
        n_blocks=1,
        prediction_horizon=2,
        dropout=0.1,
    )
    opt2 = torch.optim.AdamW(model2.parameters(), lr=0.001)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    last_done, best_val, wait, best_ep = load_full_checkpoint(
        ckpt_path, model2, opt2, sched2, torch.device("cpu")
    )
    assert last_done == 7
    assert best_val == 0.5
    assert wait == 2
    assert best_ep == 5
    assert last_done + 1 == 8
