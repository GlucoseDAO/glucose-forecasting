"""Tests for Milestone 8 personalization package."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch
from typer.testing import CliRunner

from scripts.personalization.constants import LOOP_HOLDOUT_QUALITY_USERS
from scripts.personalization.finetune import run_finetune
from scripts.personalization.prepare_personal_csv import app as prepare_app
from scripts.personalization.registry import (
    detect_model_type,
    get_model_spec,
    list_model_types,
    load_base_checkpoint,
    register_model,
)
from scripts.personalization.splits import chronological_split_labels, limit_train_days, split_meta
from scripts.personalization.sweep_utils import (
    estimate_plateau_day,
    lr_grid_from_base,
    pick_best_row,
    weight_decay_grid,
    write_summary,
)
from scripts.sugar_one.sugar_one_model import SugarOneModel
from tests.conftest import (
    TINY_D_MODEL,
    TINY_FF_UNITS,
    TINY_HORIZON,
    TINY_INPUT_STEPS,
    TINY_N_BLOCKS,
    TINY_N_HEADS,
)

runner = CliRunner()


def _write_continuous_person_csv(path: Path, *, n_rows: int = 400, user_id: str = "Subject000") -> None:
    start = datetime(2024, 1, 1, 0, 0, 0)
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        ts = start + timedelta(minutes=5 * i)
        rows.append(
            {
                "sequence_id": f"seq_{user_id}",
                "Timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "Event Type": "EGV",
                "User ID": user_id,
                "Glucose (mg/dL)": 100.0 + (i % 50) * 0.5,
                "Basal Rate (U/h)": "1.0",
                "Bolus Insulin (U)": "2.0" if i % 20 == 0 else "",
                "Carbohydrates (g)": "15.0" if i % 30 == 0 else "",
            }
        )
    pl.DataFrame(rows).write_csv(path)


def _make_tiny_base_run(tmp_path: Path, *, lr: float = 4e-4, patience: int = 10) -> Path:
    run_dir = tmp_path / "base_sugar_one"
    run_dir.mkdir(parents=True)
    meta = {
        "model_type": "sugar_one",
        "input_steps": TINY_INPUT_STEPS,
        "horizon": TINY_HORIZON,
        "d_model": TINY_D_MODEL,
        "n_heads": TINY_N_HEADS,
        "ff_units": TINY_FF_UNITS,
        "n_blocks": TINY_N_BLOCKS,
        "dropout": 0.0,
        "lr": lr,
        "patience": patience,
        "weight_decay": 3e-5,
    }
    with (run_dir / "tuning_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    model = SugarOneModel(
        n_time_steps=TINY_INPUT_STEPS,
        n_features=4,
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        ff_units=TINY_FF_UNITS,
        n_blocks=TINY_N_BLOCKS,
        prediction_horizon=TINY_HORIZON,
        dropout=0.0,
    )
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    return run_dir


def test_registry_lists_sugar_one() -> None:
    assert "sugar_one" in list_model_types()
    spec = get_model_spec("sugar_one")
    assert spec.n_features == 4
    assert "basal" in spec.value_columns


def test_lr_grid_from_base(tmp_path: Path) -> None:
    base = _make_tiny_base_run(tmp_path, lr=0.0004)
    grid = lr_grid_from_base(base, multipliers=(0.5, 1.0, 2.0))
    assert grid == [0.0002, 0.0004, 0.0008]


def test_weight_decay_grid() -> None:
    grid = weight_decay_grid((0.5, 1.0, 2.0))
    assert grid == [1.5e-5, 3e-5, 6e-5]


def test_chronological_split_and_day_limit() -> None:
    start = datetime(2024, 1, 1)
    rows = [
        {
            "Timestamp": start + timedelta(minutes=5 * i),
            "User ID": "u1",
            "Glucose (mg/dL)": 100.0,
        }
        for i in range(1000)
    ]
    df = pl.DataFrame(rows)
    labeled = chronological_split_labels(df, test_fraction=0.25, val_fraction_of_remainder=0.2)
    limited = limit_train_days(labeled, personal_days=1)
    train = limited.filter(pl.col("Recommended Split") == "train")
    assert train.height < labeled.filter(pl.col("Recommended Split") == "train").height


def test_prepare_livia_cli(tmp_path: Path) -> None:
    raw = tmp_path / "raw_person.csv"
    _write_continuous_person_csv(raw, n_rows=200)
    out_dir = tmp_path / "prepared"
    result = runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(out_dir), "--out-name", "person.csv"],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "person.csv").exists()


def test_finetune_smoke(tmp_path: Path) -> None:
    base = _make_tiny_base_run(tmp_path)
    raw = tmp_path / "raw.csv"
    _write_continuous_person_csv(raw, n_rows=300)
    prepared_dir = tmp_path / "prepared"
    prep = runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(prepared_dir), "--out-name", "p.csv"],
    )
    assert prep.exit_code == 0, prep.output

    run_dir, results = run_finetune(
        base_run_dir=base,
        personal_csv=prepared_dir / "p.csv",
        out_dir=tmp_path / "ft_runs",
        run_name="smoke_ft",
        personal_days=2,
        lwf_lambda=0.5,
        epochs=1,
        patience=0,
        batch_size=8,
        device="cpu",
        num_workers=0,
        eval_zero_shot=True,
    )
    assert (run_dir / "tuning_meta.json").exists()
    assert results.get("finetuned_test") is not None
    assert results["config"]["lwf_lambda"] == 0.5


def test_finetune_lwf_zero_smoke(tmp_path: Path) -> None:
    base = _make_tiny_base_run(tmp_path)
    raw = tmp_path / "raw.csv"
    _write_continuous_person_csv(raw, n_rows=300)
    prepared_dir = tmp_path / "prepared"
    runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(prepared_dir), "--out-name", "p.csv"],
    )
    run_dir, results = run_finetune(
        base_run_dir=base,
        personal_csv=prepared_dir / "p.csv",
        out_dir=tmp_path / "ft_no_lwf",
        run_name="no_lwf",
        lwf_lambda=0.0,
        epochs=1,
        patience=0,
        batch_size=8,
        device="cpu",
        num_workers=0,
        eval_zero_shot=False,
    )
    assert (run_dir / "personalization_metrics.json").exists()
    assert results["config"]["lwf_lambda"] == 0.0


def test_plateau_estimation() -> None:
    rows = [
        {"status": "ok", "personal_days": 1, "ft_test_mae": 15.0},
        {"status": "ok", "personal_days": 7, "ft_test_mae": 11.0},
        {"status": "ok", "personal_days": 14, "ft_test_mae": 10.6},
        {"status": "ok", "personal_days": 30, "ft_test_mae": 10.55},
    ]
    info = estimate_plateau_day(rows)
    assert info["optimal_day"] in (14, 30)
    assert info["plateau_day"] is not None


def test_sweep_utils_pick_best(tmp_path: Path) -> None:
    rows = [
        {"ft_test_mae": 12.0, "lwf_lambda": 0.0},
        {"ft_test_mae": 10.5, "lwf_lambda": 0.5},
    ]
    best = pick_best_row(rows)
    assert best is not None
    assert best["lwf_lambda"] == 0.5
    path = write_summary(rows, tmp_path / "sum")
    assert path.exists()


def test_aggregate_cli(tmp_path: Path) -> None:
    from scripts.personalization.aggregate_results import app as agg_app

    root = tmp_path / "runs"
    sweep = root / "livia" / "sweeps" / "hyperparams"
    sweep.mkdir(parents=True)
    pl.DataFrame(
        [{"subject": "livia", "lwf_lambda": 0.5, "ft_test_mae": 11.0, "status": "ok"}]
    ).write_csv(sweep / "summary.csv")
    out_json = tmp_path / "summary.json"
    result = runner.invoke(agg_app, ["--root", str(root), "--out", str(out_json)])
    assert result.exit_code == 0, result.output
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["sections"]["hyperparams"]["n_rows"] == 1


def test_holdout_constants() -> None:
    assert len(LOOP_HOLDOUT_QUALITY_USERS) == 6
