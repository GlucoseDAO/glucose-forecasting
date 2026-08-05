"""Tests for Milestone 8 personalization package."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch
from typer.testing import CliRunner

from scripts.personalization.constants import LOOP_HOLDOUT_QUALITY_USERS, SPARSE_WINDOW_STRIDE
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
from scripts.personalization.leaderboard import (
    STATUS_FAILED,
    STATUS_OK,
    build_run_combos,
    combo_hash,
    completed_hashes,
    grid_combo_hashes,
    import_existing_runs,
    write_leaderboard_csv,
)
from scripts.personalization.sweep_utils import (
    build_holdout_lr_comparison,
    estimate_plateau_day,
    holdout_run_complete,
    holdout_row_from_metrics,
    lr_grid_from_base,
    pick_best_row,
    weight_decay_grid,
    write_summary,
)
from scripts.sugar_one.train_sugar_one import SugarOneWindowDataset
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
    grid = weight_decay_grid((1.0,))
    assert grid == [3e-5]


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


def test_window_stride_reduces_train_windows(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    _write_continuous_person_csv(raw, n_rows=300)
    prepared_dir = tmp_path / "prepared"
    prep = runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(prepared_dir), "--out-name", "p.csv"],
    )
    assert prep.exit_code == 0, prep.output
    from scripts.personalization.finetune import _load_split_frames

    train_df, _, _ = _load_split_frames(prepared_dir / "p.csv")
    dense = SugarOneWindowDataset(train_df, TINY_INPUT_STEPS, TINY_HORIZON, fit_scalers=True)
    sparse = SugarOneWindowDataset(
        train_df,
        TINY_INPUT_STEPS,
        TINY_HORIZON,
        scaler_glucose=dense.scaler_glucose,
        scaler_basal=dense.scaler_basal,
        scaler_bolus=dense.scaler_bolus,
        scaler_carbs=dense.scaler_carbs,
        window_stride=SPARSE_WINDOW_STRIDE,
    )
    assert len(sparse) < len(dense)
    assert len(sparse) >= len(dense) // SPARSE_WINDOW_STRIDE - 1


def test_finetune_sparse_stride_smoke(tmp_path: Path) -> None:
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
        out_dir=tmp_path / "ft_sparse",
        run_name="sparse_ft",
        train_window_stride=SPARSE_WINDOW_STRIDE,
        lwf_lambda=0.0,
        epochs=1,
        patience=0,
        batch_size=8,
        device="cpu",
        num_workers=0,
        eval_zero_shot=False,
    )
    assert results["config"]["train_window_stride"] == SPARSE_WINDOW_STRIDE
    assert results["config"]["eval_window_stride"] == 1
    assert (run_dir / "personalization_metrics.json").exists()


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


def test_holdout_lr_comparison_vs_livia() -> None:
    rows = [
        {"status": "ok", "user_id": "154", "subject": "loop_154", "lr": 0.0001, "ft_test_mae": 14.0},
        {"status": "ok", "user_id": "154", "subject": "loop_154", "lr": 0.0002, "ft_test_mae": 13.5},
        {"status": "ok", "user_id": "154", "subject": "loop_154", "lr": 0.0004, "ft_test_mae": 13.8},
        {"status": "ok", "user_id": "556", "subject": "loop_556", "lr": 0.0001, "ft_test_mae": 12.0},
        {"status": "ok", "user_id": "556", "subject": "loop_556", "lr": 0.0002, "ft_test_mae": 12.2},
        {"status": "ok", "user_id": "556", "subject": "loop_556", "lr": 0.0004, "ft_test_mae": 11.5},
    ]
    comparison = build_holdout_lr_comparison(rows, livia_reference_lr=0.0002)
    assert len(comparison) == 2
    by_user = {c["user_id"]: c for c in comparison}
    assert by_user["154"]["optimal_lr"] == 0.0002
    assert by_user["154"]["divergence"] == "same"
    assert by_user["556"]["optimal_lr"] == 0.0004
    assert by_user["556"]["divergence"] == "higher"


def test_holdout_run_complete(tmp_path: Path) -> None:
    run_dir = tmp_path / "loop_154_lr0.0001"
    run_dir.mkdir()
    assert not holdout_run_complete(run_dir)

    metrics = {
        "config": {"lr": 0.0001},
        "zero_shot_test": {"mae": 20.0, "rmse": 30.0, "mard": 15.0},
        "finetuned_test": {"mae": 18.0, "rmse": 28.0, "mard": 14.0},
        "finetuned_val": {"mae": 17.5, "rmse": 27.0, "mard": 13.5},
    }
    (run_dir / "personalization_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    assert holdout_run_complete(run_dir)
    row = holdout_row_from_metrics(
        run_dir,
        user_id="154",
        subject="loop_154",
        lwf_lambda=0.0,
        weight_decay=3e-5,
        patience=3,
        epochs=30,
    )
    assert row is not None
    assert row["ft_test_mae"] == 18.0


def test_build_run_combos_grid() -> None:
    cfg = {
        "defaults": {"lwf_lambda": 0.3, "lr": 0.0004},
        "grid": {"lwf_lambda": [0.2, 0.3], "weight_decay": [1.5e-5, 3e-5]},
    }
    combos = build_run_combos(cfg)
    assert len(combos) == 4
    assert combos[0]["lwf_lambda"] == 0.2


def test_personalization_tune_grid_lr_only() -> None:
    """Current Step-2 TOML sweeps LR only; lwf=0 and wd fixed at default."""
    import tomllib

    cfg = tomllib.loads(
        Path("scripts/personalization/personalization_tune.toml").read_text(encoding="utf-8")
    )
    combos = build_run_combos(cfg)
    assert len(combos) == 3
    assert "lwf_lambda" not in cfg.get("grid", {})
    assert "weight_decay" not in cfg.get("grid", {})
    assert cfg["defaults"]["lwf_lambda"] == 0.0
    lrs = {c["lr"] for c in combos}
    assert lrs == {0.0001, 0.0002, 0.0004}
    assert all(c["lwf_lambda"] == 0.0 for c in combos)
    assert all(c["weight_decay"] == 3e-5 for c in combos)


def test_leaderboard_filters_to_active_grid(tmp_path: Path) -> None:
    cfg = {
        "defaults": {
            "lwf_lambda": 0.3,
            "lr": 0.0004,
            "weight_decay": 3e-5,
            "train_window_stride": 6,
            "base_run_dir": "test_model_sugar_one",
            "personal_csv": "data/p.csv",
            "patience": 3,
            "epochs": 30,
            "batch_size": 256,
            "val_every_n_epochs": 2,
            "precision": "bf16",
            "eval_zero_shot": True,
        },
        "grid": {"lwf_lambda": [0.2, 0.25], "lr": [0.0002, 0.0004]},
    }
    active = grid_combo_hashes(cfg)
    legacy_params = {
        **cfg["defaults"],
        "lwf_lambda": 0.25,
        "lr": 0.0002,
        "weight_decay": 1.5e-05,
    }
    current_params = {**cfg["defaults"], "lwf_lambda": 0.25, "lr": 0.0002}
    trials = [
        {
            "run_index": 1,
            "combo_hash": combo_hash(legacy_params),
            "status": STATUS_OK,
            "run_name": "legacy_wd_sweep",
            "ft_test_mae": 17.28,
            "params": legacy_params,
        },
        {
            "run_index": 2,
            "combo_hash": combo_hash(current_params),
            "status": STATUS_OK,
            "run_name": "current_grid",
            "ft_test_mae": 17.22,
            "params": current_params,
        },
    ]
    leaderboard_path = tmp_path / "leaderboard.csv"
    write_leaderboard_csv(leaderboard_path, trials, active_combo_hashes=active)
    text = leaderboard_path.read_text(encoding="utf-8")
    assert "legacy_wd_sweep" not in text
    assert "current_grid" in text
    assert text.count("\n") == 2
    assert combo_hash(current_params) in completed_hashes(
        trials, active_combo_hashes=active
    )
    assert combo_hash(legacy_params) not in completed_hashes(
        trials, active_combo_hashes=active
    )


def test_build_run_combos_explicit() -> None:
    cfg = {
        "defaults": {"lwf_lambda": 0.3, "lr": 0.0004},
        "runs": [
            {"name": "sparse", "train_window_stride": 6},
            {"name": "dense", "train_window_stride": 1},
        ],
    }
    combos = build_run_combos(cfg)
    assert len(combos) == 2
    assert combos[0]["train_window_stride"] == 6


def test_combo_hash_skip_completed() -> None:
    params_a = {"lwf_lambda": 0.3, "lr": 0.0004, "weight_decay": 3e-5, "train_window_stride": 6}
    params_b = {**params_a, "lr": 0.0008}
    assert combo_hash(params_a) != combo_hash(params_b)
    trials = [{"combo_hash": combo_hash(params_a), "status": STATUS_OK}]
    assert combo_hash(params_a) in completed_hashes(trials)
    assert combo_hash(params_b) not in completed_hashes(trials)


def test_import_existing_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "sparse_stride6"
    run_dir.mkdir()
    metrics = {
        "config": {
            "personalization": True,
            "base_run_dir": "test_model_sugar_one",
            "personal_csv": "data/p.csv",
            "lwf_lambda": 0.3,
            "lr": 0.0004,
            "weight_decay": 3e-5,
            "patience": 3,
            "epochs": 30,
            "batch_size": 256,
            "train_window_stride": 6,
            "precision": "bf16",
        },
        "zero_shot_test": {"mae": 19.3},
        "finetuned_test": {"mae": 17.1},
        "wall_time_s": 100.0,
    }
    (run_dir / "personalization_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    trials: list[dict] = []
    n = import_existing_runs(tmp_path, trials)
    assert n == 1
    assert trials[0]["status"] == STATUS_OK
    assert trials[0]["ft_test_mae"] == 17.1


def test_leaderboard_excludes_failed_trials(tmp_path: Path) -> None:
    params = {
        "base_run_dir": "test_model_sugar_one",
        "personal_csv": "data/p.csv",
        "lwf_lambda": 0.3,
        "lr": 0.0004,
        "weight_decay": 3e-5,
        "patience": 3,
        "epochs": 30,
        "batch_size": 256,
        "train_window_stride": 1,
        "val_every_n_epochs": 2,
        "precision": "bf16",
        "eval_zero_shot": True,
    }
    trials = [
        {
            "run_index": 1,
            "combo_hash": combo_hash({**params, "train_window_stride": 6}),
            "status": STATUS_OK,
            "run_name": "sparse_stride6",
            "ft_test_mae": 17.15,
            "params": {**params, "train_window_stride": 6},
            "run_dir": str(tmp_path / "sparse"),
        },
        {
            "run_index": 2,
            "combo_hash": combo_hash(params),
            "status": STATUS_FAILED,
            "run_name": "dense_stride1",
            "error": "CUDA out of memory.\nTraceback (most recent call last):\n  ...",
            "params": params,
        },
        {
            "run_index": 3,
            "combo_hash": combo_hash(params),
            "status": STATUS_OK,
            "run_name": "dense_stride1",
            "ft_test_mae": 17.23,
            "params": params,
            "run_dir": str(tmp_path / "dense"),
        },
    ]
    leaderboard_path = tmp_path / "leaderboard.csv"
    write_leaderboard_csv(leaderboard_path, trials)
    text = leaderboard_path.read_text(encoding="utf-8")
    assert "CUDA" not in text
    assert "Traceback" not in text
    assert text.count("\n") == 3  # header + 2 ok rows
    assert "error" not in text.splitlines()[0]


def test_tune_personal_dry_run_cli() -> None:
    from scripts.personalization.tune_personal import app as tune_app

    cfg = Path("scripts/personalization/personalization_tune_window_stride.toml")
    result = runner.invoke(tune_app, ["-c", str(cfg), "--dry-run", "--no-import-existing"])
    assert result.exit_code == 0, result.output
    assert "Pending" in result.output


def test_safe_echo_unicode_on_ascii_stdout() -> None:
    import io
    import sys

    from scripts.common.console import safe_echo

    buf = io.BytesIO()
    text_io = io.TextIOWrapper(buf, encoding="ascii", errors="strict")
    old_stdout = sys.stdout
    sys.stdout = text_io
    try:
        safe_echo("zero-shot=19.32 -> fine-tuned=17.15")
        safe_echo("arrow \u2192 test")
    finally:
        sys.stdout = old_stdout
        text_io.detach()
    assert b"fine-tuned" in buf.getvalue()


def test_holdout_constants() -> None:
    assert len(LOOP_HOLDOUT_QUALITY_USERS) == 6
    from scripts.personalization.constants import (
        HOLDOUT_LR_DEFERRED_USERS,
        HOLDOUT_LR_PILOT_USERS,
    )

    assert len(HOLDOUT_LR_PILOT_USERS) == 3
    assert len(HOLDOUT_LR_DEFERRED_USERS) == 3
    assert set(HOLDOUT_LR_PILOT_USERS) & set(HOLDOUT_LR_DEFERRED_USERS) == set()


def test_plot_data_size_curve(tmp_path: Path) -> None:
    from scripts.personalization.plot_data_size_curve import plot_data_size_curve

    rows = [
        {
            "status": "ok",
            "personal_days": "1",
            "ft_test_mae": 18.0,
            "zs_test_mae": 19.5,
        },
        {
            "status": "ok",
            "personal_days": "7",
            "ft_test_mae": 17.2,
            "zs_test_mae": 19.5,
        },
        {
            "status": "ok",
            "personal_days": "all",
            "ft_test_mae": 17.0,
            "zs_test_mae": 19.5,
        },
    ]
    out_png = tmp_path / "curve.png"
    meta = plot_data_size_curve(rows, out_png=out_png, subject="livia")
    assert out_png.is_file()
    assert meta["plateau"]["optimal_day"] in ("all", 7)


def test_plot_combined_data_size_curves(tmp_path: Path) -> None:
    from scripts.personalization.plot_data_size_curve import plot_combined_data_size_curves

    livia = [
        {"status": "ok", "personal_days": "7", "ft_test_mae": 19.5, "zs_test_mae": 19.3},
        {"status": "ok", "personal_days": "all", "ft_test_mae": 17.1, "zs_test_mae": 19.3},
    ]
    other = [
        {"status": "ok", "personal_days": "7", "ft_test_mae": 18.0, "zs_test_mae": 18.2},
        {"status": "ok", "personal_days": "all", "ft_test_mae": 17.0, "zs_test_mae": 18.2},
    ]
    out_png = tmp_path / "combined.png"
    meta = plot_combined_data_size_curves(
        [("livia", livia), ("loop_556", other)],
        out_png=out_png,
    )
    assert out_png.is_file()
    assert len(meta["subjects"]) == 2
