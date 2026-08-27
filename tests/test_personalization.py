"""Tests for the SugarOne personalization package."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch
from typer.testing import CliRunner

from personalization.constants import LOOP_HOLDOUT_QUALITY_USERS, SPARSE_WINDOW_STRIDE
from personalization.finetune import run_finetune
from personalization.prepare import app as prepare_app
from personalization.registry import (
    build_model_from_meta,
    detect_model_type,
    get_model_spec,
    list_model_types,
    load_base_checkpoint,
    register_model,
    window_steps_from_meta,
)
from personalization.splits import chronological_split_labels, limit_train_days, split_meta
from personalization.leaderboard import (
    STATUS_FAILED,
    STATUS_OK,
    build_run_combos,
    combo_hash,
    completed_hashes,
    grid_combo_hashes,
    import_existing_runs,
    write_leaderboard_csv,
)
from personalization.sweep_utils import (
    build_holdout_lr_comparison,
    estimate_plateau_day,
    holdout_run_complete,
    holdout_row_from_metrics,
    lr_grid_from_base,
    pick_best_row,
    weight_decay_grid,
    write_summary,
)
from sugar_one.train_sugar_one import SugarOneWindowDataset
from sugar_one.sugar_one_model import SugarOneModel
from tests.conftest import (
    TINY_D_MODEL,
    TINY_FF_UNITS,
    TINY_HORIZON,
    TINY_INPUT_STEPS,
    TINY_N_BLOCKS,
    TINY_N_HEADS,
)

runner = CliRunner()

# A JEPA branch inside the tiny-model budget: jepa_window must divide by
# jepa_patch_size, and jepa_embed_dim by jepa_heads. Longer than TINY_INPUT_STEPS
# on purpose, so the lookback differs from input_steps.
TINY_JEPA_WINDOW = 16

# Per-family metadata beyond the shared architecture block. Keys mirror what each
# training script writes into tuning_meta.json.
BASE_RUN_EXTRA_META: dict[str, dict[str, object]] = {
    "sugar_one": {},
    "sugar_jepa2": {
        "jepa_window": TINY_JEPA_WINDOW,
        "jepa_patch_size": 4,
        "jepa_embed_dim": TINY_D_MODEL,
        "jepa_layers": 1,
        "jepa_heads": TINY_N_HEADS,
        "jepa_norm": "instance",
        "jepa_lr": 4e-5,
        "freeze_jepa": False,
    },
}
MODEL_TYPES = sorted(BASE_RUN_EXTRA_META)


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


def _tiny_meta(model_type: str, *, lr: float = 4e-4, patience: int = 10) -> dict:
    return {
        "model_type": model_type,
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
        **BASE_RUN_EXTRA_META[model_type],
    }


def _make_tiny_base_run(
    tmp_path: Path,
    *,
    model_type: str = "sugar_one",
    lr: float = 4e-4,
    patience: int = 10,
) -> Path:
    """A base run dir as the training scripts leave it: meta + best_model.pt."""
    run_dir = tmp_path / f"base_{model_type}"
    run_dir.mkdir(parents=True)
    meta = _tiny_meta(model_type, lr=lr, patience=patience)
    with (run_dir / "tuning_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    model = build_model_from_meta(model_type, meta, torch.device("cpu"))
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    return run_dir


def _prepare_person(tmp_path: Path, *, n_rows: int = 300) -> Path:
    raw = tmp_path / "raw.csv"
    _write_continuous_person_csv(raw, n_rows=n_rows)
    prepared_dir = tmp_path / "prepared"
    prep = runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(prepared_dir), "--out-name", "p.csv"],
    )
    assert prep.exit_code == 0, prep.output
    return prepared_dir / "p.csv"


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_registry_lists_family(model_type: str) -> None:
    assert model_type in list_model_types()
    spec = get_model_spec(model_type)
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
    from personalization.finetune import _load_split_frames

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


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_finetune_sparse_stride_smoke(tmp_path: Path, model_type: str) -> None:
    base = _make_tiny_base_run(tmp_path, model_type=model_type)
    personal_csv = _prepare_person(tmp_path)
    run_dir, results = run_finetune(
        base_run_dir=base,
        personal_csv=personal_csv,
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


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_finetune_smoke(tmp_path: Path, model_type: str) -> None:
    base = _make_tiny_base_run(tmp_path, model_type=model_type)
    personal_csv = _prepare_person(tmp_path)

    run_dir, results = run_finetune(
        base_run_dir=base,
        personal_csv=personal_csv,
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
    cfg = results["config"]
    assert cfg["lwf_lambda"] == 0.5
    assert cfg["model_type"] == model_type
    # The dataset window must match what the model demands, not input_steps.
    assert cfg["window_steps"] == window_steps_from_meta(model_type, _tiny_meta(model_type))


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_type_detected_from_checkpoint(tmp_path: Path, model_type: str) -> None:
    """Both families embed basal/bolus/carbs — detection must still separate them."""
    base = _make_tiny_base_run(tmp_path, model_type=model_type)
    state = torch.load(base / "best_model.pt", map_location="cpu", weights_only=True)
    assert detect_model_type({}, state) == model_type


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_finetune_run_dir_reloads_as_a_base_run(tmp_path: Path, model_type: str) -> None:
    """A fine-tune run must be self-describing: its config alone rebuilds the model.

    This is what --resume-from and every downstream eval depend on, and it breaks
    first when a model gains constructor arguments the run config does not record.
    """
    base = _make_tiny_base_run(tmp_path, model_type=model_type)
    personal_csv = _prepare_person(tmp_path)
    run_dir, _ = run_finetune(
        base_run_dir=base,
        personal_csv=personal_csv,
        out_dir=tmp_path / "ft_reload",
        run_name="reload",
        personal_days=2,
        lwf_lambda=0.0,
        epochs=1,
        patience=0,
        batch_size=8,
        device="cpu",
        num_workers=0,
        eval_zero_shot=False,
    )
    model, meta, resolved, _ = load_base_checkpoint(run_dir, device=torch.device("cpu"))
    assert resolved == model_type
    steps = window_steps_from_meta(resolved, meta)
    assert steps == window_steps_from_meta(model_type, _tiny_meta(model_type))
    # Weights loaded into the rebuilt shape, so a forward pass runs.
    assert model(torch.zeros(2, steps, 4)).shape == (2, TINY_HORIZON)


def test_sugar_jepa2_finetune_can_freeze_encoder(tmp_path: Path) -> None:
    """--freeze-jepa overrides the base run's setting and holds encoder weights fixed."""
    base = _make_tiny_base_run(tmp_path, model_type="sugar_jepa2")
    personal_csv = _prepare_person(tmp_path)
    before = torch.load(base / "best_model.pt", map_location="cpu", weights_only=True)
    encoder_keys = [k for k in before if k.startswith("jepa_encoder.")]
    assert encoder_keys, "no JEPA encoder tensors to check"

    run_dir, results = run_finetune(
        base_run_dir=base,
        personal_csv=personal_csv,
        out_dir=tmp_path / "ft_frozen",
        run_name="frozen",
        personal_days=2,
        lwf_lambda=0.0,
        epochs=1,
        patience=0,
        batch_size=8,
        device="cpu",
        num_workers=0,
        eval_zero_shot=False,
        freeze_jepa=True,
    )
    assert results["config"]["freeze_jepa"] is True
    after = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    for key in encoder_keys:
        assert torch.equal(before[key], after[key]), f"{key} changed under --freeze-jepa"
    # The backbone around it still trained.
    assert not torch.equal(after["embed_glucose.weight"], before["embed_glucose.weight"])


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
    """Default TOML sweeps LR only; lwf=0, wd 3e-5, stride 6, SugarOne + Livia."""
    import tomllib

    cfg = tomllib.loads(Path("src/personalization/tune.toml").read_text(encoding="utf-8"))
    combos = build_run_combos(cfg)
    assert len(combos) == 3
    assert "lwf_lambda" not in cfg.get("grid", {})
    assert "weight_decay" not in cfg.get("grid", {})
    assert cfg["defaults"]["lwf_lambda"] == 0.0
    assert cfg["defaults"]["train_window_stride"] == 6
    assert cfg["paths"]["base_run_dir"] == "fixtures/checkpoints/sugar_one_1.0"
    assert cfg["paths"]["personal_csv"].endswith("livia_chronological.csv")
    lrs = {c["lr"] for c in combos}
    assert lrs == {0.0001, 0.0002, 0.0004}
    assert all(c["lwf_lambda"] == 0.0 for c in combos)
    assert all(c["weight_decay"] == 3e-5 for c in combos)
    assert all(c["train_window_stride"] == 6 for c in combos)


def test_leaderboard_filters_to_active_grid(tmp_path: Path) -> None:
    cfg = {
        "defaults": {
            "lwf_lambda": 0.3,
            "lr": 0.0004,
            "weight_decay": 3e-5,
            "train_window_stride": 6,
            "base_run_dir": "fixtures/checkpoints/sugar_one_1.0",
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
            "base_run_dir": "fixtures/checkpoints/sugar_one_1.0",
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
        "base_run_dir": "fixtures/checkpoints/sugar_one_1.0",
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


def test_tune_dry_run_cli() -> None:
    from personalization.tune import app as tune_app

    cfg = Path("src/personalization/tune_window_stride.toml")
    result = runner.invoke(tune_app, ["-c", str(cfg), "--dry-run", "--no-import-existing"])
    assert result.exit_code == 0, result.output
    assert "Pending" in result.output


def test_safe_echo_unicode_on_ascii_stdout() -> None:
    import io
    import sys

    from common.console import safe_echo

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
    from personalization.constants import (
        HOLDOUT_LR_DEFERRED_USERS,
        HOLDOUT_LR_PILOT_USERS,
    )

    assert len(HOLDOUT_LR_PILOT_USERS) == 3
    assert len(HOLDOUT_LR_DEFERRED_USERS) == 3
    assert set(HOLDOUT_LR_PILOT_USERS) & set(HOLDOUT_LR_DEFERRED_USERS) == set()


def test_product_defaults_livia_sugar_one() -> None:
    from common.paths import DEFAULT_SUGAR_ONE_CHECKPOINT, LIVIA_SUGAR_ONE_CSV
    from personalization.constants import (
        DEFAULT_BASE_RUN_DIR,
        DEFAULT_LIVIA_PREPARED_CSV,
        DEFAULT_LIVIA_SOURCE_CSV,
        DEFAULT_PERSONAL_LWF_LAMBDA,
        DEFAULT_TRAIN_WINDOW_STRIDE,
    )

    assert Path(DEFAULT_BASE_RUN_DIR) == DEFAULT_SUGAR_ONE_CHECKPOINT
    assert DEFAULT_LIVIA_SOURCE_CSV == LIVIA_SUGAR_ONE_CSV
    assert DEFAULT_LIVIA_SOURCE_CSV.as_posix().replace("\\", "/").endswith(
        "fixtures/livia_data/livia_sugar_one_ready.csv"
    )
    assert DEFAULT_LIVIA_PREPARED_CSV.as_posix().replace("\\", "/").endswith(
        "data/input/personalization/prepared/livia_chronological.csv"
    )
    assert DEFAULT_TRAIN_WINDOW_STRIDE == 6
    assert DEFAULT_PERSONAL_LWF_LAMBDA == 0.0
    assert DEFAULT_LIVIA_SOURCE_CSV.is_file()
    assert (DEFAULT_SUGAR_ONE_CHECKPOINT / "best_model.pt").is_file()
    assert (DEFAULT_SUGAR_ONE_CHECKPOINT / "scalers.json").is_file()


def test_prepare_and_finetune_cli_defaults() -> None:
    from typer.main import get_command

    from personalization.constants import (
        DEFAULT_LIVIA_PREPARED_CSV,
        DEFAULT_LIVIA_SOURCE_CSV,
        DEFAULT_TRAIN_WINDOW_STRIDE,
    )
    from personalization.finetune import app as finetune_app
    from personalization.prepare import app as prepare_cli

    livia = get_command(prepare_cli).commands["livia"]
    input_opt = next(p for p in livia.params if "--input" in p.opts)
    assert Path(str(input_opt.default)) == DEFAULT_LIVIA_SOURCE_CSV

    ft = get_command(finetune_app)
    commands = getattr(ft, "commands", {})
    ft_cmd = commands.get("main", ft)
    csv_opt = next(p for p in ft_cmd.params if "--personal-csv" in p.opts)
    stride_opt = next(p for p in ft_cmd.params if "--train-window-stride" in p.opts)
    assert Path(str(csv_opt.default)) == DEFAULT_LIVIA_PREPARED_CSV
    assert int(stride_opt.default) == DEFAULT_TRAIN_WINDOW_STRIDE


def test_personal_console_scripts() -> None:
    import tomllib

    scripts = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    expected = {
        "personal-prepare": "personalization.prepare:app",
        "personal-finetune": "personalization.finetune:app",
        "personal-tune": "personalization.tune:app",
        "personal-sweep-days": "personalization.sweep_data_size:app",
        "personal-plot": "personalization.plots:app",
        "personal-sweep-lr": "personalization.sweep_holdout_lr:app",
        "personal-sweep-lwf": "personalization.sweep_lwf:app",
        "personal-study": "personalization.study:app",
    }
    for name, target in expected.items():
        assert scripts[name] == target
    leftover = (
        "prepare-personal-csv",
        "finetune-personal",
        "tune-personal",
        "sweep-personal-hyperparams",
        "validate-personal-holdouts",
        "sweep-personal-curriculum",
        "sweep-personal-data-size",
        "plot-personal-data-size",
        "run-personal-phase4",
    )
    for name in leftover:
        assert name not in scripts


def test_ensure_holdout_csv(tmp_path: Path) -> None:
    from personalization.prepare import ensure_holdout_csv

    loop_csv = tmp_path / "loop.csv"
    _write_continuous_person_csv(loop_csv, n_rows=200, user_id="154")
    out_dir = tmp_path / "holdouts"
    first = ensure_holdout_csv(loop_csv, "154", out_dir, 0.25, 0.15)
    second = ensure_holdout_csv(loop_csv, "154", out_dir, 0.25, 0.15)
    assert first == second
    assert first.is_file()
    labeled = pl.read_csv(first)
    assert set(labeled["Recommended Split"].unique().to_list()) == {"train", "val", "test"}
    try:
        ensure_holdout_csv(loop_csv, "999", out_dir, 0.25, 0.15)
        raise AssertionError("expected missing user to raise")
    except ValueError as exc:
        assert "999" in str(exc)


def test_plot_data_size_curve(tmp_path: Path) -> None:
    from personalization.plots import plot_data_size_curve

    rows = [
        {
            "status": "ok",
            "personal_days": "1",
            "ft_test_mae": 18.0,
            "zs_test_mae": 19.5,
            "train_span_days": 345.0,
        },
        {
            "status": "ok",
            "personal_days": "7",
            "ft_test_mae": 17.2,
            "zs_test_mae": 19.5,
            "train_span_days": 345.0,
        },
        {
            "status": "ok",
            "personal_days": "all",
            "ft_test_mae": 17.0,
            "zs_test_mae": 19.5,
            "train_span_days": 345.0,
            "used_train_days": 345.0,
        },
    ]
    out_png = tmp_path / "curve.png"
    meta = plot_data_size_curve(rows, out_png=out_png, subject="livia", mode="max_days")
    assert out_png.is_file()
    assert 345.0 not in meta["x_values"]
    assert 999.0 not in meta["x_values"]
    assert all(x <= 60.0 for x in meta["x_values"])


def test_plot_combined_data_size_curves(tmp_path: Path) -> None:
    from personalization.plots import ALL_DUMMY_X, plot_combined_data_size_curves

    livia = [
        {
            "status": "ok",
            "personal_days": "7",
            "ft_test_mae": 19.5,
            "zs_test_mae": 19.3,
            "train_span_days": 345.0,
        },
        {
            "status": "ok",
            "personal_days": "all",
            "ft_test_mae": 17.1,
            "zs_test_mae": 19.3,
            "train_span_days": 345.0,
            "used_train_days": 345.0,
        },
    ]
    other = [
        {
            "status": "ok",
            "personal_days": "7",
            "ft_test_mae": 18.0,
            "zs_test_mae": 18.2,
            "train_span_days": 91.0,
        },
        {
            "status": "ok",
            "personal_days": "all",
            "ft_test_mae": 17.0,
            "zs_test_mae": 18.2,
            "train_span_days": 91.0,
            "used_train_days": 91.0,
        },
    ]
    out_png = tmp_path / "combined.png"
    meta = plot_combined_data_size_curves(
        [("livia", livia), ("loop_556", other)],
        out_png=out_png,
        mode="dummy_all",
    )
    assert out_png.is_file()
    assert len(meta["subjects"]) == 2
    all_x = [x for entry in meta["subjects"] for x in entry["x_values"]]
    assert ALL_DUMMY_X in all_x
    assert 999.0 not in all_x

    out_60 = tmp_path / "combined_60.png"
    meta_60 = plot_combined_data_size_curves(
        [("livia", livia), ("loop_556", other)],
        out_png=out_60,
        mode="max_days",
        max_days=60.0,
    )
    all_x_60 = [x for entry in meta_60["subjects"] for x in entry["x_values"]]
    assert ALL_DUMMY_X not in all_x_60
    assert 345.0 not in all_x_60
    assert all(x <= 60.0 for x in all_x_60)


def test_plot_curriculum_mae_and_lambda(tmp_path: Path) -> None:
    from personalization.plots import plot_curriculum_mae_and_lambda

    rows = [
        {
            "status": "ok",
            "personal_days": "1",
            "ft_test_mae": 18.5,
            "zs_test_mae": 18.3,
            "lwf_lambda": 0.0,
            "train_span_days": 345.0,
        },
        {
            "status": "ok",
            "personal_days": "all",
            "ft_test_mae": 17.0,
            "zs_test_mae": 18.3,
            "lwf_lambda": 0.0,
            "train_span_days": 345.0,
            "used_train_days": 345.0,
        },
    ]
    lwf_rows = [
        {**rows[0], "lwf_lambda": 0.35, "ft_test_mae": 18.6},
        {**rows[1], "lwf_lambda": 0.0, "ft_test_mae": 17.0},
    ]
    out_png = tmp_path / "mae_lambda.png"
    plot_curriculum_mae_and_lambda(
        [("livia_indep", rows), ("livia_curr_lwf", lwf_rows)],
        out_png=out_png,
    )
    assert out_png.is_file()


def test_uses_base_scalers_and_legacy_complete(tmp_path: Path) -> None:
    from personalization.sweep_utils import personalization_run_complete, uses_base_scalers

    assert uses_base_scalers({"refit_scalers_on_personal": False, "scaler_source": "fixtures/x/scalers.json"})
    assert not uses_base_scalers({})
    assert not uses_base_scalers({"refit_scalers_on_personal": None, "scaler_source": None})
    assert not uses_base_scalers({"refit_scalers_on_personal": True, "scaler_source": "personal_train"})

    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    (run_dir / "personalization_metrics.json").write_text(
        json.dumps({"finetuned_test": {"mae": 19.5}, "config": {"personal_days": 1}}),
        encoding="utf-8",
    )
    assert not personalization_run_complete(run_dir)
    assert personalization_run_complete(run_dir, require_base_scalers=False)

    new_dir = tmp_path / "new_run"
    new_dir.mkdir()
    (new_dir / "personalization_metrics.json").write_text(
        json.dumps(
            {
                "finetuned_test": {"mae": 18.0},
                "config": {
                    "refit_scalers_on_personal": False,
                    "scaler_source": "fixtures/checkpoints/sugar_one_1.0/scalers.json",
                },
            }
        ),
        encoding="utf-8",
    )
    assert personalization_run_complete(new_dir)


def test_archive_legacy_scaler_runs(tmp_path: Path) -> None:
    from personalization.sweep_utils import archive_legacy_scaler_runs

    out_dir = tmp_path / "data_size"
    run_dir = out_dir / "days_1" / "livia_days_1"
    run_dir.mkdir(parents=True)
    (run_dir / "personalization_metrics.json").write_text(
        json.dumps({"finetuned_test": {"mae": 19.5}, "config": {"personal_days": 1}}),
        encoding="utf-8",
    )
    archived = archive_legacy_scaler_runs(out_dir)
    assert archived is not None
    assert archived.is_dir()
    assert not out_dir.exists()


def test_should_skip_day_budget() -> None:
    from personalization.sweep_utils import should_skip_day_budget

    assert should_skip_day_budget(60, 37.4)
    assert not should_skip_day_budget(30, 37.4)
    assert not should_skip_day_budget(None, 37.4)


def test_joined2_test_cohort_frozen() -> None:
    from collections import Counter

    from common.data.loading import STUDY_GROUP_ORDER
    from personalization.cohort import (
        JOINED2_TEST_USERS,
        PHASE4_SUBJECTS,
        joined2_test_subjects,
        original_cohort_subjects,
        select_two_test_users_per_group,
    )

    assert len(PHASE4_SUBJECTS) == 15
    assert len(original_cohort_subjects()) == 7
    joined = joined2_test_subjects()
    assert len(joined) == 8
    assert JOINED2_TEST_USERS == tuple((s.user_id, s.study_group) for s in joined)
    counts = Counter(group for _uid, group in JOINED2_TEST_USERS)
    ai_ready_groups = [g for g in STUDY_GROUP_ORDER if g != "T1DM"]
    assert list(counts.keys()) == ai_ready_groups
    assert all(n == 2 for n in counts.values())
    assert all(not uid.startswith("loop_") for uid, _group in JOINED2_TEST_USERS)

    stats = pl.DataFrame(
        {
            "uid": [
                "ai_ready_b",
                "ai_ready_a",
                "ai_ready_c",
                "loop_z",
                "loop_y",
            ],
            "group": ["Healthy", "Healthy", "Healthy", "T1DM", "T1DM"],
            "n_rows": [100, 100, 50, 500, 400],
        }
    )
    picked = select_two_test_users_per_group(stats)
    by_group = {g: [uid for uid, grp in picked if grp == g] for g in STUDY_GROUP_ORDER}
    assert by_group["Healthy"] == ["ai_ready_a", "ai_ready_b"]
    assert by_group["T1DM"] == ["loop_z", "loop_y"]
    assert by_group["Pre-T2DM"] == []


def test_train_span_from_split_meta() -> None:
    from personalization.splits import train_span_days_from_split_meta

    span = train_span_days_from_split_meta(
        {
            "train": {
                "start": "2018-04-12 18:05:00",
                "end": "2018-05-20 04:23:00",
            }
        }
    )
    assert span is not None
    assert 37.0 < span < 38.0


def test_decaying_lwf_lambda_schedule() -> None:
    from personalization.constants import decaying_lwf_lambda
    from personalization.sweep_lwf import lwf_for_independent_kind

    assert decaying_lwf_lambda(1) == 0.5
    assert decaying_lwf_lambda(3) == 0.4
    assert decaying_lwf_lambda(7) == 0.3
    assert decaying_lwf_lambda(14) == 0.2
    assert decaying_lwf_lambda(30) == 0.0
    assert decaying_lwf_lambda(60) == 0.0
    assert decaying_lwf_lambda(None) == 0.0
    assert lwf_for_independent_kind("decay", 1) == 0.5
    assert lwf_for_independent_kind("decay", 14) == 0.2
    assert lwf_for_independent_kind("decay", 30) == 0.0
    assert lwf_for_independent_kind("const", 1) == 0.1
    assert lwf_for_independent_kind("const", 30) == 0.1
    assert lwf_for_independent_kind("const", None) == 0.1


def test_lwf_teacher_stays_global_after_student_init(tmp_path: Path) -> None:
    from personalization.finetune import attach_lwf_teacher_and_init_student

    def _model() -> SugarOneModel:
        return SugarOneModel(
            n_time_steps=TINY_INPUT_STEPS,
            n_features=4,
            d_model=TINY_D_MODEL,
            n_heads=TINY_N_HEADS,
            ff_units=TINY_FF_UNITS,
            n_blocks=TINY_N_BLOCKS,
            prediction_horizon=TINY_HORIZON,
            dropout=0.0,
        )

    device = torch.device("cpu")
    global_model = _model()
    student = _model()
    student.load_state_dict(global_model.state_dict())
    with torch.no_grad():
        for param in student.parameters():
            param.add_(0.25)
    weights = tmp_path / "best_model.pt"
    torch.save(student.state_dict(), weights)

    model = _model()
    model.load_state_dict(global_model.state_dict())
    teacher = attach_lwf_teacher_and_init_student(
        model,
        lwf_lambda=0.3,
        from_scratch=False,
        init_weights_from=weights,
        resume=False,
        device=device,
    )
    assert teacher is not None
    for left, right in zip(teacher.parameters(), global_model.parameters(), strict=True):
        assert torch.allclose(left, right)
    for left, right in zip(model.parameters(), student.parameters(), strict=True):
        assert torch.allclose(left, right)
