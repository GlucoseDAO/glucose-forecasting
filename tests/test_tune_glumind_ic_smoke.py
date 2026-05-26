"""Smoke tests for GluMindIC random tuner (global mode)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from scripts.glumind_ic.tune_glumind_ic import (
    DEFAULT_CONFIG_FILENAME,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_OK,
    STATUS_RUNNING,
    build_tune_context,
    claim_next_trial,
    combo_hash,
    derive_rng,
    finalize_trial,
    is_non_retryable_failure,
    load_user_config,
    merge_defaults_and_sample,
    reconcile_trial_state,
    resolve_config_path,
    sample_from_space,
    try_claim_resumable_trial,
    tune_loop,
)


def _write_loop_ic_csv(path: Path) -> None:
    """Minimal CSV compatible with train_glumind_ic loaders + sliding windows."""
    base = datetime(2020, 1, 1, 0, 0, 0)
    rows: list[dict[str, object]] = []

    def add_series(
        uid: str,
        split: str,
        n_rows: int,
        glucose0: float,
    ) -> None:
        for i in range(n_rows):
            ts = base + timedelta(minutes=5 * i)
            rows.append(
                {
                    "sequence_id": uid,
                    "Timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                    "Event Type": "EGV",
                    "User ID": "u_smoke",
                    "Glucose (mg/dL)": glucose0 + float(i) * 0.5,
                    "Basal Rate (U/h)": "",
                    "Bolus Insulin (U)": "",
                    "Carbohydrates (g)": "",
                    "Recommended Split": split,
                    "Study Group": "T1DM",
                }
            )

    add_series("smoke-tr-a", "train", 40, 100.0)
    add_series("smoke-val-b", "val", 35, 110.0)
    add_series("smoke-te-c", "test", 25, 105.0)

    pl.DataFrame(rows).write_csv(path)


def test_combo_hash_stable() -> None:
    h1 = combo_hash({"lr": 0.001, "batch_size": 32, "weight_decay": 1e-4})
    h2 = combo_hash({"batch_size": 32, "lr": 0.001, "weight_decay": 1e-4})
    assert h1 == h2
    assert len(h1) == 64


def test_derive_rng_deterministic() -> None:
    r1 = derive_rng(42, 7)
    r2 = derive_rng(42, 7)
    assert r1.random() == r2.random()


def test_merge_defaults_and_sample() -> None:
    defaults = {
        "d_model": 32,
        "n_heads": 4,
        "lr": 0.0005,
        "horizon": 12,
    }
    sampled = {"d_model": 64, "lr": 0.001}
    params = merge_defaults_and_sample(defaults, sampled)
    assert params["d_model"] == 64
    assert params["n_heads"] == 4
    assert params["lr"] == 0.001
    assert params["horizon"] == 12


def test_sample_from_space_respects_lists() -> None:
    rng = derive_rng(99, 0)
    space = {"lr": [0.1, 0.2], "k": [3]}
    for _ in range(20):
        s = sample_from_space(rng, space)
        assert s["lr"] in (0.1, 0.2)
        assert s["k"] == 3


def test_is_non_retryable_failure_oom() -> None:
    assert is_non_retryable_failure("OutOfMemoryError: CUDA out of memory")
    assert not is_non_retryable_failure("RuntimeError: something else")


def test_reconcile_running_and_resume_priority(tmp_path: Path) -> None:
    out = tmp_path / "tune_out"
    run_dir = out / "trial_0001_deadbeef"
    run_dir.mkdir(parents=True)
    user_cfg: dict = {
        "paths": {"csv": "x.csv", "output_dir": str(out)},
        "dataset": {
            "unique_id": "sequence_id",
            "drop_interpolated": False,
            "study_groups": "",
            "split_scheme": "classic",
            "max_train_series": 0,
            "max_eval_series": 0,
        },
        "defaults": {"horizon": 2, "input_steps": 4, "batch_size": 4},
        "tune": {
            "n_trials": 5,
            "random_seed": 1,
            "max_random_draws": 20,
            "epochs": 1,
            "patience": 0,
            "log_every": 1,
            "val_every_n_epochs": 1,
            "ckpt_every_n_epochs": 0,
            "precision": "fp32",
            "compile_mode": "none",
            "disable_tf32": False,
            "num_workers": 0,
            "prefetch_factor": 2,
            "resume_from": "",
            "space": {"lr": [0.01]},
        },
    }
    meta = tmp_path / "stub.toml"
    meta.write_text("# stub\n", encoding="utf-8")
    ctx = build_tune_context(
        user_cfg=user_cfg,
        config_path=meta,
        device_name="cpu",
        seed_override=None,
    )
    ctx.state_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    state = {
        "version": 1,
        "trials": [
            {
                "trial_index": 1,
                "combo_hash": "deadbeef",
                "params": {"lr": 0.01},
                "run_name": "trial_0001_deadbeef",
                "status": STATUS_RUNNING,
                "run_dir": str(run_dir),
            },
            {
                "trial_index": 2,
                "combo_hash": "oomhash",
                "params": {"lr": 0.02},
                "run_name": "trial_0002_oomhash",
                "status": STATUS_FAILED,
                "non_retryable": False,
                "error": "OutOfMemoryError: CUDA out of memory",
                "run_dir": str(out / "trial_0002_oomhash"),
            },
        ],
        "next_draw_index": 0,
    }
    ctx.state_path.write_text(json.dumps(state), encoding="utf-8")

    n = reconcile_trial_state(ctx)
    assert n >= 1
    data = json.loads(ctx.state_path.read_text(encoding="utf-8"))
    t1 = data["trials"][0]
    assert t1["status"] == STATUS_INTERRUPTED
    t2 = data["trials"][1]
    assert t2["non_retryable"] is True

    resumed = try_claim_resumable_trial(ctx)
    assert resumed is not None
    assert resumed.trial_index == 1
    assert resumed.is_resume is True

    again = try_claim_resumable_trial(ctx)
    assert again is None or again.trial_index != 2


def test_claim_next_trial_no_duplicate_hash(tmp_path: Path) -> None:
    out = tmp_path / "tune_out"
    user_cfg: dict = {
        "paths": {"csv": "x.csv", "output_dir": str(out)},
        "dataset": {
            "unique_id": "sequence_id",
            "drop_interpolated": False,
            "study_groups": "",
            "split_scheme": "classic",
            "max_train_series": 0,
            "max_eval_series": 0,
        },
        "defaults": {"horizon": 2, "input_steps": 4, "batch_size": 4},
        "tune": {
            "n_trials": 5,
            "random_seed": 1,
            "max_random_draws": 20,
            "epochs": 1,
            "patience": 0,
            "log_every": 1,
            "val_every_n_epochs": 1,
            "ckpt_every_n_epochs": 0,
            "precision": "fp32",
            "compile_mode": "none",
            "disable_tf32": False,
            "num_workers": 0,
            "prefetch_factor": 2,
            "resume_from": "",
            "space": {"lr": [0.01, 0.02, 0.03, 0.04]},
        },
    }
    meta = tmp_path / "stub.toml"
    meta.write_text("# stub\n", encoding="utf-8")
    ctx = build_tune_context(
        user_cfg=user_cfg,
        config_path=meta,
        device_name="cpu",
        seed_override=None,
    )
    c0 = claim_next_trial(ctx)
    assert c0 is not None
    finalize_trial(
        ctx,
        {
            "trial_index": c0.trial_index,
            "combo_hash": c0.combo_hash,
            "params": c0.trial_params,
            "run_name": c0.run_name,
            "status": STATUS_OK,
            "non_retryable": False,
            "error": None,
            "duration_seconds": 1.0,
            "val_mae": 1.0,
            "val_rmse": 1.0,
            "val_mard": 1.0,
            "run_dir": str(out / c0.run_name),
        },
    )
    c1 = claim_next_trial(ctx)
    assert c1 is not None
    assert c0.combo_hash != c1.combo_hash
    finalize_trial(
        ctx,
        {
            "trial_index": c1.trial_index,
            "combo_hash": c1.combo_hash,
            "params": c1.trial_params,
            "run_name": c1.run_name,
            "status": STATUS_OK,
            "non_retryable": False,
            "error": None,
            "duration_seconds": 1.0,
            "val_mae": 1.0,
            "val_rmse": 1.0,
            "val_mard": 1.0,
            "run_dir": str(out / c1.run_name),
        },
    )
    c2 = claim_next_trial(ctx)
    assert c2 is not None
    assert c2.combo_hash not in {c0.combo_hash, c1.combo_hash}


def test_build_tune_context_defaults_only_no_space(tmp_path: Path) -> None:
    out = tmp_path / "tune_out"
    user_cfg: dict = {
        "paths": {"csv": "x.csv", "output_dir": str(out)},
        "dataset": {
            "unique_id": "sequence_id",
            "drop_interpolated": False,
            "study_groups": "",
            "split_scheme": "classic",
            "max_train_series": 0,
            "max_eval_series": 0,
        },
        "defaults": {"lr": 0.001, "batch_size": 8, "horizon": 2, "input_steps": 4},
        "tune": {
            "n_trials": 99,
            "random_seed": 1,
            "max_random_draws": 20,
            "epochs": 1,
            "patience": 0,
            "log_every": 1,
            "val_every_n_epochs": 1,
            "ckpt_every_n_epochs": 0,
            "precision": "fp32",
            "compile_mode": "none",
            "disable_tf32": False,
            "num_workers": 0,
            "prefetch_factor": 2,
            "resume_from": "",
        },
    }
    meta = tmp_path / "stub.toml"
    meta.write_text("# stub\n", encoding="utf-8")
    ctx = build_tune_context(
        user_cfg=user_cfg,
        config_path=meta,
        device_name="cpu",
        seed_override=None,
    )
    assert ctx.space == {}
    assert ctx.n_trials_target == 1
    claim = claim_next_trial(ctx)
    assert claim is not None
    assert claim.trial_params == user_cfg["defaults"]


def test_default_config_is_production_full() -> None:
    assert DEFAULT_CONFIG_FILENAME == "tune_glumind_ic_full.toml"
    cfg_path = resolve_config_path(None)
    assert cfg_path.name == "tune_glumind_ic_full.toml"


def test_full_toml_production_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "scripts" / "glumind_ic" / "tune_glumind_ic_full.toml"
    cfg = load_user_config(cfg_path)
    assert cfg["paths"]["csv"].endswith("loop_ai_ready_joined2.csv")
    assert "space" not in cfg.get("tune", {})
    defaults = cfg["defaults"]
    assert defaults["n_blocks"] == 5
    assert defaults["input_steps"] == 128
    assert defaults["lr"] == 0.0004
    assert defaults["weight_decay"] == 0.00003
    assert int(cfg["tune"]["n_trials"]) == 1


def test_dev_toml_loads_without_tune_space() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "scripts" / "glumind_ic" / "tune_glumind_ic_dev.toml"
    cfg = load_user_config(cfg_path)
    assert "tune" in cfg
    assert "space" in cfg.get("tune", {})
    assert "defaults" in cfg


def test_tune_one_trial_cpu(tmp_path: Path) -> None:
    csv_path = tmp_path / "loop_ic_mini.csv"
    _write_loop_ic_csv(csv_path)
    out = tmp_path / "tune_out"

    user_cfg: dict = {
        "paths": {
            "csv": str(csv_path),
            "output_dir": str(out),
        },
        "dataset": {
            "unique_id": "sequence_id",
            "drop_interpolated": False,
            "study_groups": "",
            "split_scheme": "classic",
            "max_train_series": 0,
            "max_eval_series": 0,
        },
        "defaults": {
            "d_model": 16,
            "n_heads": 4,
            "n_blocks": 1,
            "ff_units": 32,
            "dropout": 0.1,
            "input_steps": 8,
            "horizon": 2,
            "weight_decay": 0.0001,
            "batch_size": 8,
        },
        "tune": {
            "n_trials": 1,
            "random_seed": 123,
            "max_random_draws": 50,
            "epochs": 1,
            "patience": 0,
            "log_every": 1,
            "val_every_n_epochs": 1,
            "ckpt_every_n_epochs": 0,
            "precision": "fp32",
            "compile_mode": "none",
            "disable_tf32": False,
            "num_workers": 0,
            "prefetch_factor": 2,
            "resume_from": "",
            "space": {
                "lr": [0.001],
            },
        },
    }

    meta_path = tmp_path / "stub.toml"
    meta_path.write_text("# stub\n", encoding="utf-8")

    tune_loop(
        user_cfg=user_cfg,
        config_path=meta_path,
        device_name="cpu",
        seed_override=None,
    )

    state_path = out / "state.json"
    report_path = out / "tune_report.md"
    assert state_path.exists()
    assert report_path.exists()
    trial_dirs = sorted(out.glob("trial_*"))
    assert len(trial_dirs) == 1
    assert (trial_dirs[0] / "val_metrics_overall.csv").exists()
