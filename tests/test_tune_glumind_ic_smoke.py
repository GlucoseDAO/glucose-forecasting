"""Smoke tests for GluMindIC random tuner (global mode)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from scripts.glumind_ic.tune_glumind_ic import (
    combo_hash,
    derive_rng,
    merge_defaults_and_sample,
    sample_from_space,
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
