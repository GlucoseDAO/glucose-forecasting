"""Smoke test for the plain SugarOne trainer CLI (train_sugar_one.py's Typer
app), on a tiny synthetic CSV, CPU, 1 epoch.

Named test_train_sugar_one_cli_smoke.py (not test_train_sugar_one_smoke.py)
to avoid collision with the existing tests/test_tune_sugar_one_smoke.py,
which covers the random-search *tuner* (tune_sugar_one.py), a different
script.
"""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from scripts.sugar_one.train_sugar_one import app
from tests.conftest import write_sugar_one_csv

runner = CliRunner()


def test_train_sugar_one_cli_smoke_cpu(tmp_path: Path) -> None:
    csv_path = tmp_path / "sugar_one_mini.csv"
    write_sugar_one_csv(
        csv_path,
        series=[
            ("s-train-a", "train", "T1DM", 40, 100.0),
            ("s-val-b", "val", "T1DM", 30, 110.0),
            ("s-test-c", "test", "T1DM", 24, 105.0),
        ],
    )
    out_dir = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "--csv", str(csv_path),
            "--mode", "global",
            "--input-steps", "8",
            "--horizon", "2",
            "--d-model", "8",
            "--n-heads", "2",
            "--n-blocks", "1",
            "--ff-units", "16",
            "--epochs", "1",
            "--batch-size", "8",
            "--patience", "0",
            "--log-every", "1",
            "--val-every-n-epochs", "1",
            "--num-workers", "0",
            "--device", "cpu",
            "--out-dir", str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    run_dirs = [p for p in out_dir.glob("sugar_one_global_*") if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "last_model.pt").exists()
    assert (run_dir / "last_checkpoint.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "val_metrics_overall.csv").exists()
    assert (run_dir / "test_metrics_overall.csv").exists()
