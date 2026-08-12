"""End-to-end smoke: train a tiny SugarOne checkpoint, then evaluate via glucose CLI."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sugar_one.train_sugar_one import app as train_sugar_one_app
from tests.conftest import write_glumind_csv, write_sugar_one_csv

runner = CliRunner()


def _train_sugar_one(tmp_path: Path) -> tuple[Path, Path]:
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
        train_sugar_one_app,
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
            "--num-workers", "0",
            "--device", "cpu",
            "--out-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    run_dirs = [p for p in out_dir.glob("sugar_one_global_*") if p.is_dir()]
    assert len(run_dirs) == 1
    return csv_path, run_dirs[0]


def test_glucose_evaluate_sugar_one_smoke(tmp_path: Path) -> None:
    from cli import app as glucose_app

    csv_path, run_dir = _train_sugar_one(tmp_path)

    result = runner.invoke(
        glucose_app,
        [
            "evaluate",
            "--run-dir", str(run_dir),
            "--data", str(csv_path),
            "--model-type", "sugar_one",
            "--test-split", "test",
            "--device", "cpu",
            "--no-plot",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "MAE" in result.output
    assert "RMSE" in result.output
    assert "MARD" in result.output


def test_glucose_evaluate_glumind_smoke(tmp_path: Path) -> None:
    from cli import app as glucose_app
    from glumind.train_glumind import main as train_glumind_main
    import sys

    csv_path = tmp_path / "glumind_mini.csv"
    write_glumind_csv(
        csv_path,
        series=[
            ("g-train-a", "train", "T1DM", 40, 100.0),
            ("g-val-b", "val", "T1DM", 30, 110.0),
            ("g-test-c", "test", "T1DM", 24, 105.0),
        ],
    )
    out_dir = tmp_path / "runs"
    argv = [
        "train_glumind.py",
        "--csv", str(csv_path),
        "--mode", "global",
        "--input_steps", "8",
        "--horizon", "2",
        "--d_model", "8",
        "--n_heads", "2",
        "--n_blocks", "1",
        "--ff_units", "16",
        "--epochs", "1",
        "--batch_size", "8",
        "--patience", "0",
        "--num_workers", "0",
        "--device", "cpu",
        "--out_dir", str(out_dir),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        train_glumind_main()
    finally:
        sys.argv = old_argv

    run_dirs = [p for p in out_dir.glob("glumind_global_*") if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    result = runner.invoke(
        glucose_app,
        [
            "evaluate",
            "--run-dir", str(run_dir),
            "--data", str(csv_path),
            "--model-type", "glumind",
            "--test-split", "test",
            "--device", "cpu",
            "--no-plot",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "MAE" in result.output
    assert "RMSE" in result.output
    assert "MARD" in result.output
