"""Smoke test for train_glumind.py's argparse-based main() on a tiny CPU run.

train_glumind.py uses argparse (not Typer) and parse_args() reads sys.argv
directly, so we patch sys.argv before calling main(), following the existing
end-to-end pattern in tests/test_tune_sugar_one_smoke.py (tiny synthetic CSV,
epochs=1, CPU).
"""
from __future__ import annotations

import sys
from pathlib import Path

from scripts.glumind.train_glumind import main
from tests.conftest import write_glumind_csv


def test_train_glumind_main_smoke_cpu(tmp_path: Path, monkeypatch) -> None:
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
        "--log_every", "1",
        "--val_every_n_epochs", "1",
        "--num_workers", "0",
        "--device", "cpu",
        "--out_dir", str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()

    run_dirs = [p for p in out_dir.glob("glumind_global_*") if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "last_model.pt").exists()
    assert (run_dir / "tuning_meta.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "val_metrics_overall.csv").exists()
    assert (run_dir / "test_metrics_overall.csv").exists()
