"""Smoke test for train_uniglumind.py's Typer app on a tiny CPU run.

train_uniglumind.py does ``from glumind_uni_model import GluMindUniModel``
(a bare, package-relative-free import) rather than
``from scripts.glumind_uni.glumind_uni_model import ...``, so it is only
importable when ``scripts/glumind_uni`` itself is on sys.path (i.e. how it's
actually run: ``cd scripts/glumind_uni && python train_uniglumind.py``). It
is also not registered under [project.scripts] in pyproject.toml. We
replicate that invocation shape here rather than modifying the script.
"""
from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

from tests.conftest import write_glumind_csv

runner = CliRunner()

_GLUMIND_UNI_DIR = str(Path(__file__).resolve().parents[1] / "scripts" / "glumind_uni")


def _import_app():
    if _GLUMIND_UNI_DIR not in sys.path:
        sys.path.insert(0, _GLUMIND_UNI_DIR)
    from train_uniglumind import app  # noqa: PLC0415

    return app


def test_train_uniglumind_cli_smoke_cpu(tmp_path: Path) -> None:
    app = _import_app()

    csv_path = tmp_path / "glumind_uni_mini.csv"
    write_glumind_csv(
        csv_path,
        series=[
            ("u-train-a", "train", "T1DM", 40, 100.0),
            ("u-val-b", "val", "T1DM", 30, 110.0),
            ("u-test-c", "test", "T1DM", 24, 105.0),
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

    run_dirs = [p for p in out_dir.glob("glumind_uni_global_*") if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "last_model.pt").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "val_metrics_overall.csv").exists()
