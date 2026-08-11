"""Phase-3 evaluation package unit tests (detect / device / compare / config)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common.evaluation.comparison import write_comparison_report
from common.evaluation.config import default_config_path, load_evaluate_config
from common.evaluation.detect import detect_run_dir
from common.evaluation.device import resolve_torch_device
from common.evaluation.readers import read_precomputed_split_metrics
from common.evaluation.types import (
    RegressionMetrics,
    RunDirKind,
    SingleModelResult,
    SplitMetrics,
)


def test_resolve_torch_device_explicit() -> None:
    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device("CUDA") == "cuda"


def test_resolve_torch_device_auto_returns_known_backend() -> None:
    assert resolve_torch_device("auto") in {"cuda", "mps", "cpu"}


def test_detect_run_dir_custom_pytorch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "best_model.pt").write_bytes(b"x")
    (run / "tuning_meta.json").write_text("{}", encoding="utf-8")
    assert detect_run_dir(run, tmp_path) == RunDirKind.CUSTOM_PYTORCH


def test_detect_run_dir_precomputed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "test_metrics_overall.csv").write_text(
        "mae,rmse,mard\n1,2,3\n", encoding="utf-8"
    )
    assert detect_run_dir(run, tmp_path) == RunDirKind.PRECOMPUTED


def test_detect_run_dir_neuralforecast(tmp_path: Path) -> None:
    run = tmp_path / "NHITS_20260101T000000Z"
    run.mkdir()
    (run / "neuralforecast").mkdir()
    (run / "run_config.json").write_text("{}", encoding="utf-8")
    (run / "val_metrics_overall.csv").write_text(
        "mae,rmse,mard\n1,2,3\n", encoding="utf-8"
    )
    assert detect_run_dir(run, tmp_path) == RunDirKind.NEURALFORECAST


def test_read_precomputed_split_metrics(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "test_metrics_overall.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mae", "rmse", "mard"])
        writer.writeheader()
        writer.writerow({"mae": "1.5", "rmse": "2.5", "mard": "3.5"})
    splits = read_precomputed_split_metrics(run)
    assert "test" in splits
    assert splits["test"].overall.mae == 1.5


def test_write_comparison_report_with_plots(tmp_path: Path) -> None:
    results = [
        SingleModelResult(
            model_name="a",
            run_dir=tmp_path / "a",
            kind=RunDirKind.PRECOMPUTED,
            split_results={
                "test": SplitMetrics(overall=RegressionMetrics(1.0, 2.0, 3.0))
            },
            model_type="glumind",
        ),
        SingleModelResult(
            model_name="b",
            run_dir=tmp_path / "b",
            kind=RunDirKind.PRECOMPUTED,
            split_results={
                "test": SplitMetrics(overall=RegressionMetrics(4.0, 5.0, 6.0))
            },
            model_type="sugar_one",
        ),
    ]
    out = write_comparison_report(results, tmp_path / "compare", plot=True)
    assert (out / "metrics_summary.csv").is_file()
    assert (out / "metrics_comparison.png").is_file()
    assert (out / "mae_comparison.png").is_file()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["results"]) == 2
    assert manifest["results"][0]["primary"]["mae"] == 1.0
    assert len(manifest["plots"]) == 2


def test_load_default_evaluate_config() -> None:
    path = default_config_path()
    assert path.is_file()
    cfg = load_evaluate_config(path)
    assert cfg.data is not None
    assert "loop_ai_ready_joined2.csv" in str(cfg.data).replace("\\", "/")
    assert cfg.out == Path("data/output/compare")
    assert cfg.plot is True
    assert len(cfg.models) >= 3
    names = {m.run_dir.name for m in cfg.models}
    assert "test_model_glumind" in names
    assert "test_model_sugar_one" in names
    assert "nf_holdout" in names
