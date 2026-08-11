"""Tests for best-per-model expansion of evaluate run roots."""
from __future__ import annotations

from pathlib import Path

import pytest

from common.evaluation.config import ModelEvalSpec
from common.evaluation.resolve_models import (
    expand_model_specs,
    infer_model_key,
    select_best_runs_by_mae,
)


def _write_run(path: Path, *, mae: float) -> None:
    path.mkdir(parents=True)
    (path / "neuralforecast").mkdir()
    (path / "run_config.json").write_text('{"evaluation":"holdout"}', encoding="utf-8")
    (path / "val_metrics_overall.csv").write_text(
        f"mae,rmse,mard\n{mae},{mae + 1},{mae + 2}\n",
        encoding="utf-8",
    )
    (path / "test_metrics_overall.csv").write_text(
        f"mae,rmse,mard\n{mae + 0.5},{mae + 1.5},{mae + 2.5}\n",
        encoding="utf-8",
    )


def test_infer_model_key_from_nf_dirname() -> None:
    assert infer_model_key(Path("NHITS_20260811T160526Z")) == "NHITS"
    assert infer_model_key(Path("NBEATSx_20260811T160552Z")) == "NBEATSx"


def test_select_best_runs_by_mae_picks_lowest_val(tmp_path: Path) -> None:
    root = tmp_path / "nf_holdout"
    _write_run(root / "__ALL__" / "NHITS_20260101T000000Z", mae=12.0)
    _write_run(root / "__ALL__" / "NHITS_20260102T000000Z", mae=10.0)
    _write_run(root / "__ALL__" / "TFT_20260101T000000Z", mae=11.0)
    # summaries must be ignored even if they contain CSVs
    summary = root / "__ALL__" / "summaries" / "20260101T000000Z"
    summary.mkdir(parents=True)
    (summary / "val_metrics_summary.csv").write_text("model,mae\nNHITS,1\n", encoding="utf-8")

    ranked = select_best_runs_by_mae(root)
    by_key = {item.model_key: item for item in ranked}
    assert set(by_key) == {"NHITS", "TFT"}
    assert by_key["NHITS"].run_dir.name == "NHITS_20260102T000000Z"
    assert by_key["NHITS"].mae == 10.0
    assert by_key["TFT"].mae == 11.0


def test_expand_model_specs_mixed_leaf_and_container(tmp_path: Path) -> None:
    leaf = tmp_path / "test_model_glumind"
    leaf.mkdir()
    (leaf / "best_model.pt").write_bytes(b"x")
    (leaf / "tuning_meta.json").write_text("{}", encoding="utf-8")

    container = tmp_path / "nf_holdout"
    _write_run(container / "__ALL__" / "NHITS_20260101T000000Z", mae=9.0)
    _write_run(container / "__ALL__" / "LSTM_20260101T000000Z", mae=15.0)

    specs = expand_model_specs(
        [
            ModelEvalSpec(run_dir=leaf, label="GluMind", model_type="glumind"),
            ModelEvalSpec(run_dir=container, label="NF"),
        ],
        project_root=tmp_path,
    )
    assert specs[0].run_dir == leaf
    assert specs[0].label == "GluMind"
    labels = [s.label for s in specs[1:]]
    assert labels == ["NF/LSTM", "NF/NHITS"]


def test_expand_empty_container_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No scored run"):
        expand_model_specs([ModelEvalSpec(run_dir=empty)], project_root=tmp_path)
