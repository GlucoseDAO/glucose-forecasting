"""Phase-4 NeuralForecast experiment unit tests (no GPU / full training)."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from nf_baselines.adapter import AI_READI_PROFILE, LOOP_PROFILE, detect_profile
from nf_baselines.benchmark import calculate_metrics, to_neuralforecast_frame
from nf_baselines.catalog import NeuralForecastModel, resolve_models
from nf_baselines.config import (
    NeuralForecastRunConfig,
    frequency_minutes,
    load_model_suites,
    neuralforecast_frequency,
)
from nf_baselines.evaluations.holdout import summarize_holdout_runs


def test_load_default_model_suites_has_auto() -> None:
    suites, text = load_model_suites()
    assert "auto" in suites.suites
    assert "version: 1" in text
    assert NeuralForecastModel.NHITS in suites.suites["auto"].models


def test_sugarone_geometry_accepted(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("sequence_id\n", encoding="utf-8")
    cfg = NeuralForecastRunConfig(
        csv=csv_path,
        holdout_protocol="sugarone-compatible",
        input_hours=128 * 5 / 60,
        h_minutes=60,
        step_size=1,
        freq="5min",
    )
    assert round(cfg.input_hours * 60 / frequency_minutes(cfg.freq)) == 128
    assert cfg.h_minutes // frequency_minutes(cfg.freq) == 12
    assert cfg.step_size == 1


def test_sugarone_geometry_rejects_bad_step(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text("sequence_id\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="sugarone-compatible"):
        NeuralForecastRunConfig(
            csv=csv_path,
            holdout_protocol="sugarone-compatible",
            step_size=12,
        )


def test_resolve_models_suite_and_names() -> None:
    suites, _ = load_model_suites()
    suite_models = {name: suite.models for name, suite in suites.suites.items()}
    auto = resolve_models("auto", suite_models=suite_models)
    assert any(d.name == NeuralForecastModel.NHITS for d in auto)
    named = resolve_models("nhits,tft", suite_models=suite_models)
    assert [d.name for d in named] == [NeuralForecastModel.NHITS, NeuralForecastModel.TFT]


def test_frequency_helpers() -> None:
    assert frequency_minutes("5min") == 5
    assert neuralforecast_frequency("5min") == "5m"
    with pytest.raises(ValueError):
        frequency_minutes("5m")


def test_detect_profile_ai_readi(tmp_path: Path) -> None:
    csv_path = tmp_path / "ai.csv"
    pl.DataFrame(
        {
            "sequence_id": ["s1"],
            "Timestamp (YYYY-MM-DDThh:mm:ss)": ["2020-01-01T00:00:00"],
            "Glucose Value (mg/dL)": [100.0],
            "Heart Rate": [70.0],
            "Step Count": [10.0],
        }
    ).write_csv(csv_path)
    assert detect_profile(csv_path) is AI_READI_PROFILE


def test_detect_profile_loop(tmp_path: Path) -> None:
    csv_path = tmp_path / "loop.csv"
    pl.DataFrame(
        {
            "sequence_id": ["s1"],
            "Timestamp": ["2020-01-01T00:00:00"],
            "Glucose (mg/dL)": [100.0],
            "Basal Rate (U/h)": [1.0],
            "Bolus Insulin (U)": [0.0],
            "Carbohydrates (g)": [0.0],
        }
    ).write_csv(csv_path)
    assert detect_profile(csv_path) is LOOP_PROFILE


def test_to_neuralforecast_frame_and_metrics() -> None:
    frame = pl.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": [1, 2],
            "y": [100.0, 110.0],
            "hr": [70.0, 71.0],
            "study_group": ["G", "G"],
            "event_type": ["EGV", "EGV"],
        }
    )
    nf = to_neuralforecast_frame(frame, ("hr",))
    assert nf.columns == ["unique_id", "ds", "y", "hr"]
    preds = frame.with_columns(pl.lit(105.0).alias("yhat"))
    metrics = calculate_metrics(preds)
    assert metrics.overall.mae == pytest.approx(5.0)
    assert metrics.by_study_group.height == 1


def test_summarize_holdout_runs(tmp_path: Path) -> None:
    shared = tmp_path / "group"
    shared.mkdir()
    configs = []
    for model, mae in (("NHITS", 10.0), ("TFT", 12.0)):
        run = shared / f"{model}_20260101T000000Z"
        run.mkdir()
        config = {
            "evaluation": "holdout",
            "holdout_protocol": "sugarone-compatible",
            "csv": "data.csv",
            "models": model,
            "out_dir": str(tmp_path),
            "step_size": 1,
        }
        (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        pl.DataFrame({"mae": [mae], "rmse": [mae + 1], "mard": [mae + 2]}).write_csv(
            run / "val_metrics_overall.csv"
        )
        pl.DataFrame({"mae": [mae], "rmse": [mae + 1], "mard": [mae + 2]}).write_csv(
            run / "test_metrics_overall.csv"
        )
        pl.DataFrame(
            {
                "unique_id": ["s1"],
                "ds": ["2020-01-01T00:00:00"],
                "y": [100.0],
                "yhat": [100.0 + mae],
                "study_group": ["G"],
                "event_type": ["EGV"],
            }
        ).write_csv(run / "val_predictions.csv")
        pl.DataFrame(
            {
                "unique_id": ["s1"],
                "ds": ["2020-01-01T00:00:00"],
                "y": [100.0],
                "yhat": [100.0 + mae],
                "study_group": ["G"],
                "event_type": ["EGV"],
            }
        ).write_csv(run / "test_predictions.csv")
        configs.append(run)

    out = summarize_holdout_runs(configs, output_dir=tmp_path / "summary", plot=False)
    summary = pl.read_csv(out / "val_metrics_summary.csv")
    assert summary["model"].to_list() == ["NHITS", "TFT"]
    assert (out / "run_config.json").is_file()
