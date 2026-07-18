"""Tests for YAML-defined NeuralForecast model suites."""
from __future__ import annotations

from glucose_forecasting.backends.neuralforecast.catalog import (
    MODEL_CATALOG,
    NeuralForecastModel,
    create_model,
    resolve_models,
)
from glucose_forecasting.backends.neuralforecast.config import load_model_suites


def test_auto_suite_is_loaded_from_packaged_yaml_and_includes_xlstm() -> None:
    suites, _ = load_model_suites()

    assert NeuralForecastModel.XLSTM in suites.suites["auto"].models
    definitions = resolve_models(
        "auto",
        suite_models={name: suite.models for name, suite in suites.suites.items()},
    )
    assert NeuralForecastModel.XLSTM in {definition.name for definition in definitions}


def test_model_suite_rejects_unknown_model_name() -> None:
    suites, _ = load_model_suites()

    try:
        resolve_models("not-a-model", suite_models={"auto": suites.suites["auto"].models})
    except ValueError as error:
        assert "unknown NeuralForecast" in str(error)
    else:
        raise AssertionError("unknown models must fail")


def test_xlstm_constructs_with_historical_covariates() -> None:
    """xLSTM is a real NeuralForecast export, not an invented AutoXLSTM alias."""
    model = create_model(
        MODEL_CATALOG[NeuralForecastModel.XLSTM],
        horizon=2,
        input_size=4,
        max_steps=1,
        historical_exogenous=("basal", "bolus", "carbohydrates"),
    )

    assert model.__class__.__name__ == "xLSTM"
    assert model.hist_exog_list == ["basal", "bolus", "carbohydrates"]
