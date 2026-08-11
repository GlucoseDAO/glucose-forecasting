"""NeuralForecast baselines experiment (holdout suites under ``nf_baselines``).

Legacy tuner: ``nf_baselines.tune_nf_baselines_by_group`` (kept until parity is verified).
Preferred path: ``uv run glucose neuralforecast train …`` (sugarone-compatible 128/12/1).
"""

from nf_baselines.catalog import (
    MODEL_CATALOG,
    ModelDefinition,
    NeuralForecastModel,
    create_model,
    resolve_models,
)

__all__ = [
    "MODEL_CATALOG",
    "ModelDefinition",
    "NeuralForecastModel",
    "create_model",
    "resolve_models",
]
