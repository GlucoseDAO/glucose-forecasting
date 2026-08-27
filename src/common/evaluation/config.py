#!/usr/bin/env python3
"""YAML defaults for ``glucose evaluate``."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_CONFIG_FILENAME = "glucose_evaluate.yaml"
# src/common/evaluation/config.py → parents[2] == src/ (or site-packages when installed)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_FILENAME

ModelTypeName = Literal["auto", "glumind", "sugar_one", "glumind_uni", "sugar_jepa", "sugar_jepa2"]
SUPPORTED_MODEL_TYPES: tuple[str, ...] = (
    "auto",
    "glumind",
    "sugar_one",
    "glumind_uni",
    "sugar_jepa",
    "sugar_jepa2",
)


@dataclass
class ModelEvalSpec:
    run_dir: Path
    label: str | None = None
    model_type: ModelTypeName = "auto"
    zero_cov: bool = False
    include_cov: str | None = None
    exclude_cov: str | None = None
    batch_size: int | None = None


@dataclass
class EvaluateConfig:
    data: Path | None = None
    train_data: Path | None = None
    out: Path = Path("data/output/compare")
    device: str = "auto"
    test_split: str | None = "test"
    batch_size: int | None = 4096
    plot: bool = True
    zero_cov: bool = False
    include_cov: str | None = None
    exclude_cov: str | None = None
    model_type: ModelTypeName = "auto"
    models: list[ModelEvalSpec] = field(default_factory=list)


def default_config_path() -> Path:
    """Prefer the packaged YAML next to ``src/`` / site-packages root."""
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


def _as_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _parse_model_type(value: Any) -> ModelTypeName:
    text = str(value or "auto").strip().lower().replace("-", "_")
    if text not in SUPPORTED_MODEL_TYPES:
        allowed = "|".join(SUPPORTED_MODEL_TYPES)
        raise ValueError(f"Invalid model_type {value!r}; expected {allowed}")
    return text  # type: ignore[return-value]


def load_evaluate_config(path: Path | None = None) -> EvaluateConfig:
    """Load evaluate defaults from YAML."""
    config_path = path if path is not None else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluate config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Evaluate config must be a mapping: {config_path}")

    global_zero = bool(raw.get("zero_cov", False))
    global_include = _as_optional_str(raw.get("include_cov"))
    global_exclude = _as_optional_str(raw.get("exclude_cov"))
    global_model_type = _parse_model_type(raw.get("model_type", "auto"))

    batch_size_raw = raw.get("batch_size", 4096)
    global_batch_size = None if batch_size_raw in (None, "") else int(batch_size_raw)

    models: list[ModelEvalSpec] = []
    for item in raw.get("models") or []:
        if not isinstance(item, dict):
            raise ValueError("Each models[] entry must be a mapping")
        run_dir = item.get("run_dir")
        if not run_dir:
            raise ValueError("Each models[] entry requires run_dir")
        item_batch = item.get("batch_size", global_batch_size)
        models.append(
            ModelEvalSpec(
                run_dir=Path(str(run_dir)),
                label=_as_optional_str(item.get("label")),
                model_type=_parse_model_type(item.get("model_type", global_model_type)),
                zero_cov=bool(item.get("zero_cov", global_zero)),
                include_cov=_as_optional_str(item.get("include_cov", global_include)),
                exclude_cov=_as_optional_str(item.get("exclude_cov", global_exclude)),
                batch_size=None if item_batch in (None, "") else int(item_batch),
            )
        )

    test_split_raw = raw.get("test_split", "test")
    if test_split_raw is None:
        test_split: str | None = None
    else:
        test_split = str(test_split_raw)

    return EvaluateConfig(
        data=_as_path(raw.get("data")),
        train_data=_as_path(raw.get("train_data")),
        out=_as_path(raw.get("out")) or Path("data/output/compare"),
        device=str(raw.get("device", "auto")),
        test_split=test_split if test_split != "" else None,
        batch_size=global_batch_size,
        plot=bool(raw.get("plot", True)),
        zero_cov=global_zero,
        include_cov=global_include,
        exclude_cov=global_exclude,
        model_type=global_model_type,
        models=models,
    )
