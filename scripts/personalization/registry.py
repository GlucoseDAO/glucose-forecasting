"""SugarOne-family model registry for personalization scripts.

Any model that accepts the glucose + basal + bolus + carbs schema and uses the
same checkpoint layout (``best_model.pt`` + ``tuning_meta.json`` / ``config.json``)
can register here. Personalization CLIs resolve architecture via this registry
rather than hard-coding ``SugarOneModel``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from scripts.common.checkpoint import strip_compile_prefix
from scripts.common.registry import load_run_meta, resolve_checkpoint
from scripts.personalization.constants import SUGAR_ONE_VALUE_COLUMNS
from scripts.sugar_one.sugar_one_model import SugarOneModel

BuildModelFn = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    """Descriptor for a SugarOne-family model type."""

    name: str
    n_features: int
    value_columns: dict[str, str]
    build: BuildModelFn
    # State-dict key substrings that identify this family in a checkpoint.
    fingerprint_keys: tuple[str, ...]


def _build_sugar_one(
    *,
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    horizon: int,
    dropout: float,
    device: torch.device,
) -> SugarOneModel:
    model = SugarOneModel(
        n_time_steps=input_steps,
        n_features=4,
        d_model=d_model,
        n_heads=n_heads,
        ff_units=ff_units,
        n_blocks=n_blocks,
        prediction_horizon=horizon,
        dropout=dropout,
    )
    return model.to(device)


_REGISTRY: dict[str, ModelSpec] = {
    "sugar_one": ModelSpec(
        name="sugar_one",
        n_features=4,
        value_columns=dict(SUGAR_ONE_VALUE_COLUMNS),
        build=_build_sugar_one,
        fingerprint_keys=(
            "embed_basal.weight",
            "embed_bolus.weight",
            "embed_carbs.weight",
        ),
    ),
}


def register_model(spec: ModelSpec) -> None:
    """Register or replace a SugarOne-family model type."""
    _REGISTRY[spec.name] = spec


def list_model_types() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_model_spec(model_type: str) -> ModelSpec:
    key = model_type.strip().lower().replace("-", "_")
    if key not in _REGISTRY:
        known = ", ".join(list_model_types())
        raise ValueError(f"Unknown model type {model_type!r}. Known: {known}")
    return _REGISTRY[key]


def detect_model_type(meta: dict[str, Any], state: dict[str, torch.Tensor]) -> str:
    """Detect SugarOne-family type from metadata or checkpoint keys."""
    explicit = meta.get("model_type") or meta.get("model")
    if explicit is not None:
        norm = str(explicit).lower().replace("-", "_")
        if norm in ("sugarone", "sugar_one", "glumind_ic", "glumindic"):
            return "sugar_one"
        if norm in _REGISTRY:
            return norm

    normalized_keys = {k.removeprefix("_orig_mod.") for k in state}
    for name, spec in _REGISTRY.items():
        if any(k in normalized_keys for k in spec.fingerprint_keys):
            return name

    # Default for this package: SugarOne-family only.
    return "sugar_one"


def arch_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture hyperparameters from a run metadata dict."""
    return {
        "input_steps": int(meta.get("input_steps", 128)),
        "d_model": int(meta.get("d_model", 32)),
        "n_heads": int(meta.get("n_heads", 8)),
        "ff_units": int(meta.get("ff_units", 128)),
        "n_blocks": int(meta.get("n_blocks", 5)),
        "horizon": int(meta.get("horizon", 12)),
        "dropout": float(meta.get("dropout", 0.1)),
    }


def build_model_from_meta(
    model_type: str,
    meta: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    spec = get_model_spec(model_type)
    arch = arch_from_meta(meta)
    return spec.build(device=device, **arch)


def load_base_checkpoint(
    run_dir: Path,
    *,
    model_type: str | None = None,
    checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any], str, Path]:
    """Load a global (or prior) run for fine-tuning.

    Returns ``(model, meta, resolved_model_type, checkpoint_path)``.
    """
    run_dir = Path(run_dir)
    meta = load_run_meta(run_dir)
    ckpt_path = resolve_checkpoint(run_dir, checkpoint)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict dict in {ckpt_path}")
    state = strip_compile_prefix(state)

    resolved = model_type or detect_model_type(meta, state)
    dev = device or torch.device("cpu")
    model = build_model_from_meta(resolved, meta, dev)
    model.load_state_dict(state)
    return model, meta, resolved, ckpt_path
