#!/usr/bin/env python3
"""Model-family descriptors: features, CSV columns, build, scaler extract.

``common`` holds only the Protocol / registry. Each model package
implements a concrete spec (e.g. ``sugar_one/sugar_one_spec.py``).
Architecture modules (``*_model.py``) stay torch-only for checkpoint reuse.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from common.scalers import ScalerLike

_FAMILY_SPECS: dict[str, ModelFamilySpec] = {}

# Lazy import paths so registering all families does not require every optional
# dependency (e.g. JEPA weights) at import time of unrelated CLIs.
_FAMILY_LAZY_IMPORTS: dict[str, str] = {
    "glumind": "glumind.glumind_spec:GLUMIND_SPEC",
    "sugar_one": "sugar_one.sugar_one_spec:SUGAR_ONE_SPEC",
    "glumind_uni": "glumind_uni.glumind_uni_spec:GLUMIND_UNI_SPEC",
    "sugar_jepa": "sugar_jepa.sugar_jepa_spec:SUGAR_JEPA_SPEC",
    "sugar_jepa2": "sugar_jepa.sugar_jepa2_spec:SUGAR_JEPA2_SPEC",
}


@runtime_checkable
class ModelFamilySpec(Protocol):
    """Per-model metadata for training, eval, and scaler persistence."""

    kind: str
    feature_names: Sequence[str]
    n_features: int
    value_columns: Mapping[str, str]
    csv_column_aliases: Mapping[str, Sequence[str]]
    covariate_aliases: Mapping[str, Sequence[str]]
    fingerprint_keys: Sequence[str]
    # Keys that must be ABSENT — what tells a family apart from one that extends
    # it and therefore carries every one of its fingerprint keys too.
    exclude_keys: Sequence[str]
    ffill_bfill_columns: Sequence[str]
    zero_fill_columns: Sequence[str]

    def build_model(self, meta: Mapping[str, Any], device: torch.device) -> nn.Module:
        """Construct an untrained model from run metadata."""

    def extract_scalers(self, dataset: Any) -> dict[str, ScalerLike]:
        """Pull fitted scalers off a window dataset for this family."""

    def build_window_dataset(
        self,
        df: Any,
        *,
        input_steps: int,
        horizon: int,
        scalers: Mapping[str, ScalerLike] | None = None,
        fit_scalers: bool = False,
        window_stride: int = 1,
        meta: Mapping[str, Any] | None = None,
    ) -> Any:
        """Build a sliding-window dataset for this family."""

    def infer_batch(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one eval batch; return ``(y_true, y_pred)`` on ``device``."""


def infer_batch_xy(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default ``(x, y)`` batch unpack + ``model(x)`` forward."""
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return y, model(x)


def infer_batch_jepa(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SugarJepa ``(x, glucose_jepa, y)`` batch unpack + ``model(x, jepa)``."""
    x, jepa, y = batch
    x = x.to(device)
    jepa = jepa.to(device)
    y = y.to(device)
    return y, model(x, jepa)


def register_family_spec(spec: ModelFamilySpec) -> None:
    """Register or replace a model-family spec."""
    key = _normalize_kind(spec.kind)
    _FAMILY_SPECS[key] = spec


def list_family_kinds() -> list[str]:
    """Return registered kinds (loads built-in lazy entries first)."""
    _ensure_builtins_registered()
    return sorted(_FAMILY_SPECS)


def get_family_spec(kind: str) -> ModelFamilySpec:
    """Resolve a family spec by kind string."""
    key = _normalize_kind(kind)
    _ensure_builtins_registered(key)
    if key not in _FAMILY_SPECS:
        # aliases
        aliases = {
            "sugarone": "sugar_one",
            "glumindic": "sugar_one",
            "glumind_ic": "sugar_one",
            "uniglumind": "glumind_uni",
            "gluminduni": "glumind_uni",
        }
        key = aliases.get(key, key)
        _ensure_builtins_registered(key)
    if key not in _FAMILY_SPECS:
        known = ", ".join(list_family_kinds())
        raise ValueError(f"Unknown model family {kind!r}. Known: {known}")
    return _FAMILY_SPECS[key]


def detect_family_kind(
    meta: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
) -> str:
    """Detect family kind from metadata and/or checkpoint state_dict keys.

    Fingerprinting requires *every* key of a family's fingerprint and none of
    its ``exclude_keys``. Matching on any one key would misread a family that
    extends another: a real ``sugar_jepa2`` checkpoint carries SugarOne's three
    covariate embeddings alongside its own encoder, so ``any()`` would call it
    ``sugar_one`` and the load would then fail under ``strict=True``.
    """
    _ensure_builtins_registered()
    explicit = meta.get("model_type") or meta.get("model") or meta.get("kind")
    if explicit is not None:
        norm = _normalize_kind(str(explicit))
        aliases = {
            "sugarone": "sugar_one",
            "glumindic": "sugar_one",
            "glumind_ic": "sugar_one",
            "uniglumind": "glumind_uni",
            "gluminduni": "glumind_uni",
        }
        norm = aliases.get(norm, norm)
        if norm in _FAMILY_SPECS or norm in _FAMILY_LAZY_IMPORTS:
            return norm

    if state is not None:
        normalized_keys = {str(k).removeprefix("_orig_mod.") for k in state}
        for kind in list_family_kinds():
            spec = get_family_spec(kind)
            # An empty fingerprint identifies nothing; `all()` over it matches
            # every checkpoint, so such a family is never detectable by keys.
            if not spec.fingerprint_keys:
                continue
            if not all(fk in normalized_keys for fk in spec.fingerprint_keys):
                continue
            if any(ek in normalized_keys for ek in spec.exclude_keys):
                continue
            return kind

    raise ValueError(
        "Could not detect model family from metadata/checkpoint. "
        "Pass an explicit model type."
    )


def arch_hparams_from_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Shared architecture hyperparameter extraction from run metadata."""
    return {
        "input_steps": int(meta.get("input_steps", 128)),
        "d_model": int(meta.get("d_model", 32)),
        "n_heads": int(meta.get("n_heads", 8)),
        "ff_units": int(meta.get("ff_units", 128)),
        "n_blocks": int(meta.get("n_blocks", 5)),
        "horizon": int(meta.get("horizon", 12)),
        "dropout": float(meta.get("dropout", 0.1)),
    }


def _normalize_kind(kind: str) -> str:
    return kind.strip().lower().replace("-", "_")


def _ensure_builtins_registered(kind: str | None = None) -> None:
    targets = [kind] if kind else list(_FAMILY_LAZY_IMPORTS)
    for key in targets:
        if key is None:
            continue
        if key in _FAMILY_SPECS:
            continue
        path = _FAMILY_LAZY_IMPORTS.get(key)
        if path is None:
            continue
        module_path, attr = path.split(":")
        module = __import__(module_path, fromlist=[attr])
        spec = getattr(module, attr)
        register_family_spec(spec)
