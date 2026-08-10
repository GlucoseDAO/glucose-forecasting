#!/usr/bin/env python3
"""Persist and restore sklearn feature scalers for glucose forecasting runs.

Schema-free: any ``dict[str, MinMaxScaler | StandardScaler]`` can be saved.
Model families declare their feature names via ``ModelFamilySpec`` in their
own package — this module does not maintain a kind → features registry.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS_FILENAME = "scalers.json"
SCALERS_SCHEMA_VERSION = 1

ScalerLike = MinMaxScaler | StandardScaler

_SCALER_ATTR_RE = re.compile(r"^scaler_(.+)$")


def scalers_path(run_dir: Path) -> Path:
    """Default sidecar path next to ``best_model.pt``."""
    return Path(run_dir) / SCALERS_FILENAME


def resolve_scalers_path(run_dir: Path, meta: Mapping[str, Any] | None = None) -> Path | None:
    """Return path to an existing scalers file, or None if absent."""
    run_dir = Path(run_dir)
    if meta is not None:
        rel = meta.get("scalers")
        if isinstance(rel, str) and rel.strip():
            candidate = Path(rel)
            if not candidate.is_file():
                candidate = run_dir / rel
            if candidate.is_file():
                return candidate
    default = scalers_path(run_dir)
    if default.is_file():
        return default
    return None


def _as_float_list(values: np.ndarray | list[float] | tuple[float, ...]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return [float(x) for x in arr]


def serialize_minmax(scaler: MinMaxScaler) -> dict[str, Any]:
    """Serialize a fitted MinMaxScaler to JSON-friendly params."""
    if not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_max_"):
        raise ValueError("MinMaxScaler is not fitted")
    feature_range = tuple(
        int(x) if float(x).is_integer() else float(x) for x in scaler.feature_range
    )
    return {
        "type": "minmax",
        "data_min": _as_float_list(scaler.data_min_),
        "data_max": _as_float_list(scaler.data_max_),
        "feature_range": [feature_range[0], feature_range[1]],
        "n_features_in": int(getattr(scaler, "n_features_in_", len(scaler.data_min_))),
        "scale": _as_float_list(scaler.scale_),
        "min": _as_float_list(scaler.min_),
    }


def deserialize_minmax(params: Mapping[str, Any]) -> MinMaxScaler:
    """Rebuild a fitted MinMaxScaler from serialized params."""
    if params.get("type", "minmax") != "minmax":
        raise ValueError(f"Expected minmax scaler, got {params.get('type')!r}")
    data_min = np.asarray(params["data_min"], dtype=np.float64)
    data_max = np.asarray(params["data_max"], dtype=np.float64)
    feature_range = tuple(params.get("feature_range", [0, 1]))
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    scaler.data_range_ = data_max - data_min
    n_features = int(params.get("n_features_in", data_min.size))
    scaler.n_features_in_ = n_features
    scaler.n_samples_seen_ = int(params.get("n_samples_seen", 1))
    if "scale" in params and "min" in params:
        scaler.scale_ = np.asarray(params["scale"], dtype=np.float64)
        scaler.min_ = np.asarray(params["min"], dtype=np.float64)
    else:
        feature_range_min, feature_range_max = feature_range
        scale = (feature_range_max - feature_range_min) / np.where(
            scaler.data_range_ == 0, 1.0, scaler.data_range_
        )
        scaler.scale_ = scale
        scaler.min_ = feature_range_min - data_min * scale
    return scaler


def serialize_standard(scaler: StandardScaler) -> dict[str, Any]:
    """Serialize a fitted StandardScaler to JSON-friendly params."""
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("StandardScaler is not fitted")
    var_list = _as_float_list(scaler.var_) if hasattr(scaler, "var_") else None
    return {
        "type": "standard",
        "mean": _as_float_list(scaler.mean_),
        "scale": _as_float_list(scaler.scale_),
        "var": var_list,
        "n_features_in": int(getattr(scaler, "n_features_in_", len(scaler.mean_))),
        "n_samples_seen": int(getattr(scaler, "n_samples_seen_", 1)),
        "with_mean": bool(getattr(scaler, "with_mean", True)),
        "with_std": bool(getattr(scaler, "with_std", True)),
    }


def deserialize_standard(params: Mapping[str, Any]) -> StandardScaler:
    """Rebuild a fitted StandardScaler from serialized params."""
    if params.get("type") != "standard":
        raise ValueError(f"Expected standard scaler, got {params.get('type')!r}")
    scaler = StandardScaler(
        with_mean=bool(params.get("with_mean", True)),
        with_std=bool(params.get("with_std", True)),
    )
    mean = np.asarray(params["mean"], dtype=np.float64)
    scale = np.asarray(params["scale"], dtype=np.float64)
    scaler.mean_ = mean
    scaler.scale_ = scale
    if params.get("var") is not None:
        scaler.var_ = np.asarray(params["var"], dtype=np.float64)
    else:
        scaler.var_ = scale * scale
    scaler.n_features_in_ = int(params.get("n_features_in", mean.size))
    scaler.n_samples_seen_ = int(params.get("n_samples_seen", 1))
    return scaler


def serialize_scaler(scaler: ScalerLike) -> dict[str, Any]:
    if isinstance(scaler, MinMaxScaler):
        return serialize_minmax(scaler)
    if isinstance(scaler, StandardScaler):
        return serialize_standard(scaler)
    raise TypeError(f"Unsupported scaler type: {type(scaler)!r}")


def deserialize_scaler(params: Mapping[str, Any]) -> ScalerLike:
    stype = params.get("type", "minmax")
    if stype == "minmax":
        return deserialize_minmax(params)
    if stype == "standard":
        return deserialize_standard(params)
    raise ValueError(f"Unsupported scaler type in params: {stype!r}")


def extract_scalers_from_dataset(
    dataset: Any,
    feature_names: Sequence[str] | None = None,
) -> dict[str, ScalerLike]:
    """Pull fitted ``scaler_*`` attributes off a window dataset.

    If ``feature_names`` is given, only those channels are required.
    Otherwise every attribute matching ``scaler_<name>`` is collected.
    """
    if feature_names is not None:
        out: dict[str, ScalerLike] = {}
        for name in feature_names:
            attr = f"scaler_{name}"
            if not hasattr(dataset, attr):
                raise AttributeError(
                    f"{type(dataset).__name__} missing attribute {attr!r}"
                )
            scaler = getattr(dataset, attr)
            if scaler is None:
                raise ValueError(f"{attr} is None; cannot serialize")
            out[name] = scaler
        return out

    discovered: dict[str, ScalerLike] = {}
    for attr, value in vars(dataset).items():
        match = _SCALER_ATTR_RE.match(attr)
        if match is None:
            continue
        if value is None:
            continue
        if not isinstance(value, (MinMaxScaler, StandardScaler)):
            continue
        discovered[match.group(1)] = value
    if not discovered:
        raise ValueError(
            f"No scaler_* attributes found on {type(dataset).__name__}"
        )
    return discovered


def dump_scalers(
    path: Path,
    *,
    scalers: Mapping[str, ScalerLike],
    kind: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``scalers.json`` (or custom path). Returns the path written."""
    if not scalers:
        raise ValueError("Cannot dump empty scalers mapping")

    payload: dict[str, Any] = {
        "version": SCALERS_SCHEMA_VERSION,
        "features": {name: serialize_scaler(scalers[name]) for name in scalers},
    }
    if kind is not None:
        payload["kind"] = kind
    if provenance:
        payload["provenance"] = dict(provenance)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_scalers(path: Path) -> tuple[str | None, dict[str, ScalerLike], dict[str, Any]]:
    """Load scalers from JSON.

    Returns ``(kind_or_none, {feature: scaler}, full_payload)``.
    Unknown / new kinds are accepted — feature set comes from the file itself.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0))
    if version != SCALERS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported scalers.json version {version}; "
            f"expected {SCALERS_SCHEMA_VERSION}"
        )
    kind = payload.get("kind")
    if kind is not None:
        kind = str(kind)
    features_blob = payload.get("features")
    if not isinstance(features_blob, dict) or not features_blob:
        raise ValueError("scalers.json missing non-empty 'features' object")
    scalers: dict[str, ScalerLike] = {
        name: deserialize_scaler(params) for name, params in features_blob.items()
    }
    return kind, scalers, payload


def save_scalers_for_run(
    run_dir: Path,
    *,
    dataset: Any | None = None,
    scalers: Mapping[str, ScalerLike] | None = None,
    feature_names: Sequence[str] | None = None,
    kind: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save ``scalers.json`` into ``run_dir`` from a dataset or scaler mapping."""
    if scalers is None:
        if dataset is None:
            raise ValueError("Provide dataset or scalers")
        scalers = extract_scalers_from_dataset(dataset, feature_names=feature_names)
    return dump_scalers(
        scalers_path(run_dir),
        scalers=scalers,
        kind=kind,
        provenance=provenance,
    )


def scalers_match_transform(
    a: ScalerLike,
    b: ScalerLike,
    sample: np.ndarray | None = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    """Whether two fitted scalers produce (nearly) identical transforms."""
    if sample is None:
        sample = np.linspace(0.0, 200.0, 16).reshape(-1, 1)
    else:
        sample = np.asarray(sample, dtype=np.float64)
        if sample.ndim == 1:
            sample = sample.reshape(-1, 1)
    ta = np.asarray(a.transform(sample), dtype=np.float64)
    tb = np.asarray(b.transform(sample), dtype=np.float64)
    return bool(np.allclose(ta, tb, rtol=rtol, atol=atol))
