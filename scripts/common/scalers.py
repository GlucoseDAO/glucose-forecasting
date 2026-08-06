#!/usr/bin/env python3
"""Persist and restore sklearn feature scalers for glucose forecasting runs.

Training fits MinMaxScaler (and optionally StandardScaler for SugarJEPA) on the
train split. Those params are written to ``scalers.json`` beside checkpoint
weights so eval/inference do not need the original training CSV.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALERS_FILENAME = "scalers.json"
SCALERS_SCHEMA_VERSION = 1

ModelKind = Literal["glumind", "sugar_one", "glumind_uni", "sugar_jepa"]

GLUMIND_FEATURES: tuple[str, ...] = ("glucose", "hr", "steps")
SUGAR_ONE_FEATURES: tuple[str, ...] = ("glucose", "basal", "bolus", "carbs")
GLUMIND_UNI_FEATURES: tuple[str, ...] = ("glucose",)
SUGAR_JEPA_FEATURES: tuple[str, ...] = (
    "glucose",
    "basal",
    "bolus",
    "carbs",
    "glucose_jepa",
)

FEATURES_BY_KIND: dict[str, tuple[str, ...]] = {
    "glumind": GLUMIND_FEATURES,
    "sugar_one": SUGAR_ONE_FEATURES,
    "glumind_uni": GLUMIND_UNI_FEATURES,
    "sugar_jepa": SUGAR_JEPA_FEATURES,
}

ScalerLike = MinMaxScaler | StandardScaler


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


def extract_scalers_from_dataset(dataset: Any, kind: ModelKind) -> dict[str, ScalerLike]:
    """Pull fitted scaler attributes off a window dataset instance."""
    features = FEATURES_BY_KIND[kind]
    out: dict[str, ScalerLike] = {}
    for name in features:
        attr = f"scaler_{name}"
        if not hasattr(dataset, attr):
            raise AttributeError(f"{type(dataset).__name__} missing attribute {attr!r}")
        scaler = getattr(dataset, attr)
        if scaler is None:
            raise ValueError(f"{attr} is None; cannot serialize")
        out[name] = scaler
    return out


def dump_scalers(
    path: Path,
    *,
    kind: ModelKind,
    scalers: Mapping[str, ScalerLike],
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``scalers.json`` (or custom path). Returns the path written."""
    expected = FEATURES_BY_KIND[kind]
    missing = [f for f in expected if f not in scalers]
    if missing:
        raise ValueError(f"Missing scalers for kind={kind!r}: {missing}")
    extra = [f for f in scalers if f not in expected]
    if extra:
        raise ValueError(f"Unexpected scalers for kind={kind!r}: {extra}")

    payload: dict[str, Any] = {
        "version": SCALERS_SCHEMA_VERSION,
        "kind": kind,
        "features": {name: serialize_scaler(scalers[name]) for name in expected},
    }
    if provenance:
        payload["provenance"] = dict(provenance)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_scalers(path: Path) -> tuple[ModelKind, dict[str, ScalerLike], dict[str, Any]]:
    """Load scalers from JSON.

    Returns ``(kind, {feature: scaler}, full_payload)``.
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
    if kind not in FEATURES_BY_KIND:
        raise ValueError(f"Unknown scaler kind: {kind!r}")
    features_blob = payload.get("features")
    if not isinstance(features_blob, dict):
        raise ValueError("scalers.json missing 'features' object")
    expected = FEATURES_BY_KIND[kind]
    scalers: dict[str, ScalerLike] = {}
    for name in expected:
        if name not in features_blob:
            raise ValueError(f"scalers.json missing feature {name!r}")
        scalers[name] = deserialize_scaler(features_blob[name])
    return kind, scalers, payload  # type: ignore[return-value]


def save_scalers_for_run(
    run_dir: Path,
    *,
    kind: ModelKind,
    dataset: Any | None = None,
    scalers: Mapping[str, ScalerLike] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Save ``scalers.json`` into ``run_dir`` from a dataset or scaler mapping."""
    if scalers is None:
        if dataset is None:
            raise ValueError("Provide dataset or scalers")
        scalers = extract_scalers_from_dataset(dataset, kind)
    return dump_scalers(
        scalers_path(run_dir),
        kind=kind,
        scalers=scalers,
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
