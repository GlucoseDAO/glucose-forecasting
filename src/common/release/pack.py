#!/usr/bin/env python3
"""Pack a training run directory into a format-1.0 inference release bundle."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from common.checkpoint import strip_compile_prefix
from common.model_spec import detect_family_kind, get_family_spec
from common.release.bundle import write_inference_bundle
from common.release.config import InferenceConfig
from common.release.manifest import ReleaseManifest
from common.release.metrics import EvaluationProtocol, MetricsSpec, SelectionMetric
from common.release.preprocessor import (
    ImputationSpec,
    PreprocessorSpec,
    ScalerSpec,
    WindowSpec,
)
from common.release.provenance import ProvenanceSpec
from common.scalers import resolve_scalers_path

_ARCH_KEYS = (
    "input_steps",
    "horizon",
    "d_model",
    "n_heads",
    "ff_units",
    "n_blocks",
    "dropout",
)

_IMPUTATION_BY_KIND: dict[str, dict[str, ImputationSpec]] = {
    "glumind": {
        "glucose": ImputationSpec(method="forward_fill"),
        "hr": ImputationSpec(method="forward_fill"),
        "steps": ImputationSpec(method="forward_fill"),
    },
    "sugar_one": {
        "glucose": ImputationSpec(method="forward_fill"),
        "basal": ImputationSpec(method="forward_fill"),
        "bolus": ImputationSpec(method="zero"),
        "carbs": ImputationSpec(method="zero"),
    },
    "glumind_uni": {
        "glucose": ImputationSpec(method="forward_fill"),
    },
}

_UNITS_BY_FEATURE: dict[str, str] = {
    "glucose": "mg/dL",
    "hr": "bpm",
    "steps": "count",
    "basal": "U/h",
    "bolus": "U",
    "carbs": "g",
}


def pack_run_dir(
    run_dir: Path,
    bundle_dir: Path,
    *,
    model_type: str | None = None,
    release_id: str | None = None,
    checkpoint: Path | None = None,
    project_root: Path | None = None,
) -> ReleaseManifest:
    """Convert a training run into a checksummed inference release bundle.

    Reads ``best_model.pt`` / ``last_model.pt``, ``tuning_meta.json`` /
    ``config.json``, ``scalers.json``, and optional ``*_metrics_overall.csv``.
    """
    run_dir = Path(run_dir)
    bundle_dir = Path(bundle_dir)
    root = project_root or Path.cwd()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    meta = _load_run_meta(run_dir)
    ckpt_path = _resolve_checkpoint(run_dir, checkpoint)
    state = _load_state_dict(ckpt_path)

    kind = model_type or detect_family_kind(meta, state)
    spec = get_family_spec(kind)
    device = torch.device("cpu")
    model = spec.build_model(meta, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    manifest = build_manifest_from_run(
        run_dir,
        meta=meta,
        kind=kind,
        release_id=release_id,
        project_root=root,
    )
    write_inference_bundle(bundle_dir, manifest=manifest, model=model)
    return manifest


def build_manifest_from_run(
    run_dir: Path,
    *,
    meta: Mapping[str, Any],
    kind: str,
    release_id: str | None = None,
    project_root: Path | None = None,
) -> ReleaseManifest:
    """Build a ``ReleaseManifest`` from run metadata without writing weights."""
    root = project_root or Path.cwd()
    family = get_family_spec(kind)
    input_steps = int(meta.get("input_steps", 128))
    horizon = int(meta.get("horizon", 12))
    cadence = int(meta.get("cadence_minutes", meta.get("freq_minutes", 5)))

    architecture = {
        key: meta[key]
        for key in _ARCH_KEYS
        if key in meta
    }
    for key, default in (
        ("input_steps", input_steps),
        ("horizon", horizon),
        ("d_model", 32),
        ("n_heads", 8),
        ("ff_units", 128),
        ("n_blocks", 5),
        ("dropout", 0.1),
    ):
        architecture.setdefault(key, default)

    config = InferenceConfig(
        model_id=str(meta.get("model_id") or kind),
        model_type=kind,
        architecture=architecture,
        feature_order=tuple(family.feature_names),
        horizon=horizon,
        cadence=cadence,
    )
    preprocessor = _preprocessor_from_run(run_dir, meta=meta, kind=kind)
    metrics = _metrics_from_run(run_dir)
    provenance = _provenance_from_run(run_dir, meta=meta, project_root=root)
    rid = release_id or _default_release_id(kind)
    return ReleaseManifest(
        release_id=rid,
        config=config,
        preprocessor=preprocessor,
        metrics=metrics,
        provenance=provenance,
    )


def _load_run_meta(run_dir: Path) -> dict[str, Any]:
    for name in ("tuning_meta.json", "config.json"):
        path = run_dir / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
            raise ValueError(f"Run metadata must be a JSON object: {path}")
    raise FileNotFoundError(
        f"No tuning_meta.json or config.json in {run_dir}"
    )


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    for name in ("best_model.pt", "last_model.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No best_model.pt or last_model.pt in {run_dir}"
    )


def _load_state_dict(checkpoint: Path) -> dict[str, Any]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state = loaded["model_state_dict"]
    elif isinstance(loaded, dict):
        state = loaded
    else:
        raise ValueError(f"Unsupported checkpoint payload in {checkpoint}")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint state_dict must be a mapping: {checkpoint}")
    return strip_compile_prefix(state)


def _preprocessor_from_run(
    run_dir: Path,
    *,
    meta: Mapping[str, Any],
    kind: str,
) -> PreprocessorSpec:
    family = get_family_spec(kind)
    input_steps = int(meta.get("input_steps", 128))
    scalers_file = resolve_scalers_path(run_dir, meta)
    scaler_specs: dict[str, ScalerSpec] = {}
    if scalers_file is not None:
        payload = json.loads(scalers_file.read_text(encoding="utf-8"))
        features = payload.get("features", {})
        if not isinstance(features, dict):
            raise ValueError(f"Invalid scalers.json features in {scalers_file}")
        for name, params in features.items():
            if isinstance(params, dict):
                scaler_specs[str(name)] = _scaler_spec_from_json(params)

    aliases: dict[str, str] = {}
    for canonical, sources in family.csv_column_aliases.items():
        for source in sources:
            aliases[str(source)] = str(canonical)
    for canonical, source in family.value_columns.items():
        aliases.setdefault(str(source), str(canonical))

    imputation = dict(_IMPUTATION_BY_KIND.get(kind, {}))
    if not imputation:
        for feature in family.feature_names:
            imputation[str(feature)] = ImputationSpec(method="forward_fill")

    units = {
        feature: _UNITS_BY_FEATURE[feature]
        for feature in family.feature_names
        if feature in _UNITS_BY_FEATURE
    }
    return PreprocessorSpec(
        scalers=scaler_specs,
        aliases=aliases,
        imputation=imputation,
        window=WindowSpec(input_steps=input_steps, stride_steps=1),
        units=units,
    )


def _scaler_spec_from_json(params: Mapping[str, Any]) -> ScalerSpec:
    raw_kind = str(params.get("type") or params.get("kind") or "custom").lower()
    if raw_kind == "minmax":
        kind: str = "minmax"
    elif raw_kind == "standard":
        kind = "standard"
    else:
        kind = "custom"

    flat: dict[str, float] = {}
    for key, value in params.items():
        if key in {"type", "kind"}:
            continue
        if isinstance(value, bool):
            flat[key] = float(value)
        elif isinstance(value, (int, float)):
            flat[key] = float(value)
        elif isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], (int, float)):
                flat[key] = float(value[0])
            else:
                for index, item in enumerate(value):
                    if isinstance(item, (int, float)):
                        flat[f"{key}_{index}"] = float(item)
        # skip nested non-numeric structures
    return ScalerSpec(kind=kind, parameters=flat)  # type: ignore[arg-type]


def _metrics_from_run(run_dir: Path) -> MetricsSpec:
    validation = _read_metrics_csv(run_dir / "val_metrics_overall.csv")
    test = _read_metrics_csv(run_dir / "test_metrics_overall.csv")
    if not validation and not test:
        validation = {"mae": 0.0}
        test = {"mae": 0.0}
    if not validation:
        validation = dict(test)
    if not test:
        test = dict(validation)
    return MetricsSpec(
        selection_metric=SelectionMetric(name="mae", direction="minimize"),
        validation=validation,
        test=test,
        protocol=EvaluationProtocol(
            name="held-out evaluation",
            split="test",
            details={"source": "run_dir_metrics_csv"},
        ),
    )


def _read_metrics_csv(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        return {}
    out: dict[str, float] = {}
    for key, value in row.items():
        if value is None or value == "":
            continue
        out[str(key)] = float(value)
    return out


def _provenance_from_run(
    run_dir: Path,
    *,
    meta: Mapping[str, Any],
    project_root: Path,
) -> ProvenanceSpec:
    csv_ref = str(meta.get("csv") or meta.get("train_csv") or run_dir.name)
    fingerprint_source = csv_ref.encode("utf-8")
    csv_path = Path(csv_ref)
    if not csv_path.is_file():
        candidate = project_root / csv_ref
        if candidate.is_file():
            csv_path = candidate
    if csv_path.is_file():
        fingerprint_source = f"{csv_path.resolve()}:{csv_path.stat().st_size}".encode(
            "utf-8"
        )
    return ProvenanceSpec(
        git_sha=_git_sha(project_root),
        lock_hash=_lock_hash(project_root),
        env={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        dataset_fingerprint="sha256:"
        + hashlib.sha256(fingerprint_source).hexdigest(),
        seed=int(meta.get("seed", 0)),
    )


def _git_sha(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        sha = completed.stdout.strip()
        return sha if sha else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _lock_hash(project_root: Path) -> str:
    lock_path = project_root / "uv.lock"
    if not lock_path.is_file():
        return "unknown"
    digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _default_release_id(kind: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{stamp}"
