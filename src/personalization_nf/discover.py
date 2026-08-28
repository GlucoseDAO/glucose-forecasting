"""Discover scored NeuralForecast holdout leaf runs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.evaluation.resolve_models import RankedRun, select_best_runs_by_mae
from common.paths import resolve_project_path
from personalization_nf.constants import DEFAULT_NF_HOLDOUT_ROOT


@dataclass(frozen=True, slots=True)
class NfHoldoutRun:
    """One global NeuralForecast holdout bundle selected for personalization."""

    model_key: str
    run_dir: Path
    bundle_dir: Path
    val_mae: float
    config: dict[str, Any]


def discover_holdout_runs(
    root: Path = DEFAULT_NF_HOLDOUT_ROOT,
    *,
    models: tuple[str, ...] | None = None,
) -> list[NfHoldoutRun]:
    """Return the best-by-val-MAE leaf run per model family under ``root``."""
    resolved = resolve_project_path(root)
    if not resolved.is_dir():
        raise FileNotFoundError(f"NeuralForecast holdout root not found: {resolved}")

    ranked: list[RankedRun] = select_best_runs_by_mae(resolved)
    wanted = {name.lower() for name in models} if models else None
    out: list[NfHoldoutRun] = []
    for item in ranked:
        if wanted is not None and item.model_key.lower() not in wanted:
            continue
        bundle = item.run_dir / "neuralforecast"
        config_path = item.run_dir / "run_config.json"
        if not bundle.is_dir() or not config_path.is_file():
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"run_config.json must contain an object: {config_path}")
        out.append(
            NfHoldoutRun(
                model_key=item.model_key,
                run_dir=item.run_dir,
                bundle_dir=bundle,
                val_mae=item.mae,
                config=payload,
            )
        )
    if not out:
        raise ValueError(f"No NeuralForecast bundles found under {resolved}")
    return out


def parse_model_filter(raw: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated model filter; ``None`` means all discovered models."""
    if raw is None or not raw.strip() or raw.strip().lower() == "all":
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return names or None
