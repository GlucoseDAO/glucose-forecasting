#!/usr/bin/env python3
"""PyTorch run-dir evaluation adapter (GluMind / SugarOne)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from common.evaluation.device import resolve_torch_device
from common.evaluation.types import (
    RegressionMetrics,
    RunDirKind,
    SingleModelResult,
    SplitMetrics,
)


def evaluate_pytorch_run(
    run_dir: Path,
    *,
    test_csv: Path,
    train_csv: Path | None = None,
    checkpoint: Path | None = None,
    model_type: Literal["auto", "glumind", "sugar_one"] = "auto",
    test_split: str | None = "test",
    batch_size: int | None = None,
    device: str = "auto",
    zero_cov: bool = False,
    include_cov: str | None = None,
    exclude_cov: str | None = None,
    refit_scalers: bool = False,
    allow_fit_on_eval: bool = False,
    log_interval: float = 10.0,
    label: str | None = None,
    project_root: Path | None = None,
) -> SingleModelResult:
    """Run inference via the shared evaluate-model path and wrap metrics."""
    # Lazy import keeps ``common.evaluation`` importable without pulling torch CLIs
    # at package import time for simple helpers / unit tests.
    from sugar_one.evaluate_model import evaluate_checkpoint

    resolved_device = resolve_torch_device(device)
    payload = evaluate_checkpoint(
        test_csv=test_csv,
        run_dir=run_dir,
        checkpoint=checkpoint,
        train_csv=train_csv,
        model_type=model_type,
        test_split=test_split,
        batch_size=batch_size,
        device=resolved_device,
        zero_cov=zero_cov,
        include_cov=include_cov,
        exclude_cov=exclude_cov,
        refit_scalers=refit_scalers,
        allow_fit_on_eval=allow_fit_on_eval,
        log_interval=log_interval,
        project_root=project_root,
        echo=True,
    )
    split_key = str(payload.get("split_used") or "all")
    metrics = RegressionMetrics(
        mae=float(payload["mae"]),
        rmse=float(payload["rmse"]),
        mard=float(payload["mard"]),
    )
    name = label or Path(payload["run_dir"]).name
    return SingleModelResult(
        model_name=name,
        run_dir=Path(payload["run_dir"]),
        kind=RunDirKind.CUSTOM_PYTORCH,
        split_results={split_key: SplitMetrics(overall=metrics)},
        model_type=str(payload.get("model_type")) if payload.get("model_type") else None,
        checkpoint=Path(payload["checkpoint"]) if payload.get("checkpoint") else None,
        test_csv=Path(payload["test_csv"]) if payload.get("test_csv") else None,
        extra={
            "windows": payload.get("windows"),
            "scaler_source": payload.get("scaler_source"),
            "active_covariates": payload.get("active_covariates"),
            "zeroed_covariates": payload.get("zeroed_covariates"),
            "zero_cov": payload.get("zero_cov"),
        },
    )
