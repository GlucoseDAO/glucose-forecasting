"""Smoke tests for the installable src-package command line interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from typer.testing import CliRunner

from glucose_forecasting.cli import app
from glucose_forecasting.config import DatasetSpec, EvaluationConfig, save_evaluation_config
from glucose_forecasting.models.registry import (
    ModelArtifact,
    ModelRegistry,
    save_registry,
)
from glucose_forecasting.release import (
    EvaluationProtocol,
    InferenceConfig,
    MetricsSpec,
    PreprocessorSpec,
    ProvenanceSpec,
    ReleaseManifest,
    SelectionMetric,
    WindowSpec,
    write_inference_bundle,
)

runner = CliRunner()


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def _release_manifest() -> ReleaseManifest:
    return ReleaseManifest(
        release_id="sugar-one-2026-07",
        config=InferenceConfig(
            model_id="sugar-one",
            model_type="sugar_one",
            architecture={"d_model": 128},
            feature_order=("glucose", "basal_rate"),
            horizon=12,
            cadence=5,
        ),
        preprocessor=PreprocessorSpec(window=WindowSpec(input_steps=72)),
        metrics=MetricsSpec(
            selection_metric=SelectionMetric(name="mae", direction="minimize"),
            validation={"mae": 18.2},
            test={"mae": 19.1},
            protocol=EvaluationProtocol(name="held-out evaluation", split="test"),
        ),
        provenance=ProvenanceSpec(
            git_sha="abc1234",
            lock_hash="def5678",
            env={"python": "3.12"},
            dataset_fingerprint="sha256:dataset",
            seed=42,
        ),
    )


def test_glucose_cli_help() -> None:
    """The modern root command must be available after package installation."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Train, evaluate, and publish glucose forecasting models." in result.output
    assert "info" in result.output
    assert "release" in result.output


def _registry(path: Path) -> Path:
    """Create a minimal real registry for CLI tests."""
    return save_registry(
        ModelRegistry(
            models=(
                ModelArtifact(
                    name="sugar-one",
                    version="1.0.0",
                    artifact_path="models/sugar-one.pt",
                    schema="loop-v1",
                    covariates=("basal", "bolus", "carbohydrates"),
                    validation_metric=8.5,
                ),
            )
        ),
        path,
    )


def test_models_commands_read_registry_and_resolve_csv(tmp_path: Path) -> None:
    """Registry-backed inspection commands return actual registered metadata."""
    registry_path = _registry(tmp_path / "registry.json")
    data_path = tmp_path / "loop.csv"
    data_path.write_text(
        "Glucose (mg/dL),Basal Rate (U/h),Bolus Insulin (U),Carbohydrates (g)\n"
        "120,1.0,0.0,30\n",
        encoding="utf-8",
    )

    listed = runner.invoke(app, ["models", "list", "--registry", str(registry_path)])
    shown = runner.invoke(
        app, ["models", "show", "sugar-one", "--registry", str(registry_path)]
    )
    resolved = runner.invoke(
        app,
        [
            "models",
            "resolve",
            "--data",
            str(data_path),
            "--registry",
            str(registry_path),
        ],
    )

    assert listed.exit_code == 0, listed.output
    assert shown.exit_code == 0, shown.output
    assert resolved.exit_code == 0, resolved.output
    assert '"name": "sugar-one"' in listed.output
    assert '"version": "1.0.0"' in shown.output
    assert '"artifact_path": "models/sugar-one.pt"' in resolved.output


def test_config_check_and_data_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config validation and bare data-name resolution use their real inputs."""
    config_path = tmp_path / "evaluation.json"
    save_evaluation_config(
        EvaluationConfig(
            dataset=DatasetSpec(
                name="loop",
                path="loop.csv",
                data_schema="loop-v1",
                covariates=("basal", "bolus", "carbohydrates"),
                cadence_minutes=5,
                horizon_steps=12,
            )
        ),
        config_path,
    )
    monkeypatch.chdir(tmp_path)

    checked = runner.invoke(app, ["config", "check", "--config", str(config_path)])
    resolved = runner.invoke(app, ["data", "path", "loop.csv"])

    assert checked.exit_code == 0, checked.output
    assert "Configuration valid: loop" in checked.output
    assert resolved.exit_code == 0, resolved.output
    assert resolved.output.strip() == str(tmp_path / "data" / "input" / "loop.csv")


def test_models_commands_explain_missing_registry(tmp_path: Path) -> None:
    """A missing default registry is reported as a useful CLI error."""
    result = runner.invoke(app, ["models", "list"], catch_exceptions=False)

    assert result.exit_code != 0
    assert "registry not found" in result.output


def test_release_check_validates_a_real_local_bundle(tmp_path: Path) -> None:
    """The release check command validates an actual inference bundle."""
    bundle_dir = tmp_path / "bundle"
    write_inference_bundle(bundle_dir, manifest=_release_manifest(), model=_TinyModel())

    result = runner.invoke(app, ["release", "check", str(bundle_dir)])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "Release bundle valid: sugar-one-2026-07"


def test_release_publish_calls_release_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release publish command forwards the bundle and repository options."""
    publish = Mock(return_value=_release_manifest())
    monkeypatch.setattr("glucose_forecasting.cli.publish_inference_bundle", publish)
    bundle_dir = tmp_path / "bundle"

    result = runner.invoke(
        app,
        ["release", "publish", str(bundle_dir), "--repo", "GlucoseDAO/sugar-one", "--private"],
    )

    assert result.exit_code == 0, result.output
    publish.assert_called_once_with(
        bundle_dir,
        repo_id="GlucoseDAO/sugar-one",
        private=True,
    )
    assert result.output.strip() == (
        "Published inference release sugar-one-2026-07 to GlucoseDAO/sugar-one"
    )


def test_release_pull_calls_release_api_with_default_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release pull command downloads main unless a revision is supplied."""
    download = Mock(return_value=_release_manifest())
    monkeypatch.setattr("glucose_forecasting.cli.download_inference_bundle", download)
    output_dir = tmp_path / "bundle"

    result = runner.invoke(
        app,
        ["release", "pull", "--repo", "GlucoseDAO/sugar-one", "--out", str(output_dir)],
    )

    assert result.exit_code == 0, result.output
    download.assert_called_once_with(
        "GlucoseDAO/sugar-one",
        revision="main",
        target_dir=output_dir,
    )
    assert result.output.strip() == (
        f"Downloaded inference release sugar-one-2026-07 to {output_dir}"
    )
