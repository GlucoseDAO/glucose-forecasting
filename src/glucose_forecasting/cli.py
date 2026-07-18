"""Top-level command line interface for glucose forecasting."""

from __future__ import annotations

import csv
from importlib.metadata import version
import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from glucose_forecasting.config import (
    DatasetSpec,
    ModelSelection,
    load_evaluation_config,
)
from glucose_forecasting.evaluation import run_evaluation
from glucose_forecasting.models.registry import (
    ModelArtifact,
    ModelRegistry,
    ModelResolutionError,
    load_registry,
    resolve_data_path,
)
from glucose_forecasting.release import (
    download_inference_bundle,
    publish_inference_bundle,
    validate_inference_bundle,
)

app = typer.Typer(
    add_completion=False,
    help="Train, evaluate, and publish glucose forecasting models.",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Inspect and resolve registered model artifacts.")
config_app = typer.Typer(help="Validate modern workflow configuration files.")
data_app = typer.Typer(help="Resolve project data paths.")
release_app = typer.Typer(help="Validate, publish, and retrieve inference releases.")

_DEFAULT_REGISTRY = Path("data/catalog/model-registry.json")
_PROFILE_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "loop": ("loop-v1", ("basal", "bolus", "carbohydrates")),
    "ai-readi": ("ai-readi-v1", ("heart_rate", "steps")),
}

app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(release_app, name="release")


@app.callback()
def main() -> None:
    """Train, evaluate, and publish glucose forecasting models."""


@app.command()
def info() -> None:
    """Print the installed glucose-forecasting version."""
    typer.echo(version("glucose-forecasting"))


@app.command("evaluate")
def evaluate(
    data: Annotated[
        list[str],
        typer.Option(help="CSV data file(s); repeat or separate values with commas."),
    ],
    models: Annotated[
        list[str],
        typer.Option(help="Model selector(s), NAME or NAME@VERSION; repeat or comma-separate."),
    ] = [],
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Immutable output directory; defaults to data/output/runs/<UTC timestamp>."),
    ] = None,
) -> None:
    """Plan a multi-model evaluation and persist long-form result records."""
    registry_path = _registry_path(registry)
    model_registry = _load_registry_or_error(registry)
    try:
        evaluation_run = run_evaluation(
            data=data,
            models=models,
            registry=model_registry,
            registry_path=registry_path,
            project_root=Path.cwd(),
            output_dir=out,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Evaluation records written to {evaluation_run.output_dir}")


def _registry_path(registry: Path | None) -> Path:
    """Return the supplied registry or the documented project-local default."""
    return registry if registry is not None else Path.cwd() / _DEFAULT_REGISTRY


def _load_registry_or_error(registry: Path | None) -> ModelRegistry:
    """Load a registry while providing command-line actionable failures."""
    registry_path = _registry_path(registry)
    if not registry_path.is_file():
        raise typer.BadParameter(
            f"registry not found at {registry_path}; pass --registry FILE or add "
            "data/catalog/model-registry.json",
            param_hint="--registry",
        )
    try:
        return load_registry(registry_path)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise typer.BadParameter(
            f"registry at {registry_path} is invalid: {error}",
            param_hint="--registry",
        ) from error


def _artifact_json(artifact: ModelArtifact) -> str:
    """Render an artifact in a stable, machine-readable form."""
    return json.dumps(artifact.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True)


def _infer_dataset_spec(data_path: Path, profile: str | None) -> DatasetSpec:
    """Build a dataset contract from a CSV header or an explicit known profile."""
    if not data_path.is_file():
        raise typer.BadParameter(f"data file not found at {data_path}", param_hint="--data")

    normalized_profile = profile.lower() if profile is not None else None
    if normalized_profile is not None:
        profile_spec = _PROFILE_SPECS.get(normalized_profile)
        if profile_spec is None:
            known_profiles = ", ".join(sorted(_PROFILE_SPECS))
            raise typer.BadParameter(
                f"unknown profile {profile!r}; choose one of: {known_profiles}",
                param_hint="--profile",
            )
        schema, covariates = profile_spec
    else:
        with data_path.open(encoding="utf-8", newline="") as data_file:
            headers = set(csv.reader(data_file).__next__())
        if {"Basal Rate (U/h)", "Bolus Insulin (U)", "Carbohydrates (g)"} & headers:
            schema, covariates = _PROFILE_SPECS["loop"]
        elif {"Heart Rate", "Step Count"} & headers:
            schema, covariates = _PROFILE_SPECS["ai-readi"]
        else:
            raise typer.BadParameter(
                "could not infer a dataset profile from CSV columns; pass "
                "--profile loop or --profile ai-readi",
                param_hint="--profile",
            )

    return DatasetSpec(
        name=data_path.stem,
        path=data_path,
        data_schema=schema,
        covariates=covariates,
        cadence_minutes=5,
        horizon_steps=12,
    )


@models_app.command("list")
def list_models(
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
) -> None:
    """List all registered artifacts."""
    model_registry = _load_registry_or_error(registry)
    artifacts = sorted(model_registry.models, key=lambda artifact: (artifact.name, artifact.version))
    typer.echo(
        json.dumps(
            [artifact.model_dump(mode="json", by_alias=True) for artifact in artifacts],
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("show")
def show_model(
    name: str,
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
) -> None:
    """Show every registered version of NAME."""
    model_registry = _load_registry_or_error(registry)
    artifacts = sorted(
        (artifact for artifact in model_registry.models if artifact.name == name),
        key=lambda artifact: artifact.version,
    )
    if not artifacts:
        raise typer.BadParameter(f"model {name!r} is not registered", param_hint="NAME")
    typer.echo(
        json.dumps(
            [artifact.model_dump(mode="json", by_alias=True) for artifact in artifacts],
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("resolve")
def resolve_model(
    data: Annotated[Path, typer.Option(help="CSV data file to match against the registry.")],
    model: Annotated[str | None, typer.Option(help="Optional model name.")] = None,
    profile: Annotated[
        str | None,
        typer.Option(help="Dataset profile: loop or ai-readi. Inferred from CSV when omitted."),
    ] = None,
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
) -> None:
    """Resolve the best compatible registered artifact for a CSV."""
    model_registry = _load_registry_or_error(registry)
    data_path = resolve_data_path(data, Path.cwd())
    dataset = _infer_dataset_spec(data_path, profile)
    try:
        artifact = model_registry.resolve(dataset, ModelSelection(name=model))
    except ModelResolutionError as error:
        raise typer.BadParameter(str(error), param_hint="--model") from error
    typer.echo(_artifact_json(artifact))


@config_app.command("check")
def check_config(
    config: Annotated[Path, typer.Option(help="Versioned evaluation configuration YAML file.")],
) -> None:
    """Validate an evaluation configuration file."""
    if not config.is_file():
        raise typer.BadParameter(f"config file not found at {config}", param_hint="--config")
    try:
        evaluation_config = load_evaluation_config(config)
    except (OSError, ValidationError, ValueError) as error:
        raise typer.BadParameter(
            f"config at {config} is invalid: {error}",
            param_hint="--config",
        ) from error
    typer.echo(f"Configuration valid: {evaluation_config.dataset.name}")


@data_app.command("path")
def data_path(name: str) -> None:
    """Show NAME resolved relative to the project's data/input directory."""
    typer.echo(resolve_data_path(name, Path.cwd()))


@release_app.command("check")
def check_release(
    bundle: Annotated[Path, typer.Argument(help="Local inference bundle directory.")],
) -> None:
    """Validate a local inference bundle."""
    manifest = validate_inference_bundle(bundle)
    typer.echo(f"Release bundle valid: {manifest.release_id}")


@release_app.command("publish")
def publish_release(
    bundle: Annotated[Path, typer.Argument(help="Local inference bundle directory.")],
    repo: Annotated[str, typer.Option(help="Hugging Face model repository (ORG/NAME).")],
    private: Annotated[bool, typer.Option(help="Create or keep the repository private.")] = False,
) -> None:
    """Publish a local inference bundle to the Hugging Face Hub."""
    manifest = publish_inference_bundle(bundle, repo_id=repo, private=private)
    typer.echo(f"Published inference release {manifest.release_id} to {repo}")


@release_app.command("pull")
def pull_release(
    repo: Annotated[str, typer.Option(help="Hugging Face model repository (ORG/NAME).")],
    out: Annotated[Path, typer.Option(help="Local directory for the downloaded bundle.")],
    revision: Annotated[
        str,
        typer.Option(help="Hub revision to download. Defaults to main."),
    ] = "main",
) -> None:
    """Download and validate an inference bundle from the Hugging Face Hub."""
    manifest = download_inference_bundle(repo, revision=revision, target_dir=out)
    typer.echo(f"Downloaded inference release {manifest.release_id} to {out}")
