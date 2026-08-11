"""Typer commands for inference release pack / check / publish / pull."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

import torch
import typer

from common.model_spec import get_family_spec
from common.release import (
    download_inference_bundle,
    load_inference_bundle,
    pack_run_dir,
    publish_inference_bundle,
    validate_inference_bundle,
)

app = typer.Typer(
    name="release",
    help="Pack, validate, publish, and retrieve inference release bundles (format 1.0).",
    add_completion=False,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)


def _factory_meta(config: Any) -> dict[str, Any]:
    """Rebuild training-meta-shaped dict for ModelFamilySpec.build_model."""
    meta = dict(config.architecture)
    meta["horizon"] = config.horizon
    meta.setdefault("input_steps", config.architecture.get("input_steps", 128))
    return meta


@app.command("pack")
def pack_release(
    run_dir: Annotated[Path, typer.Argument(help="Training run directory to export.")],
    out: Annotated[Path, typer.Option(help="Output bundle directory (must not exist).")],
    model_type: Annotated[
        Optional[str],
        typer.Option(help="Override family: auto-detect from meta/weights when omitted."),
    ] = None,
    release_id: Annotated[
        Optional[str],
        typer.Option(help="Optional release id (default: <kind>-<UTC timestamp>)."),
    ] = None,
    checkpoint: Annotated[
        Optional[Path],
        typer.Option(help="Optional weights file (default: best_model.pt / last_model.pt)."),
    ] = None,
    verify: Annotated[
        bool,
        typer.Option("--verify/--no-verify", help="Reload bundle weights after packing."),
    ] = True,
) -> None:
    """Pack a training run dir into a format-1.0 inference bundle."""
    if out.exists():
        typer.echo(f"Error: output bundle directory already exists: {out}", err=True)
        raise typer.Exit(1)
    try:
        manifest = pack_run_dir(
            run_dir,
            out,
            model_type=model_type,
            release_id=release_id,
            checkpoint=checkpoint,
        )
        if verify:
            load_inference_bundle(
                out,
                model_factory=lambda cfg: get_family_spec(cfg.model_type).build_model(
                    _factory_meta(cfg),
                    torch.device("cpu"),
                ),
            )
    except (OSError, ValueError, FileNotFoundError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Packed release {manifest.release_id} -> {out}")
    typer.echo(f"Model type: {manifest.config.model_type}")
    typer.echo(
        f"Features: {', '.join(manifest.config.feature_order)}; "
        f"horizon={manifest.config.horizon}; "
        f"input_steps={manifest.preprocessor.window.input_steps}"
    )


@app.command("check")
def check_release(
    bundle: Annotated[Path, typer.Argument(help="Local inference bundle directory.")],
) -> None:
    """Validate a local inference bundle (checksums + contract consistency)."""
    try:
        manifest = validate_inference_bundle(bundle)
    except (OSError, ValueError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Release bundle valid: {manifest.release_id}")


@app.command("publish")
def publish_release(
    bundle: Annotated[Path, typer.Argument(help="Local inference bundle directory.")],
    repo: Annotated[str, typer.Option(help="Hugging Face model repository (ORG/NAME).")],
    private: Annotated[
        bool, typer.Option("--private/--public", help="Create or keep the repository private.")
    ] = False,
) -> None:
    """Publish a local inference bundle to the Hugging Face Hub."""
    try:
        manifest = publish_inference_bundle(bundle, repo_id=repo, private=private)
    except (OSError, ValueError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Published inference release {manifest.release_id} to {repo}")


@app.command("pull")
def pull_release(
    repo: Annotated[str, typer.Option(help="Hugging Face model repository (ORG/NAME).")],
    out: Annotated[Path, typer.Option(help="Local directory for the downloaded bundle.")],
    revision: Annotated[
        str,
        typer.Option(help="Hub revision to download. Defaults to main."),
    ] = "main",
) -> None:
    """Download and validate an inference bundle from the Hugging Face Hub."""
    try:
        manifest = download_inference_bundle(repo, revision=revision, target_dir=out)
    except (OSError, ValueError, FileNotFoundError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Downloaded inference release {manifest.release_id} to {out}")
