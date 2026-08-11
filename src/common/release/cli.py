"""Typer commands for inference release check / publish / pull."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from common.release import (
    download_inference_bundle,
    publish_inference_bundle,
    validate_inference_bundle,
)

app = typer.Typer(
    name="release",
    help="Validate, publish, and retrieve inference release bundles (format 1.0).",
    add_completion=False,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
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
