#!/usr/bin/env python3
"""
download_from_huggingface.py — Download a GluMind model from Hugging Face Hub.

Downloads all model artifacts from a Hugging Face repository into a local
directory that is compatible with evaluate_glumind.py (--run-dir).

Example:
  uv run scripts/glumind/download_from_huggingface.py \\
      --repo-id GlucoseDao/glumind-global-h12 \\
      --output-dir test_model

Then evaluate with:
  uv run scripts/glumind/evaluate_glumind.py \\
      --run-dir test_model \\
      --test-csv test_data/livia_glumind_ready.csv
"""
from __future__ import annotations

from pathlib import Path

from scripts.common.network import apply_windows_tls_workarounds

apply_windows_tls_workarounds()

import typer
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

REQUIRED_FILES: list[str] = [
    "best_model.pt",
    "config.json",
]


@app.command()
def main(
    repo_id: str = typer.Option(
        ...,
        "--repo-id",
        help="Hugging Face repository ID (e.g. GlucoseDao/glumind-global-h12).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Local directory to download model files into.",
    ),
    token: str = typer.Option(
        "",
        "--token",
        help="Hugging Face access token (required for private repos).",
    ),
    revision: str = typer.Option(
        "main",
        "--revision",
        help="Branch, tag, or commit hash to download from.",
    ),
) -> None:
    typer.echo(f"Repository : {repo_id}")
    typer.echo(f"Output dir : {output_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    token_arg: str | None = token if token else None

    api = HfApi(token=token_arg)

    typer.echo("Fetching file list from Hub...")
    remote_files = list(list_repo_files(repo_id=repo_id, repo_type="model", token=token_arg, revision=revision))
    typer.echo(f"  Found {len(remote_files)} file(s) in the repository.")

    skip_prefixes = ("checkpoints/",)
    skip_exact = {"README.md"}

    files_to_download = [
        f for f in remote_files
        if f not in skip_exact and not any(f.startswith(p) for p in skip_prefixes)
    ]

    typer.echo(f"  Downloading {len(files_to_download)} file(s) (skipping checkpoints and README)...")
    typer.echo("")

    for filename in files_to_download:
        dest = output_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        typer.echo(f"  Downloading {filename}...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token_arg,
            revision=revision,
            local_dir=str(output_dir),
        )
        typer.echo(f"    -> {local_path}")

    typer.echo("")
    for req in REQUIRED_FILES:
        p = output_dir / req
        if not p.exists():
            typer.echo(f"Warning: expected file not found after download: {req}", err=True)

    typer.echo(f"Download complete. Files saved to: {output_dir.resolve()}")
    typer.echo("")
    typer.echo("Evaluate with:")
    typer.echo(
        f"  uv run scripts/glumind/evaluate_glumind.py \\\n"
        f"      --run-dir {output_dir} \\\n"
        f"      --test-csv test_data/livia_glumind_ready.csv"
    )


if __name__ == "__main__":
    app()
