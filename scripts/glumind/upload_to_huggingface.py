#!/usr/bin/env python3
"""
upload_to_huggingface.py — Upload a GluMind model run to Hugging Face Hub.

Uploads the essential artifacts from a run directory (best_model.pt,
config.json, tuning_meta.json, metrics CSVs, training log) to a public
Hugging Face repository under a given organisation.

Example:
  uv run scripts/glumind/upload_to_huggingface.py \\
      --model-dir marked_runs/glumind/ai_ready_plus_type1/glumind_global_h12_20260226_032703 \\
      --repo-name glumind-global-h12 \\
      --org GlucoseDao \\
      --token hf_xxx
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.common.network import apply_windows_tls_workarounds

apply_windows_tls_workarounds()

import typer
from huggingface_hub import HfApi, create_repo

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

UPLOAD_FILES: list[str] = [
    "best_model.pt",
    "config.json",
    "tuning_meta.json",
    "best_info.json",
    "tuning.txt",
    "test_metrics_by_study_group.csv",
    "test_metrics_overall.csv",
    "val_metrics_by_study_group.csv",
    "val_metrics_overall.csv",
]


def _build_model_card(model_dir: Path, repo_id: str) -> str:
    config: dict = {}
    for name in ("config.json", "tuning_meta.json"):
        p = model_dir / name
        if p.exists():
            with open(p) as f:
                config = json.load(f)
            break

    best_info: dict = {}
    bi = model_dir / "best_info.json"
    if bi.exists():
        with open(bi) as f:
            best_info = json.load(f)

    val_overall = ""
    vo = model_dir / "val_metrics_overall.csv"
    if vo.exists():
        val_overall = vo.read_text().strip()

    test_overall = ""
    to_ = model_dir / "test_metrics_overall.csv"
    if to_.exists():
        test_overall = to_.read_text().strip()

    lines = [
        "---",
        "license: apache-2.0",
        "tags:",
        "  - glucose-forecasting",
        "  - time-series",
        "  - transformer",
        "  - pytorch",
        "  - glumind",
        "---",
        "",
        f"# {repo_id}",
        "",
        "GluMind is a Transformer-based glucose forecasting model that predicts future",
        "blood glucose values from CGM readings combined with heart rate and step-count",
        "features.",
        "",
        "## Model Details",
        "",
        f"| Parameter       | Value |",
        f"|-----------------|-------|",
        f"| Horizon         | {config.get('horizon', 'N/A')} steps (×5 min each) |",
        f"| Input steps     | {config.get('input_steps', 'N/A')} |",
        f"| d_model         | {config.get('d_model', 'N/A')} |",
        f"| n_heads         | {config.get('n_heads', 'N/A')} |",
        f"| n_blocks        | {config.get('n_blocks', 'N/A')} |",
        f"| ff_units        | {config.get('ff_units', 'N/A')} |",
        f"| dropout         | {config.get('dropout', 'N/A')} |",
        f"| Training mode   | {config.get('mode', 'N/A')} |",
        f"| Split scheme    | {config.get('split_scheme', 'N/A')} |",
        f"| Best epoch      | {best_info.get('epoch', 'N/A')} |",
        f"| Best val loss   | {best_info.get('val_loss', 'N/A')} |",
        "",
    ]

    if val_overall:
        lines += ["## Validation Metrics (overall)", "", "```", val_overall, "```", ""]
    if test_overall:
        lines += ["## Test Metrics (overall)", "", "```", test_overall, "```", ""]

    lines += [
        "## Usage",
        "",
        "Download the model and evaluate with:",
        "",
        "```bash",
        "uv run scripts/glumind/download_from_huggingface.py \\",
        f"    --repo-id {repo_id} \\",
        "    --output-dir test_model",
        "",
        "uv run scripts/glumind/evaluate_glumind.py \\",
        "    --run-dir test_model \\",
        "    --test-csv test_data/livia_glumind_ready.csv",
        "```",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `best_model.pt` | Best model weights (PyTorch state dict) |",
        "| `config.json` | Full training configuration |",
        "| `tuning_meta.json` | Training metadata including dataset stats |",
        "| `best_info.json` | Best epoch and validation loss |",
        "| `tuning.txt` | Training log |",
        "| `*_metrics_*.csv` | Validation and test metrics breakdowns |",
    ]

    return "\n".join(lines)


@app.command()
def main(
    model_dir: Path = typer.Option(
        ...,
        "--model-dir",
        help="Path to the run directory (contains best_model.pt and config.json).",
    ),
    repo_name: str = typer.Option(
        ...,
        "--repo-name",
        help="Repository name on Hugging Face (e.g. glumind-global-h12).",
    ),
    org: str = typer.Option(
        ...,
        "--org",
        help="Hugging Face organisation name.",
    ),
    token: str = typer.Option(
        ...,
        "--token",
        help="Hugging Face access token.",
    ),
    private: bool = typer.Option(
        False,
        "--private/--public",
        help="Whether the repository should be private (default: public).",
    ),
) -> None:
    if not model_dir.exists():
        typer.echo(f"Error: model-dir does not exist: {model_dir}", err=True)
        raise typer.Exit(1)

    repo_id = f"{org}/{repo_name}"
    typer.echo(f"Repository : {repo_id}")
    typer.echo(f"Model dir  : {model_dir.resolve()}")

    api = HfApi(token=token)

    typer.echo("Creating / verifying repository...")
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        exist_ok=True,
        private=private,
    )
    typer.echo(f"  Repository ready: https://huggingface.co/{repo_id}")

    typer.echo("Uploading model card (README.md)...")
    readme_content = _build_model_card(model_dir, repo_id)
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )

    for filename in UPLOAD_FILES:
        src = model_dir / filename
        if not src.exists():
            typer.echo(f"  Skipping {filename} (not found)")
            continue
        size_kb = src.stat().st_size / 1024
        typer.echo(f"  Uploading {filename} ({size_kb:.1f} KB)...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {filename}",
        )

    typer.echo("")
    typer.echo(f"Done! Model published at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    app()
