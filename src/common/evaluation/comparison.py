#!/usr/bin/env python3
"""Write multi-run comparison summaries and optional charts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from common.evaluation.types import SingleModelResult


def _write_comparison_plots(results: list[SingleModelResult], output_dir: Path) -> list[Path]:
    """Write bar charts for MAE / RMSE / MARD. Returns created file paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, float, float, float]] = []
    for result in results:
        primary = result.primary_overall()
        if primary is None:
            continue
        rows.append((result.model_name, primary.mae, primary.rmse, primary.mard))
    if not rows:
        return []

    names = [r[0] for r in rows]
    mae = [r[1] for r in rows]
    rmse = [r[2] for r in rows]
    mard = [r[3] for r in rows]
    x = list(range(len(names)))
    created: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    metrics = [
        (axes[0], mae, "MAE (mg/dL)", "tab:blue"),
        (axes[1], rmse, "RMSE (mg/dL)", "tab:orange"),
        (axes[2], mard, "MARD (%)", "tab:green"),
    ]
    for ax, values, title, color in metrics:
        bars = ax.bar(x, values, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Model comparison")
    fig.tight_layout()
    panel_path = output_dir / "metrics_comparison.png"
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created.append(panel_path)

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.0))
    bars = ax2.barh(names, mae, color="tab:blue", alpha=0.85)
    ax2.set_xlabel("MAE (mg/dL)")
    ax2.set_title("MAE comparison")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, mae, strict=True):
        ax2.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2.0,
            f" {value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    fig2.tight_layout()
    mae_path = output_dir / "mae_comparison.png"
    fig2.savefig(mae_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    created.append(mae_path)
    return created


def write_comparison_report(
    results: list[SingleModelResult],
    output_dir: Path,
    *,
    plot: bool = False,
) -> Path:
    """Write CSV + JSON comparison artifacts under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "metrics_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "run_dir",
                "kind",
                "model_type",
                "split",
                "mae",
                "rmse",
                "mard",
            ],
        )
        writer.writeheader()
        for result in results:
            for split, split_metrics in result.split_results.items():
                writer.writerow(
                    {
                        "model_name": result.model_name,
                        "run_dir": str(result.run_dir),
                        "kind": result.kind.value,
                        "model_type": result.model_type or "",
                        "split": split,
                        "mae": split_metrics.overall.mae,
                        "rmse": split_metrics.overall.rmse,
                        "mard": split_metrics.overall.mard,
                    }
                )

    plot_paths: list[str] = []
    if plot:
        plot_paths = [str(p) for p in _write_comparison_plots(results, output_dir)]

    manifest = []
    for result in results:
        primary = result.primary_overall()
        manifest.append(
            {
                "model_name": result.model_name,
                "run_dir": str(result.run_dir),
                "kind": result.kind.value,
                "model_type": result.model_type,
                "checkpoint": str(result.checkpoint) if result.checkpoint else None,
                "test_csv": str(result.test_csv) if result.test_csv else None,
                "splits": {
                    split: metrics.overall.as_dict()
                    for split, metrics in result.split_results.items()
                },
                "primary": primary.as_dict() if primary else None,
                "extra": result.extra,
            }
        )
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"results": manifest, "plots": plot_paths}, f, indent=2)

    return output_dir
