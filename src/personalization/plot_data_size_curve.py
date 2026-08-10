#!/usr/bin/env python3
"""Plot personal train-days vs fine-tuned test MAE (Step 3 learning curves)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import polars as pl
import typer

from common.paths import DEFAULT_RUNS_ROOT
from personalization.sweep_utils import estimate_plateau_day

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

ALL_DAYS_X: float = 999.0

# Distinct colors for combined multi-subject chart (colorblind-friendly-ish).
SUBJECT_COLORS: dict[str, str] = {
    "livia": "#0072B2",  # blue
    "loop_556": "#D55E00",  # vermillion
    "loop_730": "#009E73",  # bluish green
    "556": "#D55E00",
    "730": "#009E73",
    "loop_154": "#CC79A7",  # reddish purple
    "154": "#CC79A7",
}

SUBJECT_DISPLAY: dict[str, str] = {
    "livia": "Livia",
    "loop_556": "User 556",
    "loop_730": "User 730",
    "556": "User 556",
    "730": "User 730",
    "loop_154": "User 154",
    "154": "User 154",
}


def _day_sort_key(day_label: str) -> float:
    if str(day_label).lower() == "all":
        return ALL_DAYS_X
    return float(day_label)


def _load_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"summary not found: {summary_csv}")
    df = pl.read_csv(summary_csv)
    return [dict(row) for row in df.iter_rows(named=True)]


def _ok_ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [
        r for r in rows if r.get("status") == "ok" and r.get("ft_test_mae") is not None
    ]
    return sorted(ok_rows, key=lambda r: _day_sort_key(str(r["personal_days"])))


def _zs_mae(ordered: list[dict[str, Any]]) -> float | None:
    zs_mae: float | None = None
    for row in reversed(ordered):
        if row.get("zs_test_mae") is not None:
            zs_mae = float(row["zs_test_mae"])
            if str(row.get("personal_days")).lower() == "all":
                break
    return zs_mae


def _color_for_subject(subject: str, index: int) -> str:
    key = subject.lower()
    if key in SUBJECT_COLORS:
        return SUBJECT_COLORS[key]
    return f"C{index % 10}"


def _display_name(subject: str) -> str:
    return SUBJECT_DISPLAY.get(subject.lower(), subject)


def plot_data_size_curve(
    rows: list[dict[str, Any]],
    *,
    out_png: Path,
    title: str = "Personal train days vs test MAE",
    subject: str | None = None,
) -> dict[str, Any]:
    ordered = _ok_ordered(rows)
    if not ordered:
        raise ValueError("No successful data-size runs in summary")

    x_vals = [_day_sort_key(str(r["personal_days"])) for r in ordered]
    ft_mae = [float(r["ft_test_mae"]) for r in ordered]
    zs_mae = _zs_mae(ordered)
    plateau_info = estimate_plateau_day(ordered)

    subj = subject or str(ordered[0].get("subject", "subject"))
    color = _color_for_subject(subj, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x_vals,
        ft_mae,
        marker="o",
        linewidth=2,
        color=color,
        label=f"{_display_name(subj)} fine-tuned",
    )
    if zs_mae is not None:
        ax.axhline(
            zs_mae,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"{_display_name(subj)} zero-shot ({zs_mae:.2f})",
        )

    optimal_day = plateau_info.get("optimal_day")
    if optimal_day is not None:
        opt_x = _day_sort_key(str(optimal_day))
        opt_mae = plateau_info.get("best_mae")
        if opt_mae is not None:
            opt_label = "all" if str(optimal_day).lower() == "all" else f"{optimal_day}d"
            ax.scatter(
                [opt_x],
                [float(opt_mae)],
                s=120,
                zorder=5,
                color=color,
                edgecolors="black",
                linewidths=0.8,
                label=f"Best ({opt_label})",
            )

    tick_labels = ["all" if v >= ALL_DAYS_X else str(int(v)) for v in x_vals]
    ax.set_xticks(x_vals, tick_labels)
    ax.set_xlabel("Personal train days")
    ax.set_ylabel("Test MAE (mg/dL)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    meta = {
        "subject": subj,
        "summary_rows": len(ordered),
        "plateau": plateau_info,
        "chart_path": str(out_png),
    }
    meta_path = out_png.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def plot_combined_data_size_curves(
    series: list[tuple[str, list[dict[str, Any]]]],
    *,
    out_png: Path,
    title: str = "Personal train days vs test MAE (combined)",
    show_zero_shot: bool = True,
) -> dict[str, Any]:
    """Overlay multiple subjects' data-size curves with distinct colors."""
    if not series:
        raise ValueError("No series to plot")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_x: set[float] = set()
    per_subject: list[dict[str, Any]] = []

    for i, (subject, rows) in enumerate(series):
        ordered = _ok_ordered(rows)
        if not ordered:
            continue
        x_vals = [_day_sort_key(str(r["personal_days"])) for r in ordered]
        ft_mae = [float(r["ft_test_mae"]) for r in ordered]
        all_x.update(x_vals)
        color = _color_for_subject(subject, i)
        name = _display_name(subject)
        ax.plot(
            x_vals,
            ft_mae,
            marker="o",
            linewidth=2.2,
            color=color,
            label=f"{name} fine-tuned",
        )
        if show_zero_shot:
            zs = _zs_mae(ordered)
            if zs is not None:
                ax.axhline(
                    zs,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.55,
                    label=f"{name} zero-shot ({zs:.2f})",
                )
        plateau = estimate_plateau_day(ordered)
        per_subject.append(
            {
                "subject": subject,
                "display": name,
                "color": color,
                "plateau": plateau,
                "n_points": len(ordered),
            }
        )

    if not all_x:
        raise ValueError("No successful rows in any series")

    ordered_x = sorted(all_x)
    tick_labels = ["all" if v >= ALL_DAYS_X else str(int(v)) for v in ordered_x]
    ax.set_xticks(ordered_x, tick_labels)
    ax.set_xlabel("Personal train days")
    ax.set_ylabel("Test MAE (mg/dL)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    meta = {
        "subjects": per_subject,
        "chart_path": str(out_png),
    }
    meta_path = out_png.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


@app.command("single")
def main_single(
    summary_csv: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "data_size" / "summary.csv",
        "--summary-csv",
    ),
    out_png: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "data_size" / "data_size_curve.png",
        "--out-png",
    ),
    title: str = typer.Option(
        "Personal train days vs test MAE",
        "--title",
    ),
    subject: Optional[str] = typer.Option(None, "--subject"),
) -> None:
    """Render one subject's data-size learning curve from ``summary.csv``."""
    rows = _load_summary_rows(summary_csv)
    subj = subject
    if subj is None and rows:
        subj = str(rows[0].get("subject", "subject"))
    meta = plot_data_size_curve(
        rows,
        out_png=out_png,
        title=title,
        subject=subj,
    )
    typer.echo(f"Wrote {out_png}")
    typer.echo(
        f"Optimal day={meta['plateau'].get('optimal_day')} "
        f"plateau_day={meta['plateau'].get('plateau_day')} "
        f"best_mae={meta['plateau'].get('best_mae')}"
    )


@app.command("combined")
def main_combined(
    summary_csv: list[Path] = typer.Option(
        ...,
        "--summary-csv",
        help="Repeatable: one summary.csv per subject.",
    ),
    subject: list[str] = typer.Option(
        ...,
        "--subject",
        help="Repeatable: subject name matching each --summary-csv (same order).",
    ),
    out_png: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "data_size_curves_combined.png",
        "--out-png",
    ),
    title: str = typer.Option(
        "Personal train days vs test MAE (combined)",
        "--title",
    ),
    show_zero_shot: bool = typer.Option(True, "--show-zero-shot/--hide-zero-shot"),
) -> None:
    """Overlay several subjects on one chart with distinct colors."""
    if len(summary_csv) != len(subject):
        raise typer.BadParameter(
            f"--summary-csv count ({len(summary_csv)}) must match "
            f"--subject count ({len(subject)})"
        )
    series: list[tuple[str, list[dict[str, Any]]]] = []
    for subj, path in zip(subject, summary_csv, strict=True):
        series.append((subj, _load_summary_rows(path)))
    meta = plot_combined_data_size_curves(
        series,
        out_png=out_png,
        title=title,
        show_zero_shot=show_zero_shot,
    )
    typer.echo(f"Wrote {out_png}")
    for entry in meta["subjects"]:
        p = entry["plateau"]
        typer.echo(
            f"  {entry['display']}: best_mae={p.get('best_mae')} "
            f"optimal_day={p.get('optimal_day')}"
        )


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    summary_csv: Optional[Path] = typer.Option(
        None,
        "--summary-csv",
        help="(legacy) single-curve mode when no subcommand is given.",
    ),
    out_png: Optional[Path] = typer.Option(None, "--out-png"),
    title: Optional[str] = typer.Option(None, "--title"),
    subject: Optional[str] = typer.Option(None, "--subject"),
) -> None:
    """Default entry: single-curve plot (backward compatible)."""
    if ctx.invoked_subcommand is not None:
        return
    csv_path = summary_csv or (
        DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "data_size" / "summary.csv"
    )
    png_path = out_png or (
        DEFAULT_RUNS_ROOT / "personalization" / "livia" / "sweeps" / "data_size" / "data_size_curve.png"
    )
    plot_title = title or "Personal train days vs test MAE"
    rows = _load_summary_rows(csv_path)
    subj = subject
    if subj is None and rows:
        subj = str(rows[0].get("subject", "subject"))
    meta = plot_data_size_curve(
        rows, out_png=png_path, title=plot_title, subject=subj
    )
    typer.echo(f"Wrote {png_path}")
    typer.echo(
        f"Optimal day={meta['plateau'].get('optimal_day')} "
        f"plateau_day={meta['plateau'].get('plateau_day')} "
        f"best_mae={meta['plateau'].get('best_mae')}"
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
