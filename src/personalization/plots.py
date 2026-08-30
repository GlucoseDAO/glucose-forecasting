#!/usr/bin/env python3
"""Plot personal train-days vs fine-tuned test MAE (Step 3 learning curves)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import polars as pl
import typer

from common.paths import DEFAULT_RUNS_ROOT
from personalization.sweep_utils import estimate_plateau_day

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# Dummy x for the combined "All" tick (sits just after the 60-day grid).
ALL_DUMMY_X: float = 90.0
DEFAULT_MAX_DAYS: float = 60.0

PlotMode = Literal["max_days", "dummy_all"]

# Distinct colors for combined multi-subject chart (Wong + extras).
SUBJECT_COLORS: dict[str, str] = {
    "subject_p1": "#0072B2",
    "subject_p1_indep": "#0072B2",
    "subject_p1_curr_plain": "#E69F00",
    "subject_p1_curr_lwf": "#009E73",
    "subject_p1_lwf_decay": "#E69F00",
    "subject_p1_lwf_01": "#009E73",
    "loop_154": "#CC79A7",
    "loop_154_indep": "#CC79A7",
    "loop_154_lwf_decay": "#E69F00",
    "loop_154_lwf_01": "#009E73",
    "154": "#CC79A7",
    "loop_556": "#D55E00",
    "556": "#D55E00",
    "loop_730": "#009E73",
    "730": "#009E73",
    "loop_1017": "#E69F00",
    "1017": "#E69F00",
    "loop_1029": "#56B4E9",
    "1029": "#56B4E9",
    "loop_1082": "#000000",
    "1082": "#000000",
    "ai_ready_1030": "#4daf4a",
    "ai_ready_1043": "#a6d854",
    "ai_ready_1034": "#ff7f00",
    "ai_ready_1049": "#fdbf6f",
    "ai_ready_1019": "#984ea3",
    "ai_ready_1127": "#c994c7",
    "ai_ready_1413": "#8c510a",
    "ai_ready_1036": "#bf812d",
}


def _display_name(subject: str) -> str:
    from personalization.cohort import display_name_for

    return display_name_for(subject)


def _load_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"summary not found: {summary_csv}")
    df = pl.read_csv(summary_csv)
    return [dict(row) for row in df.iter_rows(named=True)]


def _ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in rows if r.get("status") == "ok" and r.get("ft_test_mae") is not None
    ]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_all_label(day_label: Any) -> bool:
    return str(day_label).strip().lower() == "all"


def _train_span_days(row: dict[str, Any]) -> float | None:
    for key in ("train_span_days", "used_train_days"):
        span = _as_float(row.get(key))
        if span is not None and span > 0:
            return span
    return None


def _point_x(
    row: dict[str, Any],
    *,
    mode: PlotMode,
    max_days: float,
) -> float | None:
    """Return x position for a summary row, or None to drop it from this chart."""
    if _is_all_label(row.get("personal_days")):
        span = _train_span_days(row)
        if mode == "dummy_all":
            return ALL_DUMMY_X
        if span is None:
            return None
        if span > max_days:
            return None
        return span
    x = _as_float(row.get("personal_days"))
    if x is None:
        return None
    if mode == "max_days" and x > max_days:
        return None
    if mode == "dummy_all" and x > max_days:
        return None
    return x


def _ordered_points(
    rows: list[dict[str, Any]],
    *,
    mode: PlotMode,
    max_days: float,
) -> list[tuple[float, dict[str, Any]]]:
    points: list[tuple[float, dict[str, Any]]] = []
    for row in _ok_rows(rows):
        x = _point_x(row, mode=mode, max_days=max_days)
        if x is None:
            continue
        points.append((x, row))
    points.sort(key=lambda item: item[0])
    return points


def _zs_mae(rows: list[dict[str, Any]]) -> float | None:
    ordered = _ok_rows(rows)
    zs_mae: float | None = None
    for row in reversed(ordered):
        if row.get("zs_test_mae") is not None:
            zs_mae = float(row["zs_test_mae"])
            if _is_all_label(row.get("personal_days")):
                break
    return zs_mae


def _color_for_subject(subject: str, index: int) -> str:
    key = subject.lower()
    if key in SUBJECT_COLORS:
        return SUBJECT_COLORS[key]
    return f"C{index % 10}"


def display_subject_name(subject: str) -> str:
    return _display_name(subject)


def _tick_label(x: float, *, mode: PlotMode) -> str:
    if mode == "dummy_all" and x >= ALL_DUMMY_X - 1e-6:
        return "All"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.0f}"


def _apply_xticks(
    ax: Any,
    x_vals: list[float],
    *,
    mode: PlotMode,
    max_days: float,
) -> None:
    if mode == "dummy_all":
        wanted = [1.0, 3.0, 7.0, 14.0, 30.0, 60.0]
        ticks = [t for t in wanted if any(abs(v - t) < 1e-6 for v in x_vals)]
        if any(v >= ALL_DUMMY_X - 1e-6 for v in x_vals):
            ticks.append(ALL_DUMMY_X)
        ax.set_xticks(ticks, [_tick_label(t, mode=mode) for t in ticks])
        ax.set_xlim(0, ALL_DUMMY_X + 8)
        return
    ticks = sorted(set(round(v, 6) for v in x_vals))
    ax.set_xticks(ticks, [_tick_label(t, mode=mode) for t in ticks])
    ax.set_xlim(0, max_days + 4)


def plot_data_size_curve(
    rows: list[dict[str, Any]],
    *,
    out_png: Path,
    title: str = "Personal train days vs test MAE",
    subject: str | None = None,
    mode: PlotMode = "max_days",
    max_days: float = DEFAULT_MAX_DAYS,
) -> dict[str, Any]:
    points = _ordered_points(rows, mode=mode, max_days=max_days)
    if not points:
        raise ValueError("No successful data-size runs in summary for this chart")

    x_vals = [p[0] for p in points]
    ft_mae = [float(p[1]["ft_test_mae"]) for p in points]
    zs_mae = _zs_mae(rows)
    plateau_info = estimate_plateau_day([p[1] for p in points])

    subj = subject or str(points[0][1].get("subject", "subject"))
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
        opt_row = next(
            (p[1] for p in points if str(p[1].get("personal_days")) == str(optimal_day)),
            None,
        )
        opt_mae = plateau_info.get("best_mae")
        if opt_row is not None and opt_mae is not None:
            opt_x = _point_x(opt_row, mode=mode, max_days=max_days)
            if opt_x is not None:
                if _is_all_label(optimal_day):
                    span = _train_span_days(opt_row)
                    opt_label = f"all ({span:.0f}d)" if span is not None else "all"
                else:
                    opt_label = f"{optimal_day}d"
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

    _apply_xticks(ax, x_vals, mode=mode, max_days=max_days)
    xlabel = (
        "Personal train days (All = full train split)"
        if mode == "dummy_all"
        else f"Personal train days (first {int(max_days)} days)"
    )
    ax.set_xlabel(xlabel)
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
        "summary_rows": len(points),
        "mode": mode,
        "max_days": max_days,
        "plateau": plateau_info,
        "chart_path": str(out_png),
        "x_values": x_vals,
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
    mode: PlotMode = "max_days",
    max_days: float = DEFAULT_MAX_DAYS,
) -> dict[str, Any]:
    """Overlay multiple subjects' data-size curves with distinct colors."""
    if not series:
        raise ValueError("No series to plot")

    fig, ax = plt.subplots(figsize=(10, 6))
    all_x: list[float] = []
    per_subject: list[dict[str, Any]] = []

    for i, (subject, rows) in enumerate(series):
        points = _ordered_points(rows, mode=mode, max_days=max_days)
        if not points:
            continue
        x_vals = [p[0] for p in points]
        ft_mae = [float(p[1]["ft_test_mae"]) for p in points]
        all_x.extend(x_vals)
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
            zs = _zs_mae(rows)
            if zs is not None:
                ax.axhline(
                    zs,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.55,
                    label=f"{name} zero-shot ({zs:.2f})",
                )
        plateau = estimate_plateau_day([p[1] for p in points])
        per_subject.append(
            {
                "subject": subject,
                "display": name,
                "color": color,
                "plateau": plateau,
                "n_points": len(points),
                "x_values": x_vals,
            }
        )

    if not all_x:
        raise ValueError("No successful rows in any series")

    _apply_xticks(ax, all_x, mode=mode, max_days=max_days)
    xlabel = (
        "Personal train days (All = full train split)"
        if mode == "dummy_all"
        else f"Personal train days (first {int(max_days)} days)"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test MAE (mg/dL)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    meta = {
        "subjects": per_subject,
        "mode": mode,
        "max_days": max_days,
        "chart_path": str(out_png),
    }
    meta_path = out_png.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def plot_curriculum_mae_and_lambda(
    series: list[tuple[str, list[dict[str, Any]]]],
    *,
    out_png: Path,
    title: str = "Subject P1 curricula: test MAE and LwF lambda",
) -> dict[str, Any]:
    """Two-panel chart: fine-tuned test MAE (top) and lwf_lambda at each step (bottom)."""
    if not series:
        raise ValueError("No series to plot")

    fig, (ax_mae, ax_lwf) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    all_x: list[float] = []
    zs_vals: list[float] = []

    for i, (subject, rows) in enumerate(series):
        points = _ordered_points(rows, mode="dummy_all", max_days=DEFAULT_MAX_DAYS)
        if not points:
            continue
        x_vals = [p[0] for p in points]
        ft_mae = [float(p[1]["ft_test_mae"]) for p in points]
        lambdas = [_as_float(p[1].get("lwf_lambda")) or 0.0 for p in points]
        all_x.extend(x_vals)
        color = _color_for_subject(subject, i)
        name = _display_name(subject)
        ax_mae.plot(
            x_vals,
            ft_mae,
            marker="o",
            linewidth=2.2,
            color=color,
            label=name,
        )
        ax_lwf.plot(
            x_vals,
            lambdas,
            marker="s",
            linewidth=2.0,
            color=color,
            label=name,
        )
        zs = _zs_mae(rows)
        if zs is not None:
            zs_vals.append(zs)

    if not all_x:
        raise ValueError("No successful rows in any series")
    if zs_vals:
        ax_mae.axhline(
            zs_vals[-1],
            color="#888888",
            linestyle="--",
            linewidth=1.4,
            label=f"Zero-shot ({zs_vals[-1]:.2f})",
        )

    _apply_xticks(ax_mae, all_x, mode="dummy_all", max_days=DEFAULT_MAX_DAYS)
    _apply_xticks(ax_lwf, all_x, mode="dummy_all", max_days=DEFAULT_MAX_DAYS)
    ax_mae.set_ylabel("Test MAE (mg/dL)")
    ax_mae.set_title(title)
    ax_mae.grid(True, alpha=0.3)
    ax_mae.legend(loc="best", fontsize=8)
    ax_lwf.set_xlabel("Personal train days (All = full train split)")
    ax_lwf.set_ylabel("lwf_lambda")
    ax_lwf.set_ylim(-0.02, 0.42)
    ax_lwf.grid(True, alpha=0.3)
    ax_lwf.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    meta = {"chart_path": str(out_png), "mode": "dummy_all"}
    out_png.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _plot_single_from_csv(
    summary_csv: Path,
    out_png: Path,
    title: str,
    subject: str | None,
    mode: PlotMode,
    max_days: float,
) -> dict[str, Any]:
    rows = _load_summary_rows(summary_csv)
    subj = subject
    if subj is None and rows:
        subj = str(rows[0].get("subject", "subject"))
    return plot_data_size_curve(
        rows,
        out_png=out_png,
        title=title,
        subject=subj,
        mode=mode,
        max_days=max_days,
    )


@app.command("single")
def main_single(
    summary_csv: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "subject_p1" / "sweeps" / "data_size" / "summary.csv",
        "--summary-csv",
    ),
    out_png: Path = typer.Option(
        DEFAULT_RUNS_ROOT / "personalization" / "subject_p1" / "sweeps" / "data_size" / "data_size_curve.png",
        "--out-png",
    ),
    title: str = typer.Option(
        "Personal train days vs test MAE",
        "--title",
    ),
    subject: Optional[str] = typer.Option(None, "--subject"),
    max_days: float = typer.Option(DEFAULT_MAX_DAYS, "--max-days"),
    dummy_all: bool = typer.Option(
        False,
        "--dummy-all/--no-dummy-all",
        help="Plot full-train as a dummy All tick after 60 days.",
    ),
) -> None:
    """Render one subject's data-size learning curve from ``summary.csv``."""
    mode: PlotMode = "dummy_all" if dummy_all else "max_days"
    meta = _plot_single_from_csv(summary_csv, out_png, title, subject, mode, max_days)
    typer.echo(f"Wrote {out_png}")
    typer.echo(
        f"mode={mode} optimal_day={meta['plateau'].get('optimal_day')} "
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
        DEFAULT_RUNS_ROOT / "personalization" / "data_size_curves_combined_60d.png",
        "--out-png",
    ),
    title: str = typer.Option(
        "Personal train days vs test MAE (combined)",
        "--title",
    ),
    show_zero_shot: bool = typer.Option(True, "--show-zero-shot/--hide-zero-shot"),
    max_days: float = typer.Option(DEFAULT_MAX_DAYS, "--max-days"),
    dummy_all: bool = typer.Option(
        False,
        "--dummy-all/--no-dummy-all",
        help="Plot full-train as a dummy All tick so every user shares a last point.",
    ),
) -> None:
    """Overlay several subjects on one chart with distinct colors."""
    if len(summary_csv) != len(subject):
        raise typer.BadParameter(
            f"--summary-csv count ({len(summary_csv)}) must match "
            f"--subject count ({len(subject)})"
        )
    mode: PlotMode = "dummy_all" if dummy_all else "max_days"
    series: list[tuple[str, list[dict[str, Any]]]] = []
    for subj, path in zip(subject, summary_csv, strict=True):
        series.append((subj, _load_summary_rows(path)))
    meta = plot_combined_data_size_curves(
        series,
        out_png=out_png,
        title=title,
        show_zero_shot=show_zero_shot,
        mode=mode,
        max_days=max_days,
    )
    typer.echo(f"Wrote {out_png} mode={mode}")
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
    max_days: float = typer.Option(DEFAULT_MAX_DAYS, "--max-days"),
    dummy_all: bool = typer.Option(False, "--dummy-all/--no-dummy-all"),
) -> None:
    """Default entry: single-curve plot (backward compatible)."""
    if ctx.invoked_subcommand is not None:
        return
    csv_path = summary_csv or (
        DEFAULT_RUNS_ROOT / "personalization" / "subject_p1" / "sweeps" / "data_size" / "summary.csv"
    )
    png_path = out_png or (
        DEFAULT_RUNS_ROOT / "personalization" / "subject_p1" / "sweeps" / "data_size" / "data_size_curve.png"
    )
    plot_title = title or "Personal train days vs test MAE"
    mode: PlotMode = "dummy_all" if dummy_all else "max_days"
    meta = _plot_single_from_csv(csv_path, png_path, plot_title, subject, mode, max_days)
    typer.echo(f"Wrote {png_path}")
    typer.echo(
        f"mode={mode} optimal_day={meta['plateau'].get('optimal_day')} "
        f"best_mae={meta['plateau'].get('best_mae')}"
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
