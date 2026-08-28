"""Build docs/PERSONALIZATION_NF_REPORT.md from on-disk continue-fit runs."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from common.console import safe_echo
from personalization.cohort import (
    COHORT_JOINED2_TEST,
    PHASE4_SUBJECTS,
    Phase4Subject,
    joined2_test_subjects,
    original_cohort_subjects,
)
from personalization.plots import plot_combined_data_size_curves, plot_data_size_curve
from personalization.splits import load_train_span_days
from personalization_nf.constants import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_NF_PERSONALIZATION_ROOT,
    DEFAULT_REPORT_PATH,
)
from personalization_nf.discover import NfHoldoutRun
from personalization_nf.sweep import data_size_run_dir, row_from_disk


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _enrich_span(rows: list[dict[str, Any]], personal_csv: Path) -> list[dict[str, Any]]:
    span = load_train_span_days(personal_csv)
    if span is None:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("train_span_days") in (None, ""):
            item["train_span_days"] = span
        if item.get("used_train_days") in (None, ""):
            label = str(item.get("personal_days", "all"))
            if label.lower() == "all":
                item["used_train_days"] = span
            else:
                try:
                    item["used_train_days"] = min(float(label), span)
                except ValueError:
                    item["used_train_days"] = span
        out.append(item)
    return out


def _ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def collect_model_series(
    root: Path,
    model_key: str,
) -> list[tuple[Phase4Subject, list[dict[str, Any]]]]:
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]] = []
    for spec in PHASE4_SUBJECTS:
        out_dir = root / spec.subject / model_key
        summary = out_dir / "summary.csv"
        rows: list[dict[str, Any]] = []
        if summary.is_file():
            rows = [dict(row) for row in pl.read_csv(summary).iter_rows(named=True)]
        else:
            for label in ("1", "3", "7", "14", "30", "60", "all"):
                run_dir = data_size_run_dir(out_dir, spec.subject, label)
                disk_row = row_from_disk(
                    run_dir, subject=spec.subject, model_key=model_key, label=label
                )
                if disk_row is not None:
                    rows.append(disk_row)
        rows = _ok_rows(_enrich_span(rows, spec.csv))
        if rows:
            series.append((spec, rows))
    return series


def _days_table(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: float("inf")
        if str(row.get("personal_days")).lower() == "all"
        else float(row.get("personal_days", 0)),
    )
    lines = [
        "| Days | Used train days | ZS MAE | FT MAE | Δ FT vs ZS |",
        "|------|-----------------|--------|--------|------------|",
    ]
    for row in ordered:
        zs = row.get("zs_test_mae")
        ft = row.get("ft_test_mae")
        delta = None
        if zs is not None and ft is not None:
            delta = float(ft) - float(zs)
        used = row.get("used_train_days")
        label = str(row.get("personal_days"))
        if label.lower() == "all" and used is not None:
            label = f"all ({float(used):.0f}d)"
        lines.append(
            f"| {label} | {_fmt(used, 1)} | {_fmt(zs)} | {_fmt(ft)} | {_fmt(delta)} |"
        )
    return "\n".join(lines)


def _full_train_table(series: list[tuple[Phase4Subject, list[dict[str, Any]]]]) -> str:
    lines = [
        "| Subject | Cohort | Study group | Train span (d) | ZS MAE | FT MAE (all) | Δ vs ZS |",
        "|---------|--------|-------------|----------------|--------|--------------|---------|",
    ]
    for spec, rows in series:
        all_row = next(
            (row for row in rows if str(row.get("personal_days")).lower() == "all"),
            None,
        )
        if all_row is None:
            continue
        zs = all_row.get("zs_test_mae")
        ft = all_row.get("ft_test_mae")
        delta = None
        if zs is not None and ft is not None:
            delta = float(ft) - float(zs)
        span = all_row.get("train_span_days") or all_row.get("used_train_days")
        if spec.cohort == COHORT_JOINED2_TEST:
            cohort = "joined2 test"
        elif spec.cohort == "livia":
            cohort = "Livia"
        else:
            cohort = "Loop holdout"
        lines.append(
            f"| {spec.display} | {cohort} | {spec.study_group} | {_fmt(span, 1)} | "
            f"{_fmt(zs)} | {_fmt(ft)} | {_fmt(delta)} |"
        )
    return "\n".join(lines)


def _coverage_table() -> str:
    lines = [
        "| Subject | Source | Study group | Notes |",
        "|---------|--------|-------------|-------|",
        "| **Livia** | Personal CGM/pump export | T1DM | Longest history (~345 d train) |",
    ]
    for spec in original_cohort_subjects():
        if spec.cohort == "livia":
            continue
        extra = "60-day budget ≈ full train" if spec.user_id == "1082" else ""
        lines.append(
            f"| **{spec.display}** | Loop quality holdout | {spec.study_group} | {extra} |"
        )
    for spec in joined2_test_subjects():
        lines.append(
            f"| **{spec.display}** | joined2 test | {spec.study_group} | "
            "AI-READY CGM; insulin/carbs absent (zero-filled) |"
        )
    return "\n".join(lines)


def _mean_improvement(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    *,
    day_label: str,
    min_train_days: float,
) -> tuple[float, int] | None:
    deltas: list[float] = []
    for spec, rows in series:
        if spec.cohort == COHORT_JOINED2_TEST:
            continue
        all_row = next(
            (row for row in rows if str(row.get("personal_days")).lower() == "all"),
            None,
        )
        span = None
        if all_row is not None:
            span = all_row.get("train_span_days") or all_row.get("used_train_days")
        try:
            if span is None or float(span) < min_train_days:
                continue
        except (TypeError, ValueError):
            continue
        match = next(
            (row for row in rows if str(row.get("personal_days")) == day_label),
            None,
        )
        if match is None:
            continue
        zs = match.get("zs_test_mae")
        ft = match.get("ft_test_mae")
        if zs is None or ft is None:
            continue
        deltas.append(float(ft) - float(zs))
    if not deltas:
        return None
    return sum(deltas) / len(deltas), len(deltas)


def _rel_figure(path: Path, report_path: Path) -> str:
    try:
        return path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def _write_charts(
    *,
    model_key: str,
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    figures_dir: Path,
    run_root: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    original_names = {spec.subject for spec in original_cohort_subjects()}
    joined_names = {spec.subject for spec in joined2_test_subjects()}

    for spec, rows in series:
        png = figures_dir / f"{model_key}_{spec.subject}_data_size.png"
        try:
            plot_data_size_curve(
                rows,
                out_png=png,
                title=f"{spec.display} / {model_key} — personal train days vs test MAE",
                subject=spec.subject,
                mode="max_days",
                max_days=60.0,
            )
            paths[f"{spec.subject}_60d"] = png
            sweep_copy = run_root / spec.subject / model_key / "data_size_curve.png"
            _copy(png, sweep_copy)
        except ValueError as exc:
            safe_echo(f"Skip 60-day chart for {model_key}/{spec.subject}: {exc}", err=True)

    def _combined(
        name: str,
        filename: str,
        title: str,
        wanted: set[str],
        *,
        dummy_all: bool,
        show_zero_shot: bool,
    ) -> None:
        subset = [(spec.subject, rows) for spec, rows in series if spec.subject in wanted]
        if not subset:
            return
        png = figures_dir / filename
        plot_combined_data_size_curves(
            subset,
            out_png=png,
            title=title,
            show_zero_shot=show_zero_shot,
            mode="dummy_all" if dummy_all else "max_days",
            max_days=60.0,
        )
        paths[name] = png
        _copy(png, run_root / filename)

    _combined(
        "combined_all",
        f"{model_key}_data_size_curves_combined.png",
        f"{model_key}: Livia + Loop holdouts (All = full train)",
        original_names,
        dummy_all=True,
        show_zero_shot=True,
    )
    _combined(
        "combined_60d",
        f"{model_key}_data_size_curves_combined_60d.png",
        f"{model_key}: Livia + Loop holdouts (first 60 days)",
        original_names,
        dummy_all=False,
        show_zero_shot=True,
    )
    _combined(
        "combined_joined2_all",
        f"{model_key}_data_size_curves_combined_joined2.png",
        f"{model_key}: Joined2 AI-READY test users (All = full train)",
        joined_names,
        dummy_all=True,
        show_zero_shot=False,
    )
    return paths


def _per_user_sections(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    *,
    wanted: set[str],
    model_key: str,
    figures: dict[str, Path],
    report_path: Path,
) -> str:
    blocks: list[str] = []
    for spec, rows in series:
        if spec.subject not in wanted:
            continue
        fig = figures.get(f"{spec.subject}_60d")
        fig_line = ""
        if fig is not None:
            fig_line = f"\n\n![{spec.display} {model_key} data-size curve]({_rel_figure(fig, report_path)})\n"
        blocks.append(
            f"#### {spec.display}\n\n{_days_table(rows)}{fig_line}"
        )
    return "\n\n".join(blocks)


def write_personalization_nf_report(
    *,
    root: Path = DEFAULT_NF_PERSONALIZATION_ROOT,
    holdouts: list[NfHoldoutRun],
    report_path: Path = DEFAULT_REPORT_PATH,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    status: dict[str, Any] | None = None,
) -> Path:
    """Render the NeuralForecast personalization report from completed runs."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d")
    model_blocks: list[str] = []
    exec_rows: list[str] = []

    for holdout in holdouts:
        series = collect_model_series(root, holdout.model_key)
        figures = _write_charts(
            model_key=holdout.model_key,
            series=series,
            figures_dir=figures_dir,
            run_root=root,
        )
        n_subjects = len(series)
        mean_30 = _mean_improvement(series, day_label="30", min_train_days=60.0)
        mean_60 = _mean_improvement(series, day_label="60", min_train_days=60.0)
        mean_all = _mean_improvement(series, day_label="all", min_train_days=60.0)

        def _mean_cell(pair: tuple[float, int] | None) -> str:
            if pair is None:
                return "—"
            mean_delta, n = pair
            return f"{-mean_delta:.2f} (n={n})"

        exec_rows.append(
            f"| {holdout.model_key} | {n_subjects}/15 | {_fmt(holdout.val_mae)} | "
            f"{_mean_cell(mean_30)} | {_mean_cell(mean_60)} | {_mean_cell(mean_all)} |"
        )

        original_names = {spec.subject for spec in original_cohort_subjects()}
        joined_names = {spec.subject for spec in joined2_test_subjects()}
        combined = figures.get("combined_all")
        combined_60 = figures.get("combined_60d")
        combined_j = figures.get("combined_joined2_all")
        combined_md = ""
        if combined is not None:
            combined_md += (
                f"\n\n![Holdouts combined]({_rel_figure(combined, report_path)})\n"
            )
        if combined_60 is not None:
            combined_md += (
                f"\n![Holdouts 60 days]({_rel_figure(combined_60, report_path)})\n"
            )
        if combined_j is not None:
            combined_md += (
                f"\n![Joined2 combined]({_rel_figure(combined_j, report_path)})\n"
            )

        mean_table_lines = [
            "| Train budget | Mean Δ vs ZS | Mean MAE improvement | n |",
            "|--------------|--------------|----------------------|---|",
        ]
        for label, pair in (
            ("30 days", mean_30),
            ("60 days", mean_60),
            ("Full train (≥60 d)", mean_all),
        ):
            if pair is None:
                mean_table_lines.append(f"| {label} | — | — | — |")
                continue
            mean_delta, n = pair
            mean_table_lines.append(
                f"| {label} | {_fmt(mean_delta)} | {_fmt(-mean_delta)} | {n} |"
            )

        model_blocks.append(
            "\n".join(
                [
                    f"## {holdout.model_key}",
                    "",
                    f"Global holdout run: `{holdout.run_dir.as_posix()}`. "
                    f"Source val MAE **{_fmt(holdout.val_mae)}** mg/dL "
                    "(joined2 global test, not the personal chronological test).",
                    "",
                    "### Full train, continue-fit from global weights",
                    "",
                    _full_train_table(series),
                    "",
                    "### Livia and Loop quality holdouts",
                    "",
                    _per_user_sections(
                        series,
                        wanted=original_names,
                        model_key=holdout.model_key,
                        figures=figures,
                        report_path=report_path,
                    ),
                    combined_md,
                    "",
                    "### Average MAE improvement by train budget",
                    "",
                    "Mean test-MAE reduction versus zero-shot on T1DM users with at least "
                    "60 train days (Livia + Loop holdouts except User 1082 when the "
                    "budget exceeds their span). Negative Δ is better than frozen global.",
                    "",
                    "\n".join(mean_table_lines),
                    "",
                    "### Joined2 test — two users per study group",
                    "",
                    _per_user_sections(
                        series,
                        wanted=joined_names,
                        model_key=holdout.model_key,
                        figures=figures,
                        report_path=report_path,
                    ),
                    "",
                ]
            )
        )

    current = (status or {}).get("current") or "in progress"
    n_jobs = len((status or {}).get("jobs") or [])
    body = "\n".join(
        [
            "# NeuralForecast personalization — zero-shot vs continue-fit",
            "",
            f"**Date:** {generated}  ",
            "**Base models:** `data/output/runs/nf_holdout/__ALL__/`  ",
            "**Personal data:** `data/input/personalization/` (same chronological CSVs as SugarOne)  ",
            "**Horizon:** 12 steps (60 minutes at 5-minute sampling)  ",
            "**CLI:** `personal-nf-study`  ",
            f"**Status:** {current} ({n_jobs} subject×model jobs recorded)",
            "",
            "This report is the NeuralForecast counterpart of "
            "[PERSONALIZATION_REPORT.md](PERSONALIZATION_REPORT.md). "
            "There is no Learning without Forgetting and no learning-rate search. "
            "Personalization is **continue-fit** from the saved global bundle "
            "(`NeuralForecast.fit(..., use_init_models=False)`): each day budget "
            "starts from the same `nf_holdout` weights, trains on that person's "
            "chronological train slice, and is scored on the frozen personal test split.",
            "",
            "MAE is reported in mg/dL. **Δ vs zero-shot** is continue-fit MAE minus "
            "frozen-bundle MAE (negative means personalization improved on the global model).",
            "",
            "## 1. Executive summary",
            "",
            "| Model | Subjects with runs | Global val MAE | Mean MAE gain at 30 d | 60 d | Full train |",
            "|-------|--------------------|----------------|-----------------------|------|------------|",
            "\n".join(exec_rows) if exec_rows else "| — | — | — | — | — | — |",
            "",
            "**Locked recipe:** continue-fit from the global bundle, same learning rate "
            "and `max_steps` as the source holdout run, train-tail early stopping "
            "(patience 10) when the day budget still leaves one input+horizon window, "
            "sugarone-compatible dense 128/12 evaluation. A day budget only shortens "
            "**train**. Val and test never change.",
            "",
            "Mixing a few personal days into the original joined2 training CSV and "
            "retraining from scratch was rejected: that corpus is ~12 million rows, so "
            "1–60 days of one user would not move the fit, and it would cost a full "
            "global retrain per subject×day×model. Continue-fit on personal data is "
            "the transfer-learning analogue of SugarOne fine-tuning.",
            "",
            "## 2. Subjects and data coverage",
            "",
            "Same 15-person cohort as the SugarOne study. Each personal CSV already has "
            "the chronological split (last 25% test / 15% of remainder val / rest train).",
            "",
            _coverage_table(),
            "",
            "## 3. Design choices",
            "",
            "### 3.1 Continue-fit, not mix-and-retrain",
            "",
            "1. **Zero-shot.** Load `neuralforecast/` from the global holdout run and "
            "score the person's chronological test windows (`cross_validation`, "
            "`use_fitted=True`, stride 1).",
            "2. **Continue-fit.** Call `fit` on the day-limited personal train split "
            "with `use_init_models=False`, so Lightning keeps the loaded weights. "
            "Every day budget reloads the global bundle (independent, not sequential).",
            "3. **Early stopping.** NeuralForecast `val_df` requires equal-length "
            "series; personal CSVs have many `sequence_id`s of different lengths. "
            "ES therefore uses a train-tail `val_size` (≤20% of the shortest series, "
            "and never large enough to remove the last input+horizon window). "
            "The chronological val split is only used for reporting.",
            "",
            "### 3.2 No LwF, no LR grid",
            "",
            "Source holdout runs already used `learning_rate=1e-3` and `max_steps=400`. "
            "Personalization keeps those values. Short histories that overfit are a "
            "result, not something we hide with extra knobs.",
            "",
            "\n".join(model_blocks),
            "## Reproducibility and artifacts",
            "",
            "```bash",
            "uv run personal-nf-study --device auto",
            "uv run personal-nf-study --report-only",
            "```",
            "",
            "| Artifact | Path |",
            "|----------|------|",
            f"| Study root | `{root.as_posix()}` |",
            f"| This report | `{report_path.as_posix()}` |",
            f"| Figures | `{figures_dir.as_posix()}` |",
            f"| Status | `{(root / 'study_status.md').as_posix()}` |",
            "",
            "*Results from on-disk NeuralForecast personalization runs.*",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")
    safe_echo(f"Wrote {report_path}")
    return report_path
