#!/usr/bin/env python3
"""One-shot builder for docs/PERSONALIZATION_JEPA_REPORT.md.

Not a console script. Run with:

    uv run python src/sugar_jepa/jepa_report.py

Source is the colleague extract ``temp_docs/jepa_mae_by_days.csv`` (SugarOne plus
SugarJEPA2 encoder variants on the same 7 T1DM personal splits). Same table
layout and Δ convention as PERSONALIZATION_REPORT.md / PERSONALIZATION_NF_REPORT.md.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
import typer

from common.console import init_cli_console, safe_echo
from personalization.cohort import (
    Phase4Subject,
    display_name_for,
    original_cohort_subjects,
)
from personalization.plots import plot_combined_data_size_curves, plot_data_size_curve
from personalization.splits import load_train_span_days

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DEFAULT_CSV: Final[Path] = Path("temp_docs") / "jepa_mae_by_days.csv"
DEFAULT_REPORT_PATH: Final[Path] = Path("docs") / "PERSONALIZATION_JEPA_REPORT.md"
DEFAULT_FIGURES_DIR: Final[Path] = Path("docs") / "figures" / "personalization_jepa"

DAY_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("d1", "1"),
    ("d3", "3"),
    ("d7", "7"),
    ("d14", "14"),
    ("d30", "30"),
    ("d60", "60"),
    ("all", "all"),
)

# Train spans from the same chronological CSVs as Milestone 8 (used when
# split_meta.json is not on disk). User 1082 has no 60-day budget.
FALLBACK_TRAIN_SPAN_DAYS: Final[dict[str, float]] = {
    "subject_p1": 344.6,
    "loop_154": 213.6,
    "loop_556": 90.9,
    "loop_730": 84.6,
    "loop_1017": 96.7,
    "loop_1029": 136.0,
    "loop_1082": 37.4,
}

MIN_TRAIN_DAYS_FOR_MEAN: Final[float] = 60.0


@dataclass(frozen=True)
class StudySpec:
    key: str
    display: str
    window_steps: int | None
    window_label: str
    embed_dim: int | None
    notes: str


STUDIES: Final[tuple[StudySpec, ...]] = (
    StudySpec(
        key="sugarone",
        display="SugarOne",
        window_steps=128,
        window_label="128 (10.7 h backbone)",
        embed_dim=None,
        notes="No JEPA branch. Same 7 T1DM people as the SugarJEPA curves.",
    ),
    StudySpec(
        key="jepa128-64",
        display="SugarJEPA-128-64",
        window_steps=128,
        window_label="128 (10.7 h)",
        embed_dim=64,
        notes="Matched SugarOne lookback; 64-d encoder.",
    ),
    StudySpec(
        key="jepa128",
        display="SugarJEPA-128",
        window_steps=128,
        window_label="128 (10.7 h)",
        embed_dim=96,
        notes="Matched SugarOne lookback; 96-d encoder.",
    ),
    StudySpec(
        key="jepa288",
        display="SugarJEPA-288",
        window_steps=288,
        window_label="288 (1 d)",
        embed_dim=96,
        notes="Hero encoder. 1-day train cannot form a window.",
    ),
    StudySpec(
        key="jepa864",
        display="SugarJEPA-864",
        window_steps=864,
        window_label="864 (3 d)",
        embed_dim=96,
        notes="Sparse day budgets; some users only have zero-shot and full train.",
    ),
    StudySpec(
        key="jepa2016",
        display="SugarJEPA-2016",
        window_steps=2016,
        window_label="2016 (7 d)",
        embed_dim=96,
        notes="5/7 subjects. Full fine-tune can raise MAE (negative control).",
    ),
)

STUDY_BY_KEY: Final[dict[str, StudySpec]] = {spec.key: spec for spec in STUDIES}

SUBJECT_ORDER: Final[tuple[str, ...]] = tuple(
    spec.subject for spec in original_cohort_subjects()
)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _parse_mae(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _subject_spec(subject: str) -> Phase4Subject:
    for spec in original_cohort_subjects():
        if spec.subject == subject:
            return spec
    raise KeyError(f"Unknown personalization subject: {subject}")


def train_span_days(subject: str) -> float | None:
    spec = _subject_spec(subject)
    span = load_train_span_days(spec.csv)
    if span is not None:
        return span
    return FALLBACK_TRAIN_SPAN_DAYS.get(subject)


def _used_train_days(label: str, span: float | None) -> float | None:
    if span is None:
        return None
    if label.lower() == "all":
        return span
    return min(float(label), span)


def _cohort_label(spec: Phase4Subject) -> str:
    if spec.subject == "subject_p1":
        return "Subject P1"
    return "Loop holdout"


def load_jepa_mae_table(csv_path: Path) -> pl.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"JEPA MAE table not found: {csv_path}")
    df = pl.read_csv(csv_path, infer_schema_length=100)
    needed = {"subject", "study", "zero_shot_mae"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return df.filter(
        pl.col("subject").cast(pl.Utf8).str.to_lowercase() != "mean"
    )


def rows_for_subject_study(
    table: pl.DataFrame,
    *,
    subject: str,
    study: str,
) -> list[dict[str, Any]]:
    match = table.filter(
        (pl.col("subject") == subject) & (pl.col("study") == study)
    )
    if match.height == 0:
        return []
    raw = match.row(0, named=True)
    zs = _parse_mae(raw.get("zero_shot_mae"))
    if zs is None:
        return []
    span = train_span_days(subject)
    rows: list[dict[str, Any]] = []
    for column, label in DAY_COLUMNS:
        ft = _parse_mae(raw.get(column))
        if ft is None:
            continue
        rows.append(
            {
                "status": "ok",
                "subject": subject,
                "study": study,
                "personal_days": label,
                "zs_test_mae": zs,
                "ft_test_mae": ft,
                "train_span_days": span,
                "used_train_days": _used_train_days(label, span),
            }
        )
    return rows


def collect_study_series(
    table: pl.DataFrame,
    study: str,
) -> list[tuple[Phase4Subject, list[dict[str, Any]]]]:
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]] = []
    for subject in SUBJECT_ORDER:
        rows = rows_for_subject_study(table, subject=subject, study=study)
        if not rows:
            continue
        series.append((_subject_spec(subject), rows))
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


def _full_train_table(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
) -> str:
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
        lines.append(
            f"| {spec.display} | {_cohort_label(spec)} | {spec.study_group} | "
            f"{_fmt(span, 1)} | {_fmt(zs)} | {_fmt(ft)} | {_fmt(delta)} |"
        )
    return "\n".join(lines)


def mean_improvement(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    *,
    day_label: str,
    min_train_days: float = MIN_TRAIN_DAYS_FOR_MEAN,
) -> tuple[float, int] | None:
    deltas: list[float] = []
    for spec, rows in series:
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


def mean_zero_shot(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
) -> tuple[float, int] | None:
    values: list[float] = []
    for _spec, rows in series:
        if not rows:
            continue
        zs = rows[0].get("zs_test_mae")
        if zs is None:
            continue
        values.append(float(zs))
    if not values:
        return None
    return sum(values) / len(values), len(values)


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
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
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
        except ValueError as exc:
            safe_echo(f"Skip 60-day chart for {model_key}/{spec.subject}: {exc}", err=True)

    wanted = {spec.subject for spec, _rows in series}

    def _combined(
        name: str,
        filename: str,
        title: str,
        *,
        dummy_all: bool,
    ) -> None:
        subset = [(spec.subject, rows) for spec, rows in series if spec.subject in wanted]
        if not subset:
            return
        png = figures_dir / filename
        try:
            plot_combined_data_size_curves(
                subset,
                out_png=png,
                title=title,
                show_zero_shot=True,
                mode="dummy_all" if dummy_all else "max_days",
                max_days=60.0,
            )
        except ValueError as exc:
            safe_echo(f"Skip combined chart {filename}: {exc}", err=True)
            return
        paths[name] = png

    _combined(
        "combined_all",
        f"{model_key}_data_size_curves_combined.png",
        f"{model_key}: Subject P1 + Loop holdouts (All = full train)",
        dummy_all=True,
    )
    _combined(
        "combined_60d",
        f"{model_key}_data_size_curves_combined_60d.png",
        f"{model_key}: Subject P1 + Loop holdouts (first 60 days)",
        dummy_all=False,
    )
    return paths


def _per_user_sections(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    *,
    model_key: str,
    figures: dict[str, Path],
    report_path: Path,
) -> str:
    blocks: list[str] = []
    for spec, rows in series:
        fig = figures.get(f"{spec.subject}_60d")
        fig_line = ""
        if fig is not None:
            fig_line = (
                f"\n\n![{spec.display} {model_key} data-size curve]"
                f"({_rel_figure(fig, report_path)})\n"
            )
        blocks.append(f"#### {spec.display}\n\n{_days_table(rows)}{fig_line}")
    return "\n\n".join(blocks)


def _mean_table(
    mean_30: tuple[float, int] | None,
    mean_60: tuple[float, int] | None,
    mean_all: tuple[float, int] | None,
) -> str:
    lines = [
        "| Train budget | Mean Δ vs ZS | Mean MAE improvement | n |",
        "|--------------|--------------|----------------------|---|",
    ]
    for label, pair in (
        ("30 days", mean_30),
        ("60 days", mean_60),
        ("Full train (≥60 d)", mean_all),
    ):
        if pair is None:
            lines.append(f"| {label} | — | — | — |")
            continue
        mean_delta, n = pair
        lines.append(
            f"| {label} | {_fmt(mean_delta)} | {_fmt(-mean_delta)} | {n} |"
        )
    return "\n".join(lines)


def _jepa288_vs_sugarone_30d(
    table: pl.DataFrame,
) -> list[tuple[str, float, float, float]]:
    """Frozen SugarJEPA-288 vs SugarOne fine-tuned at 30 days, per user."""
    out: list[tuple[str, float, float, float]] = []
    for subject in SUBJECT_ORDER:
        jepa_rows = rows_for_subject_study(table, subject=subject, study="jepa288")
        sugar_rows = rows_for_subject_study(table, subject=subject, study="sugarone")
        if not jepa_rows or not sugar_rows:
            continue
        zs = jepa_rows[0].get("zs_test_mae")
        sugar_30 = next(
            (row for row in sugar_rows if str(row.get("personal_days")) == "30"),
            None,
        )
        if zs is None or sugar_30 is None:
            continue
        sugar_mae = sugar_30.get("ft_test_mae")
        if sugar_mae is None:
            continue
        margin = float(sugar_mae) - float(zs)
        out.append((display_name_for(subject), float(zs), float(sugar_mae), margin))
    return out


def write_jepa_personalization_report(
    *,
    csv_path: Path = DEFAULT_CSV,
    report_path: Path = DEFAULT_REPORT_PATH,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    plot: bool = True,
) -> Path:
    table = load_jepa_mae_table(csv_path)
    generated = datetime.now().strftime("%Y-%m-%d")
    if plot:
        figures_dir.mkdir(parents=True, exist_ok=True)

    exec_rows: list[str] = []
    model_blocks: list[str] = []
    n_subject_runs = 0

    for study in STUDIES:
        series = collect_study_series(table, study.key)
        n_subject_runs += len(series)
        figures: dict[str, Path] = {}
        if plot:
            figures = _write_charts(
                model_key=study.key,
                series=series,
                figures_dir=figures_dir,
            )
        mean_zs = mean_zero_shot(series)
        mean_30 = mean_improvement(series, day_label="30")
        mean_60 = mean_improvement(series, day_label="60")
        mean_all = mean_improvement(series, day_label="all")

        def _gain_cell(pair: tuple[float, int] | None) -> str:
            if pair is None:
                return "—"
            mean_delta, n = pair
            return f"{-mean_delta:.2f} (n={n})"

        zs_cell = "—" if mean_zs is None else f"{mean_zs[0]:.2f} (n={mean_zs[1]})"
        exec_rows.append(
            f"| {study.display} | {len(series)}/7 | {study.window_label} | "
            f"{zs_cell} | {_gain_cell(mean_30)} | {_gain_cell(mean_60)} | "
            f"{_gain_cell(mean_all)} |"
        )

        combined = figures.get("combined_all")
        combined_60 = figures.get("combined_60d")
        combined_md = ""
        if combined is not None:
            combined_md += (
                f"\n\n![Holdouts combined]({_rel_figure(combined, report_path)})\n"
            )
        if combined_60 is not None:
            combined_md += (
                f"\n![Holdouts 60 days]({_rel_figure(combined_60, report_path)})\n"
            )

        dim_line = (
            f"Embed dim **{study.embed_dim}**. "
            if study.embed_dim is not None
            else ""
        )
        model_blocks.append(
            "\n".join(
                [
                    f"## {study.display}",
                    "",
                    f"Encoder window **{study.window_label}**. {dim_line}{study.notes}",
                    "",
                    "### Full train, independent fine-tune from global weights",
                    "",
                    _full_train_table(series),
                    "",
                    "### Subject P1 and Loop quality holdouts",
                    "",
                    _per_user_sections(
                        series,
                        model_key=study.key,
                        figures=figures,
                        report_path=report_path,
                    ),
                    combined_md,
                    "",
                    "### Average MAE improvement by train budget",
                    "",
                    "Mean test-MAE reduction versus zero-shot on T1DM users with at least "
                    "60 train days (Subject P1 + Loop holdouts except User 1082 when the "
                    "budget exceeds their span, and except users with no run at that "
                    "budget). Negative Δ is better than frozen global. Empty cells are "
                    "not filled with zeros.",
                    "",
                    _mean_table(mean_30, mean_60, mean_all),
                    "",
                ]
            )
        )

    vs_30 = _jepa288_vs_sugarone_30d(table)
    vs_lines = [
        "| User | JEPA-288 zero-shot | SugarOne @ 30 d | Margin (mg/dL) |",
        "|------|--------------------|-----------------|----------------|",
    ]
    for name, zs, sugar, margin in vs_30:
        vs_lines.append(
            f"| {name} | {_fmt(zs)} | {_fmt(sugar)} | {_fmt(margin)} |"
        )
    all_seven = len(vs_30) == 7 and all(margin > 0 for _n, _z, _s, margin in vs_30)
    vs_note = (
        "Frozen SugarJEPA-288 has lower personal-test MAE than SugarOne fine-tuned "
        "for 30 days, for **all 7 T1DM users in this study**."
        if all_seven
        else "Frozen SugarJEPA-288 vs SugarOne at 30 days (personal test)."
    )

    coverage_lines = [
        "| Subject | Source | Study group | Notes |",
        "|---------|--------|-------------|-------|",
        "| **Subject P1** | Personal CGM/pump export | T1DM | Longest history (~345 d train) |",
    ]
    for spec in original_cohort_subjects():
        if spec.subject == "subject_p1":
            continue
        extra = "60-day budget ≈ full train; no 60-day cell" if spec.user_id == "1082" else ""
        coverage_lines.append(
            f"| **{spec.display}** | Loop quality holdout | {spec.study_group} | {extra} |"
        )

    body = "\n".join(
        [
            "# SugarJEPA personalization — zero-shot vs day-budget fine-tune",
            "",
            f"**Date:** {generated}  ",
            f"**Source table:** `{csv_path.as_posix()}`  ",
            "**Personal data:** `data/input/personalization/` (same chronological CSVs as SugarOne)  ",
            "**Horizon:** 12 steps (60 minutes at 5-minute sampling)  ",
            "**Fine-tune CLI:** `personal-*` with `--base-run-dir` on a `sugar_jepa2` checkpoint  ",
            f"**Status:** {n_subject_runs} subject×encoder rows in the source table",
            "",
            "This report is the SugarJEPA counterpart of "
            "[PERSONALIZATION_REPORT.md](PERSONALIZATION_REPORT.md) and "
            "[PERSONALIZATION_NF_REPORT.md](PERSONALIZATION_NF_REPORT.md). "
            "Each day budget is an **independent** fine-tune from that encoder's "
            "global checkpoint (not a curriculum). A day budget only shortens "
            "**train**. Val and test never change. Scalers stay the global "
            "`scalers.json`.",
            "",
            "MAE is reported in mg/dL. **Δ vs zero-shot** is fine-tuned MAE minus "
            "frozen-checkpoint MAE (negative means personalization improved on the "
            "global model).",
            "",
            "SugarOne numbers in this file come from the **same extract** as the "
            "SugarJEPA runs so the seven people and splits match. They follow the "
            "Milestone 8 protocol but are not identical to "
            "[PERSONALIZATION_REPORT.md](PERSONALIZATION_REPORT.md) (that write-up "
            "also covers 8 AI-READY users). Use this file for JEPA vs SugarOne on "
            "these seven T1DM users.",
            "",
            "## 1. Executive summary",
            "",
            "| Model | Subjects | Encoder window | Mean ZS MAE | Mean MAE gain at 30 d | 60 d | Full train |",
            "|-------|----------|----------------|-------------|-----------------------|------|------------|",
            "\n".join(exec_rows),
            "",
            "**Locked recipe:** plain fine-tune (`lwf_lambda=0`) from the matching "
            "global `sugar_jepa2` (or SugarOne) checkpoint, `weight_decay=3e-5`, "
            "`train_window_stride=6`, `precision=bf16`, chronological split "
            "(last 25% test / 15% of remainder val / rest train), base-run scalers. "
            "The JEPA encoder's LR keeps the base run's `jepa_lr / lr` ratio unless "
            "`--jepa-lr` is set. A day budget only shortens **train**.",
            "",
            "Empty cells are missing runs, not zeros. `jepa-288` has no 1-day "
            "fine-tune: lookback is already one day of CGM, so a 1-day train slice "
            "cannot build an input window. Longer encoders drop more short budgets "
            "(and, for `jepa-2016`, two of the seven people).",
            "",
            vs_note,
            "",
            "\n".join(vs_lines),
            "",
            "The same “all 7” statement is **false** against SugarOne's *full* "
            "fine-tune (Subject P1 and User 1017: full SugarOne beats frozen JEPA-288). "
            "30 days is the cutoff that holds for every user in this study.",
            "",
            "## 2. Subjects and data coverage",
            "",
            "Same 7 T1DM people as the SugarOne holdout cohort in "
            "[PERSONALIZATION_REPORT.md](PERSONALIZATION_REPORT.md). The 8 "
            "joined2 AI-READY users are **not** in this table: multi-day JEPA "
            "windows need long contiguous CGM, and those exports are ~6–9 train "
            "days.",
            "",
            "\n".join(coverage_lines),
            "",
            "## 3. Design choices",
            "",
            "### 3.1 Independent fine-tune, same as SugarOne",
            "",
            "1. **Zero-shot.** Load the global checkpoint and score the person's "
            "frozen chronological test windows (stride 1).",
            "2. **Fine-tune.** Each day budget reloads the same global weights and "
            "trains on that person's day-limited train split. Not sequential.",
            "3. **Windows.** SugarOne and `jepa-128*` use a 128-step backbone "
            "lookback. SugarJEPA2 lookback is `max(input_steps, jepa_window)`.",
            "",
            "Production personalization CLIs (`personal-finetune`, "
            "`personal-sweep-days`) already resolve `sugar_jepa2` from the base "
            "run's `tuning_meta.json`. Point `--base-run-dir` at the encoder you "
            "want. See `docs/PERSONALIZATION.md`.",
            "",
            "### 3.2 Why some day cells are empty",
            "",
            "A training window needs `lookback + horizon` contiguous rows. At "
            "`jepa_window=288` that is 300 steps (~25 h), so a **1-day** train "
            "budget cannot yield a window. At 864 / 2016 the 3-day and 7-day "
            "budgets drop in the same way. User 1082 has no 60-day cell on any "
            "model (full train ≈ 37 d).",
            "",
            "`jepa-2016` has no rows for Users 1017 and 1082. Do not average those "
            "people in as if they ran.",
            "",
            "### 3.3 No LwF arm, no new LR grid",
            "",
            "LwF on SugarOne did not rescue short-history harm "
            "([PERSONALIZATION_REPORT.md](PERSONALIZATION_REPORT.md) §6.3–6.4). "
            "This extract keeps `λ=0`. Learning rate is the frozen Subject P1 recipe "
            "(`2×10⁻⁴`) used for SugarOne day curves, with the JEPA param-group "
            "ratio inherited from the global run.",
            "",
            "\n".join(model_blocks),
            "## Reproducibility and artifacts",
            "",
            "```bash",
            "uv run python src/sugar_jepa/jepa_report.py",
            "```",
            "",
            "Fine-tune a SugarJEPA2 checkpoint on one person (same recipe as SugarOne):",
            "",
            "```bash",
            "uv run personal-sweep-days --base-run-dir <sugar_jepa2_run> \\",
            "  --personal-csv data/input/personalization/prepared/subject_p1_chronological.csv",
            "```",
            "",
            "| Artifact | Path |",
            "|----------|------|",
            f"| Source table | `{csv_path.as_posix()}` |",
            f"| This report | `{report_path.as_posix()}` |",
            f"| Figures | `{figures_dir.as_posix()}` |",
            "| SugarOne 15-person study | `docs/PERSONALIZATION_REPORT.md` |",
            "| NeuralForecast continue-fit | `docs/PERSONALIZATION_NF_REPORT.md` |",
            "",
            "*Results from the on-disk JEPA day-budget MAE table. "
            "Fact-check against run `*_metrics_overall.csv` before citing in LaTeX.*",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")
    safe_echo(f"Wrote {report_path}")
    return report_path


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    csv: Path = typer.Option(DEFAULT_CSV, "--csv", help="JEPA MAE-by-days table."),
    report_path: Path = typer.Option(DEFAULT_REPORT_PATH, "--report-path"),
    figures_dir: Path = typer.Option(DEFAULT_FIGURES_DIR, "--figures-dir"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    """Render the SugarJEPA personalization report from the day-budget CSV."""
    if ctx.invoked_subcommand is not None:
        return
    init_cli_console()
    write_jepa_personalization_report(
        csv_path=csv,
        report_path=report_path,
        figures_dir=figures_dir,
        plot=plot,
    )


if __name__ == "__main__":
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    app()
