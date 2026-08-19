#!/usr/bin/env python3
"""Rebuild Milestone 8 interim report + charts from on-disk personalization runs."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from common.console import safe_echo
from common.paths import DEFAULT_RUNS_ROOT
from personalization.cohort import (
    COHORT_JOINED2_TEST,
    LIVIA_CSV,
    PHASE4_SUBJECTS,
    Phase4Subject,
    display_name_for,
    joined2_test_subjects,
    original_cohort_subjects,
)
from personalization.plots import (
    plot_combined_data_size_curves,
    plot_curriculum_mae_and_lambda,
    plot_data_size_curve,
)
from personalization.sweep_lwf import PERSONS
from personalization.splits import load_train_span_days
from personalization.sweep_utils import (
    data_size_row_from_metrics,
    data_size_run_dir,
    load_best_recipe,
    personalization_run_complete,
    uses_base_scalers,
)

DEFAULT_ROOT = DEFAULT_RUNS_ROOT / "personalization"
DEFAULT_REPORT = Path("temp_docs") / "reports" / "MILESTONE_8_INTERIM_REPORT.md"
DEFAULT_FIGURES = Path("temp_docs") / "reports" / "figures" / "m8_interim"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    df = pl.read_csv(path)
    return [dict(row) for row in df.iter_rows(named=True)]


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


def _is_base_scaler_row(row: dict[str, Any]) -> bool:
    cfg = {
        "refit_scalers_on_personal": row.get("refit_scalers_on_personal"),
        "scaler_source": row.get("scaler_source"),
    }
    if uses_base_scalers(cfg):
        return True
    run_dir = row.get("run_dir")
    if not run_dir:
        return False
    return personalization_run_complete(Path(str(run_dir)))


def collect_data_size_series(
    root: Path,
) -> list[tuple[Phase4Subject, list[dict[str, Any]]]]:
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]] = []
    for spec in PHASE4_SUBJECTS:
        summary = root / spec.subject / "sweeps" / "data_size" / "summary.csv"
        out_dir = summary.parent
        rows = _load_csv_rows(summary)
        if not rows:
            recipe_path = root / "livia" / "best_recipe.json"
            recipe = load_best_recipe(recipe_path) if recipe_path.is_file() else {}
            rebuilt: list[dict[str, Any]] = []
            for label in ("1", "3", "7", "14", "30", "60", "all"):
                run_dir = data_size_run_dir(out_dir, spec.subject, label)
                row = data_size_row_from_metrics(
                    run_dir,
                    subject=spec.subject,
                    day_label=label,
                    lwf_lambda=float(recipe.get("lwf_lambda", 0.0)),
                    lr=float(recipe.get("lr", 0.0002)),
                    weight_decay=float(recipe.get("weight_decay", 3e-5)),
                    patience=int(recipe.get("patience", 3)),
                )
                if row is not None:
                    rebuilt.append(row)
            rows = rebuilt
        if not rows:
            continue
        rows = _enrich_span(rows, spec.csv)
        ok = [
            r
            for r in rows
            if r.get("status") == "ok"
            and r.get("ft_test_mae") is not None
            and _is_base_scaler_row(r)
        ]
        if ok:
            series.append((spec, ok))
    return series


def collect_method_series(
    root: Path,
    *,
    sweeps: tuple[tuple[str, str], ...],
    personal_csv: Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Independent λ=0 vs independent-from-global LwF curves (base scalers only)."""
    series: list[tuple[str, list[dict[str, Any]]]] = []
    for subject, rel in sweeps:
        summary = root / Path(rel) / "summary.csv"
        rows = _load_csv_rows(summary)
        if not rows:
            continue
        rows = _enrich_span(rows, personal_csv)
        ok = [
            r
            for r in rows
            if r.get("status") == "ok"
            and r.get("ft_test_mae") is not None
            and _is_base_scaler_row(r)
        ]
        if ok:
            series.append((subject, ok))
    return series


def collect_livia_method_series(
    root: Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    livia = next(p for p in PERSONS if p.name == "livia")
    return collect_method_series(
        root, sweeps=livia.method_sweeps(), personal_csv=LIVIA_CSV
    )


def collect_user154_method_series(
    root: Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    user = next(p for p in PERSONS if p.name == "154")
    return collect_method_series(
        root, sweeps=user.method_sweeps(), personal_csv=user.csv
    )


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def write_charts(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    *,
    root: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    original_names = {s.subject for s in original_cohort_subjects()}
    joined_names = {s.subject for s in joined2_test_subjects()}

    for spec, rows in series:
        png = figures_dir / f"{spec.subject}_data_size.png"
        try:
            plot_data_size_curve(
                rows,
                out_png=png,
                title=f"{spec.display} — personal train days vs test MAE (60 days)",
                subject=spec.subject,
                mode="max_days",
                max_days=60.0,
            )
            paths[f"{spec.subject}_60d"] = png
            sweep_copy = root / spec.subject / "sweeps" / "data_size" / "data_size_curve.png"
            sweep_copy.parent.mkdir(parents=True, exist_ok=True)
            _copy(png, sweep_copy)
        except ValueError as exc:
            safe_echo(f"Skip 60-day chart for {spec.subject}: {exc}", err=True)

    def _combined(
        name: str,
        filename: str,
        title: str,
        wanted: set[str],
        *,
        dummy_all: bool,
        show_zero_shot: bool,
    ) -> None:
        subset = [
            (spec.subject, rows)
            for spec, rows in series
            if spec.subject in wanted
        ]
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
        _copy(png, root / filename)

    _combined(
        "combined_all",
        "data_size_curves_combined.png",
        "Livia + Loop holdouts (All = full train)",
        original_names,
        dummy_all=True,
        show_zero_shot=True,
    )
    _combined(
        "combined_60d",
        "data_size_curves_combined_60d.png",
        "Livia + Loop holdouts (first 60 days)",
        original_names,
        dummy_all=False,
        show_zero_shot=True,
    )
    _combined(
        "combined_joined2_all",
        "data_size_curves_combined_joined2.png",
        "Joined2 AI-READY test users, 2 per group (All = full train)",
        joined_names,
        dummy_all=True,
        show_zero_shot=False,
    )
    _combined(
        "combined_joined2_60d",
        "data_size_curves_combined_joined2_60d.png",
        "Joined2 AI-READY test users, 2 per group (first 60 days)",
        joined_names,
        dummy_all=False,
        show_zero_shot=False,
    )
    def _lwf_overlay(
        methods: list[tuple[str, list[dict[str, Any]]]],
        *,
        indep_key: str,
        stem: str,
        title_person: str,
    ) -> None:
        if len(methods) < 2:
            return
        plot_series = [
            (f"{indep_key}_indep" if subject == indep_key else subject, rows)
            for subject, rows in methods
        ]
        png = figures_dir / f"{stem}_combined.png"
        plot_combined_data_size_curves(
            plot_series,
            out_png=png,
            title=f"{title_person}: independent vs LwF from global (All = full train)",
            show_zero_shot=True,
            mode="dummy_all",
            max_days=60.0,
        )
        paths[f"{stem}_all"] = png
        _copy(png, root / f"{stem}_combined.png")
        png60 = figures_dir / f"{stem}_combined_60d.png"
        plot_combined_data_size_curves(
            plot_series,
            out_png=png60,
            title=f"{title_person}: independent vs LwF from global (first 60 days)",
            show_zero_shot=True,
            mode="max_days",
            max_days=60.0,
        )
        paths[f"{stem}_60d"] = png60
        _copy(png60, root / f"{stem}_combined_60d.png")
        png_lambda = figures_dir / f"{stem}_mae_lambda.png"
        plot_curriculum_mae_and_lambda(
            plot_series,
            out_png=png_lambda,
            title=f"{title_person}: test MAE (top) and lwf_lambda (bottom)",
        )
        paths[f"{stem}_mae_lambda"] = png_lambda
        _copy(png_lambda, root / f"{stem}_mae_lambda.png")

    _lwf_overlay(
        collect_livia_method_series(root),
        indep_key="livia",
        stem="livia_lwf_indep",
        title_person="Livia",
    )
    _lwf_overlay(
        collect_user154_method_series(root),
        indep_key="loop_154",
        stem="loop_154_lwf_indep",
        title_person="User 154",
    )
    return paths


def _days_table(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(
        rows,
        key=lambda r: float("inf")
        if str(r.get("personal_days")).lower() == "all"
        else float(r.get("personal_days", 0)),
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
            f"| {label} | {_fmt(used, 1)} | {_fmt(zs)} | {_fmt(ft)} | "
            f"{_fmt(delta)} |"
        )
    return "\n".join(lines)


def _lwf_protocol_tables() -> str:
    return """Real-world question: a user shows up with **N days** of data. Should we fine-tune the global model on that slice, or keep the frozen checkpoint? Every day budget is therefore a **new** run from `fixtures/checkpoints/sugar_one_1.0/`. The LwF teacher is that same frozen checkpoint — never a shorter-day student.

**Independent (Phase 4, λ=0).** Plain fine-tune from global. No teacher.

**LwF decay.** Same independent init. λ = 0.5 / 0.4 / 0.3 / 0.2 on 1 / 3 / 7 / 14 days. From **30 days** λ=0, so those points are copied from the independent Phase 4 runs (not retrained).

**LwF λ=0.1.** Same independent init. λ=0.1 on **every** day budget, including 30 / 60 / all.

| Days | Student init | Teacher | λ independent | λ decay | λ const |
|------|--------------|---------|---------------|---------|---------|
| 1 | global | global (if λ>0) | 0 | **0.5** | **0.1** |
| 3 | global | global (if λ>0) | 0 | **0.4** | **0.1** |
| 7 | global | global (if λ>0) | 0 | **0.3** | **0.1** |
| 14 | global | global (if λ>0) | 0 | **0.2** | **0.1** |
| 30 | global | global only for λ=0.1 | 0 | **0** (copy independent) | **0.1** |
| 60 | global | global only for λ=0.1 | 0 | **0** (copy independent) | **0.1** |
| all | global | global only for λ=0.1 | 0 | **0** (copy independent) | **0.1** |

Val and test splits never change. Day budget only lengthens **train**. Scalers stay the global `scalers.json`. Recipe: lr=2e-4, stride=6, bf16, patience=3.

A previous sequential-curriculum experiment chained `best_model.pt` from shorter day budgets into longer ones. That is **not** this protocol and is not used below.

```mermaid
flowchart LR
  G[Global sugar_one_1.0]
  G --> D1[1d decay λ=0.5]
  G --> D3[3d decay λ=0.4]
  G --> D7[7d decay λ=0.3]
  G --> D14[14d decay λ=0.2]
  G --> C1[1d const λ=0.1]
  G --> CAll[all const λ=0.1]
  G --> I30[30d+ independent λ=0]
```
"""


def _lwf_methods_table(
    methods: list[tuple[str, list[dict[str, Any]]]],
    *,
    key_order: list[str],
    empty_note: str,
) -> str:
    if not methods:
        return empty_note
    day_set: set[str] = set()
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for subject, rows in methods:
        lookup: dict[str, dict[str, Any]] = {}
        for row in rows:
            label = str(row.get("personal_days"))
            lookup[label] = row
            day_set.add(label)
        by_method[subject] = lookup
    order = [d for d in ("1", "3", "7", "14", "30", "60", "all") if d in day_set]
    headers = [
        "Days",
        "Independent MAE",
        "Decay MAE",
        "Const λ=0.1 MAE",
        "Independent Δ",
        "Decay Δ",
        "Const Δ",
        "λ independent",
        "λ decay",
        "λ const",
    ]
    present = [k for k in key_order if k in by_method]
    if len(present) < 3:
        headers = ["Days"] + [display_name_for(s) for s, _ in methods]
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["------"] * len(headers)) + "|",
        ]
        for day in order:
            cells = [day]
            for subject, _rows in methods:
                row = by_method[subject].get(day)
                if row is None:
                    cells.append("—")
                    continue
                ft = row.get("ft_test_mae")
                zs = row.get("zs_test_mae")
                delta = None
                if ft is not None and zs is not None:
                    delta = float(ft) - float(zs)
                lwf = row.get("lwf_lambda")
                lwf_note = ""
                try:
                    if lwf is not None and float(lwf) > 0:
                        lwf_note = f" λ={float(lwf):g}"
                except (TypeError, ValueError):
                    pass
                cells.append(f"{_fmt(ft)} ({_fmt(delta)}){lwf_note}")
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["------"] * len(headers)) + "|",
    ]
    for day in order:
        cells = [day]
        maes: list[str] = []
        deltas: list[str] = []
        lambdas: list[str] = []
        for key in present:
            row = by_method[key].get(day)
            if row is None:
                maes.append("—")
                deltas.append("—")
                lambdas.append("—")
                continue
            ft = row.get("ft_test_mae")
            zs = row.get("zs_test_mae")
            delta = None
            if ft is not None and zs is not None:
                delta = float(ft) - float(zs)
            maes.append(_fmt(ft))
            deltas.append(_fmt(delta))
            lwf = row.get("lwf_lambda")
            try:
                lambdas.append(f"{float(lwf):g}" if lwf is not None else "0")
            except (TypeError, ValueError):
                lambdas.append("—")
        cells.extend(maes + deltas + lambdas)
        lines.append("| " + " | ".join(cells) + " |")
    return (
        "Test MAE (mg/dL). Δ is fine-tuned minus zero-shot (negative = better than the frozen global model).\n\n"
        + "\n".join(lines)
    )


def _phase_a_table(series: list[tuple[Phase4Subject, list[dict[str, Any]]]]) -> str:
    lines = [
        "| Subject | Cohort | Study group | Train span (d) | ZS MAE | FT MAE (all) | Δ vs ZS |",
        "|---------|--------|-------------|----------------|--------|--------------|---------|",
    ]
    for spec, rows in series:
        all_row = next(
            (r for r in rows if str(r.get("personal_days")).lower() == "all"),
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
        cohort = "joined2 test" if spec.cohort == COHORT_JOINED2_TEST else (
            "Livia" if spec.cohort == "livia" else "Loop holdout"
        )
        lines.append(
            f"| {spec.display} | {cohort} | {spec.study_group} | {_fmt(span, 1)} | "
            f"{_fmt(zs)} | {_fmt(ft)} | {_fmt(delta)} |"
        )
    return "\n".join(lines)


def _coverage_table() -> str:
    lines = [
        "| Subject | Source | Study group | Notes |",
        "|---------|--------|-------------|-------|",
        "| **Livia** | Personal CGM/pump export | T1DM | Longest history (~345d train) |",
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


def _per_user_sections(
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    *,
    wanted: set[str],
) -> str:
    blocks: list[str] = []
    for spec, rows in series:
        if spec.subject not in wanted:
            continue
        fig = f"figures/m8_interim/{spec.subject}_data_size.png"
        blocks.append(
            f"### {spec.display}\n\n"
            f"{_days_table(rows)}\n\n"
            f"![{spec.display} data-size curve (60 days)]({fig})\n"
        )
    return "\n".join(blocks) if blocks else "_No completed runs in this cohort yet._\n"


def _lr_tables(root: Path) -> tuple[str, str]:
    livia_board = _load_csv_rows(root / "livia" / "tune" / "leaderboard.csv")
    livia_lines = [
        "| LR | Zero-shot MAE | Fine-tuned MAE | Fine-tuned Val MAE |",
        "|----|---------------|----------------|--------------------|",
    ]
    if livia_board:
        ranked = sorted(
            [r for r in livia_board if r.get("ft_test_mae") is not None],
            key=lambda r: float(r["ft_test_mae"]),
        )
        for row in ranked:
            livia_lines.append(
                f"| {row.get('lr')} | {_fmt(row.get('zs_test_mae'), 3)} | "
                f"{_fmt(row.get('ft_test_mae'), 3)} | {_fmt(row.get('ft_val_mae'), 3)} |"
            )
    else:
        livia_lines.append("| _pending_ | | | |")

    holdout = _load_csv_rows(root / "holdout_lr_sweep" / "summary.csv")
    holdout_lines = [
        "| User | LR 1e-4 | LR 2e-4 | LR 4e-4 | Best |",
        "|------|---------|---------|---------|------|",
    ]
    by_user: dict[str, dict[float, float]] = {}
    for row in holdout:
        if row.get("status") != "ok":
            continue
        uid = str(row.get("user_id"))
        lr = float(row["lr"])
        mae = float(row["ft_test_mae"])
        by_user.setdefault(uid, {})[lr] = mae
    if by_user:
        for uid in sorted(by_user):
            grid = by_user[uid]
            best_lr = min(grid, key=grid.get)
            holdout_lines.append(
                f"| {uid} | {_fmt(grid.get(0.0001), 3)} | {_fmt(grid.get(0.0002), 3)} | "
                f"{_fmt(grid.get(0.0004), 3)} | **{best_lr:g}** |"
            )
    else:
        holdout_lines.append("| _pending_ | | | | |")
    return "\n".join(livia_lines), "\n".join(holdout_lines)


def render_report(
    *,
    root: Path,
    series: list[tuple[Phase4Subject, list[dict[str, Any]]]],
    status: dict[str, Any] | None = None,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    livia_lr, holdout_lr = _lr_tables(root)
    done_subjects = [spec.display for spec, _rows in series]
    pending = []
    if status:
        pending = list(status.get("pending_subjects") or [])
    n_expected = len(PHASE4_SUBJECTS)
    original_names = {s.subject for s in original_cohort_subjects()}
    joined_names = {s.subject for s in joined2_test_subjects()}

    original_sections = _per_user_sections(series, wanted=original_names)
    joined_sections = _per_user_sections(series, wanted=joined_names)
    livia_methods = collect_livia_method_series(root)
    user154_methods = collect_user154_method_series(root)
    livia_lwf = ""
    if livia_methods:
        curr_figs = ""
        if len(livia_methods) >= 2:
            curr_figs = (
                "![Livia MAE overlay](figures/m8_interim/livia_lwf_indep_combined.png)\n\n"
                "![Livia MAE vs lwf_lambda](figures/m8_interim/livia_lwf_indep_mae_lambda.png)\n\n"
                "![Livia first 60 days](figures/m8_interim/livia_lwf_indep_combined_60d.png)\n"
            )
        livia_lwf = (
            "### 6.3 Independent LwF on Livia — can distillation rescue short/harmful fine-tunes?\n\n"
            + _lwf_protocol_tables()
            + "\n"
            + _lwf_methods_table(
                livia_methods,
                key_order=["livia", "livia_lwf_decay", "livia_lwf_01"],
                empty_note="_No Livia independent LwF runs yet._\n",
            )
            + "\n\n"
            + curr_figs
        )
    user154_lwf = ""
    if user154_methods:
        figs154 = ""
        if len(user154_methods) >= 2:
            figs154 = (
                "![User 154 MAE overlay](figures/m8_interim/loop_154_lwf_indep_combined.png)\n\n"
                "![User 154 MAE vs lwf_lambda](figures/m8_interim/loop_154_lwf_indep_mae_lambda.png)\n\n"
                "![User 154 first 60 days](figures/m8_interim/loop_154_lwf_indep_combined_60d.png)\n"
            )
        user154_lwf = (
            "### 6.4 User 154 — same independent LwF protocol\n\n"
            "Phase 4 independent fine-tunes on this user are flat or slightly **worse** than zero-shot until full train. "
            "Same teacher (`sugar_one_1.0`) and same two λ policies as Livia.\n\n"
            + _lwf_methods_table(
                user154_methods,
                key_order=["loop_154", "loop_154_lwf_decay", "loop_154_lwf_01"],
                empty_note="_No user 154 independent LwF runs yet._\n",
            )
            + "\n\n"
            + figs154
        )

    orig_combined = ""
    if any(spec.subject in original_names for spec, _ in series):
        orig_combined = (
            "![Holdouts combined with dummy All](figures/m8_interim/data_size_curves_combined.png)\n\n"
            "![Holdouts combined, first 60 days](figures/m8_interim/data_size_curves_combined_60d.png)\n"
        )
    joined_combined = ""
    if any(spec.subject in joined_names for spec, _ in series):
        joined_combined = (
            "![Joined2 test combined with dummy All](figures/m8_interim/data_size_curves_combined_joined2.png)\n\n"
            "![Joined2 test combined, first 60 days](figures/m8_interim/data_size_curves_combined_joined2_60d.png)\n"
        )

    status_line = (
        f"Data-size curves complete for **{len(series)}/{n_expected}** subjects"
        f" ({', '.join(done_subjects) if done_subjects else 'none yet'})."
    )
    if pending:
        status_line += f" Still running/pending: {', '.join(pending)}."
    if status and status.get("current"):
        status_line += f" Independent LwF job: `{status.get('current')}`."

    body = f"""# Milestone 8 — Personalization Interim Report

**Date:** {today}  
**Project:** Glucose forecasting / SugarOne personalization  
**Base model:** `fixtures/checkpoints/sugar_one_1.0/` (global SugarOne checkpoint)  
**Horizon:** 12 steps (60 minutes at 5-minute sampling)

This report is regenerated from on-disk runs. Re-run `uv run personal-study --report-only` after interruption to refresh charts from whatever has finished.

---

## 1. Executive summary

| Question | Finding |
|----------|---------|
| Best personalization method | **Plain fine-tune** (`lwf_lambda=0`) — ~10× faster than LwF, similar MAE |
| Best train window stride | **Sparse stride=6** |
| Best LR on Livia (full train) | **2×10⁻⁴** (Step 2; personal-scaler era) |
| Scaler protocol (this recalc) | **Base-run `scalers.json`** — fine-tune stays in the pretrained input scale |
| Data-size recalc status | {status_line} |

**Protocol locked:** plain fine-tune, `weight_decay=3e-5`, `train_window_stride=6`, `precision=bf16`, chronological split (25% test / 15% of remainder val / rest train). Day budget only limits train windows. Scalers come from the **global checkpoint**, not from personal train (that was the bug that made 1-day fine-tunes look worse than zero-shot).

---

## 2. Subjects and data coverage

Two extra users were taken from **each AI-READY study group** in the `loop_ai_ready_joined2.csv` test split (largest test-split row count, then User ID). T1DM is not repeated here — Livia plus the six Loop quality holdouts already cover that group. Each CSV uses that user’s **full joined2 history**, then the same chronological split as the holdouts. AI-READY groups typically have ~10 days of CGM and no insulin/carb columns (zero-filled at train/eval).

{_coverage_table()}

---

## 3. Design choices

### 3.1 Plain fine-tune vs LwF

LwF is still in the SugarOne training loop (`loss = (1-λ)·task + λ·distill` against a frozen **global** teacher, always `fixtures/checkpoints/sugar_one_1.0`). Default personalization stays **λ=0**. The independent LwF experiment asks whether a teacher can stop short-history fine-tunes from beating (or matching) zero-shot. λ decays 0.5→0.2 on 1–14 days and is **0 from 30 days**; a second arm keeps λ=0.1 on every budget. Every day budget starts from the global student weights.

### 3.2 Sparse vs dense train windows

Sparse stride=6 is the production default (~6× fewer train windows, MAE ≈ dense).

### 3.3 Scaler protocol (the recalc)

Earlier data-size curves fitted MinMax scalers on **personal train**. With only 1–14 days that shifts the input scale away from the pretrained model, so short histories looked harmful.

**Corrected protocol:** reuse `fixtures/checkpoints/sugar_one_1.0/scalers.json`. Tables and charts below are from this recalc. Legacy personal-scaler runs are archived next to each `data_size/` folder.

---

## 4. Step 2 — Learning rate on Livia (full personal train)

These numbers are from the original Step-2 tune (not re-run in this Phase 4 recalc). Frozen recipe for days curves: **lr=2e-4**.

{livia_lr}

**Best recipe:** `data/output/runs/personalization/livia/best_recipe.json`

---

## 5. Step 2b — Does Livia’s LR transfer? (pilot holdouts)

Same LR grid on users **154, 556, 730** (full personal train). Livia reference = **2e-4**. These runs have not been re-executed with base scalers.

{holdout_lr}

Deferred LR sweep users (plan): confirm with remaining holdouts if needed. Phase 4 days curves below all use the **frozen Livia recipe** (`lr=2e-4`), not per-user best LR.

---

## 6. Step 3 / 4 — Personal train days vs test MAE

Fixed recipe: **lr=2e-4**, lwf=0, wd=3e-5, bf16, stride=6, **base-run scalers**.

Per-user charts are **limited to 60 days**. Full-train (`all`) is in the tables with the real train span, and on combined charts whose last tick is a dummy **All**. Combined charts are split (holdouts vs joined2 AI-READY) so overlays stay readable.

### 6.0 Phase A — full train, frozen Livia recipe

{_phase_a_table(series)}

### 6.1 Livia + Loop quality holdouts (60-day curves)

{original_sections}

{orig_combined if orig_combined else "_Holdout combined charts appear after the first subject in this cohort finishes._\n"}

### 6.2 Joined2 test — two users per study group

{joined_sections}

{joined_combined if joined_combined else "_Joined2 combined charts appear after the first subject in this cohort finishes._\n"}

{livia_lwf}

{user154_lwf}

---

## 7. Progress vs milestone plan

| Step | Goal | Status |
|------|------|--------|
| 1 | Prepare chronological CSVs | Done (Livia + 6 holdouts + 8 joined2 AI-READY) |
| 2 | LR search on Livia full train | Done — best LR 2e-4 |
| 2b | LR transfer check on holdouts | Partial — 3/6 users (not part of this recalc) |
| 3 | Data-size curve (Livia) | Done (base scalers) |
| 4 | Holdout + joined2 test Phase A/B | Done — 15/15 subjects |
| 5 | Aggregate + report | This file |

---

## 8. What is left to do

1. Optional: re-run holdout LR sweep with base scalers.  
2. Production LR / LwF policy from holdout + joined2 + independent LwF curves (does distillation rescue users like 154?).

---

## 9. Artifact index

| Artifact | Path |
|----------|------|
| Livia best recipe | `data/output/runs/personalization/livia/best_recipe.json` |
| Phase 4 status | `data/output/runs/personalization/phase4_status.json` |
| Holdout combined (dummy All) | `data/output/runs/personalization/data_size_curves_combined.png` |
| Holdout combined (60 days) | `data/output/runs/personalization/data_size_curves_combined_60d.png` |
| Joined2 combined (dummy All) | `data/output/runs/personalization/data_size_curves_combined_joined2.png` |
| Joined2 combined (60 days) | `data/output/runs/personalization/data_size_curves_combined_joined2_60d.png` |
| Livia independent LwF overlay | `data/output/runs/personalization/livia_lwf_indep_combined.png` |
| Livia LwF MAE + λ panels | `data/output/runs/personalization/livia_lwf_indep_mae_lambda.png` |
| User 154 independent LwF overlay | `data/output/runs/personalization/loop_154_lwf_indep_combined.png` |
| Independent LwF overnight status | `data/output/runs/personalization/lwf_indep_status.md` |
| Research plan | `docs/PERSONALIZATION.md` |
| Chart copies for this report | `temp_docs/reports/figures/m8_interim/` |

---

*Regenerated {today} by `personal-study`.*
"""
    return body


def write_milestone8_report(
    *,
    root: Path = DEFAULT_ROOT,
    report_path: Path = DEFAULT_REPORT,
    figures_dir: Path = DEFAULT_FIGURES,
    status: dict[str, Any] | None = None,
) -> Path:
    series = collect_data_size_series(root)
    write_charts(series, root=root, figures_dir=figures_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(root=root, series=series, status=status),
        encoding="utf-8",
    )
    safe_echo(f"Wrote {report_path}")
    return report_path
