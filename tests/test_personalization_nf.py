"""Tests for NeuralForecast personalization (no GPU / full continue-fit)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from personalization.sweep_utils import should_skip_day_budget
from personalization_nf.adapt import app as adapt_app
from personalization_nf.data import (
    choose_val_size,
    day_label,
    limit_train_calendar_days,
    span_days,
)
from personalization_nf.discover import parse_model_filter
from personalization_nf.report import (
    _days_table,
    _fmt,
    _full_train_table,
    _global_holdout_table,
)
from personalization_nf.study import app as study_app
from personalization_nf.sweep import parse_days_grid
from personalization.cohort import Phase4Subject


runner = CliRunner()


def _series_frame(*, n_rows: int, unique_id: str = "s1", start: datetime | None = None) -> pl.DataFrame:
    origin = start or datetime(2024, 1, 1, 0, 0, 0)
    return pl.DataFrame(
        {
            "unique_id": [unique_id] * n_rows,
            "ds": [origin + timedelta(minutes=5 * i) for i in range(n_rows)],
            "y": [100.0 + i for i in range(n_rows)],
        }
    )


def test_parse_model_filter() -> None:
    assert parse_model_filter(None) is None
    assert parse_model_filter("all") is None
    assert parse_model_filter("NHITS,TFT") == ("NHITS", "TFT")


def test_parse_days_grid_default_and_all() -> None:
    grid = parse_days_grid(None)
    assert grid[-1] is None
    assert 1 in grid
    assert parse_days_grid("1,all") == [1, None]


def test_day_label() -> None:
    assert day_label(None) == "all"
    assert day_label(7) == "7"


def test_limit_train_calendar_days_keeps_first_n() -> None:
    train = _series_frame(n_rows=288 * 3)
    limited = limit_train_calendar_days(train, 1)
    assert limited.height == 288
    assert span_days(limited) is not None
    assert span_days(limited) < 1.01


def test_limit_train_none_is_identity() -> None:
    train = _series_frame(n_rows=20)
    assert limit_train_calendar_days(train, None).height == 20


def test_choose_val_size_leaves_one_window() -> None:
    train = _series_frame(n_rows=288)
    val_size = choose_val_size(
        train,
        input_size=128,
        horizon=12,
        configured_val_size=288,
        val_tail_fraction=0.2,
    )
    assert val_size >= 12
    assert 128 + val_size + 12 <= 288


def test_choose_val_size_zero_when_too_short() -> None:
    train = _series_frame(n_rows=130)
    assert (
        choose_val_size(
            train,
            input_size=128,
            horizon=12,
            configured_val_size=288,
            val_tail_fraction=0.2,
        )
        == 0
    )


def test_should_skip_day_budget_when_span_covered() -> None:
    assert should_skip_day_budget(7, 6.3) is True
    assert should_skip_day_budget(3, 6.3) is False
    assert should_skip_day_budget(None, 6.3) is False


def test_days_table_and_full_train_table() -> None:
    rows = [
        {
            "personal_days": "1",
            "used_train_days": 1.0,
            "zs_test_mae": 12.0,
            "ft_test_mae": 12.5,
            "status": "ok",
        },
        {
            "personal_days": "all",
            "used_train_days": 90.0,
            "train_span_days": 90.0,
            "zs_test_mae": 12.0,
            "ft_test_mae": 11.0,
            "status": "ok",
        },
    ]
    table = _days_table(rows)
    assert "0.50" in table
    assert "-1.00" in table
    spec = Phase4Subject(
        user_id="subject_p1",
        subject="subject_p1",
        csv=Path("data/input/personalization/prepared/subject_p1_chronological.csv"),
        cohort="subject_p1",
        study_group="T1DM",
        display="Subject P1",
    )
    full = _full_train_table([(spec, rows)])
    assert "Subject P1" in full
    assert "-1.00" in full


def test_fmt_dash_for_missing() -> None:
    assert _fmt(None) == "—"
    assert _fmt(1.234, 2) == "1.23"


def test_adapt_cli_help() -> None:
    result = runner.invoke(adapt_app, ["--help"])
    assert result.exit_code == 0
    assert "--personal-csv" in result.output


def test_study_cli_help() -> None:
    result = runner.invoke(study_app, ["--help"])
    assert result.exit_code == 0
    assert "--holdout-root" in result.output
    assert "--report-only" in result.output


def test_discover_holdout_runs_picks_best_per_model(tmp_path: Path) -> None:
    from personalization_nf.discover import discover_holdout_runs

    root = tmp_path / "nf_holdout" / "__ALL__"
    better = root / "NHITS_20260101T000000Z"
    worse = root / "NHITS_20260102T000000Z"
    other = root / "TFT_20260101T000001Z"
    for path, mae in ((better, 10.0), (worse, 12.0), (other, 11.0)):
        path.mkdir(parents=True)
        (path / "neuralforecast").mkdir()
        (path / "run_config.json").write_text(
            '{"evaluation": "holdout", "models": "NHITS"}',
            encoding="utf-8",
        )
        (path / "val_metrics_overall.csv").write_text(
            f"mae,rmse,mard\n{mae},1,1\n",
            encoding="utf-8",
        )
    found = discover_holdout_runs(root)
    by_key = {item.model_key: item for item in found}
    assert set(by_key) == {"NHITS", "TFT"}
    assert by_key["NHITS"].run_dir == better
    filtered = discover_holdout_runs(root, models=("TFT",))
    assert [item.model_key for item in filtered] == ["TFT"]


def test_global_holdout_table_reads_val_and_test(tmp_path: Path) -> None:
    from personalization_nf.discover import NfHoldoutRun

    run = tmp_path / "NBEATSx_20260811T160552Z"
    run.mkdir()
    (run / "val_metrics_overall.csv").write_text(
        "mae,rmse,mard\n11.70753,18.37375,8.29882\n",
        encoding="utf-8",
    )
    (run / "test_metrics_overall.csv").write_text(
        "mae,rmse,mard\n11.80877,19.09758,8.05482\n",
        encoding="utf-8",
    )
    holdout = NfHoldoutRun(
        model_key="NBEATSx",
        run_dir=run,
        bundle_dir=run / "neuralforecast",
        val_mae=11.70753,
        config={},
    )
    table = _global_holdout_table([holdout])
    assert "11.71" in table
    assert "18.37" in table
    assert "8.30%" in table
    assert "11.81" in table
    assert "19.10" in table
    assert "8.05%" in table
