"""Unit tests for src/common/registry.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from common.registry import (
    find_best_run_dir,
    load_run_meta,
    resolve_checkpoint,
    resolve_csv_path,
)


def _write_registry_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv as csv_mod

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# find_best_run_dir
# ---------------------------------------------------------------------------


def test_find_best_run_dir_picks_lowest_val_mae(tmp_path: Path) -> None:
    project_root = tmp_path
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_registry_csv(
        registry_dir / "_analysis_registry.csv",
        [
            {"run_dir": "runs/a", "val_mae": "5.0"},
            {"run_dir": "runs/b", "val_mae": "2.0"},
            {"run_dir": "runs/c", "val_mae": "3.0"},
        ],
    )
    step_dir, row = find_best_run_dir(registry_dir, project_root)
    assert step_dir == project_root / "runs" / "b"
    assert row["run_dir"] == "runs/b"


def test_find_best_run_dir_resolves_final_step_subdir(tmp_path: Path) -> None:
    project_root = tmp_path
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_registry_csv(
        registry_dir / "_analysis_registry.csv",
        [{"run_dir": "runs/continual", "val_mae": "1.0", "final_step": "step_03_T1DM"}],
    )
    step_dir, row = find_best_run_dir(registry_dir, project_root)
    assert step_dir == project_root / "runs" / "continual" / "step_03_T1DM"


def test_find_best_run_dir_missing_file_exits(tmp_path: Path) -> None:
    registry_dir = tmp_path / "empty_registry"
    registry_dir.mkdir()
    with pytest.raises(typer.Exit):
        find_best_run_dir(registry_dir, tmp_path)


def test_find_best_run_dir_no_valid_rows_exits(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_registry_csv(registry_dir / "_analysis_registry.csv", [{"run_dir": "runs/a", "val_mae": ""}])
    with pytest.raises(typer.Exit):
        find_best_run_dir(registry_dir, tmp_path)


# ---------------------------------------------------------------------------
# load_run_meta
# ---------------------------------------------------------------------------


def test_load_run_meta_prefers_tuning_meta(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tuning_meta.json").write_text(json.dumps({"source": "tuning"}))
    (run_dir / "config.json").write_text(json.dumps({"source": "config"}))
    meta = load_run_meta(run_dir)
    assert meta["source"] == "tuning"


def test_load_run_meta_falls_back_to_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"source": "config"}))
    meta = load_run_meta(run_dir)
    assert meta["source"] == "config"


def test_load_run_meta_missing_exits(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(typer.Exit):
        load_run_meta(run_dir)


# ---------------------------------------------------------------------------
# resolve_checkpoint
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_explicit_path_wins(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    explicit = tmp_path / "explicit.pt"
    explicit.write_text("x")
    (run_dir / "best_model.pt").write_text("x")
    resolved = resolve_checkpoint(run_dir, explicit)
    assert resolved == explicit


def test_resolve_checkpoint_explicit_missing_exits(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    missing = tmp_path / "missing.pt"
    with pytest.raises(typer.Exit):
        resolve_checkpoint(run_dir, missing)


def test_resolve_checkpoint_prefers_best_then_last(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best_model.pt").write_text("x")
    (run_dir / "last_model.pt").write_text("x")
    resolved = resolve_checkpoint(run_dir, None)
    assert resolved == run_dir / "best_model.pt"

    (run_dir / "best_model.pt").unlink()
    resolved2 = resolve_checkpoint(run_dir, None)
    assert resolved2 == run_dir / "last_model.pt"


def test_resolve_checkpoint_none_found_exits(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(typer.Exit):
        resolve_checkpoint(run_dir, None)


# ---------------------------------------------------------------------------
# resolve_csv_path
# ---------------------------------------------------------------------------


def test_resolve_csv_path_as_is(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n")
    resolved = resolve_csv_path(csv_path, tmp_path / "project_root")
    assert resolved == csv_path


def test_resolve_csv_path_relative_to_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "root"
    project_root.mkdir()
    (project_root / "data.csv").write_text("a,b\n1,2\n")
    resolved = resolve_csv_path("data.csv", project_root)
    assert resolved == project_root / "data.csv"


def test_resolve_csv_path_missing_exits(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        resolve_csv_path("does_not_exist.csv", tmp_path)


def test_resolve_csv_path_basename_under_data_input(tmp_path: Path) -> None:
    project_root = tmp_path / "root"
    target = project_root / "data" / "input" / "train.csv"
    target.parent.mkdir(parents=True)
    target.write_text("a,b\n1,2\n")
    # Absolute Windows-style path from another machine — only basename is usable.
    legacy = r"D:\other_machine\datasets\train.csv"
    resolved = resolve_csv_path(legacy, project_root)
    assert resolved == target


def test_resolve_csv_path_prefers_data_input_over_other_data_hits(tmp_path: Path) -> None:
    project_root = tmp_path / "root"
    preferred = project_root / "data" / "input" / "shared.csv"
    other = project_root / "data" / "loop_and_ai_ready" / "shared.csv"
    preferred.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    preferred.write_text("a,b\n1,2\n")
    other.write_text("a,b\n3,4\n")
    resolved = resolve_csv_path(r"C:\legacy\shared.csv", project_root)
    assert resolved == preferred
