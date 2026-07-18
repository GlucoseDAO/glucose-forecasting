"""Compatibility checks for extracted family-specific data utilities."""
from __future__ import annotations

import glucose_forecasting.data.glumind as packaged_glumind
import glucose_forecasting.data.sugar_one as packaged_sugar_one
import glucose_forecasting.training.glumind as packaged_glumind_training
import glucose_forecasting.training.sugar_one as packaged_sugar_one_training
import scripts.glumind.train_glumind as legacy_glumind
import scripts.sugar_one.train_sugar_one as legacy_sugar_one


def test_glumind_legacy_data_exports_are_packaged_symbols() -> None:
    assert legacy_glumind.GlucoseWindowDataset is packaged_glumind.GlucoseWindowDataset
    assert legacy_glumind.load_splits_streaming is packaged_glumind.load_splits_streaming
    assert legacy_glumind.impute_and_sort is packaged_glumind.impute_and_sort
    assert legacy_glumind.build_datasets is packaged_glumind.build_datasets
    assert legacy_glumind.COL_TS == packaged_glumind.COL_TS
    assert legacy_glumind.TS_FORMAT == packaged_glumind.TS_FORMAT


def test_glumind_legacy_training_exports_are_packaged_symbols() -> None:
    for name in (
        "train_one_epoch",
        "evaluate",
        "compute_and_print_metrics",
        "save_full_checkpoint",
        "load_full_checkpoint",
        "make_optimizer_and_scheduler",
        "train_loop",
        "make_model",
        "run_train_and_eval",
        "mode_global",
        "mode_per_group",
        "mode_cohort_wise",
        "mode_continual",
    ):
        assert getattr(legacy_glumind, name) is getattr(packaged_glumind_training, name)


def test_sugar_one_legacy_data_exports_are_packaged_symbols() -> None:
    assert legacy_sugar_one.SugarOneWindowDataset is packaged_sugar_one.SugarOneWindowDataset
    assert legacy_sugar_one.load_splits_streaming is packaged_sugar_one.load_splits_streaming
    assert legacy_sugar_one.impute_and_sort is packaged_sugar_one.impute_and_sort
    assert legacy_sugar_one.apply_split_scheme is packaged_sugar_one.apply_split_scheme
    assert legacy_sugar_one.build_datasets is packaged_sugar_one.build_datasets
    assert legacy_sugar_one.COL_TS == packaged_sugar_one.COL_TS
    assert legacy_sugar_one.TS_FORMAT == packaged_sugar_one.TS_FORMAT


def test_sugar_one_legacy_training_exports_are_packaged_symbols() -> None:
    for name in (
        "train_one_epoch",
        "evaluate",
        "compute_and_print_metrics",
        "save_full_checkpoint",
        "load_full_checkpoint",
        "make_optimizer_and_scheduler",
        "train_loop",
        "make_model",
        "run_train_and_eval",
        "_mode_global",
        "_mode_per_group",
        "_mode_cohort_wise",
        "_mode_continual",
        "_model_kwargs",
    ):
        assert getattr(legacy_sugar_one, name) is getattr(packaged_sugar_one_training, name)
