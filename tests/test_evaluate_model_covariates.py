"""Tests for evaluate-model covariate selection helpers."""

from __future__ import annotations

import pytest

from scripts.glumind_ic.evaluate_model import (
    _alias_to_canonical,
    _parse_covariate_names,
    _resolve_covariate_zeroing,
)


def test_alias_to_canonical_glumind_ic() -> None:
    assert _alias_to_canonical("basal", "glumind_ic") == "basal"
    assert _alias_to_canonical("basal_rate", "glumind_ic") == "basal"
    assert _alias_to_canonical("Bolus Insulin", "glumind_ic") == "bolus"
    assert _alias_to_canonical("carbohydrates", "glumind_ic") == "carbs"


def test_alias_to_canonical_glumind() -> None:
    assert _alias_to_canonical("heart_rate", "glumind") == "hr"
    assert _alias_to_canonical("step count", "glumind") == "steps"


def test_alias_to_canonical_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown covariate"):
        _alias_to_canonical("basal", "glumind")


def test_parse_covariate_names_deduplicates() -> None:
    names = _parse_covariate_names("basal,basal_rate,bolus", "glumind_ic")
    assert names == ["basal", "bolus"]


def test_resolve_zero_cov() -> None:
    active, zeroed = _resolve_covariate_zeroing(
        "glumind_ic",
        zero_cov=True,
        include_cov=None,
        exclude_cov=None,
    )
    assert active == []
    assert zeroed == ["basal", "bolus", "carbs"]


def test_resolve_include_cov() -> None:
    active, zeroed = _resolve_covariate_zeroing(
        "glumind_ic",
        zero_cov=False,
        include_cov="basal,bolus",
        exclude_cov=None,
    )
    assert active == ["basal", "bolus"]
    assert zeroed == ["carbs"]


def test_resolve_exclude_cov() -> None:
    active, zeroed = _resolve_covariate_zeroing(
        "glumind_ic",
        zero_cov=False,
        include_cov=None,
        exclude_cov="carbs",
    )
    assert active == ["basal", "bolus"]
    assert zeroed == ["carbs"]


def test_resolve_conflicting_flags() -> None:
    with pytest.raises(ValueError, match="either --zero-cov or --include-cov"):
        _resolve_covariate_zeroing(
            "glumind_ic",
            zero_cov=True,
            include_cov="basal",
            exclude_cov=None,
        )

    with pytest.raises(ValueError, match="either --include-cov or --exclude-cov"):
        _resolve_covariate_zeroing(
            "glumind_ic",
            zero_cov=False,
            include_cov="basal",
            exclude_cov="carbs",
        )
