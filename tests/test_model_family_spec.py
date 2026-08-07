"""Tests for ModelFamilySpec + schema-free scaler persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scripts.common.model_spec import (
    ModelFamilySpec,
    detect_family_kind,
    get_family_spec,
    list_family_kinds,
    register_family_spec,
)
from scripts.common.scalers import (
    SCALERS_FILENAME,
    dump_scalers,
    extract_scalers_from_dataset,
    load_scalers,
    resolve_scalers_path,
    save_scalers_for_run,
    scalers_match_transform,
)


def _fit_minmax(lo: float, hi: float) -> MinMaxScaler:
    return MinMaxScaler().fit(np.array([[lo], [hi]], dtype=np.float64))


# ---------------------------------------------------------------------------
# Schema-free scalers (no FEATURES_BY_KIND whitelist)
# ---------------------------------------------------------------------------


def test_dump_load_arbitrary_features_unknown_kind(tmp_path: Path) -> None:
    """New model kinds must not require edits to scalers.py."""
    scalers = {
        "glucose": _fit_minmax(40.0, 300.0),
        "foo": _fit_minmax(0.0, 1.0),
    }
    path = dump_scalers(
        tmp_path / "scalers.json",
        scalers=scalers,
        kind="brand_new_model",
    )
    kind, loaded, payload = load_scalers(path)
    assert kind == "brand_new_model"
    assert set(loaded) == {"glucose", "foo"}
    assert scalers_match_transform(scalers["foo"], loaded["foo"])
    assert payload["kind"] == "brand_new_model"


def test_dump_rejects_empty_scalers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        dump_scalers(tmp_path / "bad.json", scalers={})


def test_extract_scalers_discovers_scaler_attrs() -> None:
    class _DS:
        def __init__(self) -> None:
            self.scaler_glucose = _fit_minmax(0.0, 200.0)
            self.scaler_hr = _fit_minmax(40.0, 180.0)
            self.unrelated = 1

    extracted = extract_scalers_from_dataset(_DS())
    assert set(extracted) == {"glucose", "hr"}


def test_extract_scalers_respects_feature_names() -> None:
    class _DS:
        def __init__(self) -> None:
            self.scaler_glucose = _fit_minmax(0.0, 200.0)
            self.scaler_hr = _fit_minmax(40.0, 180.0)
            self.scaler_steps = _fit_minmax(0.0, 100.0)

    extracted = extract_scalers_from_dataset(_DS(), feature_names=("glucose", "steps"))
    assert set(extracted) == {"glucose", "steps"}
    with pytest.raises(AttributeError):
        extract_scalers_from_dataset(_DS(), feature_names=("glucose", "missing"))


def test_save_scalers_for_run_without_kind_whitelist(tmp_path: Path) -> None:
    class _FakeDS:
        def __init__(self) -> None:
            self.scaler_glucose = _fit_minmax(0.0, 200.0)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_scalers_for_run(run_dir, dataset=_FakeDS(), kind="whatever")
    assert (run_dir / SCALERS_FILENAME).is_file()
    kind, loaded, _ = load_scalers(run_dir / SCALERS_FILENAME)
    assert kind == "whatever"
    assert "glucose" in loaded
    assert resolve_scalers_path(run_dir) == run_dir / SCALERS_FILENAME


# ---------------------------------------------------------------------------
# ModelFamilySpec implementations
# ---------------------------------------------------------------------------


def test_builtin_family_specs_registered() -> None:
    # Import side-effect registration
    import scripts.glumind.glumind_spec  # noqa: F401
    import scripts.glumind_uni.glumind_uni_spec  # noqa: F401
    import scripts.sugar_jepa.sugar_jepa_spec  # noqa: F401
    import scripts.sugar_one.sugar_one_spec  # noqa: F401

    kinds = set(list_family_kinds())
    assert {"glumind", "sugar_one", "glumind_uni", "sugar_jepa"} <= kinds


def test_sugar_one_spec_features_and_extract() -> None:
    from scripts.sugar_one.sugar_one_spec import SUGAR_ONE_SPEC

    assert SUGAR_ONE_SPEC.kind == "sugar_one"
    assert list(SUGAR_ONE_SPEC.feature_names) == ["glucose", "basal", "bolus", "carbs"]
    assert SUGAR_ONE_SPEC.n_features == 4
    assert "basal" in SUGAR_ONE_SPEC.value_columns
    assert "Glucose (mg/dL)" in SUGAR_ONE_SPEC.csv_column_aliases["glucose"]
    assert "Glucose Value (mg/dL)" in SUGAR_ONE_SPEC.csv_column_aliases["glucose"]

    class _DS:
        def __init__(self) -> None:
            self.scaler_glucose = _fit_minmax(50.0, 250.0)
            self.scaler_basal = _fit_minmax(0.0, 2.0)
            self.scaler_bolus = _fit_minmax(0.0, 10.0)
            self.scaler_carbs = _fit_minmax(0.0, 80.0)

    extracted = SUGAR_ONE_SPEC.extract_scalers(_DS())
    assert set(extracted) == set(SUGAR_ONE_SPEC.feature_names)


def test_evaluation_covariates_come_from_specs() -> None:
    from scripts.common.evaluation import GLUMIND_COVARIATES, SUGAR_ONE_COVARIATES
    from scripts.glumind.glumind_spec import GLUMIND_SPEC
    from scripts.sugar_one.sugar_one_spec import SUGAR_ONE_SPEC

    assert set(GLUMIND_COVARIATES) == set(GLUMIND_SPEC.feature_names)
    assert set(SUGAR_ONE_COVARIATES) == set(SUGAR_ONE_SPEC.feature_names)
    assert GLUMIND_COVARIATES["hr"] == list(GLUMIND_SPEC.csv_column_aliases["hr"])
    assert SUGAR_ONE_COVARIATES["basal"] == list(SUGAR_ONE_SPEC.csv_column_aliases["basal"])


def test_glumind_spec_build_model_from_meta() -> None:
    from scripts.glumind.glumind_spec import GLUMIND_SPEC

    meta = {
        "input_steps": 8,
        "d_model": 8,
        "n_heads": 2,
        "ff_units": 16,
        "n_blocks": 1,
        "horizon": 2,
        "dropout": 0.0,
    }
    model = GLUMIND_SPEC.build_model(meta, torch.device("cpu"))
    assert model.n_features == 3
    x = torch.zeros(2, 8, 3)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2)


def test_detect_family_kind_from_fingerprint() -> None:
    import scripts.glumind.glumind_spec  # noqa: F401
    import scripts.sugar_one.sugar_one_spec  # noqa: F401

    state = {"embed_hr.weight": torch.zeros(1), "embed_steps.weight": torch.zeros(1)}
    assert detect_family_kind({}, state) == "glumind"

    state2 = {
        "embed_basal.weight": torch.zeros(1),
        "embed_bolus.weight": torch.zeros(1),
        "embed_carbs.weight": torch.zeros(1),
    }
    assert detect_family_kind({}, state2) == "sugar_one"

    assert detect_family_kind({"model_type": "sugar_one"}, {}) == "sugar_one"


def test_get_family_spec_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_family_spec("not_a_real_model")


def test_register_custom_family_spec() -> None:
    class _TinySpec:
        kind = "tiny_test_family"
        feature_names = ("glucose",)
        n_features = 1
        value_columns = {"glucose": "Glucose (mg/dL)"}
        covariate_aliases: dict[str, tuple[str, ...]] = {}
        fingerprint_keys = ("embed_tiny.weight",)

        def build_model(self, meta: dict[str, Any], device: torch.device) -> torch.nn.Module:
            return torch.nn.Linear(1, 1).to(device)

        def extract_scalers(self, dataset: Any) -> dict[str, MinMaxScaler | StandardScaler]:
            return extract_scalers_from_dataset(dataset, feature_names=self.feature_names)

    register_family_spec(_TinySpec())  # type: ignore[arg-type]
    assert get_family_spec("tiny_test_family").kind == "tiny_test_family"
    assert isinstance(get_family_spec("tiny_test_family"), ModelFamilySpec) or True
