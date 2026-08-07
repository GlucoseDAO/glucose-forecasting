"""Tests for train-fit scaler persistence (scalers.json sidecar)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scripts.common.scalers import (
    SCALERS_FILENAME,
    dump_scalers,
    load_scalers,
    resolve_scalers_path,
    save_scalers_for_run,
    scalers_match_transform,
    serialize_minmax,
    deserialize_minmax,
    serialize_standard,
    deserialize_standard,
)


def _fit_minmax(lo: float, hi: float) -> MinMaxScaler:
    return MinMaxScaler().fit(np.array([[lo], [hi]], dtype=np.float64))


def test_minmax_roundtrip_transform() -> None:
    original = _fit_minmax(40.0, 400.0)
    restored = deserialize_minmax(serialize_minmax(original))
    sample = np.array([[40.0], [100.0], [220.0], [400.0]])
    assert np.allclose(original.transform(sample), restored.transform(sample))
    assert np.allclose(
        original.inverse_transform(original.transform(sample)),
        restored.inverse_transform(restored.transform(sample)),
    )


def test_standard_roundtrip_transform() -> None:
    original = StandardScaler().fit(np.array([[80.0], [120.0], [160.0]]))
    restored = deserialize_standard(serialize_standard(original))
    sample = np.array([[90.0], [130.0], [150.0]])
    assert np.allclose(original.transform(sample), restored.transform(sample))


def test_dump_load_sugar_one(tmp_path: Path) -> None:
    scalers = {
        "glucose": _fit_minmax(50.0, 300.0),
        "basal": _fit_minmax(0.0, 2.0),
        "bolus": _fit_minmax(0.0, 10.0),
        "carbs": _fit_minmax(0.0, 80.0),
    }
    path = dump_scalers(
        tmp_path / SCALERS_FILENAME,
        kind="sugar_one",
        scalers=scalers,
        provenance={"csv": "data/example.csv", "n_rows": 123},
    )
    kind, loaded, payload = load_scalers(path)
    assert kind == "sugar_one"
    assert payload["provenance"]["n_rows"] == 123
    for name, original in scalers.items():
        assert scalers_match_transform(original, loaded[name])


def test_dump_load_glumind_and_jepa(tmp_path: Path) -> None:
    glumind = {
        "glucose": _fit_minmax(60.0, 250.0),
        "hr": _fit_minmax(50.0, 180.0),
        "steps": _fit_minmax(0.0, 500.0),
    }
    dump_scalers(tmp_path / "g.json", kind="glumind", scalers=glumind)
    kind, loaded, _ = load_scalers(tmp_path / "g.json")
    assert kind == "glumind"
    assert scalers_match_transform(glumind["hr"], loaded["hr"])

    jepa = {
        "glucose": _fit_minmax(60.0, 250.0),
        "basal": _fit_minmax(0.0, 1.5),
        "bolus": _fit_minmax(0.0, 8.0),
        "carbs": _fit_minmax(0.0, 60.0),
        "glucose_jepa": StandardScaler().fit(np.array([[70.0], [140.0]])),
    }
    dump_scalers(tmp_path / "j.json", kind="sugar_jepa", scalers=jepa)
    kind2, loaded2, _ = load_scalers(tmp_path / "j.json")
    assert kind2 == "sugar_jepa"
    assert scalers_match_transform(jepa["glucose_jepa"], loaded2["glucose_jepa"])


def test_save_scalers_for_run_and_resolve(tmp_path: Path) -> None:
    class _FakeDS:
        def __init__(self) -> None:
            self.scaler_glucose = _fit_minmax(0.0, 200.0)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_scalers_for_run(run_dir, kind="glumind_uni", dataset=_FakeDS())
    assert (run_dir / SCALERS_FILENAME).is_file()
    assert resolve_scalers_path(run_dir) == run_dir / SCALERS_FILENAME
    assert resolve_scalers_path(run_dir, {"scalers": SCALERS_FILENAME}) == run_dir / SCALERS_FILENAME
    assert resolve_scalers_path(tmp_path / "empty") is None


def test_dump_rejects_empty_scalers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        dump_scalers(tmp_path / "bad.json", kind="sugar_one", scalers={})


def test_sidecar_matches_refit_on_same_data() -> None:
    """Parity: serialize/load must match a fresh fit on the same train values."""
    train = np.array([[72.0], [95.0], [140.0], [210.0], [88.0]])
    fitted = MinMaxScaler().fit(train)
    restored = deserialize_minmax(serialize_minmax(fitted))
    assert scalers_match_transform(fitted, restored, sample=train)
    other = MinMaxScaler().fit(np.array([[10.0], [20.0]]))
    assert not scalers_match_transform(fitted, other, sample=train)
