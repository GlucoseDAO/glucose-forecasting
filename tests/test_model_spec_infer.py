"""Tests for ModelFamilySpec.infer_batch helpers and shared columns."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from common.data.columns import (
    COL_GLU,
    COL_GLU_VALUE,
    COL_TS,
    COL_TS_SHORT,
    GLUMIND_CHANNELS,
    SUGAR_ONE_CHANNELS,
    TARGET_CHANNEL,
    TS_FORMAT,
)
from common.model_spec import (
    detect_family_kind,
    get_family_spec,
    infer_batch_jepa,
    infer_batch_xy,
)
from glumind.glumind_spec import GLUMIND_SPEC
from sugar_jepa.sugar_jepa2_spec import (
    CLI_DEFAULT_JEPA_WINDOW,
    jepa2_lookback,
    jepa2_window,
)
from sugar_jepa.sugar_jepa_spec import SUGAR_JEPA_SPEC
from sugar_one.sugar_one_spec import SUGAR_ONE_SPEC

# Vendored CGM-JEPA weights, as used by tests/test_sugar_jepa_smoke.py.
JEPA_WEIGHTS_DIR = "src/sugar_jepa/pretrained/cgm_jepa"
_TINY_ARCH = {
    "input_steps": 24,
    "d_model": 8,
    "n_heads": 2,
    "ff_units": 16,
    "n_blocks": 1,
    "horizon": 2,
    "dropout": 0.0,
}


class _XY(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, 0:1].expand(-1, 2)


class _Jepa(nn.Module):
    def forward(self, x: torch.Tensor, jepa: torch.Tensor) -> torch.Tensor:
        return x[:, -1, 0:1].expand(-1, 2) + jepa[:, -1:].mean() * 0.0


def test_infer_batch_xy_and_spec() -> None:
    model = _XY()
    x = torch.randn(3, 4, 3)
    y = torch.randn(3, 2)
    yt, yp = infer_batch_xy(model, (x, y), torch.device("cpu"))
    assert yt.shape == (3, 2)
    assert yp.shape == (3, 2)
    yt2, yp2 = GLUMIND_SPEC.infer_batch(model, (x, y), torch.device("cpu"))
    assert torch.allclose(yt, yt2)
    assert torch.allclose(yp, yp2)


def test_infer_batch_jepa_and_spec() -> None:
    model = _Jepa()
    x = torch.randn(2, 4, 4)
    jepa = torch.randn(2, 8)
    y = torch.randn(2, 2)
    yt, yp = infer_batch_jepa(model, (x, jepa, y), torch.device("cpu"))
    assert yt.shape == (2, 2)
    assert yp.shape == (2, 2)
    yt2, yp2 = SUGAR_JEPA_SPEC.infer_batch(model, (x, jepa, y), torch.device("cpu"))
    assert torch.allclose(yt, yt2)


def test_sugar_jepa2_spec_is_registered_and_distinguishable() -> None:
    """The two JEPA variants must not be confused for each other: same fusion
    block, different batch contract, different encoder — so a checkpoint of one
    loaded as the other would fail late and confusingly."""
    spec = get_family_spec("sugar_jepa2")
    assert spec.kind == "sugar_jepa2"
    # sugar_jepa carries a 5th scaler for its separate z-scored JEPA window;
    # sugar_jepa2 slices one window and needs none.
    assert "glucose_jepa" in SUGAR_JEPA_SPEC.feature_names
    assert "glucose_jepa" not in spec.feature_names
    assert not set(spec.fingerprint_keys) & set(SUGAR_JEPA_SPEC.fingerprint_keys)
    # (x, y), not (x, jepa, y)
    yt, yp = spec.infer_batch(_XY(), (torch.randn(3, 4, 4), torch.randn(3, 2)), torch.device("cpu"))
    assert yt.shape == yp.shape == (3, 2)


def test_sugar_jepa2_lookback_is_the_longer_of_the_two_views() -> None:
    """The dataset must be built at max(input_steps, jepa_window) — the model's
    forward pass rejects anything else."""
    assert jepa2_lookback(128, {"jepa_window": 288}) == 288
    assert jepa2_lookback(320, {"jepa_window": 288}) == 320


@pytest.mark.parametrize("meta", [None, {}, {"jepa_window": None}])
def test_sugar_jepa2_window_defaults_to_the_backbone_not_the_cli_default(meta) -> None:
    """Runs predating --jepa-window have no key; they ran at input_steps. Falling
    back to the CLI's 288 would build 36 encoder patches against a 16-patch
    checkpoint."""
    assert jepa2_window(128, meta) == 128
    assert jepa2_lookback(128, meta) == 128
    assert CLI_DEFAULT_JEPA_WINDOW == 288


def _real_state_keys(kind: str, meta: dict) -> dict[str, torch.Tensor]:
    """A checkpoint's actual key set — the only thing detection may rely on."""
    model = get_family_spec(kind).build_model(meta, torch.device("cpu"))
    return dict(model.state_dict())


@pytest.mark.parametrize(
    ("kind", "meta"),
    [
        ("sugar_one", _TINY_ARCH),
        ("glumind", _TINY_ARCH),
        ("sugar_jepa", {**_TINY_ARCH, "jepa_weights_dir": JEPA_WEIGHTS_DIR}),
        ("sugar_jepa2", {**_TINY_ARCH, "jepa_window": 24, "jepa_patch_size": 12}),
    ],
)
def test_detect_family_kind_on_a_real_state_dict(kind: str, meta: dict) -> None:
    """Detection must survive a FULL checkpoint, not just the fingerprint keys.

    Both SugarJEPA variants carry SugarOne's embed_basal/bolus/carbs weights on
    top of their own encoder. Fingerprinting on *any* matching key called such a
    checkpoint ``sugar_one``, and the load then blew up under ``strict=True``.
    """
    state = _real_state_keys(kind, meta)
    assert detect_family_kind({}, state) == kind
    # A torch.compile'd checkpoint carries the same keys behind a prefix.
    assert detect_family_kind({}, {f"_orig_mod.{k}": v for k, v in state.items()}) == kind


def test_detect_family_kind_prefers_explicit_model_type_over_fingerprints() -> None:
    """--model-type / tuning_meta.json wins; only a run missing it is fingerprinted."""
    state = _real_state_keys("sugar_jepa2", {**_TINY_ARCH, "jepa_window": 24, "jepa_patch_size": 12})
    assert detect_family_kind({"model_type": "sugar_jepa2"}, state) == "sugar_jepa2"
    assert detect_family_kind({"model_type": "sugar-jepa2"}, state) == "sugar_jepa2"
    # No model_type key at all (older tuning_meta.json) still resolves.
    assert detect_family_kind({"input_steps": 128}, state) == "sugar_jepa2"


def test_detect_family_kind_rejects_an_unrecognisable_checkpoint() -> None:
    """glumind_uni has no fingerprint of its own; an empty one must not match all."""
    assert not get_family_spec("glumind_uni").fingerprint_keys
    with pytest.raises(ValueError):
        detect_family_kind({}, {"linear.weight": torch.zeros(1)})


def test_get_family_spec_has_infer_batch() -> None:
    for kind in ("glumind", "sugar_one", "glumind_uni", "sugar_jepa", "sugar_jepa2"):
        spec = get_family_spec(kind)
        assert callable(spec.infer_batch)
        assert len(spec.ffill_bfill_columns) >= 1


def test_shared_columns_constants() -> None:
    assert TARGET_CHANNEL == "glucose"
    assert GLUMIND_CHANNELS[0] == "glucose"
    assert SUGAR_ONE_CHANNELS[-1] == "carbs"
    assert COL_GLU_VALUE.startswith("Glucose")
    assert COL_GLU.startswith("Glucose")
    assert COL_TS != COL_TS_SHORT
    assert "T" in TS_FORMAT
    assert SUGAR_ONE_SPEC.kind == "sugar_one"
