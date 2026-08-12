"""Tests for ModelFamilySpec.infer_batch helpers and shared columns."""
from __future__ import annotations

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
from common.model_spec import get_family_spec, infer_batch_jepa, infer_batch_xy
from glumind.glumind_spec import GLUMIND_SPEC
from sugar_jepa.sugar_jepa_spec import SUGAR_JEPA_SPEC
from sugar_one.sugar_one_spec import SUGAR_ONE_SPEC


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


def test_get_family_spec_has_infer_batch() -> None:
    for kind in ("glumind", "sugar_one", "glumind_uni", "sugar_jepa"):
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
