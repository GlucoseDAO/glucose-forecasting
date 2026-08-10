"""Cheap regression guard for the frozen model architectures.

GluMindModel, SugarOneModel, GluMindUniModel are explicitly "don't touch
without asking" files. A shape/dtype/state_dict-key regression test is the
cheapest possible guard against an accidental architecture change.
"""
from __future__ import annotations

import torch

from glumind.glumind_model import GluMindModel
from glumind_uni.glumind_uni_model import GluMindUniModel
from sugar_one.sugar_one_model import SugarOneModel

BATCH = 3
INPUT_STEPS = 8
HORIZON = 2
D_MODEL = 8
N_HEADS = 2
N_BLOCKS = 1
FF_UNITS = 16


def test_glumind_forward_shape_and_dtype(tiny_glumind_model: GluMindModel) -> None:
    x = torch.randn(BATCH, INPUT_STEPS, 3)
    out = tiny_glumind_model(x)
    assert out.shape == (BATCH, HORIZON)
    assert out.dtype == torch.float32


def test_sugar_one_forward_shape_and_dtype(tiny_sugar_one_model: SugarOneModel) -> None:
    x = torch.randn(BATCH, INPUT_STEPS, 4)
    out = tiny_sugar_one_model(x)
    assert out.shape == (BATCH, HORIZON)
    assert out.dtype == torch.float32


def test_glumind_uni_forward_shape_and_dtype(tiny_glumind_uni_model: GluMindUniModel) -> None:
    x = torch.randn(BATCH, INPUT_STEPS, 1)
    out = tiny_glumind_uni_model(x)
    assert out.shape == (BATCH, HORIZON)
    assert out.dtype == torch.float32


def test_glumind_state_dict_key_patterns_stable(tiny_glumind_model: GluMindModel) -> None:
    keys = set(tiny_glumind_model.state_dict().keys())
    expected_prefixes = [
        "embed_glucose.",
        "embed_hr.",
        "embed_steps.",
        "pos_enc.",
        "blocks.0.cross_attn.",
        "blocks.0.multiscale.",
        "flatten_fc.",
        "out_fc.",
    ]
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in keys), f"missing expected key prefix {prefix!r}"


def test_sugar_one_state_dict_key_patterns_stable(tiny_sugar_one_model: SugarOneModel) -> None:
    keys = set(tiny_sugar_one_model.state_dict().keys())
    expected_prefixes = [
        "embed_glucose.",
        "embed_basal.",
        "embed_bolus.",
        "embed_carbs.",
        "blocks.0.cross_attn.mix_logits",
        "blocks.0.multiscale.",
        "flatten_fc.",
        "out_fc.",
    ]
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in keys), f"missing expected key prefix {prefix!r}"


def test_glumind_uni_state_dict_key_patterns_stable(tiny_glumind_uni_model: GluMindUniModel) -> None:
    keys = set(tiny_glumind_uni_model.state_dict().keys())
    expected_prefixes = [
        "embed_glucose.",
        "pos_enc.",
        "blocks.0.attn_high.",
        "blocks.0.attn_low2.",
        "blocks.0.attn_low4.",
        "flatten_fc.",
        "out_fc.",
    ]
    for prefix in expected_prefixes:
        assert any(k.startswith(prefix) for k in keys), f"missing expected key prefix {prefix!r}"
    # GluMindUni intentionally has no cross-attention / covariate embeddings.
    assert not any("cross_attn" in k for k in keys)
    assert not any(k.startswith("embed_hr") or k.startswith("embed_steps") for k in keys)


def test_glumind_param_count_matches_tiny_config(tiny_glumind_model: GluMindModel) -> None:
    n_params = sum(p.numel() for p in tiny_glumind_model.parameters())
    assert n_params > 0
    # Coarse tripwire: rebuilding with identical dims should be bit-identical count.
    rebuilt = GluMindModel(
        n_time_steps=INPUT_STEPS, n_features=3, d_model=D_MODEL, n_heads=N_HEADS,
        ff_units=FF_UNITS, n_blocks=N_BLOCKS, prediction_horizon=HORIZON, dropout=0.0,
    )
    assert sum(p.numel() for p in rebuilt.parameters()) == n_params
