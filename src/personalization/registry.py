"""SugarOne-family model registry for personalization scripts.

Delegates to the family specs in ``common.model_spec`` (``sugar_one``,
``sugar_jepa2``). Personalization CLIs resolve architecture via this registry
rather than hard-coding ``SugarOneModel``.

Checkpoint fingerprints (including the ``exclude_keys`` that separate a family
from one extending it) come straight from the family spec. A family may differ
from SugarOne in two further ways that only personalization cares about, each an
optional ``ModelSpec`` hook: ``window_steps`` (a sliding window longer than
``input_steps``) and ``make_optimizer`` (extra param groups).

Only families whose dataset yields SugarOne's plain ``(x, y)`` batches belong
here — the fine-tune loop is SugarOne's. That admits ``sugar_jepa2``, which
reads one longer window, but not ``sugar_jepa``, whose dataset yields a second
``glucose_jepa`` tensor the loop would not pass on.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common.checkpoint import strip_compile_prefix
from common.model_spec import arch_hparams_from_meta, get_family_spec
from common.registry import load_run_meta, resolve_checkpoint
from sugar_one.sugar_one_spec import SUGAR_ONE_SPEC
from sugar_one.train_sugar_one import make_optimizer_and_scheduler

BuildModelFn = Callable[..., nn.Module]
WindowStepsFn = Callable[[dict[str, Any]], int]
# (model, lr, weight_decay, epochs, meta) -> (optimizer, scheduler, cfg_extras).
# ``cfg_extras`` records what the hook resolved (an encoder LR, a freeze) into
# the fine-tune run's config, so the run stays reproducible from its artifacts.
MakeOptimizerFn = Callable[
    [nn.Module, float, float, int, dict[str, Any]],
    tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, dict[str, Any]],
]


@dataclass(frozen=True)
class ModelSpec:
    """Descriptor for a SugarOne-schema model type used by personalization."""

    name: str
    n_features: int
    value_columns: dict[str, str]
    build: BuildModelFn
    fingerprint_keys: tuple[str, ...]
    # Keys that must be ABSENT — what separates a family from one that extends
    # it and therefore carries all of its fingerprint keys too.
    exclude_keys: tuple[str, ...] = ()
    # meta -> window length the dataset must emit (default: input_steps).
    window_steps: WindowStepsFn | None = None
    # Optimizer/scheduler factory (default: one AdamW group over every parameter).
    make_optimizer: MakeOptimizerFn | None = None


def _build_from_family(
    *,
    family_kind: str,
    input_steps: int,
    d_model: int,
    n_heads: int,
    ff_units: int,
    n_blocks: int,
    horizon: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    meta = {
        "input_steps": input_steps,
        "d_model": d_model,
        "n_heads": n_heads,
        "ff_units": ff_units,
        "n_blocks": n_blocks,
        "horizon": horizon,
        "dropout": dropout,
    }
    return get_family_spec(family_kind).build_model(meta, device)


def _jepa_submodule(model: nn.Module) -> nn.Module:
    """The JEPA encoder, unwrapping torch.compile if present."""
    return getattr(model, "_orig_mod", model).jepa_encoder


def _sugar_jepa2_window_steps(meta: dict[str, Any]) -> int:
    """One window long enough for both views; the model slices its own out.

    ``SugarJepaModel2.forward`` rejects anything but ``max(n_time_steps,
    jepa_window)`` steps, so the dataset must be built at that length even
    though the backbone only reads the trailing ``input_steps`` of it.
    """
    from sugar_jepa.sugar_jepa2_spec import jepa2_lookback

    return jepa2_lookback(int(meta.get("input_steps", 128)), meta)


def _sugar_jepa2_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    meta: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, dict[str, Any]]:
    """Which parameters train, and at what LR — the JEPA encoder gets its own group.

    Reuses the training script's factory rather than reimplementing it, so the
    exact-freeze semantics carry over: a frozen encoder is *dropped* from the
    optimizer, never given ``jepa_lr=0`` (the shared ``eta_min`` would anneal a
    zero-LR group upward). Freezing is applied here because it is the same
    decision as the param split. Unlike in training, it is always safe — the
    encoder comes from the base checkpoint, never from a random init.

    The encoder's LR follows the base run's ``jepa_lr / lr`` ratio rather than
    its absolute value, so an LR sweep moves both groups together instead of
    silently changing their balance. ``finetune_jepa_lr`` overrides it outright.
    """
    from sugar_jepa.train_sugar_jepa2 import (
        make_optimizer_and_scheduler as _jepa_make_optimizer,
    )

    frozen = bool(meta.get("freeze_jepa", False))
    if frozen:
        encoder = _jepa_submodule(model)
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

    override = meta.get("finetune_jepa_lr")
    if override is not None:
        jepa_lr = float(override)
    else:
        base_lr = float(meta.get("lr") or lr)
        base_jepa_lr = float(meta.get("jepa_lr", base_lr * 0.1))
        jepa_lr = lr * (base_jepa_lr / base_lr) if base_lr > 0 else base_jepa_lr

    cfg = {
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "freeze_jepa": frozen,
        "jepa_lr": jepa_lr,
    }
    optimizer, scheduler = _jepa_make_optimizer(model, cfg)
    # Record what was actually used, not what the base run used.
    return optimizer, scheduler, {"jepa_lr": jepa_lr, "freeze_jepa": frozen}


# Hooks per family, beyond what the shared family spec already provides.
_SPEC_EXTRAS: dict[str, dict[str, Any]] = {
    "sugar_jepa2": {
        "window_steps": _sugar_jepa2_window_steps,
        "make_optimizer": _sugar_jepa2_optimizer,
    },
}


def _spec_from_family(kind: str) -> ModelSpec:
    family = get_family_spec(kind)
    fields: dict[str, Any] = {
        "name": family.kind,
        "n_features": family.n_features,
        "value_columns": dict(family.value_columns),
        "build": lambda **kwargs: _build_from_family(family_kind=family.kind, **kwargs),
        "fingerprint_keys": tuple(family.fingerprint_keys),
        "exclude_keys": tuple(family.exclude_keys),
    }
    fields.update(_SPEC_EXTRAS.get(family.kind, {}))
    return ModelSpec(**fields)


# Resolved on demand: pulling in sugar_jepa2 imports the JEPA model, which the
# common sugar_one path has no reason to pay for.
_LAZY_KINDS: tuple[str, ...] = ("sugar_one", "sugar_jepa2")
_REGISTRY: dict[str, ModelSpec] = {}


def _ensure_registered(kind: str | None = None) -> None:
    for key in ([kind] if kind else list(_LAZY_KINDS)):
        if key is None or key in _REGISTRY or key not in _LAZY_KINDS:
            continue
        _REGISTRY[key] = _spec_from_family(key)


def register_model(spec: ModelSpec) -> None:
    """Register or replace a SugarOne-schema model type."""
    _REGISTRY[spec.name] = spec


def list_model_types() -> list[str]:
    _ensure_registered()
    return sorted(_REGISTRY.keys())


def get_model_spec(model_type: str) -> ModelSpec:
    key = model_type.strip().lower().replace("-", "_")
    _ensure_registered(key)
    if key not in _REGISTRY:
        known = ", ".join(list_model_types())
        raise ValueError(f"Unknown model type {model_type!r}. Known: {known}")
    return _REGISTRY[key]


def detect_model_type(meta: dict[str, Any], state: dict[str, torch.Tensor]) -> str:
    """Detect SugarOne-schema type from metadata or checkpoint keys."""
    explicit = meta.get("model_type") or meta.get("model")
    if explicit is not None:
        norm = str(explicit).lower().replace("-", "_")
        if norm in ("sugarone", "sugar_one", "glumind_ic", "glumindic"):
            return "sugar_one"
        if norm in ("sugarjepa2", "sugar_jepa_2", "sugarjepamodel2"):
            return "sugar_jepa2"
        _ensure_registered(norm)
        if norm in _REGISTRY:
            return norm

    # Fingerprinting has to consider every family, so this is where the lazy
    # JEPA import lands for a run whose metadata omits model_type.
    _ensure_registered()
    normalized_keys = {k.removeprefix("_orig_mod.") for k in state}
    for name, spec in _REGISTRY.items():
        if not all(k in normalized_keys for k in spec.fingerprint_keys):
            continue
        if any(k in normalized_keys for k in spec.exclude_keys):
            continue
        return name

    return "sugar_one"


def arch_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture hyperparameters from a run metadata dict."""
    return arch_hparams_from_meta(meta)


def window_steps_from_meta(model_type: str, meta: dict[str, Any]) -> int:
    """Sliding-window length the dataset must emit for this model.

    Usually ``input_steps``, but a model whose branches read different-length
    views of the same history (SugarJepa2) needs the longest of them.
    """
    spec = get_model_spec(model_type)
    if spec.window_steps is not None:
        return int(spec.window_steps(meta))
    return int(meta.get("input_steps", 128))


def make_finetune_optimizer(
    model_type: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    meta: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, dict[str, Any]]:
    """Optimizer + scheduler for a fine-tune run, plus cfg entries to record."""
    spec = get_model_spec(model_type)
    if spec.make_optimizer is not None:
        return spec.make_optimizer(model, lr, weight_decay, epochs, meta)
    optimizer, scheduler = make_optimizer_and_scheduler(model, lr, weight_decay, epochs)
    return optimizer, scheduler, {}


def build_model_from_meta(
    model_type: str,
    meta: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Build from the FULL metadata, not just the shared architecture block.

    ``ModelSpec.build`` takes only the seven shared hyperparameters, which would
    silently drop a family's own keys (``jepa_window`` and friends) and build the
    wrong shape. The family spec reads whatever it needs off ``meta`` itself.
    """
    get_model_spec(model_type)  # validates the type, and registers it
    return get_family_spec(model_type).build_model(meta, device)


def load_base_checkpoint(
    run_dir: Path,
    *,
    model_type: str | None = None,
    checkpoint: Path | None = None,
    device: torch.device | None = None,
    meta_overrides: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any], str, Path]:
    """Load a global (or prior) run for fine-tuning.

    ``meta_overrides`` replaces metadata keys before the model is built, for
    settings a fine-tune may choose differently from the base run (whether the
    JEPA encoder is frozen, say). The merged meta is what comes back, so callers
    record and reuse the values that were actually applied.

    Returns ``(model, meta, resolved_model_type, checkpoint_path)``.
    """
    run_dir = Path(run_dir)
    meta = load_run_meta(run_dir)
    if meta_overrides:
        meta = {**meta, **meta_overrides}
    ckpt_path = resolve_checkpoint(run_dir, checkpoint)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict dict in {ckpt_path}")
    state = strip_compile_prefix(state)

    resolved = model_type or detect_model_type(meta, state)
    dev = device or torch.device("cpu")
    model = build_model_from_meta(resolved, meta, dev)
    model.load_state_dict(state)
    return model, meta, resolved, ckpt_path


# Re-export for callers that previously imported value columns from constants only.
SUGAR_ONE_VALUE_COLUMNS = dict(SUGAR_ONE_SPEC.value_columns)
