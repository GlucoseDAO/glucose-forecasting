#!/usr/bin/env python3
"""Torch device resolution for evaluation CLIs."""
from __future__ import annotations

import torch


def resolve_torch_device(device: str = "auto") -> str:
    """Resolve ``auto`` to cuda → mps → cpu; otherwise return ``device`` unchanged."""
    text = (device or "auto").strip().lower()
    if text != "auto":
        return text
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
