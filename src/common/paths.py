#!/usr/bin/env python3
"""Shared filesystem path defaults for the project."""
from __future__ import annotations

from pathlib import Path

# Single output root for training / tuning / personalization runs.
# Do not use top-level ``runs/`` as a default destination.
DEFAULT_RUNS_ROOT: Path = Path("data") / "output" / "runs"
