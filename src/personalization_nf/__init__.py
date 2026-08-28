"""Personalization for NeuralForecast holdout models.

Zero-shot evaluation of ``data/output/runs/nf_holdout`` bundles, then continue
training (``use_init_models=False``) on one person's chronological train
slice. No LwF and no learning-rate search: fine-tuning reuses the source
run's optimizer settings and early-stops on a train-tail validation window.
"""

from __future__ import annotations

__all__: list[str] = []
