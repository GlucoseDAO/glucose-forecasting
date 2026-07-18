"""Tests for NeuralForecast terminal and persistent progress reporting."""
from __future__ import annotations

from glucose_forecasting.backends.neuralforecast.telemetry import _epoch_message


def test_epoch_message_reports_steps_and_losses() -> None:
    message = _epoch_message(
        "NHITS",
        epoch=2,
        step=100,
        max_steps=500,
        metrics={"train_loss": 12.34567, "valid_loss": 13.45678},
    )

    assert message == (
        "  Epoch 2 | step 100/500 | NHITS | "
        "train_loss=12.34567 valid_loss=13.45678"
    )
