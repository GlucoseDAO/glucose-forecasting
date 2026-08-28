#!/usr/bin/env python3
"""Zero-shot eval + continue-fit one NeuralForecast bundle on personal data."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from common.console import init_cli_console, safe_echo
from nf_baselines.adapter import PreparedSplits, filter_minimum_length
from nf_baselines.benchmark import to_neuralforecast_frame
from nf_baselines.config import NeuralForecastRunConfig, neuralforecast_frequency
from nf_baselines.evaluations.holdout import (
    evaluate_prepared_split,
    resolve_device,
)
from personalization.constants import DEFAULT_LIVIA_PREPARED_CSV
from personalization_nf.constants import (
    DEFAULT_FT_PATIENCE,
    DEFAULT_NF_HOLDOUT_ROOT,
    DEFAULT_NF_PERSONALIZATION_ROOT,
    METRICS_FILENAME,
    VAL_TAIL_FRACTION,
    ZERO_SHOT_DIRNAME,
)
from personalization_nf.data import (
    choose_val_size,
    day_label,
    filter_train_for_fit,
    limit_train_calendar_days,
    load_personal_splits,
    metrics_to_dict,
    train_span_for_csv,
    used_train_days,
    window_sizes_from_config,
)
from personalization_nf.discover import NfHoldoutRun, discover_holdout_runs, parse_model_filter

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def run_config_for_personal_csv(
    source: dict[str, Any],
    personal_csv: Path,
    *,
    device: str,
) -> NeuralForecastRunConfig:
    """Rebuild a holdout config pointing at the personal CSV."""
    payload = dict(source)
    payload["csv"] = str(personal_csv)
    payload["device"] = device
    if payload.get("holdout_protocol") is None:
        payload["holdout_protocol"] = "dense"
    return NeuralForecastRunConfig.model_validate(payload)


def load_bundle(bundle_dir: Path, freq: str) -> Any:
    from neuralforecast import NeuralForecast

    neuralforecast = NeuralForecast.load(str(bundle_dir))
    neuralforecast.freq = neuralforecast_frequency(freq)
    return neuralforecast


def configure_finetune(
    neuralforecast: Any,
    *,
    device: str,
    run_dir: Path,
    max_steps: int,
    val_check_steps: int,
    patience: int,
    enable_early_stopping: bool,
) -> None:
    """Point the loaded models at a new trainer budget without reinitializing weights."""
    trainer_kwargs = {
        "accelerator": device,
        "devices": 1,
        "default_root_dir": str(run_dir),
        "max_steps": max_steps,
        "enable_checkpointing": False,
        "logger": False,
        "enable_progress_bar": False,
    }
    resolved_val_check = min(max(1, val_check_steps), max_steps)
    for model in neuralforecast.models:
        model.max_steps = max_steps
        model.val_check_steps = resolved_val_check
        model.early_stop_patience_steps = patience if enable_early_stopping else -1
        existing = dict(getattr(model, "trainer_kwargs", None) or {})
        existing.update(trainer_kwargs)
        model.trainer_kwargs = existing


def score_split(
    neuralforecast: Any,
    *,
    frame: Any,
    splits: PreparedSplits,
    config: NeuralForecastRunConfig,
    horizon: int,
    input_size: int,
) -> dict[str, float] | None:
    scored = evaluate_prepared_split(
        neuralforecast,
        frame=frame,
        profile_exogenous=splits.profile.historical_exogenous,
        horizon=horizon,
        input_size=input_size,
        holdout_protocol=config.holdout_protocol,
        max_eval_series=config.max_eval_series,
        mask_interpolated_targets=config.mask_interpolated_targets,
    )
    if scored is None:
        return None
    metrics, _evaluated = scored
    safe_echo(
        f"MAE={metrics.overall.mae:.3f} RMSE={metrics.overall.rmse:.3f} "
        f"MARD={metrics.overall.mard:.3f}%"
    )
    return metrics_to_dict(metrics)


def zero_shot_cache_path(subject_model_dir: Path) -> Path:
    return subject_model_dir / ZERO_SHOT_DIRNAME / METRICS_FILENAME


def load_cached_zero_shot(subject_model_dir: Path) -> dict[str, Any] | None:
    path = zero_shot_cache_path(subject_model_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("zero_shot_test") is None:
        return None
    return payload


def write_cached_zero_shot(subject_model_dir: Path, payload: dict[str, Any]) -> None:
    path = zero_shot_cache_path(subject_model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def adapt_run_complete(run_dir: Path) -> bool:
    metrics_path = run_dir / METRICS_FILENAME
    if not metrics_path.is_file():
        return False
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and payload.get("finetuned_test") is not None


def run_adapt(
    *,
    holdout: NfHoldoutRun,
    personal_csv: Path,
    out_dir: Path,
    subject: str,
    personal_days: int | None,
    device: str,
    patience: int = DEFAULT_FT_PATIENCE,
    max_steps: int | None = None,
    val_check_steps: int | None = None,
    val_tail_fraction: float = VAL_TAIL_FRACTION,
    skip_completed: bool = True,
    eval_zero_shot: bool = True,
    subject_model_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Continue-train one holdout bundle on a personal day budget.

    Zero-shot metrics are computed once per subject×model and reused across
    day budgets. Fine-tuning always starts from the saved global bundle, not
    from a shorter-day student.
    """
    out_dir = Path(out_dir)
    if skip_completed and adapt_run_complete(out_dir):
        results = json.loads((out_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
        return out_dir, results

    config = run_config_for_personal_csv(holdout.config, personal_csv, device=device)
    horizon, input_size, configured_val_size = window_sizes_from_config(config)
    splits = load_personal_splits(personal_csv, source_config=config)
    full_train_span = train_span_for_csv(personal_csv, splits.train)
    train = limit_train_calendar_days(splits.train, personal_days)
    used_days = used_train_days(train, personal_days)
    val_size = choose_val_size(
        train,
        input_size=input_size,
        horizon=horizon,
        configured_val_size=configured_val_size,
        val_tail_fraction=val_tail_fraction,
    )
    train_fit = filter_train_for_fit(
        train, input_size=input_size, horizon=horizon, val_size=val_size
    )
    if train_fit.is_empty():
        raise ValueError(
            f"No train series remain for {subject} days={day_label(personal_days)} "
            f"after minimum-length filtering (input={input_size}, horizon={horizon}, "
            f"val_size={val_size})"
        )

    test_frame = filter_minimum_length(splits.test, input_size + horizon)
    if test_frame.is_empty():
        raise ValueError(f"No test windows remain for {subject}")

    resolved_device = resolve_device(config.device)
    steps = int(max_steps if max_steps is not None else config.max_steps)
    check_steps = int(
        val_check_steps if val_check_steps is not None else config.val_check_steps
    )
    cache_root = subject_model_dir if subject_model_dir is not None else out_dir.parent
    cached_zs = load_cached_zero_shot(cache_root) if eval_zero_shot else None

    out_dir.mkdir(parents=True, exist_ok=True)
    neuralforecast = load_bundle(holdout.bundle_dir, config.freq)

    zero_shot_test: dict[str, float] | None = None
    zero_shot_val: dict[str, float] | None = None
    if eval_zero_shot:
        if cached_zs is not None:
            zero_shot_test = cached_zs.get("zero_shot_test")
            zero_shot_val = cached_zs.get("zero_shot_val")
            safe_echo(f"Reusing cached zero-shot metrics from {zero_shot_cache_path(cache_root)}")
        else:
            safe_echo(f"Zero-shot eval: {holdout.model_key} on {subject}")
            zero_shot_test = score_split(
                neuralforecast,
                frame=test_frame,
                splits=splits,
                config=config,
                horizon=horizon,
                input_size=input_size,
            )
            val_frame = filter_minimum_length(splits.validation, input_size + horizon)
            if not val_frame.is_empty():
                zero_shot_val = score_split(
                    neuralforecast,
                    frame=val_frame,
                    splits=splits,
                    config=config,
                    horizon=horizon,
                    input_size=input_size,
                )
            write_cached_zero_shot(
                cache_root,
                {"zero_shot_test": zero_shot_test, "zero_shot_val": zero_shot_val},
            )

    enable_es = val_size > 0
    configure_finetune(
        neuralforecast,
        device=resolved_device,
        run_dir=out_dir,
        max_steps=steps,
        val_check_steps=check_steps,
        patience=patience,
        enable_early_stopping=enable_es,
    )
    safe_echo(
        f"Fine-tune {holdout.model_key} {subject} days={day_label(personal_days)} "
        f"device={resolved_device} max_steps={steps} val_size={val_size} "
        f"patience={patience if enable_es else 0}"
    )
    neuralforecast.fit(
        df=to_neuralforecast_frame(train_fit, splits.profile.historical_exogenous),
        val_size=val_size,
        use_init_models=False,
    )

    finetuned_test = score_split(
        neuralforecast,
        frame=test_frame,
        splits=splits,
        config=config,
        horizon=horizon,
        input_size=input_size,
    )
    if finetuned_test is None:
        raise ValueError(f"Fine-tuned eval produced no test windows for {subject}")
    val_frame = filter_minimum_length(splits.validation, input_size + horizon)
    finetuned_val = None
    if not val_frame.is_empty():
        finetuned_val = score_split(
            neuralforecast,
            frame=val_frame,
            splits=splits,
            config=config,
            horizon=horizon,
            input_size=input_size,
        )

    results: dict[str, Any] = {
        "config": {
            "model_key": holdout.model_key,
            "source_run_dir": str(holdout.run_dir),
            "personal_csv": str(personal_csv.resolve()),
            "subject": subject,
            "personal_days": day_label(personal_days),
            "train_span_days": full_train_span,
            "used_train_days": used_days,
            "device": resolved_device,
            "max_steps": steps,
            "val_check_steps": check_steps,
            "val_size": val_size,
            "patience": patience if enable_es else 0,
            "learning_rate": config.learning_rate,
            "holdout_protocol": config.holdout_protocol,
            "input_size": input_size,
            "horizon": horizon,
            "protocol": "continue_fit",
        },
        "zero_shot_test": zero_shot_test,
        "zero_shot_val": zero_shot_val,
        "finetuned_test": finetuned_test,
        "finetuned_val": finetuned_val,
    }
    (out_dir / METRICS_FILENAME).write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "run_config.json").write_text(
        json.dumps(results["config"], indent=2), encoding="utf-8"
    )
    return out_dir, results


@app.command()
def main(
    holdout_root: Path = typer.Option(DEFAULT_NF_HOLDOUT_ROOT, "--holdout-root"),
    personal_csv: Path = typer.Option(DEFAULT_LIVIA_PREPARED_CSV, "--personal-csv"),
    out_dir: Path = typer.Option(
        DEFAULT_NF_PERSONALIZATION_ROOT / "livia" / "NHITS" / "days_all",
        "--out-dir",
    ),
    subject: str = typer.Option("livia", "--subject"),
    models: Optional[str] = typer.Option(
        None, "--models", help="Comma-separated model names; default: all in holdout root."
    ),
    personal_days: Optional[int] = typer.Option(None, "--personal-days"),
    device: str = typer.Option("auto", "--device"),
    patience: int = typer.Option(DEFAULT_FT_PATIENCE, "--patience"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    skip_completed: bool = typer.Option(True, "--skip-completed/--no-skip-completed"),
) -> None:
    """Continue-fit discovered holdout models on one personal CSV."""
    init_cli_console()
    wanted = parse_model_filter(models)
    runs = discover_holdout_runs(holdout_root, models=wanted)
    for holdout in runs:
        model_out = out_dir if len(runs) == 1 else out_dir / holdout.model_key
        cache_root = model_out.parent
        run_dir, results = run_adapt(
            holdout=holdout,
            personal_csv=personal_csv,
            out_dir=model_out,
            subject=subject,
            personal_days=personal_days,
            device=device,
            patience=patience,
            max_steps=max_steps,
            skip_completed=skip_completed,
            subject_model_dir=cache_root,
        )
        ft = results.get("finetuned_test") or {}
        zs = results.get("zero_shot_test") or {}
        safe_echo(
            f"{holdout.model_key} {subject} days={day_label(personal_days)} "
            f"ZS MAE={zs.get('mae')} FT MAE={ft.get('mae')} -> {run_dir}"
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
