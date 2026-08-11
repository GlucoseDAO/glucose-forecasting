"""Tests for packing training run dirs into format-1.0 release bundles."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from common.model_spec import get_family_spec
from common.release import load_inference_bundle, pack_run_dir, validate_inference_bundle
from glumind.glumind_model import GluMindModel


def _write_minmax_scaler(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "glumind",
                "features": {
                    "glucose": {
                        "type": "minmax",
                        "data_min": [0.0],
                        "data_max": [400.0],
                        "feature_range": [0, 1],
                        "n_features_in": 1,
                        "scale": [0.0025],
                        "min": [0.0],
                    },
                    "hr": {
                        "type": "minmax",
                        "data_min": [0.0],
                        "data_max": [200.0],
                        "feature_range": [0, 1],
                        "n_features_in": 1,
                        "scale": [0.005],
                        "min": [0.0],
                    },
                    "steps": {
                        "type": "minmax",
                        "data_min": [0.0],
                        "data_max": [10000.0],
                        "feature_range": [0, 1],
                        "n_features_in": 1,
                        "scale": [0.0001],
                        "min": [0.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _make_glumind_run(run_dir: Path) -> GluMindModel:
    run_dir.mkdir(parents=True)
    meta = {
        "horizon": 2,
        "input_steps": 8,
        "d_model": 16,
        "n_heads": 4,
        "ff_units": 32,
        "n_blocks": 1,
        "dropout": 0.0,
        "seed": 7,
        "csv": "data/input/example.csv",
        "scalers": "scalers.json",
    }
    (run_dir / "tuning_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    _write_minmax_scaler(run_dir / "scalers.json")
    (run_dir / "val_metrics_overall.csv").write_text(
        "mae,rmse,mard\n10.0,12.0,8.0\n", encoding="utf-8"
    )
    (run_dir / "test_metrics_overall.csv").write_text(
        "mae,rmse,mard\n11.0,13.0,9.0\n", encoding="utf-8"
    )
    torch.manual_seed(0)
    model = GluMindModel(
        n_time_steps=8,
        n_features=3,
        d_model=16,
        n_heads=4,
        ff_units=32,
        n_blocks=1,
        prediction_horizon=2,
        dropout=0.0,
    )
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    return model


def test_pack_run_dir_glumind_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source = _make_glumind_run(run_dir)
    bundle_dir = tmp_path / "bundle"

    manifest = pack_run_dir(
        run_dir,
        bundle_dir,
        release_id="glumind-test",
        project_root=tmp_path,
    )
    assert manifest.release_id == "glumind-test"
    assert manifest.config.model_type == "glumind"
    assert manifest.config.feature_order == ("glucose", "hr", "steps")
    assert manifest.preprocessor.window.input_steps == 8
    assert manifest.preprocessor.scalers["glucose"].kind == "minmax"
    assert manifest.metrics.validation["mae"] == 10.0
    assert manifest.metrics.test["mae"] == 11.0
    assert validate_inference_bundle(bundle_dir).release_id == "glumind-test"

    loaded = load_inference_bundle(
        bundle_dir,
        model_factory=lambda cfg: get_family_spec(cfg.model_type).build_model(
            {**cfg.architecture, "horizon": cfg.horizon},
            torch.device("cpu"),
        ),
    )
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, loaded.model.state_dict()[name])
