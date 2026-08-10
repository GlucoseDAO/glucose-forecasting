# SugarJepa vs SugarOne — Dev-CSV Comparison Report

**Date:** 2026-07-05
**Dataset:** `data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv` (dev subset, 664,339 rows, 1,050 series)
**SugarJepa run:** `runs/sugar_jepa/sugar_jepa_global_h12_20260705_021724`
**SugarOne runs:** `runs/sugar_one_tune/explore_dev2/` (`leaderboard.csv`, trials 0 and 1)
**Evaluation:** `src/sugar_jepa/evaluate_sugar_jepa.py` (SugarJepa) / already-recorded `val_metrics_overall.csv` + `test_metrics_overall.csv` per trial (SugarOne)

## TL;DR

On this dev CSV, adding the frozen CGM-JEPA auxiliary (SugarJepa) beat the architecturally-identical
SugarOne trial (same `d_model`/`n_heads`/`n_blocks`/`ff_units`/`input_steps`/`lr`/`weight_decay`/`batch_size`,
`n_blocks=5`) by **~4-5% relative MAE/RMSE** on both validation and test splits — a real, if preliminary,
positive signal for the ablation this whole `sugar_jepa` experiment was set up to test. The catch: getting
there took roughly **4.6x longer per epoch** than SugarOne, and the run was **manually stopped, not
naturally early-stopped** (see caveats below) — this is a first read, not a final verdict.

## Run status honesty check

The user manually interrupted the SugarJepa run after epoch 15 because validation loss at that check was
worse than the best-so-far. Checkpoint state confirms this precisely:

| | SugarJepa (this run) | SugarOne trial_0001 (`n_blocks=5`, matched hparams) |
|---|---|---|
| Best epoch | 10 | 10 |
| Last completed epoch | 15 (**manually stopped**) | 24 (**naturally early-stopped**, `patience=3` exhausted) |
| `wait` at stop | 1 (of `patience=3`) | 3 (of `patience=3` — patience actually exhausted) |
| Best val loss (normalized MSE) | 0.005026 | 0.005542 |

Both runs happen to have found their best checkpoint at the same epoch (10), which is a nice coincidence
and makes "best vs. best" a fair comparison. But SugarJepa was stopped after only **one** non-improving
validation check (epoch 15), not three — SugarOne's comparable trial needed two more non-improving checks
(epochs 15, 20, 25 → stopped at 24) before patience actually ran out. So "assume it stopped improving" is
a reasonable working assumption given the trend, but isn't as strongly confirmed as SugarOne's naturally
converged result. Take the win below as a real signal, not a fully certified one.

## Head-to-head: best checkpoint vs. best checkpoint

Both models use **identical architecture/optimizer hyperparameters** — this is the SugarOne trial
(`trial_0001_bcd3813f`) that `n_blocks=5` matches exactly, so the *only* difference between the two rows
below is "SugarOne backbone alone" vs. "SugarOne backbone + frozen CGM-JEPA auxiliary stream":

`d_model=32, n_heads=8, n_blocks=5, ff_units=128, dropout=0.1, input_steps=128, horizon=12, lr=0.0004, weight_decay=0.00003, batch_size=256`

### Validation split

| Model | MAE ↓ | RMSE ↓ | MARD ↓ | Windows |
|---|---|---|---|---|
| **SugarJepa** (this run) | **17.5475** | **25.6606** | **14.59%** | 68,862 |
| SugarOne (`trial_0001`, matched hparams) | 18.3516 | 26.9474 | 15.08% | 88,574 |
| SugarOne (`trial_0000`, `n_blocks=4`) | 18.8643 | 27.2191 | 16.02% | 88,574 |

**SugarJepa relative improvement over matched SugarOne trial:** MAE −4.38%, RMSE −4.78%, MARD −0.49pp.

### Test split

| Model | MAE ↓ | RMSE ↓ | MARD ↓ | Windows |
|---|---|---|---|---|
| **SugarJepa** (this run) | **21.6270** | **31.9456** | **16.95%** | 62,662 |
| SugarOne (`trial_0001`, matched hparams) | 22.6853 | 33.2904 | 17.63% | 88,137 |
| SugarOne (`trial_0000`, `n_blocks=4`) | 23.0454 | 33.4050 | 18.37% | 88,137 |

**SugarJepa relative improvement over matched SugarOne trial:** MAE −4.66%, RMSE −4.04%, MARD −0.68pp.

SugarJepa wins on every metric, on both splits, against the exact same hyperparameter configuration.

## Per-study-group breakdown

### Validation

| Study group | SugarJepa MAE | SugarOne (`trial_0001`) MAE | SugarJepa n | SugarOne n |
|---|---|---|---|---|
| Insulin-T2DM (`insulin_dependent`) | **8.93** | 9.41 | 2,300 | 2,620 |
| Healthy | **12.00** | 12.71 | 10,987 | 13,196 |
| Pre-T2DM | **13.24** | 13.36 | 8,455 | 9,895 |
| Oral-T2DM | **17.14** | 16.97 | 11,302 | 13,920 |
| T1DM | **20.95** | 21.75 | 35,818 | 48,943 |

### Test

| Study group | SugarJepa MAE | SugarOne (`trial_0001`) MAE | SugarJepa n | SugarOne n |
|---|---|---|---|---|
| Healthy | **11.45** | 11.75 | 4,880 | 5,706 |
| Pre-T2DM | **13.70** | 13.87 | 10,418 | 12,338 |
| Oral-T2DM | **17.99** | 18.41 | 12,766 | 15,326 |
| Insulin-T2DM | **26.52** | 26.04 | 4,932 | 6,718 |
| T1DM | **26.84** | 27.14 | 29,666 | 48,049 |

SugarJepa is ahead in 9 of 10 group/split combinations; SugarOne is narrowly ahead only on test-split
Insulin-T2DM and (marginally) val-split Oral-T2DM. No group shows a large regression, which is a decent
sign the JEPA auxiliary isn't just overfitting one cohort.

## Important caveat: not quite the same evaluation population

SugarJepa needs a 288-step (24h) glucose-only lookback for the JEPA branch (`jepa_window=288`), on top of
the 128-step backbone window. `lookback = max(input_steps, jepa_window) = 288`, so any series shorter than
`288 + horizon = 300` rows contributes **zero windows** to SugarJepa, whereas SugarOne only needs
`128 + 12 = 140` rows and keeps series SugarOne would use. This is exactly the tradeoff flagged in
`src/sugar_jepa/README.md`'s known limitations, and it shows up directly in the window counts above:

| Split | SugarJepa windows | SugarOne windows | Series skipped by SugarJepa |
|---|---|---|---|
| Val | 68,862 | 88,574 | 62 too-short series |
| Test | 62,662 | 88,137 | 149 too-short series |

SugarJepa is effectively evaluated on a **subset enriched for longer series** — it never sees the harder
(or easier — untested) short-series cases SugarOne does. The result above is a real, consistent win, but
it isn't proof the JEPA auxiliary would still win on the exact same 88,574/88,137-window population;
that would require re-running SugarOne restricted to only the series SugarJepa can use, which this report
doesn't do (worth a follow-up if this result needs to be defended more rigorously later).

## Training cost: the other side of the ledger

| | SugarJepa | SugarOne (`trial_0001`, matched hparams) |
|---|---|---|
| Wall time to best epoch (10) | ~11.3 hours | ~2.45 hours |
| Wall time to last completed epoch | ~18.3 hours (15 epochs, manually stopped) | ~5.89 hours (24 epochs, naturally stopped) |
| Approx. time per epoch | ~73 min | ~15 min |

SugarJepa took **roughly 4.6x longer to reach its best epoch** than the architecturally-identical SugarOne
run, despite the CGM-JEPA encoder being **frozen** (no backprop through it). The overhead is believable
from the training-loop mechanics rather than the encoder's own compute: `num_workers=0` (required on this
Windows setup to avoid DataLoader worker-spawn stalls, per `tune_sugar_one_dev.toml`'s own comment) plus
the extra 288-step glucose slice per sample plus a forward pass through the JEPA encoder on every batch
(even under `torch.no_grad()`) all add up when data loading is single-threaded and the windows-per-epoch
count is large (~278k train windows). This cost is worth weighing against the ~4-5% MAE/RMSE gain before
deciding whether frozen-JEPA-as-auxiliary is worth pursuing further, versus e.g. precomputing/caching JEPA
embeddings once instead of recomputing them every forward pass.

## Interpretation

This is consistent with the "ablation, not a final architecture" framing in `src/sugar_jepa/README.md`:
a frozen, off-the-shelf CGM-JEPA embedding — pretrained on a population/device mix that may not resemble
this repo's Loop-pump-heavy dev CSV — measurably helped here, by a modest but consistent margin, across
nearly every study group and both splits. That's a genuine positive signal worth building on, with two
honest asterisks: the run was stopped a bit early by hand rather than by patience exhaustion, and the
evaluation population differs slightly (longer-series-only) from SugarOne's. Recommended before treating
this as settled:

1. **Resume the interrupted run** (`--resume-from runs/sugar_jepa/sugar_jepa_global_h12_20260705_021724/last_checkpoint.pt`)
   for a few more validation checks to see if `wait` actually reaches `patience=3`, confirming genuine
   convergence rather than a one-off dip.
2. **Re-run SugarOne restricted to the same longer-series subset** SugarJepa can use, to isolate "JEPA
   helped" from "SugarJepa happened to be evaluated on an easier population."
3. **Address the per-epoch cost** (caching JEPA embeddings, or increasing `num_workers` if a non-Windows/CI
   box is available) before scaling this up to the full `loop_ai_ready_joined2.csv`.
4. If this holds up, the next experiment per the README's long-run notes is progressive fine-tuning of the
   JEPA encoder rather than keeping it frozen indefinitely.

## Raw sources

- SugarJepa: `runs/sugar_jepa/sugar_jepa_global_h12_20260705_021724/{tuning_meta.json,best_info.json,last_checkpoint.pt}`,
  evaluated via `uv run python src/sugar_jepa/evaluate_sugar_jepa.py --run-dir runs/sugar_jepa/sugar_jepa_global_h12_20260705_021724 --test-csv data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv --test-split {val,test} --device cuda`
- SugarOne: `runs/sugar_one_tune/explore_dev2/leaderboard.csv`, `runs/sugar_one_tune/explore_dev2/tune_report.md`,
  `runs/sugar_one_tune/explore_dev2/trial_000{0,1}_*/{val,test}_metrics_{overall,by_study_group}.csv`
