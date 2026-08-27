# SugarJepa

SugarOne + a JEPA glucose embedding as a 4th cross-attention auxiliary stream
(basal / bolus / carbs / **jepa**, learnable softmax mix). See `sugar_jepa_model.py` for the
architectures.

There are **two variants** in this folder. They share the fusion block and differ only in where the
glucose encoder comes from and, consequently, in their data contract:

| | `sugar_jepa` | `sugar_jepa2` |
|---|---|---|
| Encoder | pretrained [CGM-JEPA](https://github.com/cruiseresearchgroup/CGM-JEPA), vendored under `vendor/cgm_jepa/` | `JepaEncoder` — ours, trained by us (`jepa_pretrain.py`) |
| Model class | `SugarJepaModel` | `SugarJepaModel2` |
| Trainer | `train_sugar_jepa.py` | `train_sugar_jepa2.py` |
| Batch | `(x, glucose_jepa, y)` — second tensor, own z-score scaler | `(x, y)` — one long window, no fifth scaler |
| Spec | `sugar_jepa_spec.py` | `sugar_jepa2_spec.py` |

Both are `global` training mode only, and both evaluate through the unified
`uv run glucose evaluate --model-type <kind>` path.

---

# Variant 1 — `sugar_jepa`: pretrained CGM-JEPA (proof of concept)

We port the *weights*: a frozen, off-the-shelf glucose embedding. See `vendor/cgm_jepa/NOTICE.md` for
what's vendored and why.

## Run (exact command)

```bash
uv run python src/sugar_jepa/train_sugar_jepa.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda \
  --d-model 32 --n-heads 8 --n-blocks 5 --ff-units 128 --input-steps 128 --horizon 12 \
  --lr 0.0004 --weight-decay 0.00003 --batch-size 256 \
  --epochs 30 --patience 3 --val-every-n-epochs 5 --num-workers 0 \
  --out-dir data/output/runs/sugar_jepa
```

(These are also the defaults baked into `train_sugar_jepa.py`'s CLI — you can omit all of them and just
run `--csv ... --device cuda`. They're spelled out here so it's obvious at a glance where they come from.
Flags are kebab-case, e.g. `--batch-size` not `--batch_size` — run `--help` to see the full, generated
list.)

That's it — the JEPA encoder loads its pretrained weights from the local, already-downloaded
`src/sugar_jepa/pretrained/cgm_jepa/` (no network needed), stays **frozen** by default. The
**model architecture + optimizer** hyperparameters above (`d-model`/`n-heads`/`n-blocks`/`ff-units`/
`input-steps`/`horizon`/`lr`/`weight-decay`/`batch-size`) are copied verbatim from
[`src/sugar_one/tune_sugar_one_full.toml`](../sugar_one/tune_sugar_one_full.toml) — the best-tuned
SugarOne values — so results are directly comparable to the tuned SugarOne production run. The
**training-loop control** values (`epochs`/`patience`/`val-every-n-epochs`/`num-workers`), by contrast,
come from [`tune_sugar_one_dev.toml`](../sugar_one/tune_sugar_one_dev.toml), not `..._full.toml`:
`tune_sugar_one_full.toml`'s epochs=120/patience=10 are sized for a production run against the full
~1GB CSV and would take far too long here. Since this run is against the dev CSV for a fast comparison,
it uses dev.toml's shorter loop (epochs=30, patience=3, validate every 5 epochs, `num_workers=0` to avoid
Windows DataLoader worker-spawn stalls) instead. Both source values live side by side in
[`tune_sugar_jepa_full.toml`](tune_sugar_jepa_full.toml) (`[defaults]` from full.toml, `[train]` from
dev.toml) for reference.

It checkpoints every epoch to `data/output/runs/sugar_jepa/sugar_jepa_global_h12_<timestamp>/` (`best_model.pt`,
`last_checkpoint.pt`) and writes `val_metrics_overall.csv` / `test_metrics_overall.csv` at the end, so if
it's still running later you can `Ctrl+C` and resume with `--resume-from
data/output/runs/sugar_jepa/.../last_checkpoint.pt`, or just evaluate whatever `best_model.pt` exists so far.

Rough sizing: the dev CSV has 1050 series (55MB); with a 288-step JEPA lookback, ~59% of series are long
enough to contribute windows (shorter ones are skipped — this is logged). With epochs=30/patience=3 (vs.
the production 120/10), this should finish in well under an hour on a single consumer GPU (RTX
3060-class) — validate-every-5-epochs plus early stopping at patience=3 means it can stop as early as
~15-20 epochs in.

## Evaluate a finished run

```bash
uv run glucose evaluate \
  --run-dir data/output/runs/sugar_jepa/<run_name> \
  --model-type sugar_jepa \
  --data data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda \
  --no-plot
```

Reports MAE / RMSE / MARD. For a same-CSV SugarOne baseline:

```bash
uv run glucose evaluate \
  --data data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --run-dir <a SugarOne run trained on the same dev CSV> \
  --model-type sugar_one \
  --no-plot
```

## Fast smoke test (no GPU wait, ~1 minute)

Before a long run, sanity-check the pipeline end-to-end on a tiny slice:

```bash
uv run python src/sugar_jepa/train_sugar_jepa.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda --epochs 2 --max-train-series 20 --max-eval-series 10 \
  --batch-size 32 --n-blocks 2 --out-dir data/output/runs/sugar_jepa_smoketest
```

Unit/shape tests: `uv run pytest tests/test_sugar_jepa_smoke.py -q`.

## What's different from SugarOne

- `CrossAttentionSugarJepaBlock` mixes **4** auxiliaries (basal, bolus, carbs, jepa) via a learnable
  softmax weight, vs. SugarOne's 3-way mix.
- `JepaEncoderWrapper` runs a pretrained CGM-JEPA `Encoder` (3 layers, embed_dim 96, vendored in
  `vendor/cgm_jepa/`) over a **separate, longer glucose-only lookback window** (`--jepa-window`, default
  288 steps = 24h — the window CGM-JEPA was pretrained on) each forward pass, independent of the model's
  own `--input-steps` (default 128, unchanged from SugarOne). Frozen by default (feature-extractor mode);
  pass `--finetune-jepa` to unfreeze it (with its own, smaller `--jepa-lr`, default 4e-5).
- The JEPA branch's glucose is **z-score normalized** (its own scaler, fit on train only), not
  MinMax-scaled like the rest of the model — CGM-JEPA was pretrained on z-scored inputs, and feeding it
  `[0,1]`-scaled values would be out of distribution for the pretrained weights.
- Everything else (multi-scale self-attention, flatten + output head, MSE loss, MinMax scalers for
  glucose/basal/bolus/carbs, imputation policy) is identical to `src/sugar_one/sugar_one_model.py` /
  `train_sugar_one.py`.

## Known limitations (proof of concept, not a final architecture)

- **~41% of series are too short for the native 288-step JEPA window** and contribute zero training
  windows (logged as "Skipped N series shorter than..."). SugarJepa was scored on 68,862 dev-CSV val
  windows against SugarOne's 88,574 — the two were never evaluated on the same data, which is why the
  4–5% MAE win in [`docs/SUGAR_JEPA_VS_SUGAR_ONE_DEV_COMPARISON.md`](../../docs/SUGAR_JEPA_VS_SUGAR_ONE_DEV_COMPARISON.md)
  is not usable as a baseline. Set `--jepa-window` equal to `--input-steps` to get matched populations.
- **Domain shift is untested**: CGM-JEPA's pretraining cohort/device mix may not resemble this repo's
  Loop-pump T1DM-heavy population. Treat the first run's results as an ablation ("does a frozen,
  off-the-shelf glucose embedding help at all here?"), not a verdict on the architecture. Variant 2 below
  is the answer to "then pretrain the encoder on *our* data".
- No hyperparameter tuner (`tune_sugar_jepa.py`) — `tune_sugar_jepa_full.toml` is a reference config, not
  a script input; there's no random-search loop like `tune_sugar_one.py`'s yet.

## Why local pretrained weights instead of `Encoder.from_pretrained("CRUISEResearchGroup/CGM-JEPA", ...)`

This started as a workaround for a Windows TLS crash on the original dev machine; that crash is now
root-caused and fixed (see `src/common/network.py`), so Hub-based loading works fine there too. The
vendored copy under `src/sugar_jepa/pretrained/cgm_jepa/` (`config.json` + `model.safetensors`, ~4MB,
MIT license) stays the default anyway — it's already fast, offline, and reproducible, so there's no
reason to add a network dependency for this proof of concept. Passing `--jepa-weights-dir` a Hub repo id
still works (`from_pretrained` accepts a Hub id or local path interchangeably) if you want it. See
`vendor/cgm_jepa/NOTICE.md` for the full explanation of the two TLS issues that were fixed.

---

# Variant 2 — `sugar_jepa2`: our own JEPA encoder

We port the *architecture*, not the weights. `JepaEncoder` in
[`sugar_jepa_model.py`](sugar_jepa_model.py) is a plain patch transformer — Conv1d patchify, sinusoidal
positions, pre-norm blocks — trained by us, either end-to-end from random init or from a self-supervised
checkpoint we produce ourselves with [`jepa_pretrain.py`](jepa_pretrain.py). No `from_pretrained`, no
`safetensors`, no network.

The JEPA branch reads a **longer glucose-only lookback** than the backbone (`--jepa-window`, default 288 =
24h; the backbone stays at `--input-steps`, default 128). Both views end at the same instant, so the
dataset emits ONE window of `max(input_steps, jepa_window)` steps and the model takes the trailing slice
each branch needs:

```
x            |<-------------- lookback = 288 -------------->| now
JEPA branch  |<-------------- jepa_window = 288 ----------->|
backbone                     |<-- input_steps = 128 ------->|
```

The dataset contract therefore stays SugarOne's plain `(x, y)` — no second tensor, no second scaler, no
bespoke training loop — and `SugarOneWindowDataset` is used as-is, just built at the longer lookback. The
cost is that series shorter than `lookback + horizon` (300 rows by default) contribute **no windows**, so
SugarJepa is evaluated on a population enriched for longer series relative to SugarOne. Set
`--jepa-window` equal to `--input-steps` to get the old single-window behaviour back.

Scope: `global` training mode only.

## The three commands

### 1. Train end-to-end (random init)

```bash
uv run python src/sugar_jepa/train_sugar_jepa2.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda \
  --d-model 32 --n-heads 8 --n-blocks 5 --ff-units 128 --input-steps 128 --horizon 12 \
  --lr 0.0004 --weight-decay 0.00003 --batch-size 256 \
  --epochs 30 --patience 3 --val-every-n-epochs 5 --num-workers 0 \
  --jepa-window 288 --jepa-patch-size 8 --jepa-embed-dim 96 --jepa-layers 3 --jepa-heads 6 --jepa-lr 4e-5 \
  --out-dir data/output/runs/sugar_jepa2
```

All of these are the CLI defaults, so `--csv ... --device cuda` alone reproduces it. Flags are kebab-case
(Typer); `--help` prints the full list.

By default the encoder trains alongside the backbone, in its own smaller-LR optimizer group
(`--jepa-lr`) — which is why the startup banner reports two param counts:

```
Training 367,840 out of 367,840 SugarOne params @ lr=0.0004
Training 336,576 out of 336,576 JEPA params @ lr=4e-05
```

Pass **`--freeze-jepa`** (only valid together with `--jepa-init`) to hold a pretrained encoder fixed
instead. Its parameters get `requires_grad=False` and are dropped from the optimizer entirely, so the
banner's second line reads `Training 0 out of 336,576 JEPA params (frozen)` and the encoder tensors in
`best_model.pt` are bit-identical to the `--jepa-init` file. The flag is recorded in `tuning_meta.json`.

`--jepa-lr 0` is **not** an equivalent. `CosineAnnealingLR` takes a single `eta_min` shared by every param
group, so a zero-base group anneals *upward* toward `lr * 0.01` — measured at the defaults, the encoder's
LR climbs from ~1e-8 at epoch 1 to 4e-6 by epoch 30, the same value the backbone ends at. Dropping the
group is the only exact freeze.

`jepa_window` must be divisible by `jepa_patch_size` (288 / 8 = 36 patches of 40 min); it fails up front
if not. `input_steps` is unconstrained — the backbone does not patchify.

### 2. Pretrain the encoder (JEPA self-supervision)

```bash
uv run python src/sugar_jepa/jepa_pretrain.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv --device cuda \
  --window 288 --window-stride 4 --epochs 50 --batch-size 256 \
  --patch-size 8 --embed-dim 96 --n-layers 3 --n-heads 6
```

Masked latent prediction: an EMA target encoder encodes all 36 patches, the context encoder sees only the
unmasked ones, and a narrow predictor — given just the *positions* of the masked blocks — must reproduce
their latents. Loss is smooth-L1 **in latent space**; nothing reconstructs glucose values.

`--window` is the JEPA branch's own lookback (default 288 = 24h), independent of the forecaster's
`--input-steps` and matched by `--jepa-window` at fine-tune. Target-block sizes default to a *fraction*
of the patch sequence (`n_patches/8` to `n_patches/4`), so changing the window keeps the masked fraction
at ~44-100% instead of silently making the objective easier; pass `--min-block`/`--max-block` to override.

An encoder pretrained at one window does not transfer for free to another: attention was trained over
that many patches, and the per-window z-score is computed over that span. Re-pretrain, or warm-start with
`--init-from <encoder.pt>`, which copies every learned tensor and regenerates only the sinusoidal position
buffer (the one shape that depends on window length). It refuses checkpoints that differ in any other way.

Trains on the CSV's **train split only**. Val/test rows never enter this stage, or every forecasting number
downstream is leakage-contaminated. A slice of the *train* series (`--holdout-frac`) is held out to watch
the objective.

Each run gets its own timestamped directory under `data/output/runs/`, like the trainers — so a second
pretrain cannot silently overwrite the encoder an already-fine-tuned model was initialized from:

```
data/output/runs/jepa_encoder/
├── latest.txt                                       # points at the most recent run
└── jepa_encoder_w288_p8_d96_l3_h6_20260715_002330/
    ├── config.json           # the SSL config + final latent_std / eff_rank
    ├── encoder.pt            # last epoch — this is what --jepa-init loads
    ├── encoder_best.pt       # lowest holdout loss
    ├── pretrain_metrics.csv  # the per-epoch curve
    └── plots/epoch_001.png … # 4-panel encoder diagnostics
```

The directory name carries the encoder shape (`w288_p8_d96_l3_h6` = window / patch / dim / layers /
heads), because those must match the fine-tuning config exactly — the `--jepa-init` load is
`strict=True`, and `w` must equal `--jepa-window`.

**Watch `latent_std` and `eff_rank`, not the loss.** Representation collapse — the encoder emitting nearly
the same vector for every window — drives the loss toward zero and looks like a triumph. Both are logged
every epoch and to the CSV. Your reference: a *random-init* encoder at `embed_dim=96` gives
`latent_std ≈ 0.67`. Healthy means staying the same order (0.4–1.0); a monotone slide toward 0 is collapse
in progress, and below ~0.1 the run is dead.

Then fine-tune from it — same command as (1), plus:

```bash
  --jepa-window 288 \
  --jepa-init data/output/runs/jepa_encoder/jepa_encoder_w288_p8_d96_l3_h6_<timestamp>/encoder.pt
```

Encoder weights only; `jepa_proj` and the backbone stay randomly initialized. The load is `strict=True`, so
a shape mismatch between the SSL config and the training config is an error, not a silent partial load.

### 3. Evaluate

```bash
uv run glucose evaluate \
  --run-dir data/output/runs/sugar_jepa2/<run_name> \
  --model-type sugar_jepa2 \
  --data data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda \
  --no-plot
```

Reports MAE / RMSE / MARD overall and per Study Group. Architecture hyperparameters — including
`jepa_window`, which sets the dataset lookback — come from the run's `tuning_meta.json`.

## Encoder PCA probe

The per-epoch plots answer "is it collapsing?". This answers "what did it end up organising by?" —
and how that changes with the pretraining window:

```bash
uv run python src/sugar_jepa/encoder_pca_probe.py \
  --encoder data/output/runs/jepa_encoder-288 \
  --encoder data/output/runs/jepa_encoder-864 \
  --encoder data/output/runs/jepa_encoder-2016 \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv \
  --split val --windows-per-dataset 2000 --out-dir data/output/encoder_pca
```

Same recipe as panel 3 of the training diagnostics — encode a fixed window sample, mean-pool the patch
axis, PCA to 2D — but the scatter is drawn four times over the *same* projection, coloured by **dataset**
(Loop vs AI-READI), **patient_id**, **mean glucose**, and window **trend**. Whichever colour lines up with
the geometry is what the encoder encodes. Trend is the control: it is what the pretraining plots colour by,
and without it a featureless dataset/patient/glucose panel cannot be told apart from a dead encoder.

`--encoder` takes a run directory (it follows `latest.txt`, prefers `encoder_best.pt`) or a bare
`encoder.pt`. Window, patch size, width, and depth are read off the tensors, so a checkpoint whose
`config.json` disagrees with its weights is an error rather than a mislabelled plot.

Output in `--out-dir`: `pca_w<window>.png` per encoder, plus `probe_metrics.csv` turning the colours into
numbers — `dataset_silhouette` / `patient_silhouette` in the full 96-d embedding (not the 2-d projection,
whose axes are just the top-variance directions), and `glucose_r2` / `trend_r2`, the R² of predicting each
from the top 10 PCs. There is no combined grid across encoders: a 2016-step window cannot be drawn from
the same series as a 288-step one, so each row would be a different sample and the columns would not be
comparable.

Two things to know before reading the output:

- **Windows are balanced twice** — equally per dataset, and round-robin across patients within a dataset.
  Loop has few, long series; a proportional draw hands one patient most of that cohort, and
  `patient_silhouette` then measures that one person against everyone else.
- **The Loop side is thin.** The full CSV's val split has 343 AI-READI patients against 9 Loop ones, so
  dataset separation is partly confounded with patient identity no matter how the windows are drawn. The
  script prints a CAUTION line when a dataset has fewer than 10 patients.

A high `glucose_r2` is worth a second look rather than a celebration: `JepaEncoder` instance-normalizes
every window, so absolute level is removed before the first conv. Anything recovered is level leaking back
through shape — variability, excursion frequency — not the level itself.

## Metrics CSVs

Both stages write one row per epoch, flushed immediately — plottable mid-run, and a killed run keeps
what it finished.

- `<run_dir>/training_metrics.csv` — `epoch, train_loss, val_loss, lr, lr_group1, best_val_loss,
  epoch_seconds, mix_basal, mix_bolus, mix_carbs, mix_jepa`
- `data/output/runs/jepa_encoder/<run>/pretrain_metrics.csv` — `epoch, train_loss, holdout_loss,
  latent_std, eff_rank, embed_dim, ema_momentum, lr, epoch_seconds`

A skipped validation epoch writes a **blank**, not `0.0` (which would plot as a perfect score).

`mix_jepa` is the diagnostic worth watching: it's `softmax(mix_logits)[jepa]` averaged over blocks, i.e.
how much weight the model actually gives the JEPA stream. If it decays well below 0.25, the model is
routing around the branch and whatever it learns is coming from the SugarOne backbone.

## Smoke test

```bash
uv run pytest tests/test_sugar_jepa_smoke.py tests/test_sugar_jepa_batch_first.py tests/test_jepa_pretrain.py -q

uv run python src/sugar_jepa/train_sugar_jepa2.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda --epochs 2 --max-train-series 20 --max-eval-series 10 \
  --batch-size 32 --n-blocks 2 --out-dir data/output/runs/sugar_jepa2_smoketest
```

`tests/test_sugar_jepa_smoke.py` covers both variants: the frozen CGM-JEPA `SugarJepaModel` and
`SugarJepaModel2`'s two-window slicing.

## What's different from SugarOne

- `CrossAttentionSugarJepaBlock` mixes **4** auxiliaries via a learnable softmax weight, vs. SugarOne's
  3-way mix.
- `JepaEncoder` encodes a **288-step** glucose window — longer than the backbone's 128 — into 36 patch
  embeddings (dim 96), projected to `d_model` by `jepa_proj` to serve as the 4th K/V stream. The K/V
  sequence is therefore a different length from the query, which cross-attention allows.
- The encoder **instance-normalizes each window** (per-window z-score) inside its own forward pass. A
  z-score is invariant to affine maps, and MinMax scaling is affine — so an encoder pretrained on raw
  mg/dL sees an identical input distribution when fine-tuned on MinMax-scaled `x[..., 0]`. That is what
  removes the need for a separate JEPA scaler (variant 1 needs one).
- Its blocks run `batch_first=True` (the flag defaults to `False` for `SugarJepaModel`, which still uses
  the `(seq, batch, d_model)` contract).
- Everything else — multi-scale self-attention, output head, MSE loss, MinMax scalers, imputation, the
  training loop — is SugarOne's, imported not copied.

## Known gaps

- **No baseline on disk.** The SugarOne run variant 1 was compared against is gone, so a plain SugarOne
  run at matched config on matched window counts is a prerequisite for any claim here.
- **The JEPA encoder is ~336k params against a ~368k backbone**, so a random-init win could just be extra
  capacity. A capacity-matched SugarOne control (bump `n-blocks`/`d-model` until params match) is not
  optional. `--jepa-embed-dim 64 --jepa-layers 2` gets the branch to ~100k if you'd rather shrink it.
- **No `--ablate-jepa`.** Doing it right means renormalizing the 4-way softmax over the three remaining
  auxiliaries — otherwise `w[3]` still multiplies a nonzero term and "ablating" perturbs basal/bolus/carbs
  too.
- No hyperparameter tuner; `tune_sugar_jepa_full.toml` is a reference config, not a script input.

