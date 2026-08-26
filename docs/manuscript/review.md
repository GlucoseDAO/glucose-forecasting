# Open items — things I could not fix by editing

Everything fixable by editing has been fixed in `manuscript.tex` / `references.bib` (see
"Already fixed" at the bottom for the log). What remains needs **new runs, a decision only
you can make, or information I do not have**.

Ordered by whether it blocks submission.

---

## Must verify before submission

### T0 — Sensor provenance behind the "two-week floor" claim

The manuscript now argues (Section 5.5, "The two-week floor may be a sensor artifact") that the
personalization data floor may be a sensor-replacement effect rather than a sample-size effect.
**Two premises of that argument are unverified and you flagged both.**

What I actually measured, and stand behind:

| Observation | Value | Source |
|---|---|---|
| Ceiling on contiguous sequence duration | **10.22 d** | `loop_ai_ready_joined2_dev.csv`, computed |
| Mode in the duration histogram | **9.5–10 d** (65 of 351 Loop-side sequences ≥1 d) | computed |
| Between-sequence SD of mean glucose (users with ≥3 sequences, n=11) | **11.1 mg/dL** | computed |
| Within-sequence SD of glucose | **42.0 mg/dL** | computed |
| Sequences spanned by a 7-day personal budget | median **1** | computed |

What is **not** verified:

1. **Device composition of Loop.** I originally attributed the 10.2-day ceiling to the Dexcom G6
   session length. You are right that Loop may include Medtronic (and other) sensors with
   different wear periods — Medtronic Guardian is 7 d, Dexcom G6 is 10 d, Libre is 14 d. The
   joined CSV has **no device column** (`sequence_id, Timestamp, Event Type, User ID, Glucose,
   Basal Rate, Bolus Insulin, Carbohydrates, Recommended Split, Study Group`), and `Event Type`
   only distinguishes `EGV / AI_READY / Interpolated / BGM`. **I removed the "Dexcom G6" claim
   from the manuscript** — the text now says only "close to a sensor wear period" and cites the
   measured ceiling and mode, which hold regardless of device mix. If Loop is in fact
   multi-device, a mixed histogram with modes at 7 and 10 d would *strengthen* the argument,
   since the floor should then track device type. Worth checking against the raw Loop export,
   which may carry a device field the joined schema drops.

2. **Whether the Loop export ran through calibration removal.** I had written that preprocessing
   "already excises calibration periods and the following 24 hours", which would have made the
   effect a *persistent session offset* rather than early-wear noise — a stronger claim. I read
   those defaults in `glucose_data_processing` (`calibration_period_minutes = 165`,
   `remove_after_calibration_hours = 24`), but **never verified the Loop CSV was produced with
   them.** I have removed that claim; the manuscript now says the data does not separate
   transient early-wear instability from a persistent calibration offset, and notes the two
   imply different mitigations.

**What to check:** which preprocessing invocation produced `loop_ai_ready_joined2.csv`, whether
calibration removal was on, and whether the raw Loop export identifies sensor device or session.
If session boundaries are recoverable, T-new below becomes runnable and the hypothesis becomes a
result instead of a caveat.

### T0b — Session-stratified personalization (makes T0 a result)

If sensor sessions are identifiable, fine-tune each subject on 14 days drawn from **one** session
versus 14 days drawn from **two**, holding total training days fixed. If two-session budgets
generalize better at equal sample size, the two-week floor is a sensor artifact and per-session
normalization is the fix — a considerably more interesting finding than "we need more data", and
cheap, since it reuses existing checkpoints and the existing fine-tuning path.

---

## Must run before submission

### T1 — Re-score the baseline on each long encoder's series subset

**This is the one that decides whether the JEPA section stands.** The manuscript now states
the confound honestly, but "honestly stated" is weaker than "controlled".

Longer JEPA windows need longer contiguous series, so each encoder is scored on a different
population. Reconstructed window counts:

| Encoder | Windows | % of baseline | T1DM share |
|---|---|---|---|
| jepa-128-64, jepa-128 | 1,667,437 | 100% | 49.1% |
| jepa-288, CGM-JEPA | 1,469,436 | 88% | 47.8% |
| jepa-864 | 967,521 | 58% | 44.7% |
| jepa-2016 | 295,933 | **18%** | **37.8%** |

**What to run:** for each of `jepa-864` and `jepa-2016`, evaluate SugarOne (and ideally
`jepa-128`) restricted to exactly the series that encoder can use — i.e. filter to series with
`≥ jepa_window + horizon` contiguous rows, then re-score. No retraining; evaluation only,
existing checkpoints.

**What it settles:** if the long-context advantage survives on matched series, it is real and
the paper gets a much stronger result than 3.0%. If it collapses, the current framing is
already correct and you have the control to prove it. Either outcome is publishable; not
knowing is not.

Cohort reweighting (which I *did* compute, and which is in Table 7) accounts for only ~1pp of
the 16pp, so it does not substitute for this.

### T2 — Persistence and linear baselines at 60 min

No naive baseline exists anywhere in the paper. For CGM at a 60-minute horizon, last-value
persistence is a genuinely strong competitor, and its absence is the first thing a time-series
reviewer will check. Currently listed as limitation (4) in the Discussion, which is honest but
costs credibility.

**What to run:** ŷ = last observed glucose for all 12 steps; and a linear extrapolation from
the last 2–4 points, clamped to a physiological range. On the exact joined2 test split,
`input_steps=128`, `horizon=12`, stride 1, reported per cohort. Pure numpy over the CSV — no
GPU, no training. Add as the top rows of the consolidated Table 2 and of Table 7.

I wrote such a script during this review and deleted it when you said not to run experiments;
it is trivial to rewrite. Note `uv` is not on the sandbox PATH from my side, so I cannot run it
even if asked.

### T3 — Confirm the variant → results-row mapping

I inferred, from PR #4, that the `jepa-*` rows in Table 7 come from `sugar_jepa2`
(encoder pretrained in-house) and the `CGM-JEPA` row from `sugar_jepa` (vendored released
weights). The manuscript now asserts this. **If the mapping is the other way round, or if some
rows are mixed, the paper's second headline finding — "borrowed pretraining ties in-house
pretraining" — is wrong and must be rewritten.**

Please confirm: which trainer produced each row, at which context length, and whether the
vendored encoder was frozen (dropped from the optimizer) or fine-tuned.

### T4 — Was any reported "frozen" run frozen by zeroing its learning rate?

PR #4 documents that `--freeze-jepa` must drop the encoder from the optimizer, because
`CosineAnnealingLR` shares one `eta_min` across parameter groups, so a zero-base group anneals
*upward* and ends at the backbone's final LR. The manuscript now states the correct method.

If any run reported here was "frozen" by setting `--jepa-lr 0` under the cosine schedule, that
encoder was **not** frozen and the run is mislabelled. Needs checking against the run configs.

---

## Should run — strengthens the paper materially

### T5 — Clinically-weighted metrics and hypoglycemia breakdown

The metric set is MAE/RMSE/MARD only. Missing, and expected at a clinical venue:
- Error-grid zone occupancy (Clarke or Parkes), or gMSE
- Hypoglycemia sensitivity and lead time for <70 mg/dL events, for T1DM and Insulin-T2DM

The paper's own motivation is hypoglycemia avoidance, so a reviewer will notice that no
hypoglycemia metric appears. Evaluation-only on existing predictions if `test_predictions.csv`
was retained per run.

Related: MARD is near-meaningless for the Healthy cohort (narrow range by construction), so
the pooled 8.25% headline is partly an artifact of cohort mix. The Discussion says this; a
per-cohort MARD table would remove the objection entirely.

### T6 — Capacity-matched control for the JEPA branch

The JEPA branch adds ~336K parameters against a ~368K backbone — nearly doubling the model.
Some of the 3.0% may be capacity, not pretraining. PR #4 states the branch is not
capacity-matched.

**What to run:** SugarOne widened to ~700K parameters (more blocks or larger `d_model`), same
1,667,437 windows, versus jepa-128. If the widened baseline closes the 3.0%, the only defensible
JEPA claim in the paper goes away — worth knowing before a reviewer asks.

### T7 — Random-init encoder control at each window length

I used the random-init reference of latent_std ≈ 0.67 at `E=96` from PR #4's diagnostics. It
would be better to have it *per window length*, since jepa-2016's 0.83 is currently compared
against a reference measured at a different context size. Cheap: no training, just instantiate
and measure.

### T8 — Decide whether the dev matched-hyperparameter run goes in the supplement

`docs/SUGAR_JEPA_VS_SUGAR_ONE_DEV_COMPARISON.md` has the only SugarJEPA result with matched
hyperparameters *and* a documented cost ledger (4.6× per epoch). It is small and on the dev
subset, but it is closer to a controlled ablation than the full-benchmark encoder sweep.

My view: include it in the appendix, explicitly as the matched-hyperparameter comparison, with
the dev/full distinction stated. Your call — and note the unresolved discrepancy in that
document (SugarJEPA reported as both +4.66% and behind SugarOne on the dev subset) should be
settled first, since a reviewer who reads both numbers will not trust either.

---

## Information I need from you

### T9 — Encoder sweep protocol

The manuscript cannot currently state whether the five encoders were fused into (a) one shared
SugarOne training run, or (b) five separate runs, and whether any hyperparameter was tuned per
encoder. If (b) with per-encoder tuning, the sweep has a second confound on top of the
population one.

### T10 — AI-READI participant count

The paper says 896 participants for AI-READI. `docs/GLUMIND_VS_SUGARONE_COMPARISON.md` says
2,232 series for the `ai_ready` export, and the encoder probe splits reference 343 AI-READI
patients in validation. These are different quantities (participants vs series vs val-split
patients) but I could not verify 896 against any file in the repo. Please confirm the source.

### T11 — GluMind-original comparison basis

Table 2 compares against "GluMind orig. 12.95 / 18.19". I computed 12.95 as the *unweighted
mean* of the four published per-cohort MAEs (10.58, 11.08, 13.74, 16.41). If the published
paper reports a pooled figure directly, use that instead — an unweighted cohort mean is not the
same as a window-weighted pooled metric, and a GluMind author reviewing this would notice.

### T12 — Trade-secret check before submission

Per project IP policy: the manuscript now describes the pretraining objective, masking ratios,
optimizer grouping, the scaler-inheritance protocol for personalization, and the corpus
composition. **The scaler-inheritance detail and the fine-tuning protocol are the ones I would
flag** — the personalization methodology is listed as a trade secret. Section 5.5 currently
states it explicitly because it is load-bearing for the result. Livia's and Anton's call, not
mine.

---

## Already fixed (no action needed — logged for traceability)

- **Citations.** Two entries pointed at unrelated papers (`2309.01843` = a dark-matter physics
  paper, cited as Gluformer; `2410.09250` = a quantum deepfake-audio paper, cited as
  GlucoBench, with an 11-author list that does not exist). Corrected to `2209.04526` and
  `2410.05780` with verified authors. GluMind's first author was wrong ("Bian" → Farahmand);
  AttenGluco had wrong authors, venue and year; the two wearable-foundation-model citations had
  conflated titles; `EventGlucoseBench` had no findable preprint and was removed; `GlySim` and
  `kirkpatrick2017ewc` were uncited and removed. `li2017lwf` was typed as `@inproceedings` with
  a journal as `booktitle`. All IDs verified against the arXiv API and Crossref.
- **False monotonicity claim.** "Error increases monotonically with severity, consistent across
  all tested models" — true for the two multimodal models, false for both baselines, which
  invert at Pre-T2DM and again at T1DM. The figure printed alongside showed the violation.
  Rewritten.
- **Sensor-accuracy conflation.** Abstract, results and conclusion compared forecast MARD to
  *sensor measurement* MARD and concluded predictions "approach the accuracy of the sensor
  itself". Different quantities against different references. Reframed as a ceiling argument.
- **JEPA method section** rewritten against PR #4: both variants named and distinguished, the
  scaler asymmetry explained, the actual pretraining objective and masking scheme described,
  collapse diagnostics with the random-init reference, the freeze/cosine-anneal subtlety, and
  an explicit statement that there is no action conditioning, decoder, or counterfactual
  evaluation.
- **Results table** now carries window count, cohort share, and both pooled and
  cohort-reweighted deltas per row, with the two internally valid comparisons promoted.
- **Stale figure caption.** The personalization figure caption described solid/dashed
  zero-shot lines that no longer exist after the figure was redrawn as deltas.
- **Page budget.** 18pp → main text now ends ~p10 (last verified build: main text p1–12 before
  the final round of cuts; re-render to confirm). Continual learning, experimental setup, data
  pipeline, encoder diagnostics, and four of six figures moved to appendices, which do not
  count. Introduction, Related Work, Discussion and Conclusion tightened; the three overlapping
  baseline tables consolidated into one.
- Numerous smaller items: exact ablation figures with correct arithmetic (70.5%), the
  best-personalization-gain caveat, `T=80`/`T=128` config ambiguity, multi-scale resolution
  wording, batch-size ambiguity between configurations.
