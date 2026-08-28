# Paper plan

This is a writing plan, not yet a section outline. The previous `docs/manuscript/` draft tried to be several papers at once. This plan keeps one theme and makes every section serve it.

Venue constraint: EASRP 2026 template, **8 pages of main text**, references and appendix unlimited. Prefer a shorter paper over leftover sections.

---

## 1. Locked theme

**Fine-tuning stability of a personal 60-minute glucose model, together with zero-shot accuracy. Not “the best frozen model.”**

A global model is what you can ship on day zero. Fine-tuning is what you do as that person’s days arrive. Most papers report only the two endpoints: zero-shot, and fine-tune on all history. The neglected object is the **path between them**. If 7 or 30 days of fine-tuning raise MAE, you cannot “always adapt.” You need a gate (keep the frozen model until enough days exist). That gate is the practical problem.

**Claim:** SugarOne plus a CGM-JEPA embedding (SugarJEPA) is a better *personalization system* because it scores on two axes at once:

1. **Zero-shot level** — usable before much personal history exists.
2. **Fine-tune smoothness** — adapting on more days should not make the frozen model worse, and gains should appear before a 60-day wait.

Architecture, the joined dataset, and a short global-test table exist only so this comparison is honest.

**SugarOne has no paper.** Readers have never seen it. Methods must define it as: GluMind’s parallel cross-attention + multi-scale self-attention, retargeted to commodity CGM + insulin covariates, with different mixing, hyperparameters, and training corpus. Cite GluMind for the block design. SugarOne is the **backbone we introduce**, not a citation. It is still not a second contribution: we do not run a GluMind-vs-SugarOne leaderboard.

### What this is not

- Not a contest for the single best zero-shot MAE (that temptation is how long JEPA windows sneak in and change the test population).
- Not a new foundation model paper.
- Not a GluMind reproduction, and not a “we publish SugarOne” paper. SugarOne is specified so SugarJEPA is intelligible.
- Not a preprocessing-methods paper.
- Not a “longer context is better” scaling paper.

---

## 2. Why this is the paper

Personal fine-tuning of glucose forecasters is thin in the literature. The usual report is: train a global model, maybe fine-tune once on the whole personal train set. That hides the deployment question: **after N days, is the fine-tuned checkpoint safe to serve?**

What we already measured (mean personal-test MAE, 7 T1DM users, numbers still to fact-check against run CSVs):

| Model | Zero-shot | Path 3–30 days (main figure) | Full history |
|-------|-----------|------------------------------|--------------|
| SugarOne | 19.5 | Often **worse** than frozen; report says gains show up around 30–60 days | Better (~18.7) |
| NBEATSx / N-HiTS / LSTM continue-fit | — | **Harmful** at 30 days (MAE up several mg/dL) | Recovers and can beat zero-shot |
| TFT continue-fit | — | Helps at 30 days (a smooth exception among NF) | Helps more |
| SugarJEPA-288 (hero) | Better than SugarOne (~18.1) | Mean MAE at or below zero-shot | Still better than its own zero-shot |

So the story is not “JEPA wins the frozen leaderboard.” It is: **SugarOne and several strong time-series models need a fine-tune gate; SugarJEPA mostly does not, and it starts from a higher zero-shot floor.**

A useful negative control is already in the CSV: `jepa-2016` can have a competitive zero-shot and then **hurt** at full fine-tune. More context is not the same as a smoother adapter. That belongs in the **appendix**, not as the hero.

The notes’ line “JEPA could benefit finetuning” is the right topic. The precise statement is: JEPA-as-a-feature raises the day-zero floor **and** makes the adaptation path non-harmful, which is the thing a personalization system actually needs.

---

## 3. What “smooth fine-tune” means in this paper

Define it in Methods so Results is not vague. Three checks, in this order:

1. **Non-harmful.** At each day budget, mean personal-test MAE is not worse than that model’s own zero-shot. (Primary. This is the gate: can we always deploy the new weights?)
2. **Early gain.** MAE drops below zero-shot before the 60-day mark where SugarOne usually becomes useful.
3. **Terminal gain.** Full-history fine-tune still beats that model’s zero-shot (no late collapse).

Plot the **curve**, not only the last point. Zero-shot is the first point on that curve, not a separate contest.

Honesty bounds (do not over-claim “monotonic for every user”):

- Means over the 7 T1DM users, plus a note that single users can dip.
- **No 1-day fine-tune on the main figure.** `jepa-288` lookback is already one day of CGM (288 × 5 min), so a 1-day train budget cannot build an input window. All models on the main curve start at **zero-shot, then 3 / 7 / 14 / 30 / 60 / full**. SugarOne / NF / `jepa-128` 1-day numbers, if used at all, stay in the appendix.
- User 1082 (~37 days train): SugarOne full fine-tune is worse than zero-shot; SugarJEPA stays roughly flat. That user is evidence for the theme, not an outlier to hide.
- TFT is a NeuralForecast model whose continue-fit path is also non-harmful. Do not write “all baselines are unstable.” Contrast SugarJEPA with SugarOne (same family, bad path) and with N-HiTS or NBEATSx (harmful short continue-fit). TFT is the honest exception.

---

## 4. Argument chain

Each block has a job. If a paragraph does not advance this chain, it does not belong in the main text.

| Order | Job | What the reader must believe afterwards |
|-------|-----|----------------------------------------|
| 1 | Gap | 60-minute CGM forecast is useful; the open problem is **safe personal adaptation as days arrive**, not only a better global MAE. |
| 2 | Data | Named corpus `loop_ai_ready_joined2.csv` from commodity CGM + pump records via `glucose_data_processing`. Global test and personal test are different splits. |
| 3 | Backbone | SugarOne is **defined here**: GluMind blocks, commodity covariates (glucose, basal, bolus, carbs), learnable mixing, our hyperparameters, trained on `loop_ai_ready_joined2`. There is no SugarOne citation. |
| 4 | Competence | On the **global** test split, SugarOne / SugarJEPA / NF are serious models. We are not fine-tuning a weak toy. |
| 5 | Method | CGM-JEPA-style encoder, pretrained on our glucose, attached as a fourth SugarOne stream. Fine-tune protocol is the same for everyone: global scalers, day budget shortens **train** only, frozen personal test. |
| 6 | Main result | **Curves** on the same 7 people: zero-shot, then 3…60…full. SugarOne and N-HiTS/NBEATSx are non-harmful only after a long wait (or never, at 30 days). SugarJEPA-288 is non-harmful earlier and still improves at full history. TFT is the smooth NF exception. |
| 7 | Conclusion | The contribution is a personalization *path*, enabled by JEPA as a feature. Limitations: small T1DM cohort, matched windows, smoothness is empirical not theoretically guaranteed. |

Nothing else is a contribution.

---

## 5. What goes in, and why

### In (needed for the theme)

- **Task.** Horizon 12 steps = 60 minutes at 5-minute sampling. Metrics: MAE (lead), RMSE, MARD.
- **Data pipeline, short.** Raw CGM/pump → [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing) → `loop_ai_ready_joined2.csv`. Name sources, study groups, and **which split is “the test.”**
- **SugarOne, specified (unpublished).** Not a footnote. One Methods subsection plus a tiny GluMind-vs-SugarOne difference table:
  - **Kept:** GluMind parallel cross-attention (glucose queries, each auxiliary as keys/values) and multi-scale self-attention.
  - **Covariates:** heart rate + steps → basal rate, bolus, carbohydrates (what a commodity CGM + insulin record has).
  - **Mixing:** GluMind’s fixed average → learnable softmax weights.
  - **Hyperparameters:** not GluMind’s defaults (e.g. lookback 128 vs 80, more heads/blocks — lock exact numbers from the SugarOne checkpoint when drafting).
  - **Data:** `loop_ai_ready_joined2.csv`, not the wearable AI-READI table GluMind was trained on.
  Cite Farahmand et al. for GluMind. Never write “as we showed in the SugarOne paper.”
- **JEPA, compact.** Origin: [CGM-JEPA](https://github.com/cruiseresearchgroup/CGM-JEPA). Pretrained on our train split (glucose only). Fourth cross-attention branch. Longer encoder context ≠ automatically smoother fine-tune.
- **Fine-tune protocol.** Independent run per day budget from the global checkpoint (not a curriculum). Reuse global `scalers.json`. Same recipe for SugarOne, SugarJEPA, and NF continue-fit.
- **Global test table (one, small).** Proves competence. Not the climax.
- **Personalization curves (the climax).** Same people, same personal splits, same horizon. Sources:
  - SugarOne: `docs/PERSONALIZATION_REPORT.md`
  - NeuralForecast: `docs/PERSONALIZATION_NF_REPORT.md`
  - SugarJEPA: `temp_docs/jepa_mae_by_days.csv` (must be checked against run CSVs)
- **LwF, one paragraph.** Distillation does **not** fix SugarOne’s short-budget harm. It is the failed stability baseline, not a second paper.
- **Patient-ID probe:** **skip.** No section, no appendix table. The claim is the forecast curve.

### Out

- Sugar-Sugar human game
- Dual wearable (HR/steps) vs pump **as a Results comparison**. The difference belongs in Methods when defining SugarOne. Do not add a GluMind checkpoint to the personalization figure.
- “Glucose world model” language
- JEPA context-length scaling as a finding (longer windows change the **test population**)
- Two-week floor as a sensor-wear theory
- Encoder PCA / effective-rank QC as a main result
- Per-study-group global tables for every encoder
- Mixing-and-retraining NeuralForecast from scratch
- Treating zero-shot SugarJEPA beating fine-tuned SugarOne as the *title* result (it can be one sentence in Results; it is not the theme)

---

## 6. Locked comparison (fair curves)

**Main figure (personal test, 7 T1DM users):**

- SugarOne
- SugarJEPA-288 (hero)
- N-HiTS **or** NBEATSx (harmful short continue-fit — pick one when we lock numbers)
- TFT (smooth NF exception)

**Budgets on that figure:** zero-shot, 3, 7, 14, 30, 60, full. No 1-day point.

**Appendix only:** `jepa-128` / `jepa-128-64`, `jepa-864`, `jepa-2016`, extra NF models, 1-day budgets for models that can form a 128-step window, AI-READY short-wear users.

Do not average empty cells. Do not mix the 8 short AI-READY users into the main mean.

---

## 7. Two tests, never one number

| Name in the paper | Data | Split | Question it answers |
|-------------------|------|-------|---------------------|
| **Global test** | `loop_ai_ready_joined2.csv` | dataset `test` | Is this a competent population model? |
| **Personal test** | `data/input/personalization/` CSVs | last 25% of that user; train shortened by day budget | Is the **path** from day 0 to full history safe and useful? |

Zero-shot personal MAE is not global test MAE. Every results paragraph names which test it is.

---

## 8. Evidence we already have

| Piece | Where | Status |
|-------|--------|--------|
| SugarOne day-budget MAE, gate at 30–60 days, LwF failure | `docs/PERSONALIZATION_REPORT.md` | Ready; copy from run CSVs |
| NF continue-fit paths (harmful vs smooth) | `docs/PERSONALIZATION_NF_REPORT.md` | Ready |
| SugarJEPA vs SugarOne curves | `jepa_paper/easrp2026.tex` table + `temp_docs/jepa_mae_by_days.csv` | **JEPA paper is SoT** (protocol, freeze, architecture). CSV is MAE only. |
| Global SugarJEPA tables | `docs/manuscript2/jepa_paper/easrp2026.tex` | Primary for JEPA global numbers; still drop 864/2016 from the main global table (population shift) |
| Dataset / join | `docs/DATA.md`, `docs/GLUMIND_VS_SUGARONE_COMPARISON.md` | Ready |
| Old draft mistakes | `docs/manuscript/review.md` | Do-not-repeat list |

No new training in the writing phase unless a hole in the hero **curve** is blocking. There is no `jepa-288` 1-day run to fill: lookback is already one day. If a job is needed, check that none is already running.

---

## 9. Writing rules

1. **One claim per paragraph.** First sentence is the point.
2. **Name the test.**
3. **Lead with the path, not the leaderboard.** Introduction: fine-tuning can hurt; we want a model you can keep adapting. Results: curves. Discussion: the gate vs no-gate implication.
4. **Zero-shot is axis 1 of the same figure**, not a separate victory lap.
5. **SugarOne is defined, not cited.** Related work points at GluMind. Methods states the four changes (covariates, mixing, hyperparameters, dataset). Results do not add a GluMind personalization curve.
6. **Prefer one architecture figure and one curve figure** in the main text, plus a small global table and a day-budget table. The architecture figure is SugarJEPA = SugarOne + JEPA branch; SugarOne’s GluMind lineage is a small table or callout on that figure.
7. Do not reuse `docs/manuscript/manuscript.tex` as source of truth.

Public names: **SugarOne**, **SugarJEPA**, **CGM-JEPA**. Cite **GluMind** as the published architecture. Avoid “Sugar I.”

---

## 10. Risks (must not ship)

- Selling SugarJEPA as “best zero-shot” while the theme is smoothness.
- Population confound on long JEPA windows.
- Cohort mismatch (15 SugarOne users vs 7 JEPA users).
- Claiming every user is monotonic.
- Ignoring TFT’s smooth NF path.
- Sign errors on Δ (FT MAE − zero-shot MAE; negative = better).
- Describing scaler refit on 1–14 personal days (that protocol was wrong).
- Publishing `temp_docs/jepa_mae_by_days.csv` without matching run metrics.
- Unverified sensor-wear / calibration claims from the old review.

---

## 11. Work order

1. Theme and comparison settings: **locked** (`plan.md`).
2. Paper structure: **written** (`structure.md`).
3. Lock hero numbers from run CSVs.
4. Draft LaTeX from `docs/manuscript2/template.tex`.
5. Compile and cut to 8 pages by deleting. Claims we must not make will be caught while drafting.

---

## 12. Locked decisions

| # | Decision |
|---|----------|
| Hero encoder | **SugarJEPA-288** on the main curve. |
| Long encoders (3 d / 7 d) | **Appendix only.** |
| NeuralForecast on the main figure | **NBEATSx** (harmful through 30 d) **and** **TFT** (helps from 30 d; harmful at 3--14 d). Not all five. |
| Personal cohort | **7 T1DM users** with JEPA curves. AI-READY short wear = limitation only. |
| 1-day fine-tune | **Drop from the main figure for every model.** `jepa-288` needs a 1-day lookback, so a 1-day train budget cannot yield a window. Curves: zero-shot, then 3 / 7 / 14 / 30 / 60 / full. |
| Venue | **EASRP 2026**, anonymous, 8-page A4 main text. |
| SugarOne publication | **None.** Introduce it in Methods as GluMind blocks + different covariates, mixing, hyperparameters, and dataset. Not a second thesis. |
| Patient-ID probe | **Skip.** |
| JEPA source of truth | **`docs/manuscript2/jepa_paper/`.** Personalization JEPA report is CSV MAE only; it does not override freeze/protocol. |
| Personal JEPA fine-tune | **Encoder frozen**; only SugarOne weights update. Global training: encoder unfrozen, lower LR. |
| Forbidden claims | Not pre-listed. Catch during drafting (no clinical deployment, no guaranteed-safe adaptation, no world model). |

### Headline result (not a theme change)

Colleague comment, checked on `jepa_mae_by_days.csv` (still to fact-check against run CSVs): **frozen SugarJEPA-288 has lower personal-test MAE than SugarOne fine-tuned for 30 days, for all 7 T1DM users.**

| User | JEPA-288 zero-shot | SugarOne @ 30 d | Margin (mg/dL) |
|------|--------------------|-----------------|----------------|
| Livia | 17.64 | 18.06 | 0.42 |
| 154 | 23.13 | 24.84 | 1.70 |
| 556 | 17.22 | 17.65 | 0.43 |
| 730 | 16.02 | 18.23 | 2.21 |
| 1017 | 17.41 | 18.30 | 0.90 |
| 1029 | 20.30 | 22.81 | 2.51 |
| 1082 | 15.17 | 17.60 | 2.43 |

This is the **abstract punchline** and the first sentence of Results. It is not the paper’s theme.

Why it stays a result, not the spine:

- It is one slice of the curve (frozen A vs 30-day B). The research gap is the *path* (can you keep adapting?).
- A reviewer can say: you compared a better global model to a weaker model’s short fine-tune. The smoothness story answers what happens when you *do* fine-tune JEPA-288.
- The same “all 7” statement is **false** vs SugarOne’s *full* fine-tune (Livia and 1017: full SugarOne beats frozen JEPA-288). So 30 days is the right cutoff, but it must sit on the figure, not replace it.
- Write **“all 7 T1DM users in this study,”** never “all patients.”

Where it lives: abstract; opening of Results; a mark on the main curve figure at the 30-day SugarOne point. The rest of the paper still argues zero-shot level **plus** a non-harmful fine-tune path.
