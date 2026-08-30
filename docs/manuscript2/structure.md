# Paper structure

Follows `plan.md`. This is a section map, not prose. Target: Anonymous Conference 2026, **8 pages** main text (11pt A4). Appendix unlimited. Draft later from `template.tex`.

Working title (change when we have a better one):

**Fine-tuning paths for 60-minute glucose forecasts: a JEPA feature on a GluMind-style backbone**

Names in the paper: **SugarOne**, **SugarJEPA**, **CGM-JEPA**. Cite **GluMind**. Never “Sugar I.” Never “the SugarOne paper.”

Numbers in this outline are placeholders. Lock them from run CSVs / `docs/PERSONALIZATION_JEPA_REPORT.md` before drafting Results.

JEPA draft: `docs/manuscript2/jepa_paper/` (`easrp2026.tex`, `sugar_jepa.png`, `references.bib`). **This folder is the source of truth for anything JEPA** (architecture, pretraining, how the encoder is attached, global vs personal training protocol, JEPA tables and wording). `docs/PERSONALIZATION_JEPA_REPORT.md` is only a CSV replay of MAE-by-days; it does not override the paper on protocol (including freeze). Salvage map is at the end of this file. We still do **not** take their thesis (patient-ID probe, encoder PCA as a result, “longer window is better”).

---

## Page budget

| Block | Target | Job |
|-------|--------|-----|
| Abstract | 180–220 words | Punchline + two axes |
| 1 Introduction | ~0.9 p | Gap is the **path**, not a better frozen MAE |
| 2 Related work | ~0.7 p | What exists; what they do not report (day-budget curves) |
| 3 Methods | ~2.6 p | Define SugarOne (unpublished), data, JEPA-288, fine-tune protocol, smoothness |
| 4 Results | ~2.2 p | Short global table, then curves; 30-day slice is the first sentence |
| 5 Discussion | ~0.9 p | Gate vs no-gate; limitations |
| 6 Conclusion | ~0.25 p | Two sentences |
| **Main text** | **~8 p** | Cut Discussion and Related work first if over |
| Appendix | free | Long encoders, extra NF, 1-day, per-user tables |

Two figures and three tables in the main text. Nothing else in the main text.

| ID | Where | What |
|----|--------|------|
| **Fig. 1** | §3.4 | SugarJEPA = SugarOne + JEPA branch (`jepa_paper/sugar_jepa.png`) |
| **Fig. 2** | §4.2 | Personal-test MAE vs train days (the paper) |
| **Table 1** | §3.3 | GluMind vs SugarOne (covariates, mixing, lookback, heads/blocks, dataset) |
| **Table 2** | §4.1 | Global test MAE/RMSE/MARD: SugarOne, SugarJEPA-288, N-HiTS or NBEATSx, TFT |
| **Table 3** | §4.3 | Seven users: frozen JEPA-288 vs SugarOne @ 30 d |

---

## Abstract (~200 words)

**Job.** Problem, method, one number, one path claim. A reader who stops here should still have the paper.

**Order (one sentence each, then stop):**

1. 60-minute CGM forecast is useful; a global model is what you can serve on day zero.
2. Fine-tuning on one person is usually reported as two endpoints. Short fine-tunes can **raise** error, so a system needs a gate.
3. We use SugarOne (GluMind blocks, commodity CGM + insulin covariates, our hyperparameters, joined Loop + AI-READI corpus) and add a CGM-JEPA embedding as a fourth stream (SugarJEPA-288).
4. **Punchline:** frozen SugarJEPA-288 has lower personal-test MAE than SugarOne fine-tuned for 30 days, for all 7 T1DM users in this study.
5. SugarJEPA-288’s own fine-tune path stays at or below its zero-shot mean from 3 days onward and still improves at full history; SugarOne and N-HiTS/NBEATSx do not.
6. TFT continue-fit is a smooth exception among NeuralForecast models.
7. Limitation: seven T1DM users; smoothness is empirical.

No architecture lecture. No “all patients.” No long-encoder MAE.

---

## 1 Introduction (~0.9 page)

**Job.** Make the reader care about the **path**. End with three contributions. Do not explain JEPA yet.

**Paragraphs (one claim each):**

1. **Why forecast.** CGM reports an excursion after it has started. A 60-minute, 12-step forecast (5-minute sampling) is the task.
2. **Day zero vs later.** A population model can be served immediately. Personal history arrives slowly. The practical question is: after *N* days, is the fine-tuned checkpoint safe to serve?
3. **Gap.** Glucose papers report zero-shot and/or fine-tune on all available personal data. They rarely show the budgets in between. If 3–30 days make MAE worse, “always fine-tune” is wrong.
4. **This paper.** We introduce SugarOne (unpublished GluMind retargeting; one sentence, details in §3.3) and SugarJEPA-288 (JEPA embedding as a feature). We compare day-budget curves on the same 7 T1DM people against SugarOne and NeuralForecast continue-fit.
5. **Contributions** (bullets, not a paragraph):
   - SugarOne: GluMind architecture on commodity covariates and our corpus — specified here because it has no paper.
   - Evaluation of personalization as a **curve** (zero-shot, 3, 7, 14, 30, 60, full), with a named smoothness check.
   - Result: JEPA-as-a-feature raises the day-zero floor **and** makes short fine-tunes non-harmful on this cohort; frozen JEPA-288 beats 30-day SugarOne for all 7 users.

**Must not:** wearable vs pump as a finding; world models; Sugar-Sugar; encoder PCA; promising clinical deployment.

---

## 2 Related work (~0.7 page)

**Job.** Point at GluMind and CGM-JEPA. Say clearly that day-budget fine-tune **paths** are missing. Three short blocks, not a survey.

**2.1 Glucose transformers.** GluMind (parallel cross-attention + multi-scale self-attention, HR/steps). Gluformer and similar attention forecasters in one or two citations. Close: we keep GluMind’s blocks and change covariates, mixing, hyperparameters, and data — defined in §3.3, not compared as a leaderboard here.

**2.2 Self-supervised CGM.** CGM-JEPA (joint-embedding predictive objective). We use that idea as an extra **supervised-forecast** stream, pretrained on our train split. Not a new foundation model.

**2.3 Personalization and baselines.** Fine-tuning / meta-learning for T1DM exists; typical report is one adapted model, not a 3–60 day curve. N-HiTS, NBEATSx, TFT as the non-transformer continue-fit baselines we actually plot. GlucoBench etc.: one sentence if we need a benchmark pointer, not a subsection.

**Must not:** reproduce GluMind results; list every NeuralForecast model; patient-ID probing.

---

## 3 Methods (~2.6 pages)

**Job.** A competent reader could repeat the comparison. SugarOne is **defined**, not cited.

### 3.1 Task (~0.2 p)

Window → 12-step (60 min) glucose. Lead metric MAE (mg/dL); RMSE and MARD in tables. Same horizon for every model.

### 3.2 Data and two tests (~0.5 p)

- Pipeline in one paragraph: raw CGM/pump → `glucose_data_processing` (5-minute grid, gap fill) → join → `loop_ai_ready_joined2.csv`. Sources: AI-READI-style wearable CGM and Loop T1DM pump records; roughly balanced row mass; study groups named. No device archaeology.
- **Global test:** dataset `test` split of the joined CSV. Question: competent population model?
- **Personal test:** 7 T1DM users (Author1 + six Loop holdouts). Chronological split: last 25% test, 15% of remainder val, rest train. A day budget **only shortens train**. Val/test frozen.
- Why not the 8 short AI-READY users in the main curve: ~6–9 train days and empty insulin/carb channels; JEPA-288 needs a 1-day lookback. Mention as a limitation, not a second cohort table.

Name both tests every time they appear later.

### 3.3 SugarOne (~0.6 p) — unpublished backbone

GluMind blocks (cite Farahmand et al.). Then **Table 1**, four rows of difference:

| | GluMind | SugarOne |
|--|---------|----------|
| Auxiliaries | Heart rate, steps | Basal, bolus, carbohydrates |
| Fusion | Fixed average of cross-attention heads | Learnable softmax mixing |
| Lookback / size | 80 steps, 4 heads, 3 blocks (lock from GluMind paper/checkpoint) | 128 steps, 8 heads, 5 blocks, \(d=32\) (lock from SugarOne checkpoint) |
| Training data | Wearable AI-READI-style table | `loop_ai_ready_joined2.csv` |

One sentence: we do not evaluate a GluMind checkpoint on the personalization curves. SugarOne is the control in the same family as SugarJEPA.

### 3.4 SugarJEPA (~0.6 p)

**Lift from `jepa_paper/easrp2026.tex` (SugarOne Integration + Experiment Setup + Additional Regularization).** Rewrite to our length; keep the facts.

CGM-JEPA-style encoder \cite{muhammad2026cgm}, pretrained on **our train split, glucose only**, with a 5% encoder-val holdout. Loss: SmoothL1 latent prediction + EMA teacher + VICReg-style variance penalty (their equation). Attach as a **fourth** SugarOne branch: layer-norm, project 96 → 32, same cross-attention pattern as basal/bolus/carbs (**Fig. 1** = `jepa_paper/sugar_jepa.png`). If JEPA wants \(m\) steps and SugarOne \(n=128\), the window is \(m\) long; JEPA sees all \(m\), SugarOne the last \(n\).

During **global** SugarOne+JEPA training, the encoder is **not** frozen; lower LR (\(4 \times 10^{-5}\) vs \(4 \times 10^{-4}\)). During **personal fine-tuning**, the JEPA encoder **stays frozen**; only the SugarOne weights are updated (`jepa_paper`: “JEPA encoder remained frozen and only the main model's weight was updated”). Hero: **jepa-288** (96-d, 1 day). Other windows: one sentence, details in appendix.

Do not copy `PERSONALIZATION_JEPA_REPORT.md` on encoder LR during personalization — that report was rebuilt from the MAE CSV and had no freeze information.

### 3.5 Fine-tune protocol and smoothness (~0.5 p)

- Each day budget is an **independent** run from the global checkpoint. Not a curriculum.
- Global `scalers.json`. Do not refit on personal train.
- SugarJEPA personalization: **JEPA encoder frozen**; SugarOne (and mixing) weights update. Plain fine-tune, \(\lambda=0\). LwF on SugarOne did not fix short-budget harm (one sentence; details can sit in appendix if we keep any LwF number).
- NeuralForecast: continue-fit from the global bundle, same idea (train shortened, test frozen).
- **No 1-day budget on the main figure.** JEPA-288 lookback is already one day, so a 1-day train slice cannot form a window. All models: zero-shot, then 3, 7, 14, 30, 60, full.
- **Smoothness** (this is the definition Results will use):
  1. Non-harmful: mean personal MAE at budget \(t\) \(\le\) that model’s zero-shot.
  2. Early gain: mean MAE below zero-shot before 60 days.
  3. Terminal gain: full-history MAE below that model’s zero-shot.

User 1082: no 60-day cell (full train ≈ 37 d). Means at 60 days use \(n=6\).

### 3.6 Baselines on the main figure (~0.2 p)

Four lines: SugarOne, SugarJEPA-288, **either N-HiTS or NBEATSx** (harmful 30-day continue-fit — pick when locking numbers), TFT (smooth NF exception). Other NF models and other JEPA windows: appendix.

---

## 4 Results (~2.2 pages)

**Job.** Prove competence quickly, then spend the pages on Fig. 2. First sentence of 4.2 is the colleague punchline.

### 4.1 Global test (~0.4 p)

**Table 2.** Joined `test` split. SugarOne, SugarJEPA-288, the two NF models from §3.6. One paragraph: these are not weak toys. Do not interpret per-study-group gradients. Do not bring `jepa-864` / `jepa-2016` into this table (different scored population).

### 4.2 Personalization paths (~1.0 p) — climax

**Fig. 2.** Mean personal-test MAE vs train days, 7 T1DM users (60 d and some cells: \(n=6\)). Four curves. Mark the 30-day SugarOne point.

**First sentence:** Frozen SugarJEPA-288 has lower personal-test MAE than SugarOne fine-tuned for 30 days, for all 7 T1DM users in this study. Then: that is one slice; the figure is the path.

Then walk the three smoothness checks for each of the four curves. SugarOne: often at or above its zero-shot through 30 days; useful later. N-HiTS/NBEATSx: 30-day continue-fit harmful, later recovers. TFT: non-harmful early (exception). SugarJEPA-288: mean at or below zero-shot from 3 days; still better at full history.

User 1082 in **one sentence**: SugarOne full fine-tune worse than frozen; SugarJEPA-288 roughly flat. Evidence for the gate, not an outlier to drop.

**Must not:** claim every user is monotonic; “all patients”; mix AI-READY into the mean.

### 4.3 The 30-day slice (~0.4 p)

**Table 3.** Seven rows, two columns, margin. One paragraph: 30 days is the budget where a clinic might first try to personalize, and where SugarOne’s mean gain is still ~0. The same “all 7” line is **false** vs SugarOne’s **full** fine-tune (Author1, User 1017). So we do not replace Fig. 2 with this table; we show why 30 days is the honest quote.

### 4.4 Fine-tuning SugarJEPA-288 (~0.4 p)

Does the better frozen model still adapt? Mean Δ vs its own zero-shot at 3 / 30 / 60 / full (MAE from the CSV / `PERSONALIZATION_JEPA_REPORT.md`; protocol from `jepa_paper`). Point: smoothness is not “refuse to fine-tune”; the path is usable **and** full history still helps, with the JEPA branch held fixed. One clause: this is still a better *system* than “wait 60 days, then fine-tune SugarOne.”

No new encoder variants in this subsection.

---

## 5 Discussion (~0.9 page)

**Job.** Gate vs no-gate. Honest limits. No new experiment.

**Paragraphs:**

1. **Deployment reading.** SugarOne (and N-HiTS/NBEATSx) need a rule: keep frozen weights until enough days exist. SugarJEPA-288 can be adapted from 3 days on this cohort without a mean MAE penalty. TFT shows a smooth path is possible without JEPA; JEPA-288 still starts from a better personal zero-shot than SugarOne.
2. **Why not title the paper with the 30-day slice.** It is the cleanest single sentence. It compares two different models at two different budgets. The path is the contribution.
3. **Limitations.** \(n=7\) T1DM; no short-wear AI-READY in the main curve; JEPA-288 lookback is one day vs SugarOne’s 10.7 h (matched-window `jepa-128` in appendix); smoothness is not a theorem; we did not show a GluMind personalization curve; no clinical deployment claim.
4. **Future (three short items max).** More people; other study groups if contiguous CGM allows; whether the same protocol holds for other JEPA trainings. Do not invent a world model.

---

## 6 Conclusion (~0.25 page)

Two to three sentences: (1) personalization should be scored as a path; (2) a CGM-JEPA feature on SugarOne raises zero-shot accuracy and keeps short fine-tunes non-harmful on 7 T1DM users; (3) frozen JEPA-288 beats 30-day SugarOne for every user in that set.

---

## Appendix (does not count)

Use so the main text stays 8 pages. Each item is a table or a short caption, not a second thesis.

- **A.** Other SugarJEPA windows: take the encoder-variant table and the big per-user fine-tune table from `jepa_paper/easrp2026.tex` (`tab:model_comparison`, `tab:sugar-jepa-finetuning`). One warning: `jepa-2016` can **hurt** at full fine-tune; longer context ≠ smoother adapter. Global MAE for 864/2016 only with an explicit “different test population” sentence (`n` windows drop in their per-study-group tables — use that as the caveat). Encoder QC / PCA figures (`jepa-864-encoder.png`, `jepa-2016-encoder.png`, `pca_w864.png`) stay here if anywhere, not in the main text.
- **B.** Other NeuralForecast models (LSTM, TiDE, the NF model not chosen for Fig. 2).
- **C.** 1-day fine-tune for SugarOne / NF / `jepa-128*` (models that can form a 128-step window).
- **D.** Folded into **A** (`tab:sugar-jepa-finetuning` already has per-user grids). Main text Table 3 is only the 30-day slice.
- **E.** SugarOne LwF short-budget numbers (failed stability control), if we want a number behind the Methods sentence.
- **F.** Hyperparameters and checkpoint IDs.

**Patient-ID probe stays out** (main and appendix), as locked. Colleague PCA/effective-rank is optional in appendix A only, not a story.

---

## What each section is *for* (argument chain)

| Section | Reader should believe |
|---------|------------------------|
| Intro | The open problem is a safe fine-tune **path**. |
| Related | GluMind and CGM-JEPA exist; day-budget curves mostly do not. |
| 3.2–3.3 | We know the data and what SugarOne is. |
| 3.4–3.5 | SugarJEPA-288 is a feature on that backbone; comparison protocol is fair (no 1-day). |
| 4.1 | We are not fine-tuning a weak model. |
| 4.2–4.4 | Frozen JEPA-288 beats 30-day SugarOne for all 7; the path is non-harmful and still improves. |
| Discussion | Gate vs no-gate; n=7; 30-day slice is not the whole claim. |

If a paragraph does not serve a row above, delete it.

---

## Salvage from `jepa_paper/`

Source: `docs/manuscript2/jepa_paper/easrp2026.tex` (colleague draft). Keep the file as-is; copy into our manuscript, do not rewrite their folder.

### Take (rewrite to our theme, keep facts)

| Colleague passage | Our section | Note |
|-------------------|-------------|------|
| SugarOne Integration (LN, 96→32, \(m\) vs \(n\) windows, Fig. sugar_jepa.png) | §3.4 + **Fig. 1** | Best written part. Use almost as-is. |
| Additional Regularization (EMA + VICReg loss) | §3.4 | One short paragraph + equation. Cite `vicreg`, `muhammad2026cgm`. |
| Experiment Setup (Loop+AIREADY, 5% holdout, LR \(4\mathrm{e}{-5}/4\mathrm{e}{-4}\), encoder table) | §3.4; encoder table → appendix A | Main text names **jepa-288** only. |
| `tab:sugar-jepa-test-metrics` row **jepa-288** (+ SugarOne global from our reports) | **Table 2** | Do not put 864/2016 in Table 2 (window counts change). CGM-JEPA row can be a one-line “released encoder, matched 288 window” if we fact-check it. |
| `tab:sugar-jepa-288-study-group-metrics` | skip main; optional appendix | Not the climax. |
| Opening of “SugarJepa finetuning” + `tab:sugar-jepa-finetuning` | §4.2–4.3, appendix A | Punchline they already have: zero-shot JEPA vs 30-day SugarOne. **Restrict the quote to jepa-288 and all 7 users.** Drop 1-day column from the main figure. |
| Discussion sentence on zero-shot JEPA vs 30-day SugarOne FT | Abstract + §4.2 first sentence | Tighten: “all 7 T1DM users,” not “30+ days,” not “patients.” |
| `references.bib` (`muhammad2026cgm`, `vicreg`, `cgm-accuracy`) | our bib | Merge. Garg G7 citation only if we actually talk sensor noise; do not claim we measured the 9% MARD floor. |

### Do not take (conflicts with locked plan)

- Intro bullets (patient encapsulation, “standalone Glucose JEPA,” unweighted patient loss).
- Whole **JEPA as Patient Embedding** subsection + patient-ID table (10–68%). We skipped that experiment.
- Encoder Evaluation (effective rank, PCA figures) as a main Results subsection. Optional appendix A only.
- Global “15–18% MARD” / “jepa-864 is optimal” / “gains vanish vs sensor noise” as the paper’s result. That is the long-window leaderboard we rejected.
- Wording “JEPA embedding computed from **only one day of patient data**.” Zero-shot JEPA-288 uses a 1-day **lookback on each test window**, not a 1-day personal train set. Our punchline is frozen JEPA-288 vs SugarOne fine-tuned 30 days.
- “3-day encoder seems optimal” as conclusion. Hero is **jepa-288**. 864/2016 go to appendix, including their FT regression.
- Blood vs CGM / LeJEPA future work as a limitation paragraph (optional one sentence in Discussion at most).
- Template dummy citations (Vaswani, BERT, …).

### Fact (locked): encoder freeze

| Phase | JEPA encoder | Source |
|-------|----------------|--------|
| Global SugarJEPA training | **Unfrozen**, lower LR (\(4\times10^{-5}\) vs \(4\times10^{-4}\)) | `jepa_paper` |
| Personal fine-tune | **Frozen**; only SugarOne weights update | `jepa_paper` |

`PERSONALIZATION_JEPA_REPORT.md` must not be used for this. It was rebuilt from `jepa_mae_by_days.csv` and invented an LR-ratio protocol that the paper does not state.

---

## Drafting order (after this structure)

1. Lock Fig. 2 / Table 2 / Table 3 **MAE** from the CSV / reports. JEPA **protocol and wording** from `jepa_paper`. Pick N-HiTS vs NBEATSx.
2. Write Methods 3.3–3.5: SugarOne from us; **3.4 from `jepa_paper`**.
3. Write Results 4.2–4.3 (climax), using their fine-tune table trimmed to SugarOne + jepa-288.
4. Abstract + Introduction (once the numbers are frozen).
5. Related work, Discussion, Conclusion.
6. Compile; cut Related work and Discussion first if over 8 pages.

Do not start `manuscript.tex` until step 1 is done, except for empty section headings if useful. When we do start it, copy `jepa_paper/sugar_jepa.png` and merge `jepa_paper/references.bib` — do not rewrite the colleague folder.
