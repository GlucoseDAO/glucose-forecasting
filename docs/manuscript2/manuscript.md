# Introduction

Because a CGM reports an excursion only after it has begun (Schmelzeisen-Redeker et al. 2015), a 60-minute forecast—12 steps at 5-minute sampling—gives a person time to act. A population model can be served immediately, but personalizing it requires the wearer’s own data, which accumulates slowly. After N days of one person’s history, is the fine-tuned checkpoint safe to serve?

Most studies report a frozen global model, or one adapted on all available data, or both (Sergazinov et al. 2023; Zhu et al. 2023), but rarely examine what happens in between. If fine-tuning on a partial history can raise MAE, “always adapt” is the wrong default. The system needs a gate: keep the population weights until adaptation is non-harmful.

We treat personalization as a *curve*, scored at 3, 7, 14, 30, 60, and all available days. We introduce **SugarOne**: an encoder-based transformer with parallel dual-attention and learnable softmax mixing, targeted at the covariates any commodity CGM-plus-insulin setup provides: glucose, carbohydrates, and basal/bolus insulin (Section <a href="#sec:sugarone" data-reference-type="ref" data-reference="sec:sugarone">3.3</a>). **SugarJEPA-288** is a variation that adds a frozen self-supervised glucose embedding (Muhammad et al. 2026) as a fourth stream alongside those three. We compare personalization curves for both on seven T1DM users against NeuralForecast continue-fit baselines (Olivares et al. 2022, 2023; Lim et al. 2021).

# Related work

#### Attention models for CGM.

Attention came to CGM forecasting with Gluformer, a probabilistic transformer with subject-level personalization (Sergazinov et al. 2023). Multimodal successors add non-glucose channels: AttenGluco fuses CGM with activity on AI-READI (Farahmand, Azghan, Chatrudi, Kim, et al. 2025), and GluMind combines parallel cross-attention over glucose, heart rate, and steps with multi-scale self-attention (Farahmand, Azghan, Chatrudi, Ansu-Baidoo, et al. 2025). Scaling the same recipe to large unlabeled corpora yields models that describe themselves as CGM foundation models—GluFormer (Lutsker et al. 2026) (distinct from Gluformer above, despite the name), CGMformer (Lu et al. 2025), CGM-LSM (<span class="nocase">Luo et al.</span> 2025), CGM-JEPA (Muhammad et al. 2026) and GlucoFM (Li et al. 2026). Neither group reports a personal-history day-budget curve, which is the axis we score.

#### Our backbone.

SugarOne borrows many elements from GluMind’s transformer design: parallel dual-attention block, glucose embeddings as queries against per-auxiliary keys and values, and multi-scale self-attention over glucose.

#### Fine-tuning sensitivity.

Whether adaptation helps at a given budget is unsettled beyond glucose: break-even against classical baselines ranges from 24 to 8,361 samples on 9 of 30 datasets (Tan Jerome and Simon 2026), and fine-tuned foundation models do not consistently beat smaller dedicated ones once their parameter and memory cost is counted (Karaouli et al. 2025). In CGM, GlucoFM-Bench scores eight architectures on 15 datasets and 1,117 individuals under zero-, few- (5%) and full-shot protocols and finds adaptation non-monotone: at a 12-hour context and 30-minute horizon, TimesFM2.5 and Moirai2.0 degrade from zero- to full-shot (18.75 → 19.47 and 19.48 → 20.30 mg/dL RMSE) (Lu et al. 2026).

All of these studies vary the size of a pooled training set, drawn across many subjects. We vary a different quantity: the number of days of a single person’s own history.

#### Benchmarks and personalization.

GlucoBench documents why so few of these models can be compared: of 45 catalogued methods, 38 ship no public implementation (Sergazinov et al. 2024). Fine-tuning and meta-learning for T1DM typically yield one adapted model (Zhu et al. 2023); closest to us, Rigamonti et al. (2026) ablate shrinking patient-specific training sets.

# Method

## Data

Most of our training data comes from AI-READI (AI-READI Consortium 2024), a wearable CGM study covering Healthy, Pre-T2DM, Oral-T2DM, and Insulin-T2DM groups but no Type 1 participants. Because forecast errors in T1DM can directly affect insulin dosing, we add the Loop observational study from the JAEB Center for Health Research: a public dataset of DIY closed-loop pump users with CGM, basal/bolus insulin, and carbohydrate channels (Appendix <a href="#app:datasets" data-reference-type="ref" data-reference="app:datasets">8</a>). Loop has fewer unique patients than AI-READI, but each contributes many more sequences; by row mass the joined table is roughly half T1DM. `glucose_data_processing` (Anonymous 2026) resamples both sources to a 5-minute grid, gap-fills short dropouts, and joins them into one table (12.1 million rows). AI-READI rows have no insulin or carbohydrate data (zero-filled); sequence IDs are prefixed to prevent collisions.

The personal test also includes *Author1*: a personal Dexcom CGM recording collected by a co-author (~345 train days), ingested from the raw Dexcom export onto the 5-minute grid and held out from the joined training table.

## Task and two tests

From this joined table every model forecasts H=12 glucose values (60 min at 5-min sampling). The lead metric is MAE in mg/dL, alongside RMSE and MARD—the standard CGM accuracy measure. All models share this horizon.

As a secondary experiment, we also trained a 120-minute forecasting variant (H=24) after Google released GlucoFM (Li et al. 2026), which reports 2-hour glucose prediction. We did not fine-tune the 120-minute models but evaluated them at the global (zero-shot) level. On this horizon, our generic SugarOne already reaches MAE 17.99 mg/dL and SugarJEPA 16.32, against GlucoFM’s 21.88. While our metrics look better, GlucoFM scores narrower meal-related segments and adds nutrition and subject covariates, so the comparison is not direct. These numbers are preliminary—no personalization curves were run at H=24—but they suggest the architecture is competitive beyond the 60-minute horizon that is the focus of this paper.

Because the table mixes populations with and without insulin channels, we use two evaluations that must not be combined.

**Global test.** The `test` split of the joined table: is this a competent population model?

**Personal test.** Seven T1DM users with long history—the Author1 export and six Loop holdouts (Users 154, 556, 730, 1017, 1029, 1082)—each split chronologically (last 25% test, 15% of the remainder validation, rest train). A day budget shortens *train* only; validation and test stay fixed. User 1082 has ~37 train days and no 60-day cell; 60-day means use n=6.

Eight short-wear AI-READI users (~6–9 train days, no insulin/carb channels) fall outside the main curve: SugarJEPA-288’s one-day lookback exceeds their usable history.

## SugarOne

SugarOne started from improving on GluMind’s parallel dual-attention blocks (Farahmand, Azghan, Chatrudi, Ansu-Baidoo, et al. 2025). A linear layer maps each scalar channel to d_model and adds sinusoidal positional encoding. Each block has (i) cross-attention in which glucose embeddings are queries and each auxiliary supplies its own keys and values, and (ii) multi-scale self-attention on glucose at downsampling factors 1, 2, and 4. A two-layer MLP decodes 12 steps. The SugarOne trunk is the non-JEPA part of Figure <a href="#fig:arch" data-reference-type="ref" data-reference="fig:arch">2</a> (appendix).

While GluMind was built for wearable extras, SugarOne is the same design aimed at data any CGM records—glucose values and carbohydrates—plus an insulin record. Table <a href="#tab:glumind-vs-sugarone" data-reference-type="ref" data-reference="tab:glumind-vs-sugarone">1</a> is the difference. Fusion is learnable softmax mixing over the three auxiliary streams,
``` math
\mathbf{C}=\sum_{i=1}^{3}w_i\mathbf{C}_i,\qquad
w_i=\frac{\exp(\alpha_i)}{\sum_{j}\exp(\alpha_j)},
```
with α initialized at zero (equal mix). We do not evaluate a GluMind checkpoint on the personalization curves. SugarOne is the control in the same family as SugarJEPA.

<div id="tab:glumind-vs-sugarone">

|                 |           GluMind           |           SugarOne           |
|:----------------|:---------------------------:|:----------------------------:|
| Auxiliaries     |      Heart rate, steps      | Basal, bolus, carbohydrates  |
| Fusion          |        Fixed average        |      Learnable softmax       |
| Lookback / size | 80 steps, 4 heads, 3 blocks | 128 steps, 8 heads, 5 blocks |
| Width           |          d=32           |           d=32           |
| Training table  |   Wearable AI-READI-style   |    Joined Loop + AI-READI    |

SugarOne is unpublished. It keeps GluMind’s blocks and changes covariates, mixing, size, and data.

</div>

## SugarJEPA

We pretrain a CGM-JEPA-style encoder (Muhammad et al. 2026) on the joined `train` split, glucose only. A random 5% of that split is held out for encoder validation and is never used for encoder training. The pretraining objective is SmoothL1 latent prediction, with an EMA teacher for weight updates and a VICReg-style variance penalty (Bardes et al. 2022):
``` math
L=\mathrm{SmoothL1}(\hat{z},z)
+\lambda\frac{1}{E}\sum_{j=1}^{E}
\mathrm{ReLU}\bigl(\sigma_{\mathrm{target}}-\sigma_{c,j}\bigr),
```
where E is the embedding size and σ_c,j is the standard deviation of context-block representations on dimension j.

The encoder is attached as a fourth SugarOne branch, on the same footing as basal, bolus, and carbohydrates (Figure <a href="#fig:arch" data-reference-type="ref" data-reference="fig:arch">2</a>). Embeddings are layer-normalized and projected from 96 to 32 dimensions before cross-attention. If the encoder wants m steps and SugarOne wants n=128, with m>n, each training window has length m: the encoder sees all m points; SugarOne sees the last n.

During *global* SugarJEPA training the encoder is not frozen. It is updated at 4×10^-5; the rest of the model at 4×10^-4. The main encoder, the one this paper reports, is **jepa-288** (96 dimensions, 288 steps, one day of CGM). Other windows (128, 864, 2016) are in Appendix <a href="#app:windows" data-reference-type="ref" data-reference="app:windows">9</a>. Longer windows change which series can be scored; they are not the claim of this paper.

## Fine-tune protocol and smoothness

Every day budget starts fresh from the global checkpoint; we do not chain budgets into a curriculum. Min–max scalers fit on the global training split are reused at personalization and never refit on personal data. When personalizing SugarJEPA we freeze the JEPA encoder and update only the SugarOne weights, using plain fine-tuning (λ=0). We tried Learning-without-Forgetting distillation (Li and Hoiem 2018), but it did not remove SugarOne’s short-budget harm, so we dropped it. NeuralForecast models follow the same idea: continue-fit from the saved global bundle (`use_init_models=False`), shorten training to the day budget, keep the personal test split fixed (Olivares et al. 2022).

Figure <a href="#fig:curves" data-reference-type="ref" data-reference="fig:curves">1</a> shows zero-shot, then 3, 7, 14, 30, 60, and full history. We omit a 1-day point because SugarJEPA-288 already looks back one full day of CGM, and a 1-day training slice cannot supply enough context for a window. Models whose shorter lookback does allow a 1-day fine-tune appear in Appendix <a href="#app:oneday" data-reference-type="ref" data-reference="app:oneday">11</a>.

<figure id="fig:curves" data-latex-placement="t">
<img src="fig_personalization_curves.png" style="width:92.0%" />
<figcaption>Mean personal-test MAE versus train-day budget on seven T1DM users (60-day means: <span class="math inline"><em>n</em> = 6</span>). Dotted line: 30-day SugarOne point. There is no 1-day budget: SugarJEPA-288 cannot form a window from one day of train.</figcaption>
</figure>

We call a personalization path *smooth* when it passes three checks on mean personal-test MAE across the seven users (60-day averages use n=6 because one user’s history is shorter):

1.  **Non-harmful.** MAE at budget t stays at or below that model’s own zero-shot.

2.  **Early gain.** The mean drops below zero-shot before 60 days.

3.  **Terminal gain.** Full-history MAE beats zero-shot.

These are checks on the group mean; individual users may dip.

The main figure shows four curves: SugarOne, SugarJEPA-288, NBEATSx (whose short continue-fit is harmful), and TFT (which helps from about 30 days). Other NeuralForecast models appear in Appendix <a href="#app:nf" data-reference-type="ref" data-reference="app:nf">10</a>.

# Results

## Global test

Table <a href="#tab:global" data-reference-type="ref" data-reference="tab:global">3</a> is the joined-corpus holdout, not the personal chronological test. All four models are scored on the dataset `test` split. NBEATSx and TFT are the global NeuralForecast bundles used for continue-fit.

<div class="minipage">

<div id="tab:global">

| Model         |  MAE  | RMSE  | MARD  |
|:--------------|:-----:|:-----:|:-----:|
| SugarOne      | 12.41 | 19.05 | 9.90% |
| SugarJEPA-288 | 11.37 | 17.63 | 9.08% |
| NBEATSx       | 11.81 | 19.10 | 8.05% |
| TFT           | 12.69 | 20.36 | 8.47% |

**Left:** Joined-corpus holdout (H\!=\!12, 60 min). NBEATSx and TFT are the global NeuralForecast bundles used for continue-fit. SugarJEPA-864/2016 are omitted; they score a smaller population of long series. **Right:** H\!=\!24 global test (120 min), trained after the release of GlucoFM (Li et al. 2026) (GlucoFM’s 21.88 is on narrower meal-related segments). Generic models, no fine-tuning. The 60 min rows show the first 12 steps of the H\!=\!24 forecast.

</div>

</div>

<div class="minipage">

<div id="tab:global">

<table>
<caption><strong>Left:</strong> Joined-corpus holdout (<span class="math inline"><em>H</em> = 12</span>, 60 min). NBEATSx and TFT are the global NeuralForecast bundles used for continue-fit. SugarJEPA-864/2016 are omitted; they score a smaller population of long series. <strong>Right:</strong> <span class="math inline"><em>H</em> = 24</span> global test (120 min), trained after the release of GlucoFM <span class="citation" data-cites="li2026glucofm">(Li et al. 2026)</span> (GlucoFM’s 21.88 is on narrower meal-related segments). Generic models, no fine-tuning. The 60 min rows show the first 12 steps of the <span class="math inline"><em>H</em> = 24</span> forecast.</caption>
<thead>
<tr>
<th style="text-align: left;"></th>
<th style="text-align: center;">MAE</th>
<th style="text-align: center;">RMSE</th>
<th style="text-align: center;">MARD</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="4" style="text-align: left;"><em>120 min (<span class="math inline"><em>H</em> = 24</span>)</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">17.99</td>
<td style="text-align: center;">27.45</td>
<td style="text-align: center;">13.61%</td>
</tr>
<tr>
<td style="text-align: left;">SugarJEPA-288</td>
<td style="text-align: center;">16.32</td>
<td style="text-align: center;">25.23</td>
<td style="text-align: center;">12.57%</td>
</tr>
<tr>
<td colspan="4" style="text-align: left;"><em>60 min (<span class="math inline"><em>H</em> = 24</span> model)</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">13.02</td>
<td style="text-align: center;">18.99</td>
<td style="text-align: center;">9.66%</td>
</tr>
<tr>
<td style="text-align: left;">SugarJEPA-288</td>
<td style="text-align: center;">11.56</td>
<td style="text-align: center;">16.81</td>
<td style="text-align: center;">8.93%</td>
</tr>
</tbody>
</table>

</div>

</div>

## Personalization paths

Frozen SugarJEPA-288 has lower personal-test MAE than SugarOne fine-tuned for 30 days, for all seven T1DM users in this study (Table <a href="#tab:slice" data-reference-type="ref" data-reference="tab:slice">4</a>). That is one slice. Figure <a href="#fig:curves" data-reference-type="ref" data-reference="fig:curves">1</a> is the path.

SugarOne starts at 19.48 mg/dL. Mean MAE stays at or above that floor through 30 days (19.64) and only then falls (19.09 at 60 days; 18.67 at full history). Check 1 fails at 3–30 days. Checks 2–3 hold only after the 60-day mark.

NBEATSx starts worse on this personal test (23.05) despite a competitive global holdout. Continue-fit is harmful through 30 days (25.66) and still above zero-shot at 60 days (23.91). Full history recovers (21.58). The gate is mandatory.

TFT starts at 24.41. At 3–14 days the mean is *higher* than zero-shot (32.78, 29.56, 27.04). By 30 days the mean is below zero-shot (22.65) and full history is 19.87. TFT is not a smooth path from day 3. It is a delayed-gain exception among the NeuralForecast models we plot: useful from about 30 days, costly before that.

SugarJEPA-288 starts at 18.13—already below 30-day SugarOne. From 3 days the mean stays at or below its own zero-shot (18.08, 17.99, 17.92, 17.82; 18.09 at 60 days, still ≤ 18.13) and ends at 17.51. All three checks hold on the mean. Single users can dip; we do not claim monotonicity per person.

User 1082 is the short T1DM history (~37 train days). SugarOne’s full fine-tune is worse than frozen (17.00 → 17.79). SugarJEPA-288 stays flat (15.17 → 15.19). That is evidence for a gate, not a reason to drop the user.

## The 30-day slice

Thirty days is a budget at which a clinic might first try to personalize, and at which SugarOne’s mean gain is still about zero. Table <a href="#tab:slice" data-reference-type="ref" data-reference="tab:slice">4</a> holds for every user in this study. The same “all seven” sentence is false against SugarOne’s *full* fine-tune: Author1 (16.98) and User 1017 (16.95) beat frozen SugarJEPA-288 (17.64 and 17.41). Thirty days is the honest quote because it sits on Figure <a href="#fig:curves" data-reference-type="ref" data-reference="fig:curves">1</a>, not because it replaces the figure.

<div id="tab:slice">

| User    | SugarJEPA-288 ZS | SugarOne @ 30 d | Margin |
|:--------|:----------------:|:---------------:|:------:|
| Author1 |      17.64       |      18.06      |  0.42  |
| 154     |      23.13       |      24.84      |  1.70  |
| 556     |      17.22       |      17.65      |  0.43  |
| 730     |      16.02       |      18.23      |  2.21  |
| 1017    |      17.41       |      18.30      |  0.90  |
| 1029    |      20.30       |      22.81      |  2.51  |
| 1082    |      15.17       |      17.60      |  2.43  |

Personal-test MAE (mg/dL). Frozen SugarJEPA-288 versus SugarOne fine-tuned for 30 days. All seven T1DM users in this study.

</div>

## Fine-tuning SugarJEPA-288

A better frozen model can still adapt. With the JEPA encoder held fixed, mean personal MAE versus SugarJEPA-288’s own zero-shot drops slightly at 3 days (-0.05), then more at 30 days (-0.31), only a little at 60 days (-0.04, n=6), and most at full history (-0.62). The frozen-encoder path is a usable route, and more personal data keeps paying off. That beats waiting 60 days before fine-tuning SugarOne at all.

# Discussion

The introduction asks when a fine-tuned checkpoint becomes safe to serve, and the answer is easiest to read as a number of sensors. A current sensor lasts ten to fifteen days, so the sixty days SugarOne needs before adaptation helps is four or more consecutive wears. That is not an unreasonable thing to ask of a motivated user, but it is weeks of wear spent serving a model that could already have been personalized. SugarJEPA-288 crosses the same line three days into the first sensor.

The reason to care is that the risk is asymmetric. Adapting too early can add more error than adapting ever removes, and the cost is paid immediately while the benefit arrives months later. How large that cost is depends on the model and the fine-tuning recipe rather than on the day budget, so the safe day count has to be measured per model; a strong score on the joined corpus does not predict it.

What the JEPA feature contributes is a floor rather than a faster learner. Attaching the frozen embedding improves day-zero accuracy by more than fine-tuning SugarOne on a person’s entire history does, and the two then add up: the corpus supplies glucose dynamics, the person supplies their own response, and neither substitutes for the other. Two of seven users still do better fully fine-tuned, so this shifts the average starting point rather than settling every case.

Two failures say where the effect comes from. Distilling from the population checkpoint during fine-tuning left the short-budget harm intact, so that harm is not simple forgetting; changing what the model sees is what removed it. Nor is the gain merely a longer view of the past, since an encoder matched to SugarOne’s own lookback keeps about half of it while the longest encoder we trained is the least stable one to adapt (Appendix <a href="#app:windows" data-reference-type="ref" data-reference="app:windows">9</a>).

Limits. Seven T1DM users, and smoothness is a property of averages rather than of individuals. The short-wear users fall outside Figure <a href="#fig:curves" data-reference-type="ref" data-reference="fig:curves">1</a> because our best encoder needs a day of context; serving them means the shorter encoders, which we have not scored on this axis. Nothing here is a clinical claim.

# Conclusion

How a 60-minute glucose forecaster personalizes cannot be judged from two snapshots, one with no personal data and one with a full history. What matters is the path between them, because that is where adaptation can quietly make a model worse while both endpoints still look healthy.

Scored that way, a frozen CGM-JEPA feature does two things for SugarOne. It improves accuracy before the model has seen any of the person’s own data, and it keeps early fine-tuning—on the handful of days a new wearer actually has—from doing harm. The practical difference is a number of sensors: SugarJEPA-288 can be adapted three days into someone’s first wear, while SugarOne and the NeuralForecast baselines need four or more consecutive sensors before adaptation stops hurting.

The clearest single result is that frozen SugarJEPA-288, with no fine-tuning at all, was better than SugarOne fine-tuned for thirty days for every one of the seven people with type 1 diabetes we tested. Read together with the rest of the paper, that points at improving what a forecaster sees rather than how it adapts.

# Architecture

<figure id="fig:arch" data-latex-placement="H">
<img src="sugar_jepa.png" style="width:92.0%" />
<figcaption>SugarJEPA-288. One shared window of 288 steps (24 h) ends at forecast time. The JEPA encoder reads all 288 glucose steps (36 patches of 8) and projects them to a <span class="math inline">(<em>B</em>, 36, 32)</span> K/V stream; the SugarOne trunk embeds the last 128 steps of glucose, basal, bolus, and carbohydrates. Cross-attention lets the 128-step glucose queries attend to the shorter JEPA sequence. Personal fine-tunes freeze the encoder and update the SugarOne weights.</figcaption>
</figure>

# Datasets

Table <a href="#tab:datasets" data-reference-type="ref" data-reference="tab:datasets">5</a> summarizes the two public sources. Loop is the observational study of open-source DIY closed-loop pump users collected by the JAEB Center for Health Research (~1,000 T1DM participants). `glucose_data_processing` (Anonymous 2026) supports ingestion of over 50 public CGM datasets across 9 device formats; the full catalog is in that repository’s documentation.

<div id="tab:datasets">

| Dataset  |     Participants      |     Diabetes types      | CGM | Insulin | Carbs |
|:---------|:---------------------:|:-----------------------:|:---:|:-------:|:-----:|
| AI-READI | ~4,000 | Healthy, Pre-T2DM, T2DM | yes |    —    |   —   |
| Loop     | ~1,000 |          T1DM           | yes |   yes   |  yes  |

Datasets used in this study. AI-READI provides wearable CGM across metabolic health groups but no Type 1 participants and no insulin/carbohydrate records. Loop adds T1DM users with full pump telemetry. Author1 is a personal Dexcom CGM recording from a co-author, not counted in the Loop row.

</div>

# Other JEPA windows

Table <a href="#tab:encoders" data-reference-type="ref" data-reference="tab:encoders">6</a> lists the encoders that were trained. The main text uses jepa-288 only. Table <a href="#tab:all-ft" data-reference-type="ref" data-reference="tab:all-ft">7</a> is the full personal-test grid from the JEPA source draft. Empty cells are missing runs, not zeros: a window needs lookback plus horizon, so 1-day train is undefined for jepa-288 and short budgets drop for 864/2016. User 1082 has no 60-day cell. jepa-2016 has no rows for Users 1017 and 1082; do not average those people in. jepa-2016 can *raise* MAE at full fine-tune (mean 19.77 versus zero-shot 18.96). Longer context is not a smoother adapter. Global MAE for 864/2016 looks better in part because those encoders score fewer, longer series (Healthy windows fall from 194k at 288 steps to 49k at 2016). That is why they are not in Table <a href="#tab:global" data-reference-type="ref" data-reference="tab:global">3</a>.

<div id="tab:encoders">

| Encoder     | Embedding dim. |   Context    |
|:------------|:--------------:|:------------:|
| jepa-128-64 |       64       | 128 (10.7 h) |
| jepa-128    |       96       | 128 (10.7 h) |
| jepa-288    |       96       |  288 (1 d)   |
| jepa-864    |       96       |  864 (3 d)   |
| jepa-2016   |       96       |  2016 (7 d)  |

JEPA encoder variants. Only jepa-288 is in the main figure.

</div>

Mean personal zero-shot MAE: SugarOne 19.48; jepa-128-64 19.00; jepa-128 18.87; jepa-288 18.13. The 128-step encoders match SugarOne’s lookback and still start lower. They are a fairer architecture comparison; jepa-288 is the smoother hero curve we locked.

<div id="tab:all-ft">

<table>
<caption>Personal-test MAE (mg/dL) by user and day budget. Source: JEPA draft table. Means do not fill empty cells with zeros.</caption>
<thead>
<tr>
<th style="text-align: left;">Model</th>
<th style="text-align: center;">ZS</th>
<th style="text-align: center;">1d</th>
<th style="text-align: center;">3d</th>
<th style="text-align: center;">7d</th>
<th style="text-align: center;">14d</th>
<th style="text-align: center;">30d</th>
<th style="text-align: center;">60d</th>
<th style="text-align: center;">All</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9" style="text-align: left;"><em>Author1</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">18.31</td>
<td style="text-align: center;">18.42</td>
<td style="text-align: center;">18.60</td>
<td style="text-align: center;">18.88</td>
<td style="text-align: center;">18.48</td>
<td style="text-align: center;">18.06</td>
<td style="text-align: center;">17.54</td>
<td style="text-align: center;">16.98</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">18.05</td>
<td style="text-align: center;">19.14</td>
<td style="text-align: center;">17.56</td>
<td style="text-align: center;">17.30</td>
<td style="text-align: center;">17.26</td>
<td style="text-align: center;">17.31</td>
<td style="text-align: center;">16.90</td>
<td style="text-align: center;">16.76</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">17.67</td>
<td style="text-align: center;">18.10</td>
<td style="text-align: center;">17.55</td>
<td style="text-align: center;">17.24</td>
<td style="text-align: center;">17.24</td>
<td style="text-align: center;">17.20</td>
<td style="text-align: center;">16.95</td>
<td style="text-align: center;">16.69</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">17.64</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">17.61</td>
<td style="text-align: center;">17.21</td>
<td style="text-align: center;">17.09</td>
<td style="text-align: center;">17.12</td>
<td style="text-align: center;">16.83</td>
<td style="text-align: center;">16.53</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">17.97</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">17.91</td>
<td style="text-align: center;">17.69</td>
<td style="text-align: center;">17.96</td>
<td style="text-align: center;">17.42</td>
<td style="text-align: center;">16.69</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">18.61</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">18.07</td>
<td style="text-align: center;">18.15</td>
<td style="text-align: center;">17.46</td>
<td style="text-align: center;">17.79</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;"><em>User 154</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">24.61</td>
<td style="text-align: center;">24.74</td>
<td style="text-align: center;">24.57</td>
<td style="text-align: center;">24.57</td>
<td style="text-align: center;">24.57</td>
<td style="text-align: center;">24.84</td>
<td style="text-align: center;">24.81</td>
<td style="text-align: center;">24.12</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">25.40</td>
<td style="text-align: center;">24.89</td>
<td style="text-align: center;">24.30</td>
<td style="text-align: center;">24.30</td>
<td style="text-align: center;">24.30</td>
<td style="text-align: center;">24.47</td>
<td style="text-align: center;">24.01</td>
<td style="text-align: center;">23.64</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">25.48</td>
<td style="text-align: center;">25.22</td>
<td style="text-align: center;">24.59</td>
<td style="text-align: center;">24.59</td>
<td style="text-align: center;">24.59</td>
<td style="text-align: center;">24.67</td>
<td style="text-align: center;">24.33</td>
<td style="text-align: center;">23.90</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">23.13</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">22.84</td>
<td style="text-align: center;">22.84</td>
<td style="text-align: center;">22.84</td>
<td style="text-align: center;">22.82</td>
<td style="text-align: center;">22.87</td>
<td style="text-align: center;">22.66</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">22.74</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">22.58</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">21.93</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">27.28</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;"><em>User 556</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">18.10</td>
<td style="text-align: center;">17.74</td>
<td style="text-align: center;">18.00</td>
<td style="text-align: center;">17.89</td>
<td style="text-align: center;">18.40</td>
<td style="text-align: center;">17.65</td>
<td style="text-align: center;">17.25</td>
<td style="text-align: center;">17.39</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">17.52</td>
<td style="text-align: center;">17.47</td>
<td style="text-align: center;">17.40</td>
<td style="text-align: center;">17.36</td>
<td style="text-align: center;">17.50</td>
<td style="text-align: center;">17.19</td>
<td style="text-align: center;">16.90</td>
<td style="text-align: center;">16.92</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">17.46</td>
<td style="text-align: center;">17.79</td>
<td style="text-align: center;">17.52</td>
<td style="text-align: center;">17.55</td>
<td style="text-align: center;">17.64</td>
<td style="text-align: center;">17.61</td>
<td style="text-align: center;">17.25</td>
<td style="text-align: center;">17.01</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">17.22</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">16.94</td>
<td style="text-align: center;">16.94</td>
<td style="text-align: center;">16.96</td>
<td style="text-align: center;">17.11</td>
<td style="text-align: center;">17.03</td>
<td style="text-align: center;">16.65</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">16.46</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">16.38</td>
<td style="text-align: center;">16.62</td>
<td style="text-align: center;">16.52</td>
<td style="text-align: center;">16.60</td>
<td style="text-align: center;">16.00</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">15.30</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">19.69</td>
<td style="text-align: center;">15.18</td>
<td style="text-align: center;">15.49</td>
<td style="text-align: center;">15.49</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;"><em>User 730</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">18.06</td>
<td style="text-align: center;">18.27</td>
<td style="text-align: center;">18.42</td>
<td style="text-align: center;">18.31</td>
<td style="text-align: center;">18.02</td>
<td style="text-align: center;">18.23</td>
<td style="text-align: center;">16.52</td>
<td style="text-align: center;">16.50</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">16.35</td>
<td style="text-align: center;">16.43</td>
<td style="text-align: center;">16.50</td>
<td style="text-align: center;">16.24</td>
<td style="text-align: center;">16.23</td>
<td style="text-align: center;">16.39</td>
<td style="text-align: center;">16.13</td>
<td style="text-align: center;">16.06</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">16.36</td>
<td style="text-align: center;">16.44</td>
<td style="text-align: center;">16.46</td>
<td style="text-align: center;">16.24</td>
<td style="text-align: center;">16.31</td>
<td style="text-align: center;">16.47</td>
<td style="text-align: center;">16.24</td>
<td style="text-align: center;">16.14</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">16.02</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">15.90</td>
<td style="text-align: center;">15.92</td>
<td style="text-align: center;">15.80</td>
<td style="text-align: center;">15.73</td>
<td style="text-align: center;">15.71</td>
<td style="text-align: center;">15.62</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">16.17</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">16.05</td>
<td style="text-align: center;">16.05</td>
<td style="text-align: center;">16.04</td>
<td style="text-align: center;">15.91</td>
<td style="text-align: center;">15.83</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">14.73</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">14.95</td>
<td style="text-align: center;">14.93</td>
<td style="text-align: center;">15.75</td>
<td style="text-align: center;">15.39</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;"><em>User 1017</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">17.69</td>
<td style="text-align: center;">17.85</td>
<td style="text-align: center;">17.89</td>
<td style="text-align: center;">17.91</td>
<td style="text-align: center;">18.01</td>
<td style="text-align: center;">18.30</td>
<td style="text-align: center;">17.38</td>
<td style="text-align: center;">16.95</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">18.20</td>
<td style="text-align: center;">17.88</td>
<td style="text-align: center;">17.95</td>
<td style="text-align: center;">17.96</td>
<td style="text-align: center;">17.93</td>
<td style="text-align: center;">17.72</td>
<td style="text-align: center;">16.89</td>
<td style="text-align: center;">16.62</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">17.97</td>
<td style="text-align: center;">17.81</td>
<td style="text-align: center;">17.80</td>
<td style="text-align: center;">17.78</td>
<td style="text-align: center;">17.88</td>
<td style="text-align: center;">17.63</td>
<td style="text-align: center;">17.00</td>
<td style="text-align: center;">16.75</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">17.41</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">17.37</td>
<td style="text-align: center;">17.37</td>
<td style="text-align: center;">16.93</td>
<td style="text-align: center;">16.94</td>
<td style="text-align: center;">16.58</td>
<td style="text-align: center;">16.38</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">16.62</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">16.85</td>
<td style="text-align: center;">17.01</td>
<td style="text-align: center;">16.83</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;"><em>User 1029</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">22.62</td>
<td style="text-align: center;">22.66</td>
<td style="text-align: center;">22.67</td>
<td style="text-align: center;">22.68</td>
<td style="text-align: center;">22.22</td>
<td style="text-align: center;">22.81</td>
<td style="text-align: center;">21.04</td>
<td style="text-align: center;">20.94</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">20.93</td>
<td style="text-align: center;">20.97</td>
<td style="text-align: center;">20.88</td>
<td style="text-align: center;">20.49</td>
<td style="text-align: center;">20.57</td>
<td style="text-align: center;">20.40</td>
<td style="text-align: center;">19.96</td>
<td style="text-align: center;">19.90</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">20.60</td>
<td style="text-align: center;">20.81</td>
<td style="text-align: center;">20.56</td>
<td style="text-align: center;">20.41</td>
<td style="text-align: center;">20.36</td>
<td style="text-align: center;">20.24</td>
<td style="text-align: center;">20.04</td>
<td style="text-align: center;">20.01</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">20.30</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">20.28</td>
<td style="text-align: center;">20.28</td>
<td style="text-align: center;">20.78</td>
<td style="text-align: center;">19.94</td>
<td style="text-align: center;">19.52</td>
<td style="text-align: center;">19.55</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">19.37</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">20.38</td>
<td style="text-align: center;">19.45</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">24.22</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">22.91</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;"><em>User 1082</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">17.00</td>
<td style="text-align: center;">16.97</td>
<td style="text-align: center;">17.08</td>
<td style="text-align: center;">17.13</td>
<td style="text-align: center;">17.20</td>
<td style="text-align: center;">17.60</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">17.79</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">16.54</td>
<td style="text-align: center;">16.55</td>
<td style="text-align: center;">16.55</td>
<td style="text-align: center;">16.36</td>
<td style="text-align: center;">16.41</td>
<td style="text-align: center;">16.56</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">16.65</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">16.57</td>
<td style="text-align: center;">16.54</td>
<td style="text-align: center;">16.54</td>
<td style="text-align: center;">16.48</td>
<td style="text-align: center;">16.51</td>
<td style="text-align: center;">16.62</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">16.66</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">15.17</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">15.59</td>
<td style="text-align: center;">15.36</td>
<td style="text-align: center;">15.04</td>
<td style="text-align: center;">15.10</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">15.19</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">14.78</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">14.83</td>
<td style="text-align: center;">14.97</td>
<td style="text-align: center;">14.97</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">14.97</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td colspan="9" style="text-align: left;"><em>Mean</em></td>
</tr>
<tr>
<td style="text-align: left;">SugarOne</td>
<td style="text-align: center;">19.48</td>
<td style="text-align: center;">19.52</td>
<td style="text-align: center;">19.61</td>
<td style="text-align: center;">19.62</td>
<td style="text-align: center;">19.56</td>
<td style="text-align: center;">19.64</td>
<td style="text-align: center;">19.09</td>
<td style="text-align: center;">18.67</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128-64</td>
<td style="text-align: center;">19.00</td>
<td style="text-align: center;">19.05</td>
<td style="text-align: center;">18.74</td>
<td style="text-align: center;">18.57</td>
<td style="text-align: center;">18.60</td>
<td style="text-align: center;">18.58</td>
<td style="text-align: center;">18.47</td>
<td style="text-align: center;">18.08</td>
</tr>
<tr>
<td style="text-align: left;">jepa-128</td>
<td style="text-align: center;">18.87</td>
<td style="text-align: center;">18.96</td>
<td style="text-align: center;">18.72</td>
<td style="text-align: center;">18.61</td>
<td style="text-align: center;">18.65</td>
<td style="text-align: center;">18.63</td>
<td style="text-align: center;">18.64</td>
<td style="text-align: center;">18.16</td>
</tr>
<tr>
<td style="text-align: left;">jepa-288</td>
<td style="text-align: center;">18.13</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">18.08</td>
<td style="text-align: center;">17.99</td>
<td style="text-align: center;">17.92</td>
<td style="text-align: center;">17.82</td>
<td style="text-align: center;">18.09</td>
<td style="text-align: center;">17.51</td>
</tr>
<tr>
<td style="text-align: left;">jepa-864</td>
<td style="text-align: center;">17.73</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">16.29</td>
<td style="text-align: center;">16.33</td>
<td style="text-align: center;">16.47</td>
<td style="text-align: center;">17.46</td>
<td style="text-align: center;">17.48</td>
</tr>
<tr>
<td style="text-align: left;">jepa-2016</td>
<td style="text-align: center;">18.96</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">—</td>
<td style="text-align: center;">17.57</td>
<td style="text-align: center;">16.09</td>
<td style="text-align: center;">16.23</td>
<td style="text-align: center;">19.77</td>
</tr>
</tbody>
</table>

</div>

# Other NeuralForecast models

N-HiTS (Challu et al. 2023) on the same six T1DM users with ≥60 train days has mean continue-fit Δ +2.24 at 30 days (harmful), +0.16 at 60 days, and -1.92 at full history—the same shape as NBEATSx (Olivares et al. 2023). Its joined-corpus test MAE is 11.94 (RMSE 19.38, MARD 8.08%), close to NBEATSx. LSTM (Hochreiter and Schmidhuber 1997) is worse at 30 days (Δ +6.00) and weaker globally (test MAE 17.37, RMSE 26.30, MARD 11.57%). TiDE’s (Das et al. 2023) 30-day mean Δ is -7.01 (helpful) but its global test MAE is 16.12 (RMSE 24.01, MARD 11.07%), well above the models in Table <a href="#tab:global" data-reference-type="ref" data-reference="tab:global">3</a>. We left them off Figure <a href="#fig:curves" data-reference-type="ref" data-reference="fig:curves">1</a> so the harmful-path contrast is one curve (NBEATSx), not five.

# One-day fine-tune

SugarOne, jepa-128\*, and the NeuralForecast models can form a 128-step window from one day of train. SugarJEPA-288 cannot. Mean SugarOne MAE at 1 day is 19.52 (slightly above zero-shot 19.48). NBEATSx and TFT 1-day continue-fit are sharply harmful on several users (e.g. TFT User 1017: 21.70 → 52.81). Those points are omitted from Figure <a href="#fig:curves" data-reference-type="ref" data-reference="fig:curves">1</a> so every model shares the same x-axis.

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-aireadi2024" class="csl-entry">

AI-READI Consortium. 2024. “AI-READI: Rethinking AI Data Collection, Preparation and Sharing in Diabetes Research and Beyond.” *Nature Metabolism* 6 (12): 2210–12. <https://doi.org/10.1038/s42255-024-01165-x>.

</div>

<div id="ref-glucosedataprocessing" class="csl-entry">

Anonymous. 2026. *Glucose_data_processing*. <a href="https://anonymous.4open.science/r/glucose_data_processing" class="uri">Https://anonymous.4open.science/r/glucose_data_processing</a>.

</div>

<div id="ref-bardes2022vicreg" class="csl-entry">

Bardes, Adrien, Jean Ponce, and Yann LeCun. 2022. “VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.” *International Conference on Learning Representations*. <https://arxiv.org/abs/2105.04906>.

</div>

<div id="ref-challu2023nhits" class="csl-entry">

Challu, Cristian, Kin G. Olivares, Boris N. Oreshkin, Federico Garza Ramirez, Max Mergenthaler Canseco, and Artur Dubrawski. 2023. “NHITS: Neural Hierarchical Interpolation for Time Series Forecasting.” *Proceedings of the AAAI Conference on Artificial Intelligence* 37: 6989–97. <https://doi.org/10.1609/aaai.v37i6.25854>.

</div>

<div id="ref-das2023tide" class="csl-entry">

Das, Abhimanyu, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen, and Rose Yu. 2023. “Long-Term Forecasting with TiDE: Time-Series Dense Encoder.” *arXiv Preprint arXiv:2304.08424*. <https://arxiv.org/abs/2304.08424>.

</div>

<div id="ref-farahmand2025glumind" class="csl-entry">

Farahmand, Ebrahim, Reza Rahimi Azghan, Nooshin Taheri Chatrudi, Velarie Yaa Ansu-Baidoo, et al. 2025. “GluMind: Multimodal Parallel Attention and Knowledge Retention for Robust Cross-Population Blood Glucose Forecasting.” *arXiv Preprint arXiv:2509.18457*. <https://arxiv.org/abs/2509.18457>.

</div>

<div id="ref-farahmand2025attengluco" class="csl-entry">

Farahmand, Ebrahim, Reza Rahimi Azghan, Nooshin Taheri Chatrudi, Eric Kim, Gautham Krishna Gudur, and Edison Thomaz. 2025. “AttenGluco: Multimodal Transformer-Based Blood Glucose Forecasting on AI-READI Dataset.” *arXiv Preprint arXiv:2502.09919*. <https://arxiv.org/abs/2502.09919>.

</div>

<div id="ref-hochreiter1997lstm" class="csl-entry">

Hochreiter, Sepp, and Jürgen Schmidhuber. 1997. “Long Short-Term Memory.” *Neural Computation* 9 (8): 1735–80. <https://doi.org/10.1162/neco.1997.9.8.1735>.

</div>

<div id="ref-karaouli2025foundational" class="csl-entry">

Karaouli, Nouha, Denis Coquenet, Elisa Fromont, Martial Mermillod, and Marina Reyboz. 2025. “How Foundational Are Foundation Models for Time Series Forecasting?” *arXiv Preprint arXiv:2510.00742*. <https://arxiv.org/abs/2510.00742>.

</div>

<div id="ref-li2026glucofm" class="csl-entry">

Li, Zechen, Keerthana Natarajan, Weizhi Zhang, et al. 2026. “GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring.” *arXiv Preprint arXiv:2605.30865*. <https://arxiv.org/abs/2605.30865>.

</div>

<div id="ref-li2018lwf" class="csl-entry">

Li, Zhizhong, and Derek Hoiem. 2018. “Learning Without Forgetting.” *IEEE Transactions on Pattern Analysis and Machine Intelligence* 40 (12): 2935–47. <https://doi.org/10.1109/TPAMI.2017.2773081>.

</div>

<div id="ref-lim2021tft" class="csl-entry">

Lim, Bryan, Sercan Ö Arık, Nicolas Loeff, and Tomas Pfister. 2021. “Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting.” *International Journal of Forecasting* 37 (4): 1748–64. <https://doi.org/10.1016/j.ijforecast.2021.03.012>.

</div>

<div id="ref-lu2026glucofmbench" class="csl-entry">

Lu, Baiying, Zhaohui Liang, Ryan Pontius, Shengpu Tang, and Temiloluwa Prioleau. 2026. “GlucoFM-Bench: Benchmarking Time-Series Foundation Models for Blood Glucose Forecasting.” *arXiv Preprint arXiv:2606.06881*. <https://arxiv.org/abs/2606.06881>.

</div>

<div id="ref-lu2025cgmformer" class="csl-entry">

Lu, Yurun, Dan Liu, Zhongming Liang, et al. 2025. “A Pretrained Transformer Model for Decoding Individual Glucose Dynamics from Continuous Glucose Monitoring Data.” *National Science Review* 12 (5): nwaf039. <https://doi.org/10.1093/nsr/nwaf039>.

</div>

<div id="ref-luo2025cgmlsm" class="csl-entry">

<span class="nocase">Luo, Junjie, Abhimanyu Kumbara, Mansur Shomali, et al.</span> 2025. “A Large Sensor Foundation Model Pretrained on Continuous Glucose Monitor Data for Diabetes Management.” *Npj Health Systems* 2 (1). <https://doi.org/10.1038/s44401-025-00039-y>.

</div>

<div id="ref-lutsker2026gluformer" class="csl-entry">

Lutsker, Guy, Gal Sapir, Smadar Shilo, et al. 2026. “A Foundation Model for Continuous Glucose Monitoring Data.” *Nature* 650 (8103): 978–86. <https://doi.org/10.1038/s41586-025-09925-9>.

</div>

<div id="ref-muhammad2026cgmjepa" class="csl-entry">

Muhammad, Hada Melino, Zechen Li, Flora Salim, and Ahmed A. Metwally. 2026. “CGM-JEPA: Learning Consistent Continuous Glucose Monitor Representations via Predictive Self-Supervised Pretraining.” *arXiv Preprint arXiv:2605.00933*. <https://arxiv.org/abs/2605.00933>.

</div>

<div id="ref-olivares2023nbeatsx" class="csl-entry">

Olivares, Kin G., Cristian Challu, Grzegorz Marcjasz, Rafał Weron, and Artur Dubrawski. 2023. “Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with NBEATSx.” *International Journal of Forecasting* 39 (2): 884–900. <https://doi.org/10.1016/j.ijforecast.2022.03.001>.

</div>

<div id="ref-neuralforecast2022" class="csl-entry">

Olivares, Kin G., Cristian Challú, Azul Garza, Max Mergenthaler Canseco, and Artur Dubrawski. 2022. *NeuralForecast: User Friendly State-of-the-Art Neural Forecasting Models*. PyCon Salt Lake City, Utah, US. <https://github.com/Nixtla/neuralforecast>.

</div>

<div id="ref-rigamonti2026patientspecific" class="csl-entry">

Rigamonti, Giorgia, Mirko Paolo Barbato, Davide Marelli, and Paolo Napoletano. 2026. “Tailoring Adverse Event Prediction in Type 1 Diabetes with Patient-Specific Deep Learning Models.” *arXiv Preprint arXiv:2601.14917*. <https://arxiv.org/abs/2601.14917>.

</div>

<div id="ref-schmelzeisen2015delay" class="csl-entry">

Schmelzeisen-Redeker, Günther, Michael Schoemaker, Harald Kirchsteiger, Guido Freckmann, Lutz Heinemann, and Luigi del Re. 2015. “Time Delay of CGM Sensors: Relevance, Causes, and Countermeasures.” *Journal of Diabetes Science and Technology* 9 (5): 1006–15. <https://doi.org/10.1177/1932296815590154>.

</div>

<div id="ref-sergazinov2022gluformer" class="csl-entry">

Sergazinov, Renat, Mohammadreza Armandpour, and Irina Gaynanova. 2023. “Gluformer: Transformer-Based Personalized Glucose Forecasting with Uncertainty Quantification.” *ICASSP 2023 — 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 1–5. <https://doi.org/10.1109/ICASSP49357.2023.10096419>.

</div>

<div id="ref-sergazinov2024glucobench" class="csl-entry">

Sergazinov, Renat, Elizabeth Chun, Valeriya Rogovchenko, Nathaniel Fernandes, Nicholas Kasman, and Irina Gaynanova. 2024. “GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks.” *International Conference on Learning Representations*. <https://arxiv.org/abs/2410.05780>.

</div>

<div id="ref-jerome2026breakeven" class="csl-entry">

Tan Jerome, Nicholas, and Frank Simon. 2026. “When Do Foundation Models Pay Off? A Break-Even Analysis of Pretrained Time Series Forecasters.” *arXiv Preprint arXiv:2607.04919*. <https://arxiv.org/abs/2607.04919>.

</div>

<div id="ref-zhu2023personalized" class="csl-entry">

Zhu, Taiyu, Kezhi Li, Pau Herrero, and Pantelis Georgiou. 2023. “Personalized Blood Glucose Prediction for Type 1 Diabetes Using Evidential Deep Learning and Meta-Learning.” *IEEE Transactions on Biomedical Engineering* 70 (1): 193–204. <https://doi.org/10.1109/TBME.2022.3187703>.

</div>

</div>
