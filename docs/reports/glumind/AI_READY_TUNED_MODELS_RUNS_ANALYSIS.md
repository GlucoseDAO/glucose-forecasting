# GluMind Run Analysis (ai_ready)

- Scope: **test-only analysis**.
- Rule: when test is unavailable (`trainval_test_as_val`), `val` is used as test-equivalent.
- Metadata normalization: missing `split_scheme` was inferred from available metrics (`test` present -> `classic`; `test` missing + `val` present -> `trainval_test_as_val`).
- Total parent runs analyzed: **13** (global=5, continual=8).

## Global Mode (Test-Only)

| run_name | split_scheme | lr | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- |
| glumind_global_h12_20260223_195526 | trainval_test_as_val | 0.0010 | val_as_test | 11.3336 | 17.7312 | 8.2476 |
| glumind_global_h12_20260222_194108 | classic | 0.0010 | test | 11.3357 | 17.8744 | 8.1687 |
| glumind_global_h12_20260223_010201 | classic | 0.0010 | test | 11.4252 | 17.7949 | 8.3216 |
| glumind_global_h12_20260223_183806 | trainval_test_as_val | 0.0010 | val_as_test | 11.5006 | 18.0872 | 8.3579 |

## Continual Mode (Final Step, Test-Only)

| run_name | split_scheme | lr | lwf_lambda | continual_val_scope | final_step | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glumind_continual_h12_20260223_104653 | classic | 0.0010 | 0.2000 | None | step_04_Insulin_T2DM_20260223_121739 | test | 11.4803 | 17.9541 | 8.3980 |
| glumind_continual_h12_20260223_133040 | classic | 0.0007 | 0.2000 | None | step_04_Insulin_T2DM_20260223_150057 | test | 11.6304 | 18.0403 | 8.5813 |
| glumind_continual_h12_20260223_024502 | classic | 0.0010 | 0.3000 | None | step_04_Insulin_T2DM_20260223_041541 | test | 11.6393 | 18.0230 | 8.5828 |
| glumind_continual_h12_20260222_224757 | classic | 0.0010 | 0.5000 | None | step_04_Insulin_T2DM_20260223_004254 | test | 11.7846 | 18.0474 | 8.7412 |
| glumind_continual_h12_20260224_020530 | trainval_test_as_val | 0.0010 | 0.3000 | all_groups | step_04_Insulin_T2DM_20260224_035131 | val_as_test | 11.8019 | 18.1233 | 8.7474 |
| glumind_continual_h12_20260223_235557 | trainval_test_as_val | 0.0010 | 0.2000 | all_groups | step_04_Insulin_T2DM_20260224_014317 | val_as_test | 11.8031 | 18.1057 | 8.7894 |
| glumind_continual_h12_20260223_123121 | classic | 0.0005 | 0.2000 | None | step_04_Insulin_T2DM_20260223_132056 | test | 11.8376 | 18.1078 | 8.8157 |
| glumind_continual_h12_20260223_215047 | trainval_test_as_val | 0.0010 | 0.2000 | None | step_04_Insulin_T2DM_20260223_233206 | val_as_test | 13.5420 | 20.9165 | 8.2876 |

## Best Runs by Test Metric

- Best global: `glumind_global_h12_20260223_195526` | split=`trainval_test_as_val` | lr=0.001 | source=val_as_test | MAE=11.3336, RMSE=17.7312, MARD=8.2476%.
- Best continual: `glumind_continual_h12_20260223_104653` | split=`classic` | lr=0.001 | lwf_lambda=0.2 | scope=None | source=test | MAE=11.4803, RMSE=17.9541, MARD=8.3980%.

## Global vs Continual (Best-by-Split, Test-Only)

| split_scheme | global_run | global_test_mae | continual_run | continual_test_mae | delta_cont_minus_global |
| --- | --- | --- | --- | --- | --- |
| classic | glumind_global_h12_20260222_194108 | 11.3357 | glumind_continual_h12_20260223_104653 | 11.4803 | 0.1446 |
| trainval_test_as_val | glumind_global_h12_20260223_195526 | 11.3336 | glumind_continual_h12_20260224_020530 | 11.8019 | 0.4683 |

- Positive `delta_cont_minus_global` means continual is worse.

## Exact Per-Group Comparison For Best Checkpoints

### Split: classic (source=test)

| study_group | mae_global | mae_continual | delta_mae_cont_minus_global | rmse_global | rmse_continual | delta_rmse_cont_minus_global | mard_global | mard_continual | delta_mard_cont_minus_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 9.490091 | 9.987715 | 0.497623 | 14.702601 | 15.427317 | 0.724715 | 8.008856 | 8.551748 | 0.542892 |
| Insulin-T2DM | 13.826124 | 13.516909 | -0.309216 | 21.572123 | 20.989176 | -0.582947 | 8.347141 | 8.214952 | -0.132189 |
| Oral-T2DM | 12.454449 | 12.451815 | -0.002634 | 19.440022 | 19.328644 | -0.111378 | 8.497314 | 8.582141 | 0.084826 |
| Pre-T2DM | 9.819306 | 10.166709 | 0.347403 | 15.272351 | 15.783174 | 0.510822 | 7.835508 | 8.222585 | 0.387076 |

### Split: trainval_test_as_val (source=val_as_test)

| study_group | mae_global | mae_continual | delta_mae_cont_minus_global | rmse_global | rmse_continual | delta_rmse_cont_minus_global | mard_global | mard_continual | delta_mard_cont_minus_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 9.579968 | 10.396302 | 0.816334 | 14.704187 | 15.748480 | 1.044292 | 8.161717 | 9.023278 | 0.861561 |
| Insulin-T2DM | 13.702629 | 13.793583 | 0.090954 | 21.276945 | 21.105230 | -0.171715 | 8.344365 | 8.438518 | 0.094152 |
| Oral-T2DM | 12.396898 | 12.661195 | 0.264297 | 19.220505 | 19.334797 | 0.114292 | 8.519620 | 8.840012 | 0.320392 |
| Pre-T2DM | 9.890280 | 10.556857 | 0.666577 | 15.270064 | 16.075815 | 0.805751 | 7.971352 | 8.655773 | 0.684422 |

## LR Analysis (Test-Only)

| mode | split_scheme | lr | n_runs | mean_test_mae | best_test_mae |
| --- | --- | --- | --- | --- | --- |
| continual | classic | 0.0007 | 1 | 11.6304 | 11.6304 |
| continual | classic | 0.0010 | 3 | 11.6347 | 11.4803 |
| continual | classic | 0.0005 | 1 | 11.8376 | 11.8376 |
| continual | trainval_test_as_val | 0.0010 | 3 | 12.3824 | 11.8019 |
| global | classic | 0.0010 | 2 | 11.3804 | 11.3357 |
| global | trainval_test_as_val | 0.0010 | 2 | 11.4171 | 11.3336 |

## Continual `lwf_lambda` Analysis (Test-Only)

| split_scheme | lwf_lambda | n_runs | mean_test_mae | best_test_mae |
| --- | --- | --- | --- | --- |
| classic | 0.3000 | 1 | 11.6393 | 11.6393 |
| classic | 0.2000 | 3 | 11.6494 | 11.4803 |
| classic | 0.5000 | 1 | 11.7846 | 11.7846 |
| trainval_test_as_val | 0.3000 | 1 | 11.8019 | 11.8019 |
| trainval_test_as_val | 0.2000 | 2 | 12.6726 | 11.8031 |

## Validation Against Existing `AI_READY_TUNED_MODELS_REPORT.md`

| check | computed_run | computed_rmse | matches_existing_report |
| --- | --- | --- | --- |
| classic global best RMSE | glumind_global_h12_20260223_010201 | 17.7949 | True |
| classic continual best RMSE | glumind_continual_h12_20260223_104653 | 17.9541 | True |
| trainval_test_as_val global best RMSE | glumind_global_h12_20260223_195526 | 17.7312 | True |
| trainval_test_as_val continual best RMSE | glumind_continual_h12_20260223_235557 | 18.1057 | True |

- Validation status: **PASS**.

## Incomplete Runs (No Test-Equivalent Metric)

| run_name | mode | split_scheme | lr | lwf_lambda | continual_val_scope |
| --- | --- | --- | --- | --- | --- |
| glumind_global_h12_20260223_154930 | global | trainval_test_as_val | 0.0010 | 0.5000 | None |

## Files

- Registry: `_analysis_registry.csv`
- Existing report checked: `AI_READY_TUNED_MODELS_REPORT.md`
- This report: `RUNS_ANALYSIS.md`