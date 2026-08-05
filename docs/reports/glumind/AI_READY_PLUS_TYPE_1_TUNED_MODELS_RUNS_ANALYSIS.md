# GluMind Run Analysis (ai_ready_plus_type1)

- Scope: **test-only analysis**.
- Rule: when `test` is unavailable (`trainval_test_as_val`), we use **`val` as test-equivalent**.
- Total parent runs: **13**

## Global Mode (Test-Only)

| run_name | split_scheme | lr | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- |
| glumind_global_h12_20260226_032703 | classic | 0.0010 | test | 11.6987 | 18.4568 | 8.5202 |
| glumind_global_h12_20260225_124003 | trainval_test_as_val | 0.0010 | val_as_test | 11.7764 | 18.5579 | 8.5493 |
| glumind_global_h12_20260225_163842 | trainval_test_as_val | 0.0010 | val_as_test | 11.8205 | 18.4367 | 8.7155 |

## Continual Mode (Final Step, Test-Only)

| run_name | split_scheme | lr | lwf_lambda | final_step | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glumind_continual_h12_20260226_011733 | trainval_test_as_val | 0.0007 | 0.3000 | step_05_T1DM_20260226_030422 | val_as_test | 12.1975 | 19.2909 | 8.8975 |
| glumind_continual_h12_20260226_205202 | trainval_test_as_val | 0.0007 | 0.3000 | step_05_T1DM_20260226_223745 | val_as_test | 12.1975 | 19.2909 | 8.8975 |
| glumind_continual_h12_20260226_161154 | trainval_test_as_val | 0.0006 | 0.3000 | step_05_T1DM_20260226_175859 | val_as_test | 12.1991 | 19.3590 | 8.8763 |
| glumind_continual_h12_20260226_183741 | trainval_test_as_val | 0.0008 | 0.3000 | step_05_T1DM_20260226_203528 | val_as_test | 12.3209 | 19.3261 | 9.0165 |
| glumind_continual_h12_20260226_120621 | trainval_test_as_val | 0.0007 | 0.3500 | step_05_T1DM_20260226_135137 | val_as_test | 12.3347 | 19.2248 | 9.1491 |
| glumind_continual_h12_20260226_140741 | trainval_test_as_val | 0.0007 | 0.2500 | step_05_T1DM_20260226_155318 | val_as_test | 12.4004 | 19.3563 | 9.1835 |
| glumind_continual_h12_20260225_201531 | trainval_test_as_val | 0.0010 | 0.3000 | step_05_T1DM_20260225_224412 | val_as_test | 12.4162 | 19.4021 | 9.1010 |
| glumind_continual_h12_20260226_225124 | classic | 0.0007 | 0.3000 | step_05_T1DM_20260227_003537 | test | 12.4673 | 19.4760 | 9.0768 |
| glumind_continual_h12_20260226_100145 | trainval_test_as_val | 0.0007 | 0.2000 | step_05_T1DM_20260226_114747 | val_as_test | 12.5037 | 19.4107 | 9.2956 |
| glumind_continual_h12_20260225_184746 | trainval_test_as_val | 0.0010 | 0.2000 | step_05_T1DM_20260225_195615 | val_as_test | 12.5701 | 19.5564 | 9.2253 |

## Best Runs by Test Metric

- Best global: `glumind_global_h12_20260226_032703` | split=`classic` | lr=0.001 | source=test | MAE=11.6987, RMSE=18.4568, MARD=8.5202%.
- Best continual: `glumind_continual_h12_20260226_011733` | split=`trainval_test_as_val` | lr=0.0007 | lwf_lambda=0.3 | source=val_as_test | MAE=12.1975, RMSE=19.2909, MARD=8.8975%.

## Global vs Continual (Best-by-Split, Test-Only)

| split_scheme | global_run | global_test_mae | continual_run | continual_test_mae | delta_cont_minus_global |
| --- | --- | --- | --- | --- | --- |
| classic | glumind_global_h12_20260226_032703 | 11.6987 | glumind_continual_h12_20260226_225124 | 12.4673 | 0.7685 |
| trainval_test_as_val | glumind_global_h12_20260225_124003 | 11.7764 | glumind_continual_h12_20260226_011733 | 12.1975 | 0.4211 |

- Positive `delta_cont_minus_global` means continual is worse.

## Exact Per-Group Comparison For Best Checkpoints

- Below tables compare the **best global vs best continual** checkpoint per split, using exact group metrics and deltas.

### Split: classic (source=test)

| study_group | mae_global | mae_continual | delta_mae_cont_minus_global | rmse_global | rmse_continual | delta_rmse_cont_minus_global | mard_global | mard_continual | delta_mard_cont_minus_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 9.572092 | 10.786868 | 1.214776 | 14.786479 | 16.540195 | 1.753716 | 8.147040 | 9.288304 | 1.141264 |
| Insulin-T2DM | 13.582047 | 14.259144 | 0.677097 | 21.146038 | 22.197296 | 1.051258 | 8.218802 | 8.410919 | 0.192117 |
| Oral-T2DM | 12.347430 | 12.955923 | 0.608493 | 19.231108 | 20.120304 | 0.889196 | 8.452931 | 8.821866 | 0.368935 |
| Pre-T2DM | 9.887127 | 10.890294 | 1.003167 | 15.365112 | 16.805870 | 1.440758 | 7.953131 | 8.825160 | 0.872029 |
| T1DM | 15.063043 | 14.921614 | -0.141429 | 23.564716 | 23.212223 | -0.352493 | 11.214652 | 10.970834 | -0.243818 |

### Split: trainval_test_as_val (source=val_as_test)

| study_group | mae_global | mae_continual | delta_mae_cont_minus_global | rmse_global | rmse_continual | delta_rmse_cont_minus_global | mard_global | mard_continual | delta_mard_cont_minus_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 9.602149 | 10.555895 | 0.953746 | 14.765375 | 16.444754 | 1.679379 | 8.171497 | 9.028942 | 0.857445 |
| Insulin-T2DM | 13.697993 | 13.800728 | 0.102735 | 21.301912 | 21.791611 | 0.489698 | 8.254066 | 8.215623 | -0.038444 |
| Oral-T2DM | 12.431044 | 12.789638 | 0.358594 | 19.320158 | 20.059608 | 0.739450 | 8.489195 | 8.715504 | 0.226309 |
| Pre-T2DM | 9.928284 | 10.660645 | 0.732362 | 15.363420 | 16.701799 | 1.338380 | 7.979074 | 8.589665 | 0.610591 |
| T1DM | 15.232848 | 14.613951 | -0.618897 | 23.906088 | 22.914139 | -0.991949 | 11.232810 | 10.950115 | -0.282695 |

## LR Analysis (Test-Only)

| mode | split_scheme | lr | n_runs | mean_test_mae | best_test_mae |
| --- | --- | --- | --- | --- | --- |
| continual | classic | 0.0007 | 1 | 12.4673 | 12.4673 |
| continual | trainval_test_as_val | 0.0006 | 1 | 12.1991 | 12.1991 |
| continual | trainval_test_as_val | 0.0008 | 1 | 12.3209 | 12.3209 |
| continual | trainval_test_as_val | 0.0007 | 5 | 12.3268 | 12.1975 |
| continual | trainval_test_as_val | 0.0010 | 2 | 12.4931 | 12.4162 |
| global | classic | 0.0010 | 1 | 11.6987 | 11.6987 |
| global | trainval_test_as_val | 0.0010 | 2 | 11.7985 | 11.7764 |

## Continual `lwf_lambda` Analysis (Test-Only)

| split_scheme | lwf_lambda | n_runs | mean_test_mae | best_test_mae |
| --- | --- | --- | --- | --- |
| classic | 0.3000 | 1 | 12.4673 | 12.4673 |
| trainval_test_as_val | 0.3000 | 5 | 12.2662 | 12.1975 |
| trainval_test_as_val | 0.3500 | 1 | 12.3347 | 12.3347 |
| trainval_test_as_val | 0.2500 | 1 | 12.4004 | 12.4004 |
| trainval_test_as_val | 0.2000 | 2 | 12.5369 | 12.5037 |

## Recommendation (Test-Only)

- Best global overall: `glumind_global_h12_20260226_032703` (MAE 11.6987).
- Best continual overall: `glumind_continual_h12_20260226_011733` (MAE 12.1975, lwf_lambda=0.3).

## Files

- Registry: `_analysis_registry.csv`
- This report: `RUNS_ANALYSIS.md`