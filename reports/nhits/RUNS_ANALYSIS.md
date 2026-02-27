# NHITS Runs Analysis

- Folder: `runs/nhits`
- Scope: NHITS runs only (`model==nhits`).
- Primary ranking uses **effective test metrics**: `test` when available, otherwise `val_as_test`.
- Checkpoint rule: when `eval_checkpoints` exists, each run is scored by its **best checkpoint** (lowest effective test MAE). Otherwise run summary metrics are used.
- Runs discovered: total=14, non-smoke=12, smoke=2.

## Leaderboard (Non-Smoke, Best-Checkpoint-Adjusted)

| run_name | group | dataset_hint | split_inferred | lr | max_steps | step_size | selected_ckpt | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800 | nf_nhits_type1_only | type1_only_testmirror | classic | 0.0010 | 300 | 12 | step-step=150 | test | 15.1124 | 21.0525 | 11.2350 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013605 | nf_nhits_type1_trainval | type1_trainval_only | trainval_no_test | 0.0010 | 300 | 12 | step-step=150 | val_as_test | 15.1124 | 21.0525 | 11.2350 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014043 | nf_nhits_ai_plus_type1_classic | ai_ready_plus_type1 | classic | 0.0010 | 300 | 12 | step-step=150 | test | 20.2081 | 33.7321 | 13.1145 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014332 | nf_nhits_ai_plus_type1_trainval_test_as_val | ai_ready_plus_type1 | trainval_test_as_val | 0.0010 | 300 | 12 | step-step=50 | val_as_test | 20.4023 | 33.5683 | 13.5963 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012925 | __ALL__ | ai_ready_processed | val_only_unknown | 0.0010 | 300 | 12 | step-step=100 | val_as_test | 20.5951 | 34.4466 | 13.3342 |
| nhits_lr0.001_ms1000_bs8_ws256_ss12_20260227_011855 | __ALL__ | ai_ready_processed | classic | 0.0010 | 1000 | 12 | step-step=100 | test | 20.6506 | 34.5552 | 13.2925 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012111 | __ALL__ | ai_ready_processed | classic | 0.0010 | 300 | 12 | step-step=100 | test | 20.6506 | 34.5552 | 13.2925 |
| nhits_lr0.0005_ms300_bs8_ws256_ss12_20260227_012441 | __ALL__ | ai_ready_processed | classic | 0.0005 | 300 | 12 | step-step=100 | test | 20.7509 | 34.7694 | 13.2649 |
| nhits_lr0.001_ms200_bs8_ws256_ss12_20260227_012258 | __ALL__ | ai_ready_processed | classic | 0.0010 | 200 | 12 | last | test | 20.8183 | 35.1674 | 13.1163 |
| nhits_lr0.001_ms800_bs8_ws256_ss12_20260227_011501 | __ALL__ | ai_ready_processed | classic | 0.0010 | 800 | 12 | summary | test | 20.9752 | 35.4575 | 13.1247 |
| nhits_lr0.001_ms300_bs8_ws256_ss6_20260227_012620 | __ALL__ | ai_ready_processed | classic | 0.0010 | 300 | 6 | last | test | 20.9796 | 35.5356 | 13.0686 |
| nhits_lr0.001_ms2000_bs8_ws256_ss12_20260227_011113 | __ALL__ | ai_ready_processed | classic | 0.0010 | 2000 | 12 | summary | test | 21.0637 | 35.7546 | 13.0925 |

## Best Runs

- Best overall non-smoke run: `nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800` (group=`nf_nhits_type1_only`, split=`classic`, lr=0.001, max_steps=300, step_size=12, ckpt=`step-step=150`, source=`test`) -> MAE=15.1124, RMSE=21.0525, MARD=11.2350%.
- Runner-up: `nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013605` -> MAE=15.1124, RMSE=21.0525, MARD=11.2350%.

## Best By Split

| split_inferred | run_name | group | dataset_hint | lr | max_steps | step_size | selected_ckpt | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classic | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800 | nf_nhits_type1_only | type1_only_testmirror | 0.0010 | 300 | 12 | step-step=150 | test | 15.1124 | 21.0525 | 11.2350 |
| trainval_no_test | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013605 | nf_nhits_type1_trainval | type1_trainval_only | 0.0010 | 300 | 12 | step-step=150 | val_as_test | 15.1124 | 21.0525 | 11.2350 |
| trainval_test_as_val | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014332 | nf_nhits_ai_plus_type1_trainval_test_as_val | ai_ready_plus_type1 | 0.0010 | 300 | 12 | step-step=50 | val_as_test | 20.4023 | 33.5683 | 13.5963 |
| val_only_unknown | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012925 | __ALL__ | ai_ready_processed | 0.0010 | 300 | 12 | step-step=100 | val_as_test | 20.5951 | 34.4466 | 13.3342 |

## Best By Dataset Hint

| dataset_hint | run_name | group | split_inferred | lr | max_steps | step_size | selected_ckpt | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ai_ready_plus_type1 | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014043 | nf_nhits_ai_plus_type1_classic | classic | 0.0010 | 300 | 12 | step-step=150 | test | 20.2081 | 33.7321 | 13.1145 |
| ai_ready_processed | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012925 | __ALL__ | val_only_unknown | 0.0010 | 300 | 12 | step-step=100 | val_as_test | 20.5951 | 34.4466 | 13.3342 |
| type1_only_testmirror | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800 | nf_nhits_type1_only | classic | 0.0010 | 300 | 12 | step-step=150 | test | 15.1124 | 21.0525 | 11.2350 |
| type1_trainval_only | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013605 | nf_nhits_type1_trainval | trainval_no_test | 0.0010 | 300 | 12 | step-step=150 | val_as_test | 15.1124 | 21.0525 | 11.2350 |

## Exact Per-Group Comparison (Best vs Runner-Up)

| study_group | mae_best | mae_second | delta_mae_second_minus_best | rmse_best | rmse_second | delta_rmse_second_minus_best | mard_best | mard_second | delta_mard_second_minus_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1DM | 15.112361 | 15.112361 | 0.000000 | 21.052547 | 21.052547 | 0.000000 | 11.234989 | 11.234989 | 0.000000 |

## Hyperparameter Trends (Non-Smoke)

### LR

| split_inferred | lr | n_runs | mean_test_mae | best_test_mae |
| --- | --- | --- | --- | --- |
| classic | 0.0010 | 8 | 20.0573 | 15.1124 |
| classic | 0.0005 | 1 | 20.7509 | 20.7509 |
| trainval_no_test | 0.0010 | 1 | 15.1124 | 15.1124 |
| trainval_test_as_val | 0.0010 | 1 | 20.4023 | 20.4023 |
| val_only_unknown | 0.0010 | 1 | 20.5951 | 20.5951 |

### Step Size

| split_inferred | step_size | n_runs | mean_test_mae | best_test_mae |
| --- | --- | --- | --- | --- |
| classic | 12 | 8 | 20.0287 | 15.1124 |
| classic | 6 | 1 | 20.9796 | 20.9796 |
| trainval_no_test | 12 | 1 | 15.1124 | 15.1124 |
| trainval_test_as_val | 12 | 1 | 20.4023 | 20.4023 |
| val_only_unknown | 12 | 1 | 20.5951 | 20.5951 |

## All Runs (Including Smoke)

| run_name | group | dataset_hint | split_inferred | is_smoke | lr | max_steps | step_size | selected_ckpt | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800 | nf_nhits_type1_only | type1_only_testmirror | classic | False | 0.0010 | 300 | 12 | step-step=150 | test | 15.1124 | 21.0525 | 11.2350 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013605 | nf_nhits_type1_trainval | type1_trainval_only | trainval_no_test | False | 0.0010 | 300 | 12 | step-step=150 | val_as_test | 15.1124 | 21.0525 | 11.2350 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014043 | nf_nhits_ai_plus_type1_classic | ai_ready_plus_type1 | classic | False | 0.0010 | 300 | 12 | step-step=150 | test | 20.2081 | 33.7321 | 13.1145 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014332 | nf_nhits_ai_plus_type1_trainval_test_as_val | ai_ready_plus_type1 | trainval_test_as_val | False | 0.0010 | 300 | 12 | step-step=50 | val_as_test | 20.4023 | 33.5683 | 13.5963 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012925 | __ALL__ | ai_ready_processed | val_only_unknown | False | 0.0010 | 300 | 12 | step-step=100 | val_as_test | 20.5951 | 34.4466 | 13.3342 |
| nhits_lr0.001_ms1000_bs8_ws256_ss12_20260227_011855 | __ALL__ | ai_ready_processed | classic | False | 0.0010 | 1000 | 12 | step-step=100 | test | 20.6506 | 34.5552 | 13.2925 |
| nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012111 | __ALL__ | ai_ready_processed | classic | False | 0.0010 | 300 | 12 | step-step=100 | test | 20.6506 | 34.5552 | 13.2925 |
| nhits_lr0.0005_ms300_bs8_ws256_ss12_20260227_012441 | __ALL__ | ai_ready_processed | classic | False | 0.0005 | 300 | 12 | step-step=100 | test | 20.7509 | 34.7694 | 13.2649 |
| nhits_lr0.001_ms200_bs8_ws256_ss12_20260227_012258 | __ALL__ | ai_ready_processed | classic | False | 0.0010 | 200 | 12 | last | test | 20.8183 | 35.1674 | 13.1163 |
| nhits_lr0.001_ms800_bs8_ws256_ss12_20260227_011501 | __ALL__ | ai_ready_processed | classic | False | 0.0010 | 800 | 12 | summary | test | 20.9752 | 35.4575 | 13.1247 |
| nhits_lr0.001_ms300_bs8_ws256_ss6_20260227_012620 | __ALL__ | ai_ready_processed | classic | False | 0.0010 | 300 | 6 | last | test | 20.9796 | 35.5356 | 13.0686 |
| nhits_lr0.001_ms2000_bs8_ws256_ss12_20260227_011113 | __ALL__ | ai_ready_processed | classic | False | 0.0010 | 2000 | 12 | summary | test | 21.0637 | 35.7546 | 13.0925 |
| nhits_lr0.001_ms1_bs8_ws256_ss12_20260227_013458 | _smoke_nhits_type1_trainval | type1_trainval_only | trainval_no_test | True | 0.0010 | 1 | 12 | summary | val_as_test | 365.4057 | 611.3821 | 306.0947 |
| nhits_lr0.001_ms1_bs8_ws256_ss12_20260227_013641 | __ROOT__ | unknown | classic | True | 0.0010 | 1 | 12 | summary | test | 365.4057 | 611.3821 | 306.0947 |

## Validation Notes

| check | computed_run | computed_rmse |
| --- | --- | --- |
| ai_ready classic best rmse in __ALL__ | nhits_lr0.001_ms1000_bs8_ws256_ss12_20260227_011855 | 34.5552 |

## Files

- Registry: `_analysis_registry.csv`
- This report: `RUNS_ANALYSIS.md`