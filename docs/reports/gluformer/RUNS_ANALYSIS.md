# GluFormer Run Analysis (ai_ready_plus_type1)

- Scope: **test-only analysis**.
- Source folder: `data/output/runs/gluformer/ai_ready_plus_type1/gluformer_20260227_005453`
- Total runs analyzed: **1**
- Total predictions: **6,435** (`unique_id` count: **560**)

## Runs (Test-Only)

| run_name | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- |
| gluformer_20260227_005453 | test | 19.5332 | 33.2846 | 13.0314 |

## Best Run

- Best run: `gluformer_20260227_005453` | source=test | MAE=19.5332, RMSE=33.2846, MARD=13.0314%.

## Per-Group Test Metrics

| study_group | n_points | share_pct | mae | rmse | mard |
| --- | --- | --- | --- | --- | --- |
| pre_diabetes_lifestyle_controlled | 1356 | 21.07 | 14.212372 | 21.794701 | 12.116766 |
| T1DM | 287 | 4.46 | 15.463229 | 22.857195 | 15.096645 |
| healthy | 1447 | 22.49 | 17.078192 | 26.142618 | 14.639007 |
| oral_medication_and_or_non_insulin_injectable_medication_controlled | 1557 | 24.20 | 19.360189 | 32.966671 | 13.325925 |
| insulin_dependent | 1788 | 27.79 | 26.359039 | 45.206299 | 11.836184 |

## Group Ranking by MAE (Best to Worst)

| rank | study_group | mae |
| --- | --- | --- |
| 1 | pre_diabetes_lifestyle_controlled | 14.212372 |
| 2 | T1DM | 15.463229 |
| 3 | healthy | 17.078192 |
| 4 | oral_medication_and_or_non_insulin_injectable_medication_controlled | 19.360189 |
| 5 | insulin_dependent | 26.359039 |

## Prediction Coverage

- Timestamp range in predictions: `2018-06-27 23:00:00` to `2025-05-11 16:10:00`.

| event_type | n_points |
| --- | --- |
| AI_READY | 6148 |
| EGV | 136 |
| HUPA | 60 |
| Sleep | 59 |
| Activity | 31 |
| Meal | 1 |

## Notes

- This run folder contains only test artifacts (`test_metrics_overall.csv`, `test_metrics_by_study_group.csv`, `test_predictions.csv`).
- No hyperparameter comparison is possible from this folder alone.

## Files

- Registry: `_analysis_registry.csv`
- This report: `RUNS_ANALYSIS.md`
