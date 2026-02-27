# GluMind Run Analysis (type1_only)

- Scope: **test-only analysis**.
- Rule: if a run has no `test`, use `val` as test-equivalent.
- Total parent runs analyzed: **2**

## Runs (Global Mode, Test-Only)

| run_name | split_scheme | lr | effective_test_source | effective_test_mae | effective_test_rmse | effective_test_mard |
| --- | --- | --- | --- | --- | --- | --- |
| glumind_global_h12_20260225_120905 | classic | 0.0007 | test | 14.5090 | 23.0004 | 10.9902 |
| glumind_global_h12_20260225_115118 | classic | 0.0010 | test | 14.7174 | 23.1519 | 11.1783 |

## Best Run

- Best run: `glumind_global_h12_20260225_120905` | split=`classic` | lr=0.0007 | source=test | MAE=14.5090, RMSE=23.0004, MARD=10.9902%.

## Exact Per-Group Comparison (Best vs Other Runs)

### `glumind_global_h12_20260225_120905` (best) vs `glumind_global_h12_20260225_115118`

| study_group | mae_best | mae_other | delta_mae_other_minus_best | rmse_best | rmse_other | delta_rmse_other_minus_best | mard_best | mard_other | delta_mard_other_minus_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1DM | 14.509039 | 14.717361 | 0.208323 | 23.000402 | 23.151949 | 0.151546 | 10.990230 | 11.178273 | 0.188044 |

## Notes

- This folder currently contains only global runs; no continual runs were found to compare LwF settings.
- `lwf_lambda` appears in metadata but is not used in global training.

## Files

- Registry: `_analysis_registry.csv`
- This report: `RUNS_ANALYSIS.md`