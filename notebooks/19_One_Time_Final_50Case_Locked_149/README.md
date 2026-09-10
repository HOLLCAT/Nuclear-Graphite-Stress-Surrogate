# Locked 149-Case Model: One-Time 50-Case Final Evaluation

This package is the only final-test implementation for the integrated
149-development-case result. Do not run the older notebooks `04` or `14` for
the formal result.

## Frozen roles

- Primary: `mean_calibrated_consensus`.
- Secondary research-only: `full_consensus_tail`, rotations 1/2/3, gain 1.0.
- Final-test results cannot change these roles.

## Metric contract

All 400,360 elements in every one of the 50 final cases are used for all
reported metrics. The fixed 5,000-row sample per case is used only for figures.

Primary reporting thresholds are macro RMSE <= 2.0, macro R2 >= 0.45,
worst-case RMSE <= 3.5, mean P95 relative error <= 10%, mean P99 relative error
<= 25%, mean P99 underprediction <= 25%, mean Top-1% hotspot overlap >= 45%,
mean Top-1% recall inside predicted Top-5% >= 80%, and maximum absolute
prediction ratio <= 1.5. These are research-reporting thresholds, not nuclear
safety limits.

## Thesis figures

The completed run creates a `figures_for_thesis` directory containing:

1. overall metric dashboard with locked thresholds;
2. case RMSE and R2 rankings;
3. P95, P99 and maximum-stress calibration;
4. tail error and hotspot overlap by case;
5. prediction and residual density plots;
6. development-to-final generalisation comparison;
7. best, median and worst spatial case studies.

`figure_manifest_for_thesis.csv` states the data basis and suggested thesis use
for every image.

## One-time rule

Static validation does not read final element values. The formal job requires
the explicit environment authorisation `CT3_AUTHORISE_LOCKED_FINAL_TEST=YES`.
An interrupted exact-signature run may resume case caches; once completion is
recorded, the evaluator and Slurm job both refuse a second run.
