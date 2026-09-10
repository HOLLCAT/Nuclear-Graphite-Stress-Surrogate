# Frozen V8 candidate-5 baseline

V10 uses this copied artifact set instead of whichever V8 output happens to
exist on the machine running the job. This prevents the local candidate-5 and
CSF3 candidate-11 searches from silently defining different residual targets.

The baseline was selected for V10 because it has the lower iteration-1
validation macro RMSE (2.3473) and positive macro R2 (0.1804). Its high-stress
limitations remain explicit: P95 relative error is 0.1286 and P99 relative
error is 0.2148. No final-test case was read when selecting it.

`baseline_manifest.json` records SHA-256 hashes. V10 verifies every hash before
building a training cache or reading any FEM case.
