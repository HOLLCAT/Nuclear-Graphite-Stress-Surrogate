# V12 Formal Case-Adaptive Tail Gain

V12 freezes the V10/V11 stress formula and tests one constrained extension:
a bounded case-dependent tail gain predicted only from deployment-available
predictor summaries.

The formal run uses all 149 non-final development cases, similarity-group
isolated nested validation and complete element-level evaluation. It does not
read the 50 final-test case files. If the adaptive gain fails any promotion
gate, the selected deployment artifact retains V11's global gain of `0.90`.

The oracle gain is a stress-derived supervised development target and a
non-deployable upper-bound diagnostic. The deployed gain formula does not use
stress, stress P95, stress P99, or any other target summary.
