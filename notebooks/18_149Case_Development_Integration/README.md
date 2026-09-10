# 149-Case Development Integration

This stage integrates the four frozen symbolic-regression rotations after their
independent searches and held-out audits have completed.

## Statistical role

- The 149 cases are the complete development pool.
- All 400,360 elements of every development case are used.
- The 50 final-test cases remain sealed. Only their inventory, headers and row
  counts are checked; no predictor or target values are parsed or used for
  refitting or model selection.
- Metrics reported here are fit diagnostics. They are not the final unbiased
  generalisation result.

## Locked integration rule

1. Average the four frozen V10 predictions with fixed weights of 0.25.
2. Refit a six-feature RidgeCV case-mean correction on all 149 cases.
3. Exclude any tail formula that is constant or zero after complete-case
   centring. Rotation 4 is expected to be excluded for this reason.
4. Average the remaining centred symbolic tail corrections.
5. Fit one complete-element weighted gain and clip it to `[0, 1]`.
6. Apply the predeclared engineering gates and freeze one candidate.

The final formula remains symbolic but is stored in componentised form so it
is inspectable and deployable without expanding several large expressions into
one unreadable line.

## Recovery

The job saves one checkpoint per complete FEM case. Resubmission reuses valid
checkpoints. Case caches are removed only after every output and completion
marker has been written successfully.

## Final-test protection

Successful completion writes `final_test_release_gate.json` with
`human_approval_required=true` and `approved=false`. A separate one-time
final-test notebook must be created only after the integration diagnostics have
been reviewed.
