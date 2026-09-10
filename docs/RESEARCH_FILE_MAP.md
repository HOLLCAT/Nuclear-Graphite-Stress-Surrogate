# Research File Map

## Data and Frozen Splits

| Material | Location | Purpose |
|---|---|---|
| Original FEM data | `FE_Results_Cases_All/` | 199 original TXT files, preserving all elements and physical fields |
| Shared reading and metrics | `src/ct3_common.py` | Column mapping, coordinates, complete-case reading, weights and metric definitions |
| Pre-split 199-case analysis | `notebooks/Multi-Case_FEM_Parametric_and_Position_Sensitivity_Stability_Analysis_199Cases.ipynb`; `outputs/multi_case_sensitivity_199_cases/` | Historical exploratory statistics, not an independent final-test model comparison |
| Grouping code | `notebooks/PreSplit_Grouping_and_Frozen_Manifest.ipynb` | Actual preparation cells extracted from the earlier notebook, without its old symbolic-search loop |
| Original grouping outputs | `outputs/symbolic_regression_199_cases/` | Distances, thresholds, group members, boundary anchors and cross-split checks |
| Frozen assignments | `shared/frozen_case_split_manifest_199cases.csv`; `shared/frozen_similarity_group_map_199cases.csv` | 156 groups, 128 singletons, largest group of six; four rotations with assignment seed 42 |

There are 149 development cases and 50 final cases. Each development rotation uses 119 training, 15 validation and 15 internal-test cases. Similarity groups do not cross splits within a rotation. The four rotations do not represent four cumulative training passes of one model.

## Development and Outputs

| Stage | Notebook or source | Output and role |
|---|---|---|
| Feature selection | `notebooks/00_Data_QC_Sensitivity_and_Feature_Ablation.ipynb` | `outputs/00_qc_sensitivity_ablation/`; seven locked periodic features |
| Conventional baselines | `notebooks/01_Four_Iteration_199Case_Baseline.ipynb` | `outputs/01_baseline_199cases/`; Ridge, HistGradientBoosting and ExtraTrees |
| Hierarchical feasibility | Four `notebooks/02A_*.ipynb` files; `src/hierarchical_feasibility.py` | `outputs/05_hierarchical_feasibility_v5/`; completed feasibility evidence |
| Prerequisite case-context formulas | `src/hierarchical_symbolic_regression.py`; `notebooks/06_Completed_Context_Stage_Artifacts.ipynb` | `outputs/06_hierarchical_symbolic_regression/iteration_1/`; completed mean/scale stages and reused sampling/scaling preparation, excluding the failed shape output |
| Recoverable single-shape search | `notebooks/08_Recoverable_16000_Single_Formula_Symbolic_Regression/` | `outputs/08_recoverable_16000_single_formula/`; completed local result |
| Frozen common base | `shared/v10_authoritative_v8_candidate5/` | The base actually reused downstream, with hashes; not the candidate 11 from a separate CSF3 run |
| Single-residual comparison | `notebooks/09_Residual_Shape_Symbolic_Regression_Pilot/` | `outputs/09_residual_shape_symbolic_pilot/`; completed local experiment used in the thesis comparison, not a mixture with the separate CSF3 result |
| Staged signed residuals | `src/signed_staged_residual_symbolic.py`; `notebooks/10_*/` | `outputs/10_signed_staged_residual_symbolic_pilot/iteration_1..4/` |
| Tail correction and calibration | `src/v11_tail_aware_localised_symbolic.py`; `notebooks/11_*/` | `outputs/11_tail_aware_localised_symbolic/iteration_1..4/` |
| Gain selection and stability | `src/v11_gain_stability_audit.py`; `notebooks/12_*/` | `outputs/12_v11_gain_stability_audit/iteration_1..4/` |
| Formal four-rotation orchestration | `notebooks/17_Formal_V11_Four_Rotation/` | `outputs/17_v11_four_rotation_formal/`; registers the reviewed first rotation and runs the same method for rotations two to four |
| 149-case integration | `notebooks/18_149Case_Development_Integration/` | `outputs/18_149case_development_integration/formal_149case_integration/` |
| One-time final evaluation | `notebooks/19_One_Time_Final_50Case_Locked_149/` | `outputs/19_one_time_final_50case_locked_149/formal_final_50case/`; already completed |
| Three-case spatial export | `src/export_three_spatial_comparison_cases.py` | `outputs/21_three_case_fem_vs_frozen_prediction/full_400360_element_exports/` |

Numeric directory prefixes are historical identifiers, not instructions to retrain every folder sequentially. Symbolic segment searches use separate worker processes through `scripts/run_*segment.py`; logs, segment audits, frontiers, selected formulas and completion records are retained.

## Required Supporting Artifacts

- `outputs/02_symbolic_search_v4_context_interaction/.../discovery_manifest_iteration_1.csv.gz` and its iteration-2 counterpart: reused discovery-sample records only, not the obsolete model results.
- `outputs/10_*/iteration_*/training_cache/`: features, targets, weights, scaling and sample records. Some frozen-state loaders check these files; they are not disposable runtime clutter.
- `outputs/11_*/iteration_*/training_cache/`: mean-calibration models, tail scaling and local features.
- `outputs/19_*/formal_final_50case/case_cache/` and `final_fixed_uniform_plot_sample.csv.gz`: final metric caches and plotting samples. Final metrics use complete cases; plotting samples do not reduce the evaluated test population.
- `shared/v10_authoritative_v8_candidate5/baseline_manifest.json`: common-base provenance and verification.
- `provenance/original_code/`: versions of adapted code and notebooks, including their originally saved outputs. The annotation review explicitly identifies translated copies that are no longer byte-identical originals.
- `provenance/notebook_annotation_edits.json`: original/current hashes and code/output-preservation checks for three translated archival notebooks and one serialization-only change; curation evidence, not a new experiment.

## Non-Deployed Comparisons

`13_V12_Case_Adaptive_Tail_Gain`, `15_V13_Structural_Tail_Residual` and `16_V13_Residual_Gain_Audit`, with their corresponding outputs, document selection decisions. They are not additional terms in the final deployed model. The formal registration reads the residual-study decision, and that dependency chain refers to adaptive-gain comparison evidence.

`src/robust_chunked_symbolic.py` retains only the three unchanged frontier-conversion helpers used by later recoverable searches. The failed old `07` notebook and outputs are excluded.

## Figures and Formulas

- Nine final-evaluation PNG figures are in `outputs/19_one_time_final_50case_locked_149/formal_final_50case/figures_for_thesis/`. Their generator is `src/final_locked_149_evaluation.py`.
- Chapter 3 data figures are `reports/thesis_figures/chapter3/figure_3_2_*` and `figure_3_3_*`. The retained generator `scripts/generate_chapter3_figures.py` produces the data figures, not the conceptual workflow illustration.
- Stage-specific charts remain with their source tables and generating code. Duplicate images copied into thesis folders are not added separately.
- `provenance/figure_source_map.csv` links the 65 retained PNG files to generating code and copied-file hashes. This is provenance evidence, not a fresh rerun of every chart.
- The frozen expanded expression is in `outputs/18_149case_development_integration/formal_149case_integration/locked_model_formula.txt` and `.csv`.
- `formula_appendix/` contains transcription and typesetting, not new fitting. `formula_transcription_checks.json` records numerical transcription checks. External Word-image extraction and its images are excluded.

## Three-Case Exports

Cases 80, 178 and 31 are the best, median and worst cases by the frozen primary model's per-case RMSE, not by R-squared.

- Original professor-supplied data: `FE_Results_Cases_All/FE_Results_Case_80.txt`, and the corresponding files for 178 and 31; these retain physical fields and FEM stress.
- `*_FEM_field.csv.gz`: compact export of the original FEM field.
- `*_frozen_primary_prediction_field.csv.gz`: separate frozen-primary-model predictions.
- `*_FEM_vs_prediction_full.csv.gz`: full-element comparison with inputs, FEM stress, predictions and errors.

Each case contains 400,360 elements, totalling 1,201,080 export locations. The archive retains the gzip files actually written by the export code, not duplicate decompressed CSVs or secondary TXT conversions without a retained generating script.
