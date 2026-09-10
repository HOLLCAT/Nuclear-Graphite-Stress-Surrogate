# Nuclear Graphite Project

This directory is a curated copy of the research code, FEM data and completed outputs associated with the thesis. The original experiment directories have not been moved or overwritten. Curation date: 2026-09-09.

The completed task is surrogate prediction of FEM maximum principal stress fields, not graphite-brick lifetime prediction. Deployment requires the post-deformation coordinates and physical predictor fields. The model does not independently replace the entire FEM workflow when these inputs are unavailable.

## Start Here

1. Read the [research file map](docs/RESEARCH_FILE_MAP.md) to distinguish the deployed method, required prerequisites and non-deployed comparison studies.
2. Read the [path and execution guide](docs/RUN_AND_PATHS.md). Quote paths because the project directory name contains spaces.
3. Review the [delivery validation](provenance/delivery_validation.json), [source manifest](provenance/source_file_manifest.csv) and [English-edition audit](provenance/english_edition_validation.json).
4. Inspect the [frozen final metrics](outputs/19_one_time_final_50case_locked_149/formal_final_50case/final_50case_scope_metrics.csv) and [thesis result figures](outputs/19_one_time_final_50case_locked_149/formal_final_50case/figures_for_thesis).
5. Use the [GitHub upload guide](docs/GITHUB_UPLOAD.md) for the private `HOLLCAT/Nuclear-Graphite-Stress-Surrogate` repository. Do not stage large files until Git LFS is installed.

With the scientific Python dependencies installed, run these commands from this directory:

```bash
python scripts/inspect_results.py
python scripts/inspect_results.py --formula
python scripts/validate_delivery.py --full-hashes --smoke
```

The validator checks the archive and reproduces frozen predictions on all 400,360 elements of development case_01. It does not train a model, repeat the final 50-case evaluation or overwrite experimental outputs.

## Data and Method

- 199 original FEM TXT files: identifiers 0 through 200, excluding 25 and 30.
- 400,360 elements per case; 79,671,640 element records in total.
- 156 similarity groups: 128 singleton groups, with a maximum group size of six cases.
- 149 development cases and 50 fixed final-test cases.
- Four frozen development rotations, each with 119 training, 15 validation and 15 internal-test cases. The case-assignment seed is 42; groups remain intact within each rotation.
- Seven locked element-level features: fluence rate, temperature, weight-loss rate, rho, sin(theta), cos(theta) and z.
- Conventional baselines: Ridge, HistGradientBoosting and ExtraTrees, implemented with scikit-learn.
- Symbolic development: a frozen common base, staged signed-residual PySR searches, mean calibration, local-tail expressions and validation-based gain selection, followed by development-set integration.

The frozen primary model is `mean_calibrated_consensus`. The predeclared secondary research model is `full_consensus_tail`; it must not be relabelled as the primary model because of its final-test performance.

The baseline and feature-ablation fits use complete training cases. Later symbolic searches use recorded discovery samples, including the 5,000-element-per-training-case design, rather than evaluating every candidate on all training elements. Complete-case evaluation and sampled discovery are different operations. Four rotations are separate development comparisons, not four cumulative training passes of a single model.

## Retained Research Material

The archive includes pre-split statistics and grouping evidence; development-only ablation and baselines; hierarchical feasibility studies; reused mean, scale and spatial-shape artifacts; formal four-rotation symbolic development; 149-case integration; the completed final 50-case evaluation; and full-element exports for cases 80, 178 and 31.

Non-deployed gain and residual studies are retained where they document model-selection decisions or are required by the recorded dependency chain. Worker logs, segment records, checkpoints, candidate frontiers, selected formulas, executed notebooks and required caches remain associated with their generating stages.

External papers, meeting notes, thesis prose, AI illustrations, manual presentation documents, untraceable secondary conversions, duplicate archives and incomplete obsolete training outputs are not treated as research evidence in this directory.

## English Edition and Provenance

The runnable code, notebook explanations, README and handover guides are in English. Raw data, result tables, formulas, saved scientific outputs and historical logs are preserved unchanged. Those archival records may still contain Chinese text or Chinese directory names from the original machine.

The copied experimental results retain their original bytes, including historical absolute paths. Runtime path adaptation applies to the executable edition, not to frozen evidence. Versions of adapted source files are stored in `provenance/original_code/`. Three archival notebooks later received Markdown translations and are no longer byte-identical originals. The [annotation review](provenance/notebook_annotation_edits.json) records their original and current hashes and verifies unchanged code, saved outputs, metadata and cell order. Their directory names are retained for compatibility, not as a claim that translated prose is original.

The executable edition is not byte-identical to every original source file. Changes include portable root discovery, Slurm path defaults, read-only validation, protection against rewriting an existing rotation registration, extraction of reused helpers and English explanations. These are documented in the path guide. The formal prediction formulas, scientific calculations, model settings and frozen datasets have not been retuned.

This README, `docs/` and the delivery audit utilities are curation material, not new experimental results. `formula_appendix/` converts frozen expressions into thesis typesetting and includes numerical transcription checks; its introductory template is explanatory prose.

## Environment

Create a new environment on the target machine rather than copying another machine's `.venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name notebookct3 --display-name "NotebookCT3"
```

A first PySR import may initialise Julia. Inspecting archived results or validating frozen predictions does not require a new PySR search.

## Known Limits

- `requirements.txt` records dependency ranges, not a complete historical environment lock. The validation report describes the current checking environment, not the original CSF3 environment.
- Of 57 compact artifacts in the original final-test signature, 56 match the available files. The exception is the first-rotation completion registration, whose time/path metadata changed during historical re-registration. Predictive artifacts remain checked. The original signature has not been rewritten to hide this discrepancy.
- A historical `run_status.json` may be stale. Use `final_evaluation_complete.json` together with the complete case metrics as completion evidence.
- The final test has already been opened and completed. Do not delete completion markers or submit it again as an untouched test set.
- This curation checks paths, imports, hashes and a complete development-case prediction. It does not rerun all training, long PySR searches or remote Slurm jobs.
- Confirm FEM data-release permission with the data owner before publishing. This is a local archive, not a GitHub upload. Plan separate storage for raw data and large caches before committing the package.

## Before Uploading to GitHub

The complete local archive is approximately 8.70 GB. It is not prepared for a single browser upload: 220 archived files exceed 25 MiB. No archived file exceeds 100 MiB. Git LFS tracking rules are now configured for original FEM TXT files and all NPY, NPZ, PKL and GZ files. Install the Git LFS client before the first `git add`; attributes alone do not install the client. Small tables, formulas, notebooks and figures remain directly readable in Git.

The `.gitattributes` rules disable newline conversion for original FEM data, frozen outputs and provenance records. Preserve these rules when configuring Git LFS, because hashes refer to the original bytes, including CRLF line endings. Use Git for transfer rather than manually copying file contents. After cloning and retrieving any LFS objects or separately stored data, run the delivery validator again.

The runnable edition is English. A full-archive upload also includes unmodified historical Chinese text, local usernames, cluster account identifiers and machine paths in provenance or logs. These are not runtime dependencies, but they will be visible to people with repository access. Do not rewrite frozen evidence to remove them; decide whether the archive should remain private or whether a separately documented public edition is needed.

The owner has confirmed permission to upload the complete archive to a private repository. This is not permission to make the data public or to grant downstream reuse rights. No licence has been invented or assigned. Confirm new permissions before changing repository visibility or publishing a licence.

The publication audit checks text and recognised credential patterns, not arbitrary binary payloads. Saved pickle and checkpoint files must be treated as trusted research artifacts, not loaded from unknown replacements. A passing technical audit is not confirmation of publication permission.

To regenerate the publication audit without training or uploading:

```bash
python scripts/audit_publication.py --report provenance/github_upload_audit.json
```

This scan reads the full text archive, including original FEM files and compressed CSVs, so it can take several minutes. Review `status` and the reported publication decisions, not only the command's exit code.
