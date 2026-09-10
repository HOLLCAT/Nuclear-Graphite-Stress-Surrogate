# Paths and Execution

## Path Rules

The project root is discovered using the `src/ct3_common.py` marker, not a parent directory named `NotebookCT3`. Raw data defaults to `FE_Results_Cases_All/` inside that root. The shared path configuration supports `CT3_CASE_DIR` for an alternative data directory; inspect individual historical notebooks before assuming every entry uses that override.

Notebook paths are resolved relative to the project root. Slurm scripts use `CT3_PACKAGE_ROOT`, falling back to the submission directory. Submit from the project root, not from `cluster/`. Quote all paths containing spaces.

Historical JSON, CSV, logs and executed notebooks deliberately retain their original `/Users/...` and `/net/scratch/...` paths. The checked frozen-state loaders rebuild runtime locations from the current root. Old paths copied manually from a historical log must be replaced with the current location.

## Inspect and Validate Completed Results

From the project root, with scientific Python dependencies available:

```bash
python scripts/inspect_results.py
python scripts/inspect_results.py --formula
python scripts/validate_delivery.py --full-hashes --smoke --report provenance/latest_delivery_validation.json
python scripts/validate_presplit_reproduction.py
```

Delivery validation checks raw-file row counts and hashes, frozen groups and splits, Python/notebook/Slurm syntax, imports and root discovery. With `--smoke`, it also reproduces frozen predictions for all elements of development case_01. It does not write to `outputs/`.

The pre-split reproduction script executes the extracted preparation cells in a temporary output directory and compares the resulting group map and four-rotation manifest with the archived versions. Neither command retrains the final model or reopens the final test.

## Local Notebooks

```bash
source .venv/bin/activate
python -m jupyter lab
```

Select the `NotebookCT3` kernel. Runnable notebook copies have cleared outputs. Originally saved outputs remain in `provenance/original_code/notebooks/` or stage-specific `executed_notebooks/` directories. A blank entry notebook does not mean its experiment was never completed.

Do not use Run All on 00, 01 or early preparation notebooks in this evidence archive: historical writers can overwrite stage outputs. Use a separate working copy for new experiments and review each stage's existing-output and resume checks. The final test has already been evaluated; deleting completion markers does not make it an unopened test set.

## CSF3 Transfer and Environment

The following is a transfer template, not an instruction to publish data. Replace `YOUR_USER` and the local path. Run it in a local terminal:

```bash
rsync -avhP --exclude='.venv/' --exclude='.cache/' \
  "/absolute/path/Nuclear Graphite Project/" \
  YOUR_USER@csf3.itservices.manchester.ac.uk:/scratch/YOUR_USER/NuclearGraphiteProject/
```

In the remote terminal:

```bash
cd /scratch/YOUR_USER/NuclearGraphiteProject
module load apps/binapps/conda/miniforge3/25.9.1
bash cluster/setup_environment.sh
source .venv/bin/activate
export CT3_PACKAGE_ROOT="$PWD"
export CT3_CASE_DIR="$PWD/FE_Results_Cases_All"
python scripts/validate_delivery.py
```

The module name, partition and resource settings are retained historical configuration. Their current availability has not been checked on CSF3 during curation.

Long training runs belong in Slurm allocations, not on login nodes. The archived four rotations and final evaluation are already complete; migration does not require resubmitting them. For a deliberate experiment in a separate working copy, the historical rotation entries are:

```bash
sbatch --export=ALL cluster/run_formal_v11_iteration2.slurm
sbatch --export=ALL cluster/run_formal_v11_iteration3.slurm
sbatch --export=ALL cluster/run_formal_v11_iteration4.slurm
```

Keep the job ID returned by Slurm. Disappearance from `squeue` is not proof of failure: use `sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode`, then inspect the matching stdout, stderr and stage completion marker. Available original scheduler logs are in `logs/csf3/`; worker logs remain in stage outputs. Not every historical submission has a complete retained scheduler log.

## Changes from the Original Code

The runnable edition contains curation changes; it is not an assertion that every original source file is byte-identical.

- `ct3_common.resolve_package_root`: locate the root by its source marker rather than directory names.
- Slurm entry points: use configurable package/data locations rather than a fixed user scratch path.
- Notebook setup cells and the integration-notebook builder: use the relocated root and data paths.
- First-rotation registration: verify an existing completed registration and its artifacts before returning it, instead of rewriting time/path metadata on each validation.
- `validate_locked_149_final_test.py`: delegate to read-only delivery validation, avoiding writes to completed final-test outputs.
- `robust_chunked_symbolic.py`: retain only the three reused helper functions, with their function bodies unchanged.
- Chapter 3 and formula-appendix builders: retain data-figure and formula-transcription work; remove unrelated conceptual plotting and external Word-image extraction.
- Two notebook explanations, three handover documents and a historical CSF3 runbook: translate to English. Correct the historical explanatory total to 79,671,640 elements and label the old exploratory feature pool as historical, not the later locked feature set. Rename the translated runbook to `CSF3_RUNBOOK.md` and remove machine-specific command paths.
- Git attributes: disable newline conversion for archived data, results and provenance so CRLF content and recorded hashes survive cross-platform checkouts. Additional LFS rules track raw FEM TXT files and NPY, NPZ, PKL and GZ artifacts without changing their working-tree paths or scientific content.
- Annotation audit: record later Markdown translations in three archived notebooks and a JSON-only reserialization of one runnable notebook. Preserve original source hashes while separately checking the reviewed copies; no notebook was rewritten by this reconciliation.

The English translation does not change executable notebook statements. The historical sensitivity notebook's obsolete data-location comment is corrected without changing its Python syntax tree. Scientific function bodies, formal search settings, formulas, weights and metric calculations are not retuned by this language pass.

## Known Frozen-Signature Difference

The original final-test signature disagrees with the available source file for `rotation_1_formal_completion`; the other 56 compact artifacts match. Historical re-registration rewrote the first-rotation completion record's time and absolute-path metadata. The relocated registration function now checks valid existing records instead of rewriting them.

The original signature is preserved. Delivery validation uses the existing three-case export's post-evaluation preflight in temporary storage, reports the metadata exception separately and checks the split, gain and completion state. Changes to predictive formulas, scaling or model configuration still fail validation.

The original final-test validator wrote preflight files before failing on this historical mismatch. Its relocated wrapper is read-only. The evaluator source itself remains unchanged to preserve its source signature. Do not rerun the completed final-test notebook; use `inspect_results.py` to inspect the archived result.

## English Edition and Reproducibility Limits

Runnable explanations and handover documentation are English. Archival material may retain Chinese prose or machine-path names. Three archived notebooks were subsequently translated in place by the owner; they are explicitly recorded as annotated copies, not byte-identical originals, in `provenance/notebook_annotation_edits.json`. The review compared them against the verified original sources or original extraction procedure and confirmed that code, outputs and metadata did not change. All other source snapshots continue to require their original byte hashes.

The checks establish copy integrity, tested runtime paths, imports, frozen-artifact consistency subject to the documented metadata exception, exact pre-split reproduction and a full development-case numerical check. They do not reproduce every training run. Seed 42 does not guarantee bitwise-identical parallel PySR searches across machines. Historical checkpoints also depend on PySR/Julia versions.

Current validation versions are in `provenance/delivery_validation.json`, not a historical training lock file. For a new run, record Python package versions, Julia/PySR versions, CPU/thread settings, Slurm resources and loaded modules in that run's own directory.

Original FEM fields and applicability assumptions are preserved. Provisional numerical boundary flags do not establish full physical validity. Confirm units and obtain the data owner's permission before public data release.
