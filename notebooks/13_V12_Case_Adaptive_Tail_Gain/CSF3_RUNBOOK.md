# Case-Adaptive Tail-Gain CSF3 Runbook

This is a retained non-deployed comparison study, not a required new training step. Its outputs are already archived. Use a separate working copy for any deliberate rerun; do not overwrite the research evidence.

## 1. Transfer the Working Copy

Run in the local terminal. Replace the local path and `YOUR_USER`. For an initial transfer, include the required data, frozen artifacts and earlier-stage dependencies as described in the root README.

```bash
rsync -avhP --exclude='.venv/' --exclude='.cache/' \
  "/absolute/path/Nuclear Graphite Project/" \
  YOUR_USER@csf3.itservices.manchester.ac.uk:/scratch/YOUR_USER/NuclearGraphiteProject/
```

## 2. Validate Remotely

After setting up the remote environment, run in the CSF3 terminal:

```bash
cd /scratch/YOUR_USER/NuclearGraphiteProject
source .venv/bin/activate
export CT3_PACKAGE_ROOT="$PWD"
export CT3_CASE_DIR="$PWD/FE_Results_Cases_All"
python scripts/validate_v12_case_adaptive_tail_gain.py
```

Expected success text: `Fast V12 package validation passed`. This checks structure, the manifest, frozen prerequisites, candidate configuration and group isolation; it does not execute the complete 149-case analysis.

## 3. Submit a Deliberate Rerun

The stage may reject incompatible existing outputs. Preserve completed evidence and review resume/output checks before rerunning in a working copy.

```bash
JOBID=$(sbatch --parsable --export=ALL cluster/run_v12_case_adaptive_tail_gain.slurm)
echo "$JOBID"
```

Record the returned job ID. The retained script requests 8 CPUs, 32 GB and a 12-hour limit. These are historical settings, not a current availability guarantee.

## 4. Monitor

```bash
squeue -j "$JOBID" -o "%.18i %.10P %.22j %.2t %.10M %.10l %R"
tail -f "slurm-ct3-v12-gain-${JOBID}.out"
```

`PD` means pending; `R` means running. Logs may not exist until the job starts. In another terminal, inspect stderr:

```bash
tail -f "slurm-ct3-v12-gain-${JOBID}.err"
```

Ctrl+C exits log viewing without cancelling the job.

## 5. Inspect Completion

```bash
sacct -j "$JOBID" --format=JobID,JobName,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
ROOT="outputs/13_v12_case_adaptive_tail_gain/iteration_1"
cat "$ROOT/v12_complete.json"
cat "$ROOT/v12_promotion_decision.json"
column -s, -t < "$ROOT/v12_promotion_gates.csv"
column -s, -t < "$ROOT/v12_comparison_scope_metrics.csv"
cat "$ROOT/selected_deployment_formula.txt"
```

Check scheduler state `COMPLETED`, exit code `0:0` and the stage's completion record. The status `promote_v12_adaptive_gain` means this study's promotion gates passed. The status `retain_v11_global_gain_0_90` means the study completed but retained the parent model's global gain. Neither status changes the later frozen primary-model role by itself.

## 6. Download

Run in the local terminal, downloading into the appropriate working copy:

```bash
rsync -avhP \
  YOUR_USER@csf3.itservices.manchester.ac.uk:/scratch/YOUR_USER/NuclearGraphiteProject/outputs/13_v12_case_adaptive_tail_gain/ \
  "/absolute/path/Nuclear Graphite Project/outputs/13_v12_case_adaptive_tail_gain/"
```

Retain the completion record, promotion decision and gates, scope metrics, nested out-of-fold gain results, formulas, diagnostic figures and executed notebook. Do not merge outputs from distinct runs without preserving their provenance.
