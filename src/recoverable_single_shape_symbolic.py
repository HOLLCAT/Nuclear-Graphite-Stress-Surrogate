"""Recoverable 16,000-step single-shape symbolic regression for CT3.

This module preserves the original Iteration-1 mathematical design:

    stress = case_mean + exp(case_log_scale) * single_shape_formula

The only material change is execution control.  The original 2,000 PySR
iterations with eight populations are run as 20 sequential warm-start
segments of 100 iterations.  Every segment runs in an isolated process,
writes a checkpoint, and is monitored for loss of activity.  A stalled
worker can therefore be terminated and resumed without discarding completed
segments or splitting the final shape into several formula components.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from ct3_common import (  # noqa: E402
    RANDOM_SEED,
    aggregate_case_metrics,
    assert_no_final_cases,
    save_json,
)
from hierarchical_feasibility import (  # noqa: E402
    FeasibilityConfig,
    MODEL_FEATURES,
    assemble_training_sample,
    load_inputs,
)
from hierarchical_symbolic_regression import (  # noqa: E402
    CASE_FORMULA_FEATURES,
    SHAPE_FORMULA_FEATURES,
    _add_composite_selection_score,
    _composite_formula_record,
    _evaluate_shape_candidates,
    _load_scaling,
    _reference_metrics,
    _save_composite_text,
    _save_plots,
    _save_scaling,
    _scaling_payload,
    _shortlist_shape_frontier,
    _tail_adaptation,
)
from robust_chunked_symbolic import frontier_from_hall_of_fame  # noqa: E402


@dataclass(frozen=True)
class RecoverableSingleShapeConfig:
    """Settings for the minimally changed Iteration-1 replacement."""

    iteration: int = 1
    output_subdir: str = "iteration_1"
    rows_per_training_case: int = 5_000

    # Original scientific search budget: 2,000 * 8 = 16,000 population
    # iterations.  Segmentation changes recovery behaviour, not the formula.
    total_niterations: int = 2_000
    populations: int = 8
    segment_niterations: int = 100
    population_size: int = 40
    ncycles_per_iteration: int = 100
    batch_size: int = 50_000
    maxsize: int = 28
    maxdepth: int = 10
    julia_threads: int = 8

    # A successful segment normally took about one hour in the previous run.
    # These limits allow slow segments while still detecting a genuine stall.
    no_activity_timeout_seconds: int = 45 * 60
    segment_wall_timeout_seconds: int = 3 * 60 * 60
    watchdog_poll_seconds: int = 60
    max_attempts_per_segment: int = 3

    max_shape_candidates_for_full_validation: int = 14
    include_old_partial_frontier: bool = True
    force_rebuild_training_cache: bool = False

    def validate(self) -> None:
        if self.iteration != 1:
            raise ValueError("This recoverable notebook is locked to Iteration 1")
        if self.rows_per_training_case != 5_000:
            raise ValueError("The locked discovery design uses 5,000 rows per case")
        if self.total_niterations % self.segment_niterations != 0:
            raise ValueError("total_niterations must be divisible by segment_niterations")
        if self.total_niterations != 2_000 or self.populations != 8:
            raise ValueError("The reviewed search target is fixed at 2,000 x 8 = 16,000")
        if self.batch_size != 50_000:
            raise ValueError("The original shape-search batch size is fixed at 50,000")
        if self.maxsize != 28 or self.maxdepth != 10:
            raise ValueError("The original shape complexity limits must remain unchanged")
        if self.segment_wall_timeout_seconds <= self.no_activity_timeout_seconds:
            raise ValueError("The wall timeout must exceed the no-activity timeout")
        if self.max_attempts_per_segment < 1:
            raise ValueError("At least one attempt per segment is required")

    @property
    def n_segments(self) -> int:
        return self.total_niterations // self.segment_niterations

    @property
    def target_population_iterations(self) -> int:
        return self.total_niterations * self.populations

    @property
    def population_iterations_per_segment(self) -> int:
        return self.segment_niterations * self.populations


def package_output_dir(
    package_root: Path,
    config: RecoverableSingleShapeConfig,
) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "08_recoverable_16000_single_formula"
        / config.output_subdir
    )


def old_iteration_dir(package_root: Path) -> Path:
    return (
        Path(package_root)
        / "outputs"
        / "06_hierarchical_symbolic_regression"
        / "iteration_1"
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _load_reused_case_formulas(package_root: Path) -> dict:
    source = old_iteration_dir(package_root)
    paths = {
        "mean_row": source / "selected_case_mean_formula.csv",
        "mean_scaling": source / "case_mean_scaling.csv",
        "scale_row": source / "selected_case_log_scale_formula.csv",
        "scale_scaling": source / "case_log_scale_scaling.csv",
        "shape_scaling": source / "shape_scaling.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The completed original Iteration-1 case-level outputs are required: "
            f"{missing}"
        )
    return {
        "mean_row": pd.read_csv(paths["mean_row"]).iloc[0],
        "mean_scaling": _load_scaling(paths["mean_scaling"]),
        "scale_row": pd.read_csv(paths["scale_row"]).iloc[0],
        "scale_scaling": _load_scaling(paths["scale_scaling"]),
        "shape_scaling": _load_scaling(paths["shape_scaling"]),
        "source_paths": paths,
    }


def preflight_recoverable_iteration(
    package_root: Path,
    config: RecoverableSingleShapeConfig,
) -> dict:
    """Check split isolation and prerequisites without reading final-test elements."""

    config.validate()
    package_root = Path(package_root).resolve()
    output_dir = package_output_dir(package_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    feasibility_config = FeasibilityConfig(
        iteration=config.iteration,
        rows_per_case=config.rows_per_training_case,
        output_subdir="iteration_1_pilot",
    )
    inputs = load_inputs(package_root, feasibility_config)
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )
    reused = _load_reused_case_formulas(package_root)
    worker_path = package_root / "scripts" / "run_recoverable_shape_segment.py"
    checks = [
        {"check": "training_cases", "value": len(inputs["train_ids"]), "expected": 119},
        {"check": "validation_cases", "value": len(inputs["validation_ids"]), "expected": 15},
        {"check": "internal_test_cases", "value": len(inputs["internal_ids"]), "expected": 15},
        {"check": "sealed_final_cases", "value": len(inputs["final_ids"]), "expected": 50},
        {"check": "shape_features", "value": len(SHAPE_FORMULA_FEATURES), "expected": 27},
        {"check": "worker_script_exists", "value": worker_path.exists(), "expected": True},
        {
            "check": "target_population_iterations",
            "value": config.target_population_iterations,
            "expected": 16_000,
        },
        {"check": "segments", "value": config.n_segments, "expected": 20},
    ]
    table = pd.DataFrame(checks)
    table["pass"] = table["value"] == table["expected"]
    table.to_csv(output_dir / "preflight_checks.csv", index=False)
    if not table["pass"].all():
        raise RuntimeError("Recoverable Iteration-1 preflight failed")

    return {
        "package_root": package_root,
        "output_dir": output_dir,
        "inputs": inputs,
        "feasibility_config": feasibility_config,
        "reused": reused,
        "worker_path": worker_path,
        "checks": table,
    }


def prepare_training_cache(preflight: dict, config: RecoverableSingleShapeConfig) -> dict:
    """Build the same 595,000-row, 27-feature discovery arrays once."""

    cache_dir = preflight["output_dir"] / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "X": cache_dir / "shape_features.npy",
        "y": cache_dir / "shape_target.npy",
        "weights": cache_dir / "discovery_weights.npy",
        "scaling": cache_dir / "shape_scaling.csv",
        "metadata": cache_dir / "cache_metadata.json",
    }
    expected_rows = len(preflight["inputs"]["train_ids"]) * config.rows_per_training_case
    can_reuse = (
        all(path.exists() for path in paths.values())
        and not config.force_rebuild_training_cache
    )
    if can_reuse:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("n_rows") == expected_rows
            and metadata.get("features") == list(SHAPE_FORMULA_FEATURES)
            and metadata.get("train_ids") == preflight["inputs"]["train_ids"]
        ):
            return {"paths": paths, "metadata": metadata, "reused": True}

    sample = assemble_training_sample(
        preflight["inputs"], preflight["feasibility_config"]
    )
    shape_indices = [
        MODEL_FEATURES.index(feature) for feature in SHAPE_FORMULA_FEATURES
    ]
    X = np.ascontiguousarray(sample["X"][:, shape_indices], dtype=np.float32)
    y = np.ascontiguousarray(sample["y_shape"], dtype=np.float32)
    weights = np.ascontiguousarray(sample["weights"], dtype=np.float32)
    if len(X) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(X)}")

    # Recompute and compare scaling so the new cache is independently checked
    # against the original shape-search inputs.
    computed = _scaling_payload(X, y, SHAPE_FORMULA_FEATURES, weights)
    original = preflight["reused"]["shape_scaling"]
    audit = {
        "max_abs_x_mean_difference": float(
            np.max(np.abs(np.asarray(computed["x_mean"]) - np.asarray(original["x_mean"])))
        ),
        "max_abs_x_std_difference": float(
            np.max(np.abs(np.asarray(computed["x_std"]) - np.asarray(original["x_std"])))
        ),
        "abs_y_mean_difference": abs(float(computed["y_mean"]) - float(original["y_mean"])),
        "abs_y_std_difference": abs(float(computed["y_std"]) - float(original["y_std"])),
    }
    # Use the recomputed values for a self-contained new run.  The audit makes
    # any unexpected departure from the original search visible.
    np.save(paths["X"], X)
    np.save(paths["y"], y)
    np.save(paths["weights"], weights)
    _save_scaling(computed, "normalised_spatial_shape", paths["scaling"])
    sample["audit"].to_csv(cache_dir / "training_sample_audit.csv", index=False)
    sample["manifest"].to_csv(
        cache_dir / "training_sample_manifest.csv.gz",
        index=False,
        compression="gzip",
    )
    metadata = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": list(SHAPE_FORMULA_FEATURES),
        "train_ids": preflight["inputs"]["train_ids"],
        "rows_per_case": config.rows_per_training_case,
        "random_seed": RANDOM_SEED,
        "sampling": "same fixed 5,000-row-per-case discovery design as version 06",
        "complete_case_validation": True,
        "scaling_comparison_with_version_06": audit,
    }
    _atomic_json(paths["metadata"], metadata)
    del sample, X, y, weights
    gc.collect()
    return {"paths": paths, "metadata": metadata, "reused": False}


def _start_sleep_inhibitor(output_dir: Path) -> subprocess.Popen | None:
    if platform.system() != "Darwin" or shutil.which("caffeinate") is None:
        return None
    process = subprocess.Popen(
        ["caffeinate", "-dimsu", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _atomic_json(output_dir / "sleep_inhibitor.json", {
        "enabled": True,
        "parent_pid": os.getpid(),
        "caffeinate_pid": process.pid,
    })
    return process


def _stop_sleep_inhibitor(process: subprocess.Popen | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover
            process.terminate()
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
            process.wait(timeout=10)


def _file_activity_signature(paths: Sequence[Path]) -> tuple:
    signature = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)


def _append_attempt_audit(path: Path, record: dict) -> None:
    row = pd.DataFrame([record])
    if path.exists():
        existing = pd.read_csv(path)
        row = pd.concat([existing, row], ignore_index=True, sort=False)
    row.to_csv(path, index=False)


def _search_signature(config: RecoverableSingleShapeConfig, cache: dict) -> dict:
    return {
        "features": list(SHAPE_FORMULA_FEATURES),
        "n_rows": cache["metadata"]["n_rows"],
        "train_ids": cache["metadata"]["train_ids"],
        "total_niterations": config.total_niterations,
        "populations": config.populations,
        "segment_niterations": config.segment_niterations,
        "population_size": config.population_size,
        "ncycles_per_iteration": config.ncycles_per_iteration,
        "batch_size": config.batch_size,
        "maxsize": config.maxsize,
        "maxdepth": config.maxdepth,
        "random_seed": RANDOM_SEED,
        "parallelism": "multithreading",
    }


def _completed_segment_indices(output_dir: Path, n_segments: int) -> list[int]:
    return [
        segment
        for segment in range(1, n_segments + 1)
        if (output_dir / "segments" / f"segment_{segment:02d}" / "complete.json").exists()
    ]


def _write_progress(output_dir: Path, config: RecoverableSingleShapeConfig) -> dict:
    completed = _completed_segment_indices(output_dir, config.n_segments)
    contiguous = 0
    for segment in range(1, config.n_segments + 1):
        if segment in completed:
            contiguous = segment
        else:
            break
    payload = {
        "completed_segments": completed,
        "contiguous_completed_segments": contiguous,
        "total_segments": config.n_segments,
        "planned_population_iterations_completed": (
            contiguous * config.population_iterations_per_segment
        ),
        "target_population_iterations": config.target_population_iterations,
        "planned_progress_fraction": contiguous / config.n_segments,
        "note": (
            "Only successful segments count toward the planned 16,000. Aborted "
            "attempts consume additional wall time but are rolled back to the "
            "latest stable segment checkpoint before retry."
        ),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_json(output_dir / "search_progress.json", payload)
    return payload


def run_search_segment(
    preflight: dict,
    config: RecoverableSingleShapeConfig,
    cache: dict,
    segment: int,
) -> Path:
    """Run or resume one segment and return its canonical frontier path."""

    output_dir = preflight["output_dir"]
    segment_dir = output_dir / "segments" / f"segment_{segment:02d}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    completion_path = segment_dir / "complete.json"
    canonical_frontier = segment_dir / "frontier.csv"
    if completion_path.exists() and canonical_frontier.exists():
        print(f"Segment {segment}/{config.n_segments}: completed result reused", flush=True)
        return canonical_frontier

    run_id = "recoverable_i1_single_shape"
    run_directory = output_dir / "search_state" / "pysr_runs" / run_id
    checkpoint_path = run_directory / "checkpoint.pkl"
    hall_path = run_directory / "hall_of_fame.csv"
    snapshot_root = output_dir / "search_state" / "stable_snapshots"
    attempt_audit_path = output_dir / "segment_attempt_audit.csv"

    for attempt in range(1, config.max_attempts_per_segment + 1):
        attempt_dir = segment_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = attempt_dir / "worker_stdout.log"
        stderr_path = attempt_dir / "worker_stderr.log"
        worker_status_path = attempt_dir / "worker_status.json"
        resume = checkpoint_path.exists()
        command = [
            sys.executable,
            str(preflight["worker_path"]),
            "--features", str(cache["paths"]["X"]),
            "--target", str(cache["paths"]["y"]),
            "--weights", str(cache["paths"]["weights"]),
            "--scaling", str(cache["paths"]["scaling"]),
            "--feature-names", json.dumps(list(SHAPE_FORMULA_FEATURES)),
            "--state-root", str(output_dir / "search_state"),
            "--segment-output", str(attempt_dir),
            "--run-id", run_id,
            "--segment", str(segment),
            "--attempt", str(attempt),
            "--niterations", str(config.segment_niterations),
            "--populations", str(config.populations),
            "--population-size", str(config.population_size),
            "--ncycles", str(config.ncycles_per_iteration),
            "--batch-size", str(config.batch_size),
            "--maxsize", str(config.maxsize),
            "--maxdepth", str(config.maxdepth),
            "--seed", str(RANDOM_SEED),
        ]
        if resume:
            command.append("--resume")

        environment = os.environ.copy()
        environment["JULIA_NUM_THREADS"] = str(config.julia_threads)
        environment["PYTHONPATH"] = os.pathsep.join([
            str(preflight["package_root"] / "src"),
            environment.get("PYTHONPATH", ""),
        ]).rstrip(os.pathsep)
        started = time.time()
        reason = "process_exit"
        timed_out = False
        stalled = False
        print(
            f"Segment {segment}/{config.n_segments}, attempt {attempt}: "
            f"starting ({'checkpoint resume' if resume else 'fresh search'})",
            flush=True,
        )
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=preflight["package_root"],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=(os.name == "posix"),
            )
            activity_paths = [
                stdout_path,
                stderr_path,
                worker_status_path,
                hall_path,
                checkpoint_path,
            ]
            last_signature = _file_activity_signature(activity_paths)
            last_activity = time.time()
            last_report = 0.0
            try:
                while process.poll() is None:
                    now = time.time()
                    elapsed = now - started
                    signature = _file_activity_signature(activity_paths)
                    if signature != last_signature:
                        last_signature = signature
                        last_activity = now
                    inactive = now - last_activity
                    if inactive >= config.no_activity_timeout_seconds:
                        reason = "no_file_activity_timeout"
                        stalled = True
                        _terminate_process_tree(process)
                        break
                    if elapsed >= config.segment_wall_timeout_seconds:
                        reason = "segment_wall_timeout"
                        timed_out = True
                        _terminate_process_tree(process)
                        break
                    if elapsed - last_report >= 300 or last_report == 0:
                        print(
                            f"Segment {segment}/{config.n_segments}, attempt {attempt}: "
                            f"elapsed={elapsed / 60:.1f} min, "
                            f"inactive={inactive / 60:.1f} min, "
                            f"checkpoint={checkpoint_path.exists()}",
                            flush=True,
                        )
                        last_report = elapsed
                    _atomic_json(segment_dir / "watchdog_status.json", {
                        "segment": segment,
                        "attempt": attempt,
                        "pid": process.pid,
                        "elapsed_seconds": elapsed,
                        "inactive_seconds": inactive,
                        "checkpoint_exists": checkpoint_path.exists(),
                        "hall_of_fame_exists": hall_path.exists(),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })
                    time.sleep(config.watchdog_poll_seconds)
            except KeyboardInterrupt:
                _terminate_process_tree(process)
                raise
            return_code = process.poll()

        worker_frontier = attempt_dir / "frontier.csv"
        audit_record = {
            "segment": segment,
            "attempt": attempt,
            "resume_from_checkpoint": resume,
            "return_code": return_code,
            "termination_reason": reason,
            "stalled": stalled,
            "wall_timed_out": timed_out,
            "elapsed_seconds": time.time() - started,
            "checkpoint_exists_after": checkpoint_path.exists(),
            "hall_of_fame_exists_after": hall_path.exists(),
            "worker_frontier_exists": worker_frontier.exists(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _append_attempt_audit(attempt_audit_path, audit_record)

        if return_code == 0 and worker_frontier.exists():
            if not checkpoint_path.exists():
                raise RuntimeError(
                    f"Segment {segment} returned successfully but did not save "
                    f"{checkpoint_path}"
                )
            frontier = pd.read_csv(worker_frontier)
            frontier["completed_segment"] = segment
            frontier["completed_attempt"] = attempt
            frontier.to_csv(canonical_frontier, index=False)
            stable_snapshot = snapshot_root / f"segment_{segment:02d}"
            stable_snapshot.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, stable_snapshot / "checkpoint.pkl")
            if hall_path.exists():
                shutil.copy2(hall_path, stable_snapshot / "hall_of_fame.csv")
            _atomic_json(completion_path, {
                "status": "complete",
                "segment": segment,
                "attempt": attempt,
                "segment_niterations": config.segment_niterations,
                "populations": config.populations,
                "planned_population_iterations": (
                    config.population_iterations_per_segment
                ),
                "resumed_from_checkpoint": resume,
                "elapsed_seconds": time.time() - started,
                "frontier": str(canonical_frontier),
            })
            _write_progress(output_dir, config)
            print(
                f"Segment {segment}/{config.n_segments}: complete; "
                f"planned cumulative progress "
                f"{segment * config.population_iterations_per_segment}/"
                f"{config.target_population_iterations}",
                flush=True,
            )
            return canonical_frontier

        # Do not trust a checkpoint that may have been interrupted while it was
        # being written.  Preserve it for diagnosis, then roll back to the most
        # recent successful segment.  For segment 1 there is no stable state,
        # so a failed attempt is preserved and the next attempt starts fresh.
        if run_directory.exists():
            failed_state = attempt_dir / "interrupted_state"
            failed_state.mkdir(parents=True, exist_ok=True)
            if checkpoint_path.exists():
                shutil.copy2(checkpoint_path, failed_state / "checkpoint.pkl")
            if hall_path.exists():
                shutil.copy2(hall_path, failed_state / "hall_of_fame.csv")
        previous_snapshot = snapshot_root / f"segment_{segment - 1:02d}"
        if segment > 1 and (previous_snapshot / "checkpoint.pkl").exists():
            run_directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous_snapshot / "checkpoint.pkl", checkpoint_path)
            previous_hall = previous_snapshot / "hall_of_fame.csv"
            if previous_hall.exists():
                shutil.copy2(previous_hall, hall_path)
            recovery_action = f"rolled_back_to_stable_segment_{segment - 1:02d}"
        else:
            if run_directory.exists():
                preserved = attempt_dir / "failed_run_directory"
                if preserved.exists():
                    raise RuntimeError(f"Recovery destination already exists: {preserved}")
                shutil.move(str(run_directory), str(preserved))
            recovery_action = "no_stable_checkpoint; next_attempt_starts_fresh"
        audit_record["recovery_action"] = recovery_action
        # Rewrite the audit row with the recovery decision added.  Keeping a
        # second row would falsely imply an additional worker attempt.
        audit_table = pd.read_csv(attempt_audit_path)
        audit_table.loc[
            (audit_table["segment"] == segment)
            & (audit_table["attempt"] == attempt),
            "recovery_action",
        ] = recovery_action
        audit_table.to_csv(attempt_audit_path, index=False)
        print(
            f"Segment {segment}, attempt {attempt} did not finish ({reason}); "
            f"recovery={recovery_action}.",
            flush=True,
        )

    raise RuntimeError(
        f"Segment {segment} failed {config.max_attempts_per_segment} times. "
        "All earlier segments and the current checkpoint remain saved; rerun the "
        "same notebook after reviewing the latest worker log."
    )


def run_recoverable_shape_search(
    preflight: dict,
    config: RecoverableSingleShapeConfig,
    cache: dict,
) -> pd.DataFrame:
    output_dir = preflight["output_dir"]
    signature_path = output_dir / "search_signature.json"
    signature = _search_signature(config, cache)
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError(
                "Existing search state belongs to a different configuration. "
                "Use a new output_subdir rather than mixing experiments."
            )
    else:
        _atomic_json(signature_path, signature)

    latest_frontier = None
    for segment in range(1, config.n_segments + 1):
        latest_frontier = run_search_segment(
            preflight, config, cache, segment
        )
    if latest_frontier is None:
        raise RuntimeError("No search segment was executed or recovered")
    frontier = pd.read_csv(latest_frontier)
    frontier["frontier_source"] = "new_recoverable_16000_search"

    if config.include_old_partial_frontier:
        old_dir = old_iteration_dir(preflight["package_root"])
        old_halls = sorted(
            (old_dir / "pysr_runs").glob("hierarchical_i1_shape_*/hall_of_fame.csv")
        )
        if old_halls:
            old_frontier = frontier_from_hall_of_fame(
                old_halls[-1],
                SHAPE_FORMULA_FEATURES,
                _load_scaling(old_dir / "shape_scaling.csv"),
                "shape",
                old_halls[-1].parent.name,
            )
            old_frontier["frontier_source"] = "version_06_partial_search"
            frontier = pd.concat([frontier, old_frontier], ignore_index=True, sort=False)

    frontier = (
        frontier.sort_values(["loss", "complexity"])
        .drop_duplicates("formula_scaled_sympy", keep="first")
        .reset_index(drop=True)
    )
    frontier["source_candidate_index"] = frontier["candidate_index"]
    frontier["candidate_index"] = np.arange(len(frontier), dtype=int)
    frontier.to_csv(output_dir / "shape_frontier_combined.csv", index=False)
    return frontier


def _select_relaxed_prototype_formula(
    candidate_metrics: pd.DataFrame,
    reference: pd.Series,
    output_dir: Path,
) -> pd.Series:
    """Select a finite prototype by validation fit; engineering gates are diagnostics."""

    valid_mask = candidate_metrics["candidate_valid"].fillna(False)
    if not valid_mask.any():
        raise RuntimeError("No candidate produced finite predictions for every validation case")
    scored = _add_composite_selection_score(candidate_metrics[valid_mask].copy(), reference)
    finite = scored[
        np.isfinite(scored["validation_macro_rmse"])
        & np.isfinite(scored["validation_mean_p99_relative_error"])
    ].copy()
    if finite.empty:
        raise RuntimeError("No candidate has finite validation selection metrics")

    best_rmse = float(finite["validation_macro_rmse"].min())
    tolerance = 0.03 * max(abs(best_rmse), 1e-12)
    competitive = finite[
        finite["validation_macro_rmse"] <= best_rmse + tolerance
    ].copy()
    selected = competitive.sort_values([
        "complexity",
        "validation_mean_p99_relative_error",
        "validation_macro_rmse",
        "candidate_index",
    ]).iloc[0].copy()
    selected["selection_pool_status"] = "prototype_engineering_gates_diagnostic_only"
    selected["selection_method"] = (
        "finite_full_validation_candidates; within_3pct_of_best_macro_rmse; "
        "then_minimum_complexity; p99_relative_error_tiebreak"
    )

    for column in scored.columns:
        if column not in candidate_metrics.columns:
            candidate_metrics[column] = np.nan
    candidate_metrics.loc[scored.index, scored.columns] = scored
    candidate_metrics["selected_candidate"] = candidate_metrics["candidate_index"].eq(
        selected["candidate_index"]
    )
    candidate_metrics["engineering_gates_used_as_hard_filter"] = False
    candidate_metrics["selection_pool_status"] = selected["selection_pool_status"]
    candidate_metrics["selection_method"] = selected["selection_method"]
    candidate_metrics.to_csv(
        output_dir / "shape_candidate_validation_metrics.csv", index=False
    )
    pd.DataFrame([selected]).to_csv(
        output_dir / "selected_shape_formula.csv", index=False
    )
    return selected


def _copy_reused_formula_artifacts(preflight: dict) -> None:
    output_dir = preflight["output_dir"]
    reused = preflight["reused"]
    pd.DataFrame([reused["mean_row"]]).to_csv(
        output_dir / "selected_case_mean_formula.csv", index=False
    )
    pd.DataFrame([reused["scale_row"]]).to_csv(
        output_dir / "selected_case_log_scale_formula.csv", index=False
    )
    shutil.copy2(
        reused["source_paths"]["mean_scaling"], output_dir / "case_mean_scaling.csv"
    )
    shutil.copy2(
        reused["source_paths"]["scale_scaling"], output_dir / "case_log_scale_scaling.csv"
    )


def run_recoverable_iteration_one(
    package_root: Path,
    config: RecoverableSingleShapeConfig,
) -> dict:
    """Run/resume search, relaxed validation selection and one internal test."""

    started = time.time()
    preflight = preflight_recoverable_iteration(package_root, config)
    output_dir = preflight["output_dir"]
    completion_path = output_dir / "round_complete.json"
    if completion_path.exists():
        return json.loads(completion_path.read_text(encoding="utf-8"))

    save_json(output_dir / "run_configuration.json", asdict(config))
    _copy_reused_formula_artifacts(preflight)
    pd.DataFrame([
        {"feature": feature, "role": "single_shape_formula"}
        for feature in SHAPE_FORMULA_FEATURES
    ]).to_csv(output_dir / "feature_registry.csv", index=False)
    _atomic_json(output_dir / "run_status.json", {
        "status": "running",
        "stage": "prepare_training_cache",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_test_cases_read": 0,
    })

    sleep_inhibitor = _start_sleep_inhibitor(output_dir)
    try:
        cache = prepare_training_cache(preflight, config)
        _atomic_json(output_dir / "run_status.json", {
            "status": "running",
            "stage": "recoverable_single_shape_search",
            "elapsed_seconds": time.time() - started,
            "final_test_cases_read": 0,
        })
        frontier = run_recoverable_shape_search(preflight, config, cache)
        shortlist = _shortlist_shape_frontier(
            frontier, config.max_shape_candidates_for_full_validation
        ).copy()
        shortlist.to_csv(
            output_dir / "shape_full_validation_shortlist.csv", index=False
        )

        shape_scaling = _load_scaling(cache["paths"]["scaling"])
        reused = preflight["reused"]
        validation_metrics, validation_cases = _evaluate_shape_candidates(
            candidates=shortlist,
            shape_scaling=shape_scaling,
            mean_row=reused["mean_row"],
            mean_scaling=reused["mean_scaling"],
            scale_row=reused["scale_row"],
            scale_scaling=reused["scale_scaling"],
            inputs=preflight["inputs"],
            case_ids=preflight["inputs"]["validation_ids"],
            split="validation",
            iteration=config.iteration,
            hard_deadline=None,
        )
        reference_validation = _reference_metrics(
            preflight["inputs"], config.iteration, "validation"
        )
        selected = _select_relaxed_prototype_formula(
            validation_metrics, reference_validation, output_dir
        )
        selected_index = int(selected["candidate_index"])
        selected_validation = validation_cases[
            validation_cases["candidate_index"] == selected_index
        ].copy()
        selected_validation.to_csv(
            output_dir / "selected_formula_validation_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )

        selected_table = pd.DataFrame([selected])
        _, internal_cases = _evaluate_shape_candidates(
            candidates=selected_table,
            shape_scaling=shape_scaling,
            mean_row=reused["mean_row"],
            mean_scaling=reused["mean_scaling"],
            scale_row=reused["scale_row"],
            scale_scaling=reused["scale_scaling"],
            inputs=preflight["inputs"],
            case_ids=preflight["inputs"]["internal_ids"],
            split="internal_test",
            iteration=config.iteration,
            hard_deadline=None,
        )
        internal_cases.to_csv(
            output_dir / "selected_formula_internal_test_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        all_cases = pd.concat(
            [selected_validation, internal_cases], ignore_index=True
        )
        all_cases.to_csv(
            output_dir / "selected_formula_case_metrics.csv.gz",
            index=False,
            compression="gzip",
        )
        split_metrics = aggregate_case_metrics(
            all_cases, ["iteration", "split", "model"]
        )
        split_metrics.to_csv(
            output_dir / "selected_formula_split_metrics.csv", index=False
        )
        tail_rows = [
            {"iteration": config.iteration, "split": split, **_tail_adaptation(group)}
            for split, group in all_cases.groupby("split")
        ]
        pd.DataFrame(tail_rows).to_csv(
            output_dir / "selected_formula_tail_adaptation.csv", index=False
        )

        composite = _composite_formula_record(
            config.iteration,
            reused["mean_row"],
            reused["scale_row"],
            selected,
        )
        composite["status"] = (
            "Iteration-1 prototype; engineering gates reported but not used as hard filters"
        )
        pd.DataFrame([composite]).to_csv(
            output_dir / "selected_composite_formula.csv", index=False
        )
        _save_composite_text(composite, output_dir)
        reference_rows = pd.DataFrame([
            reference_validation,
            _reference_metrics(preflight["inputs"], config.iteration, "internal_test"),
        ])
        _save_plots(all_cases, split_metrics, reference_rows, output_dir)

        progress = _write_progress(output_dir, config)
        payload = {
            "status": "complete",
            "iteration": config.iteration,
            "prototype_only": True,
            "formula_shape_components": 1,
            "train_cases": len(preflight["inputs"]["train_ids"]),
            "validation_cases": len(preflight["inputs"]["validation_ids"]),
            "internal_test_cases": len(preflight["inputs"]["internal_ids"]),
            "final_test_cases_read": 0,
            "training_rows": cache["metadata"]["n_rows"],
            "planned_population_iterations": config.target_population_iterations,
            "completed_segments": progress["contiguous_completed_segments"],
            "selected_shape_candidate": selected_index,
            "selection_pool_status": selected["selection_pool_status"],
            "elapsed_seconds": time.time() - started,
            "output_directory": str(output_dir),
        }
        _atomic_json(completion_path, payload)
        _atomic_json(output_dir / "run_status.json", payload)
        return payload
    except Exception as exc:
        _atomic_json(output_dir / "run_status.json", {
            "status": "failed_or_interrupted",
            "stage": "exception",
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
            "recovery_instruction": "Rerun the same notebook; completed segments are reused.",
            "final_test_cases_read": 0,
        })
        raise
    finally:
        _stop_sleep_inhibitor(sleep_inhibitor)
