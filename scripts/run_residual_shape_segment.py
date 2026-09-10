#!/usr/bin/env python3
"""Run one recoverable PySR segment for the V9 residual-shape pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import sympy as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--feature-names", required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--segment-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--niterations", type=int, required=True)
    parser.add_argument("--populations", type=int, required=True)
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--ncycles", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--maxsize", type=int, required=True)
    parser.add_argument("--maxdepth", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.segment_output.mkdir(parents=True, exist_ok=True)
    status_path = args.segment_output / "worker_status.json"
    started = time.time()
    atomic_json(status_path, {
        "status": "loading",
        "segment": args.segment,
        "attempt": args.attempt,
        "pid": os.getpid(),
    })

    package_root = Path(__file__).resolve().parents[1]
    source_dir = package_root / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from pysr import PySRRegressor
    from hierarchical_symbolic_regression import (
        _formula_exports,
        _load_scaling,
        _scale_arrays,
    )

    feature_names = json.loads(args.feature_names)
    X = np.load(args.features, mmap_mode="r")
    y = np.load(args.target, mmap_mode="r")
    weights = np.load(args.weights, mmap_mode="r")
    scaling = _load_scaling(args.scaling)
    if feature_names != scaling["features"]:
        raise ValueError("Feature names and residual scaling file are inconsistent")
    X_scaled, y_scaled = _scale_arrays(X, y, scaling)

    binary_operators = ["+", "-", "*", "/"]
    unary_operators = ["square", "abs", "gauss(x) = exp(-x^2)"]
    constraints = {
        "/": (-1, 8),
        "square": 12,
        "abs": 12,
        "gauss": 12,
    }
    nested_constraints = {
        "square": {"square": 0, "gauss": 1},
        "abs": {"abs": 0},
        "gauss": {"gauss": 0, "square": 1},
    }
    operator_complexity = {"/": 3, "square": 2, "abs": 3, "gauss": 4}
    extra_sympy_mappings = {"gauss": lambda value: sp.exp(-(value ** 2))}
    run_root = args.state_root / "pysr_runs"
    run_directory = run_root / args.run_id
    common = dict(
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        ncycles_per_iteration=args.ncycles,
        maxsize=args.maxsize,
        maxdepth=args.maxdepth,
        warmup_maxsize_by=0.5,
        constraints=constraints,
        nested_constraints=nested_constraints,
        complexity_of_operators=operator_complexity,
        extra_sympy_mappings=extra_sympy_mappings,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        batching=True,
        batch_size=args.batch_size,
        precision=32,
        random_state=args.seed,
        deterministic=False,
        parallelism="multithreading",
        progress=True,
        verbosity=1,
        input_stream="devnull",
        update=False,
    )
    if args.resume:
        checkpoint = run_directory / "checkpoint.pkl"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot warm-start without {checkpoint}")
        model = PySRRegressor.from_file(
            run_directory=run_directory,
            warm_start=True,
            **common,
        )
        mode = "warm_start_checkpoint"
    else:
        if run_directory.exists():
            raise RuntimeError(
                f"Fresh search requested but state already exists at {run_directory}"
            )
        model = PySRRegressor(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            warm_start=False,
            output_directory=str(run_root),
            run_id=args.run_id,
            **common,
        )
        mode = "fresh_search"

    atomic_json(status_path, {
        "status": "searching",
        "segment": args.segment,
        "attempt": args.attempt,
        "pid": os.getpid(),
        "mode": mode,
        "n_rows": int(len(X_scaled)),
        "n_features": int(X_scaled.shape[1]),
        "niterations": args.niterations,
        "populations": args.populations,
        "planned_population_iterations": args.niterations * args.populations,
        "operators": binary_operators + ["square", "abs", "gauss"],
        "julia_threads": os.environ.get("JULIA_NUM_THREADS"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    try:
        model.fit(
            X_scaled,
            y_scaled,
            weights=np.asarray(weights, dtype=np.float32),
            variable_names=[f"{feature}_scaled" for feature in feature_names],
        )
        frontier = _formula_exports(
            model,
            feature_names,
            scaling,
            args.run_id,
            "residual_shape",
        )
        frontier["segment"] = args.segment
        frontier["attempt"] = args.attempt
        frontier["search_mode"] = mode
        frontier.to_csv(args.segment_output / "frontier.csv", index=False)
        atomic_json(status_path, {
            "status": "complete",
            "segment": args.segment,
            "attempt": args.attempt,
            "mode": mode,
            "n_candidates": int(len(frontier)),
            "elapsed_seconds": time.time() - started,
            "checkpoint": str(run_directory / "checkpoint.pkl"),
        })
    except Exception as exc:
        atomic_json(status_path, {
            "status": "failed",
            "segment": args.segment,
            "attempt": args.attempt,
            "mode": mode,
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
        })
        raise


if __name__ == "__main__":
    main()
