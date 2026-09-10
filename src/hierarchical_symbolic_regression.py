"""Restartable hierarchical symbolic regression for the CT3 FEM surrogate.

The final deployable expression is factorised as

    stress(case, element) = mean(case) + exp(log_scale(case)) * shape(case, element)

All three terms are discovered by PySR.  The locked final-test cases are never
read here.  Formula selection uses validation cases only; the selected
composite is then reported on the internal-test cases without further tuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
import json
import math
import os
import re
import sys
import threading
import time
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from ct3_common import (  # noqa: E402
    RANDOM_SEED,
    TARGET_COL,
    aggregate_case_metrics,
    assert_no_final_cases,
    cases_for_role,
    evaluate_prediction_arrays,
    read_complete_case,
    save_json,
)
from hierarchical_feasibility import (  # noqa: E402
    FeasibilityConfig,
    LOCAL_FEATURES,
    MODEL_FEATURES,
    STANDARDISED_LOCAL_FEATURES,
    _case_summary_row,
    assemble_training_sample,
    build_model_matrix,
    load_inputs,
)


# These case-context variables were consistently useful in the four completed
# RidgeCV feasibility rotations.  The list is fixed before the four symbolic
# searches so every rotation uses the same search space.
CASE_FORMULA_FEATURES = [
    "fluence_rate_mean",
    "fluence_rate_p95",
    "temperature_mean",
    "temperature_p95",
    "weight_loss_rate_mean",
    "weight_loss_rate_std",
    "rho_mean",
    "rho_std",
    "rho_max",
    "z_mean",
    "z_std",
    "z_max",
    "theta_sin_std",
]

# Shape receives local absolute values, their within-case z scores, and the
# fixed case context.  No stress statistic is present in this predictor list.
SHAPE_FORMULA_FEATURES = (
    LOCAL_FEATURES + STANDARDISED_LOCAL_FEATURES + CASE_FORMULA_FEATURES
)

CASE_BINARY_OPERATORS = ["+", "-", "*", "/"]
CASE_UNARY_OPERATORS = ["square", "abs"]
SHAPE_BINARY_OPERATORS = ["+", "-", "*", "/"]
SHAPE_UNARY_OPERATORS = ["square", "cube", "abs"]

OPERATOR_CONSTRAINTS = {
    "/": (-1, 8),
    "square": 12,
    "cube": 10,
    "abs": 12,
}
NESTED_CONSTRAINTS = {
    "square": {"square": 0, "cube": 0},
    "cube": {"square": 0, "cube": 0},
    "abs": {"abs": 0},
}
OPERATOR_COMPLEXITY = {"/": 3, "square": 2, "cube": 3, "abs": 3}


@dataclass(frozen=True)
class HierarchicalSymbolicConfig:
    """Configuration for one frozen outer rotation.

    The search window is limited to 22 hours, leaving two hours inside the
    requested 24-hour notebook budget for complete-case candidate evaluation,
    exports and small PySR timeout overruns.
    """

    iteration: int
    output_subdir: str
    rows_per_training_case: int = 5_000
    total_wall_seconds: int = 24 * 60 * 60
    evaluation_reserve_seconds: int = 2 * 60 * 60
    case_mean_timeout_seconds: int = 45 * 60
    case_log_scale_timeout_seconds: int = 45 * 60
    minimum_shape_timeout_seconds: int = 30 * 60
    niterations_case: int = 500
    niterations_shape: int = 2_000
    populations: int = 8
    population_size: int = 40
    ncycles_per_iteration: int = 100
    case_maxsize: int = 18
    case_maxdepth: int = 7
    shape_maxsize: int = 28
    shape_maxdepth: int = 10
    shape_batching: bool = True
    shape_batch_size: int = 50_000
    max_shape_candidates_for_full_validation: int = 14
    pysr_parallelism: str = "multiprocessing"
    pysr_processes: int = 8
    status_heartbeat_seconds: int = 60
    force_rerun_completed_iteration: bool = False
    force_rebuild_training_sample: bool = False

    def validate(self) -> None:
        if self.iteration not in {1, 2, 3, 4}:
            raise ValueError("iteration must be one of 1, 2, 3 or 4")
        if self.rows_per_training_case != 5_000:
            raise ValueError("The formal design is locked to 5,000 rows per training case")
        if self.total_wall_seconds <= self.evaluation_reserve_seconds:
            raise ValueError("Evaluation reserve must be smaller than the total wall time")
        if self.pysr_parallelism not in {"serial", "multithreading", "multiprocessing"}:
            raise ValueError("Unsupported PySR parallelism")
        if self.pysr_processes < 1:
            raise ValueError("pysr_processes must be positive")
        if self.max_shape_candidates_for_full_validation < 1:
            raise ValueError("At least one shape candidate must be evaluated")


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _weighted_mean_std(
    X: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if weights is None:
        mean = values.mean(axis=0)
        variance = values.var(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = w / w.sum()
        mean = np.sum(values * w[:, None], axis=0)
        variance = np.sum((values - mean) ** 2 * w[:, None], axis=0)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.maximum(std, 1e-8)
    return mean, std


def _scaling_payload(
    X: np.ndarray,
    y: np.ndarray,
    features: Sequence[str],
    weights: np.ndarray | None = None,
) -> dict:
    x_mean, x_std = _weighted_mean_std(X, weights)
    y_mean_array, y_std_array = _weighted_mean_std(np.asarray(y), weights)
    return {
        "features": list(features),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": float(y_mean_array[0]),
        "y_std": float(y_std_array[0]),
    }


def _scale_arrays(X: np.ndarray, y: np.ndarray, scaling: dict) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = np.ascontiguousarray(
        (np.asarray(X, dtype=np.float32) - np.asarray(scaling["x_mean"], dtype=np.float32))
        / np.asarray(scaling["x_std"], dtype=np.float32),
        dtype=np.float32,
    )
    y_scaled = np.ascontiguousarray(
        (np.asarray(y, dtype=np.float32) - np.float32(scaling["y_mean"]))
        / np.float32(scaling["y_std"]),
        dtype=np.float32,
    )
    return X_scaled, y_scaled


def _save_scaling(scaling: dict, target_name: str, path: Path) -> None:
    rows = [
        {
            "variable": feature,
            "role": "input",
            "mean": float(mean),
            "std": float(std),
        }
        for feature, mean, std in zip(
            scaling["features"], scaling["x_mean"], scaling["x_std"]
        )
    ]
    rows.append({
        "variable": target_name,
        "role": "target",
        "mean": float(scaling["y_mean"]),
        "std": float(scaling["y_std"]),
    })
    pd.DataFrame(rows).to_csv(path, index=False)


def _load_scaling(path: Path) -> dict:
    table = pd.read_csv(path)
    inputs = table[table["role"] == "input"]
    target = table[table["role"] == "target"]
    if len(target) != 1:
        raise ValueError(f"Expected one target row in {path}")
    return {
        "features": inputs["variable"].tolist(),
        "x_mean": inputs["mean"].to_numpy(dtype=np.float64),
        "x_std": inputs["std"].to_numpy(dtype=np.float64),
        "y_mean": float(target.iloc[0]["mean"]),
        "y_std": float(target.iloc[0]["std"]),
    }


def _stage_status(
    output_dir: Path,
    stage: str,
    status: str,
    started_at: float,
    **extra,
) -> dict:
    payload = {
        "stage": stage,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started_at,
    }
    payload.update(extra)
    _atomic_json(payload, output_dir / f"stage_status_{stage}.json")
    return payload


def _start_heartbeat(
    output_dir: Path,
    run_root: Path,
    stage: str,
    run_id: str,
    started_at: float,
    interval_seconds: int,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    run_directory = run_root / run_id

    def monitor() -> None:
        while not stop.wait(interval_seconds):
            hall = run_directory / "hall_of_fame.csv"
            _stage_status(
                output_dir,
                stage,
                "search_running",
                started_at,
                run_id=run_id,
                hall_of_fame_exists=hall.exists(),
                hall_of_fame_size_bytes=hall.stat().st_size if hall.exists() else 0,
                hall_of_fame_modified=(
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%S%z",
                        time.localtime(hall.stat().st_mtime),
                    )
                    if hall.exists() else None
                ),
            )

    _stage_status(output_dir, stage, "search_starting", started_at, run_id=run_id)
    thread = threading.Thread(target=monitor, name=f"ct3-{stage}-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def _pysr_model(
    config: HierarchicalSymbolicConfig,
    stage: str,
    timeout_seconds: int,
    run_root: Path,
):
    try:
        from pysr import PySRRegressor
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PySR could not be imported. Activate the NotebookCT3 environment first."
        ) from exc

    is_shape = stage == "shape"
    run_id = (
        f"hierarchical_i{config.iteration}_{stage}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )
    unary_operators = SHAPE_UNARY_OPERATORS if is_shape else CASE_UNARY_OPERATORS
    active_operators = set(
        (SHAPE_BINARY_OPERATORS if is_shape else CASE_BINARY_OPERATORS)
        + unary_operators
    )
    constraints = {
        operator: value
        for operator, value in OPERATOR_CONSTRAINTS.items()
        if operator in active_operators
    }
    nested_constraints = {
        outer: {
            inner: value
            for inner, value in inner_constraints.items()
            if inner in active_operators
        }
        for outer, inner_constraints in NESTED_CONSTRAINTS.items()
        if outer in active_operators
    }
    operator_complexity = {
        operator: value
        for operator, value in OPERATOR_COMPLEXITY.items()
        if operator in active_operators
    }
    model_kwargs = dict(
        niterations=config.niterations_shape if is_shape else config.niterations_case,
        populations=config.populations,
        population_size=config.population_size,
        ncycles_per_iteration=config.ncycles_per_iteration,
        timeout_in_seconds=int(timeout_seconds),
        maxsize=config.shape_maxsize if is_shape else config.case_maxsize,
        maxdepth=config.shape_maxdepth if is_shape else config.case_maxdepth,
        warmup_maxsize_by=0.5,
        binary_operators=SHAPE_BINARY_OPERATORS if is_shape else CASE_BINARY_OPERATORS,
        unary_operators=unary_operators,
        constraints=constraints,
        nested_constraints=nested_constraints,
        complexity_of_operators=operator_complexity,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        batching=config.shape_batching if is_shape else False,
        precision=32,
        random_state=RANDOM_SEED,
        deterministic=False,
        parallelism=config.pysr_parallelism,
        procs=(
            config.pysr_processes
            if config.pysr_parallelism == "multiprocessing"
            else None
        ),
        warm_start=False,
        progress=True,
        verbosity=1,
        input_stream="devnull",
        update=False,
        output_directory=str(run_root),
        run_id=run_id,
    )
    if is_shape and config.shape_batching:
        model_kwargs["batch_size"] = config.shape_batch_size
    model = PySRRegressor(**model_kwargs)
    return model, run_id


def _formula_exports(
    model,
    features: Sequence[str],
    scaling: dict,
    run_id: str,
    stage: str,
) -> pd.DataFrame:
    equations = model.equations_.copy()
    records = []
    for candidate_index, row in equations.iterrows():
        expression = model.sympy(index=int(candidate_index))
        replacements = {
            sp.Symbol(f"{feature}_scaled"): (
                sp.Symbol(feature) - sp.Float(scaling["x_mean"][index])
            ) / sp.Float(scaling["x_std"][index])
            for index, feature in enumerate(features)
        }
        original = (
            sp.Float(scaling["y_mean"])
            + sp.Float(scaling["y_std"]) * expression.xreplace(replacements)
        )
        constant_free = re.sub(
            r"Float\([^\)]*\)",
            "CONST",
            sp.srepr(sp.factor(expression)),
        )
        operators = sorted({
            node.func.__name__
            for node in sp.preorder_traversal(expression)
            if getattr(node, "args", ())
        })
        support = sorted(str(symbol) for symbol in expression.free_symbols)
        records.append({
            "stage": stage,
            "run_id": run_id,
            "candidate_index": int(candidate_index),
            "complexity": int(row.get("complexity", 0)),
            "loss": float(row.get("loss", np.nan)),
            "score": float(row.get("score", np.nan)),
            "equation": str(row.get("equation", expression)),
            "formula_scaled_sympy": str(expression),
            "formula_original_variables": str(original),
            "feature_support_json": json.dumps(support),
            "structure_signature": constant_free,
            "family_signature": json.dumps({
                "features": support,
                "operators": operators,
            }, sort_keys=True),
        })
    if not records:
        raise RuntimeError(f"PySR returned no equations for stage {stage}")
    return pd.DataFrame(records).sort_values(["complexity", "loss"])


def _run_search_stage(
    *,
    config: HierarchicalSymbolicConfig,
    stage: str,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    features: Sequence[str],
    scaling: dict,
    timeout_seconds: int,
    output_dir: Path,
    run_root: Path,
) -> pd.DataFrame:
    frontier_path = output_dir / f"{stage}_frontier.csv"
    if frontier_path.exists() and not config.force_rerun_completed_iteration:
        print(f"{stage}: existing frontier reused: {frontier_path}", flush=True)
        return pd.read_csv(frontier_path)

    X_scaled, y_scaled = _scale_arrays(X, y, scaling)
    model, run_id = _pysr_model(config, stage, timeout_seconds, run_root)
    started_at = time.time()
    stop, heartbeat = _start_heartbeat(
        output_dir,
        run_root,
        stage,
        run_id,
        started_at,
        config.status_heartbeat_seconds,
    )
    try:
        model.fit(
            X_scaled,
            y_scaled,
            weights=weights,
            variable_names=[f"{feature}_scaled" for feature in features],
        )
        frontier = _formula_exports(model, features, scaling, run_id, stage)
        frontier.to_csv(frontier_path, index=False)
        _stage_status(
            output_dir,
            stage,
            "search_complete",
            started_at,
            run_id=run_id,
            timeout_seconds=int(timeout_seconds),
            n_candidates=len(frontier),
        )
        return frontier
    except Exception as exc:
        _stage_status(
            output_dir,
            stage,
            "search_failed",
            started_at,
            run_id=run_id,
            error=repr(exc),
        )
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=5)
        del model, X_scaled, y_scaled
        gc.collect()


def _compiled_formula(row: pd.Series, scaling: dict) -> Callable[[np.ndarray], np.ndarray]:
    features = list(scaling["features"])
    symbols = [sp.Symbol(f"{feature}_scaled") for feature in features]
    formula_text = (
        row["formula_scaled_sympy"]
        if isinstance(row, (pd.Series, dict))
        else getattr(row, "formula_scaled_sympy")
    )
    expression = sp.sympify(formula_text)
    function = sp.lambdify(symbols, expression, modules="numpy")

    def predict(X: np.ndarray) -> np.ndarray:
        X_scaled = (
            np.asarray(X, dtype=np.float64) - np.asarray(scaling["x_mean"])
        ) / np.asarray(scaling["x_std"])
        result = np.asarray(
            function(*[X_scaled[:, index] for index in range(X_scaled.shape[1])]),
            dtype=np.float64,
        )
        if result.ndim == 0:
            result = np.full(len(X_scaled), float(result), dtype=np.float64)
        result = result.reshape(-1)
        result = scaling["y_mean"] + scaling["y_std"] * result
        return result

    return predict


def _case_context_matrix(summary: pd.DataFrame, case_ids: Sequence[str]) -> np.ndarray:
    return np.ascontiguousarray(
        summary.loc[list(case_ids), CASE_FORMULA_FEATURES].to_numpy(dtype=np.float32)
    )


def _select_case_mean_formula(
    frontier: pd.DataFrame,
    scaling: dict,
    summary: pd.DataFrame,
    validation_ids: Sequence[str],
    output_dir: Path,
) -> pd.Series:
    X = _case_context_matrix(summary, validation_ids)
    actual = summary.loc[list(validation_ids), "stress_mean"].to_numpy(dtype=np.float64)
    records = []
    for _, candidate in frontier.iterrows():
        try:
            predicted = _compiled_formula(candidate, scaling)(X)
            finite = bool(np.isfinite(predicted).all())
            rmse = float(np.sqrt(np.mean((predicted - actual) ** 2))) if finite else np.inf
            mae = float(np.mean(np.abs(predicted - actual))) if finite else np.inf
            spearman = pd.Series(actual).corr(pd.Series(predicted), method="spearman") if finite else np.nan
        except Exception as exc:
            predicted = np.full(len(actual), np.nan)
            finite, rmse, mae, spearman = False, np.inf, np.inf, np.nan
            error = repr(exc)
        else:
            error = ""
        records.append({
            **candidate.to_dict(),
            "candidate_valid": finite,
            "invalid_reason": error,
            "validation_offset_rmse": rmse,
            "validation_offset_mae": mae,
            "validation_offset_spearman": spearman,
        })
    metrics = pd.DataFrame(records)
    valid = metrics[metrics["candidate_valid"]].copy()
    if valid.empty:
        raise RuntimeError("No finite case-mean formula was found")
    best_rmse = float(valid["validation_offset_rmse"].min())
    competitive = valid[
        valid["validation_offset_rmse"] <= 1.10 * max(best_rmse, 1e-12)
    ]
    selected = competitive.sort_values(
        ["complexity", "validation_offset_rmse", "candidate_index"]
    ).iloc[0]
    metrics["selected_candidate"] = metrics["candidate_index"].eq(selected["candidate_index"])
    metrics.to_csv(output_dir / "case_mean_candidate_metrics.csv", index=False)
    pd.DataFrame([selected]).to_csv(output_dir / "selected_case_mean_formula.csv", index=False)
    return selected


def _select_case_log_scale_formula(
    frontier: pd.DataFrame,
    scaling: dict,
    mean_row: pd.Series,
    mean_scaling: dict,
    summary: pd.DataFrame,
    validation_ids: Sequence[str],
    output_dir: Path,
) -> pd.Series:
    X = _case_context_matrix(summary, validation_ids)
    actual_mean = summary.loc[list(validation_ids), "stress_mean"].to_numpy(dtype=np.float64)
    actual_p95 = summary.loc[list(validation_ids), "stress_p95"].to_numpy(dtype=np.float64)
    actual_scale = actual_p95 - actual_mean
    actual_log_scale = np.log(actual_scale)
    predicted_mean = _compiled_formula(mean_row, mean_scaling)(X)
    records = []
    for _, candidate in frontier.iterrows():
        try:
            predicted_log_scale = _compiled_formula(candidate, scaling)(X)
            # The validation expression must match the exported deployable
            # expression exactly.  Reject explosive logarithmic scales instead
            # of silently clipping them during formula selection.
            if (
                not np.isfinite(predicted_log_scale).all()
                or np.any(np.abs(predicted_log_scale) > 20.0)
            ):
                raise FloatingPointError("non-finite or explosive predicted log scale")
            predicted_scale = np.exp(predicted_log_scale)
            predicted_p95 = predicted_mean + predicted_scale
            finite = bool(np.isfinite(predicted_scale).all())
            log_rmse = float(np.sqrt(np.mean((predicted_log_scale - actual_log_scale) ** 2))) if finite else np.inf
            scale_relative_error = float(np.mean(
                np.abs(predicted_scale - actual_scale) / np.maximum(actual_scale, 1e-12)
            )) if finite else np.inf
            p95_relative_error = float(np.mean(
                np.abs(predicted_p95 - actual_p95) / np.maximum(np.abs(actual_p95), 1e-12)
            )) if finite else np.inf
            p95_under = float(np.mean(
                np.maximum(actual_p95 - predicted_p95, 0.0)
                / np.maximum(np.abs(actual_p95), 1e-12)
            )) if finite else np.inf
            p95_spearman = pd.Series(actual_p95).corr(
                pd.Series(predicted_p95), method="spearman"
            ) if finite else np.nan
        except Exception as exc:
            finite = False
            log_rmse = scale_relative_error = p95_relative_error = p95_under = np.inf
            p95_spearman = np.nan
            error = repr(exc)
        else:
            error = ""
        records.append({
            **candidate.to_dict(),
            "candidate_valid": finite,
            "invalid_reason": error,
            "validation_log_scale_rmse": log_rmse,
            "validation_mean_scale_relative_error": scale_relative_error,
            "validation_mean_p95_relative_error": p95_relative_error,
            "validation_mean_p95_underprediction_fraction": p95_under,
            "validation_p95_spearman": p95_spearman,
        })
    metrics = pd.DataFrame(records)
    valid = metrics[metrics["candidate_valid"]].copy()
    if valid.empty:
        raise RuntimeError("No finite case log-scale formula was found")
    for column in [
        "validation_log_scale_rmse",
        "validation_mean_scale_relative_error",
        "validation_mean_p95_relative_error",
        "validation_mean_p95_underprediction_fraction",
    ]:
        valid[f"{column}_rank"] = valid[column].rank(pct=True, method="average")
    valid["case_scale_selection_score"] = (
        0.25 * valid["validation_log_scale_rmse_rank"]
        + 0.25 * valid["validation_mean_scale_relative_error_rank"]
        + 0.35 * valid["validation_mean_p95_relative_error_rank"]
        + 0.15 * valid["validation_mean_p95_underprediction_fraction_rank"]
    )
    best_score = float(valid["case_scale_selection_score"].min())
    competitive = valid[
        valid["case_scale_selection_score"] <= 1.05 * max(best_score, 1e-12)
    ]
    selected = competitive.sort_values(
        ["complexity", "case_scale_selection_score", "candidate_index"]
    ).iloc[0]
    for column in valid.columns:
        if column not in metrics.columns:
            metrics[column] = np.nan
    metrics.loc[valid.index, valid.columns] = valid
    metrics["selected_candidate"] = metrics["candidate_index"].eq(selected["candidate_index"])
    metrics.to_csv(output_dir / "case_log_scale_candidate_metrics.csv", index=False)
    pd.DataFrame([selected]).to_csv(output_dir / "selected_case_log_scale_formula.csv", index=False)
    return selected


def _shortlist_shape_frontier(frontier: pd.DataFrame, limit: int) -> pd.DataFrame:
    table = frontier.sort_values(["complexity", "loss"]).drop_duplicates("complexity")
    if len(table) <= limit:
        return table
    positions = np.unique(np.linspace(0, len(table) - 1, limit, dtype=int))
    lowest_loss_position = int(np.argmin(table["loss"].to_numpy(dtype=np.float64)))
    positions = np.unique(np.r_[positions, lowest_loss_position])
    if len(positions) > limit:
        removable = [position for position in positions if position != lowest_loss_position]
        positions = np.asarray([lowest_loss_position] + removable[: limit - 1])
    return table.iloc[np.sort(positions)].copy()


def _tail_adaptation(case_metrics: pd.DataFrame) -> dict:
    result = {}
    for tail in ["p95", "p99"]:
        actual = case_metrics[f"actual_{tail}"].to_numpy(dtype=np.float64)
        predicted = case_metrics[f"predicted_{tail}"].to_numpy(dtype=np.float64)
        actual_std = float(np.std(actual, ddof=0))
        predicted_std = float(np.std(predicted, ddof=0))
        spearman = pd.Series(actual).corr(pd.Series(predicted), method="spearman")
        result.update({
            f"case_{tail}_spearman": float(spearman) if pd.notna(spearman) else np.nan,
            f"case_{tail}_std_ratio": predicted_std / max(actual_std, 1e-12),
            f"case_{tail}_mean_bias": float(np.mean(predicted - actual)),
        })
    return result


def _reference_metrics(inputs: dict, iteration: int, split: str) -> pd.Series:
    path = (
        inputs["paths"].output_root
        / "05_hierarchical_feasibility_v5"
        / f"iteration_{iteration}_pilot"
        / "split_metrics.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing hierarchical feasibility reference: {path}")
    table = pd.read_csv(path)
    rows = table[
        (table["split"] == split)
        & (table["model"] == "deployable_hierarchical")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one deployable feasibility row for {split}")
    return rows.iloc[0]


def _safe_ratio(value: float, reference: float, floor: float = 1e-4) -> float:
    return float(value) / max(abs(float(reference)), floor)


def _add_composite_selection_score(
    table: pd.DataFrame,
    reference: pd.Series,
) -> pd.DataFrame:
    out = table.copy()
    out["reference_relative_engineering_score"] = (
        0.18 * out["validation_macro_rmse"].map(lambda x: _safe_ratio(x, reference["macro_rmse"]))
        + 0.08 * out["validation_worst_case_rmse"].map(lambda x: _safe_ratio(x, reference["worst_case_rmse"]))
        + 0.10 * out["validation_mean_top5_actual_rmse"].map(lambda x: _safe_ratio(x, reference["mean_top5_actual_rmse"]))
        + 0.12 * out["validation_mean_p95_relative_error"].map(lambda x: _safe_ratio(x, reference["mean_p95_relative_error"]))
        + 0.20 * out["validation_mean_p99_relative_error"].map(lambda x: _safe_ratio(x, reference["mean_p99_relative_error"]))
        + 0.08 * out["validation_mean_p95_underprediction_fraction"].map(
            lambda x: _safe_ratio(x, reference["mean_p95_underprediction_fraction"], 0.02)
        )
        + 0.12 * out["validation_mean_p99_underprediction_fraction"].map(
            lambda x: _safe_ratio(x, reference["mean_p99_underprediction_fraction"], 0.02)
        )
        + 0.06 * (1.0 - out["validation_mean_top1pct_hotspot_overlap"]) / max(
            1.0 - float(reference["mean_top1pct_hotspot_overlap"]), 0.02
        )
        + 0.06 * (1.0 - out["validation_mean_top1_recall_in_predicted_top5"]) / max(
            1.0 - float(reference["mean_top1_recall_in_predicted_top5"]), 0.01
        )
    )
    out["gate_macro_r2"] = out["validation_macro_r2"] > 0.0
    out["gate_rmse"] = out["validation_macro_rmse"] <= 1.50 * reference["macro_rmse"]
    out["gate_p95_error"] = out["validation_mean_p95_relative_error"] <= 0.15
    out["gate_p99_error"] = out["validation_mean_p99_relative_error"] <= 0.15
    out["gate_p95_underprediction"] = out[
        "validation_mean_p95_underprediction_fraction"
    ] <= 0.10
    out["gate_p99_underprediction"] = out[
        "validation_mean_p99_underprediction_fraction"
    ] <= 0.10
    out["gate_top1_overlap"] = out[
        "validation_mean_top1pct_hotspot_overlap"
    ] >= 0.70
    out["gate_top1_recall"] = out[
        "validation_mean_top1_recall_in_predicted_top5"
    ] >= 0.90
    out["gate_p95_spearman"] = out["validation_case_p95_spearman"] >= 0.60
    out["gate_p99_spearman"] = out["validation_case_p99_spearman"] >= 0.60
    out["gate_p95_spread"] = out["validation_case_p95_std_ratio"].between(0.50, 1.50)
    out["gate_p99_spread"] = out["validation_case_p99_std_ratio"].between(0.50, 1.50)
    out["gate_numerical_guardrail"] = out[
        "validation_max_prediction_abs_max_ratio"
    ] <= 5.0
    gate_columns = [column for column in out if column.startswith("gate_")]
    out["all_provisional_acceptance_gates_pass"] = out[gate_columns].all(axis=1)
    return out


def _evaluate_shape_candidates(
    *,
    candidates: pd.DataFrame,
    shape_scaling: dict,
    mean_row: pd.Series,
    mean_scaling: dict,
    scale_row: pd.Series,
    scale_scaling: dict,
    inputs: dict,
    case_ids: Sequence[str],
    split: str,
    iteration: int,
    hard_deadline: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape_functions = {
        int(row.candidate_index): _compiled_formula(row, shape_scaling)
        for row in candidates.itertuples(index=False)
    }
    mean_function = _compiled_formula(mean_row, mean_scaling)
    scale_function = _compiled_formula(scale_row, scale_scaling)
    model_indices = [MODEL_FEATURES.index(feature) for feature in SHAPE_FORMULA_FEATURES]
    metric_rows = []
    invalid = {}

    for position, case_id in enumerate(case_ids, start=1):
        if hard_deadline is not None and time.time() >= hard_deadline:
            raise TimeoutError(
                "The 24-hour run budget was reached during complete-case "
                "evaluation. Saved search frontiers will be reused on rerun."
            )
        print(
            f"[{position}/{len(case_ids)}] {split} full-case symbolic evaluation: {case_id}",
            flush=True,
        )
        frame, _ = read_complete_case(inputs["path_by_case"][case_id])
        summary = _case_summary_row(inputs["case_summary"], case_id)
        actual = frame[TARGET_COL].to_numpy(dtype=np.float64)
        context = summary[CASE_FORMULA_FEATURES].to_numpy(dtype=np.float64)[None, :]
        predicted_mean = float(mean_function(context)[0])
        predicted_log_scale = float(scale_function(context)[0])
        if not np.isfinite(predicted_log_scale) or abs(predicted_log_scale) > 20.0:
            raise FloatingPointError(
                f"{case_id}: non-finite or explosive predicted log scale"
            )
        predicted_scale = float(np.exp(predicted_log_scale))
        full_matrix = build_model_matrix(frame, summary)
        shape_matrix = np.ascontiguousarray(full_matrix[:, model_indices], dtype=np.float32)

        for candidate_index, function in shape_functions.items():
            if candidate_index in invalid:
                continue
            try:
                predicted_shape = function(shape_matrix)
                predicted = predicted_mean + predicted_scale * predicted_shape
                if not np.isfinite(predicted).all():
                    raise FloatingPointError("non-finite composite prediction")
                metric_rows.append({
                    "iteration": iteration,
                    "split": split,
                    "model": "hierarchical_symbolic",
                    "candidate_index": candidate_index,
                    "case_id": case_id,
                    "predicted_case_mean": predicted_mean,
                    "predicted_case_scale": predicted_scale,
                    **evaluate_prediction_arrays(actual, predicted),
                })
            except Exception as exc:
                invalid[candidate_index] = repr(exc)
            finally:
                if "predicted_shape" in locals():
                    del predicted_shape
                if "predicted" in locals():
                    del predicted
        del frame, actual, full_matrix, shape_matrix
        gc.collect()

    case_metrics = pd.DataFrame(metric_rows)
    records = []
    for candidate in candidates.itertuples(index=False):
        index = int(candidate.candidate_index)
        group = case_metrics[case_metrics["candidate_index"] == index]
        if index in invalid or group["case_id"].nunique() != len(case_ids):
            records.append({
                **candidate._asdict(),
                "candidate_valid": False,
                "invalid_reason": invalid.get(index, "incomplete_case_evaluation"),
            })
            continue
        aggregate = aggregate_case_metrics(group, ["candidate_index"]).iloc[0].to_dict()
        aggregate.pop("candidate_index", None)
        adaptation = _tail_adaptation(group)
        records.append({
            **candidate._asdict(),
            "candidate_valid": True,
            "invalid_reason": "",
            **{f"{split}_{key}": value for key, value in aggregate.items()},
            **{f"{split}_{key}": value for key, value in adaptation.items()},
        })
    return pd.DataFrame(records), case_metrics


def _select_shape_formula(
    candidate_metrics: pd.DataFrame,
    reference: pd.Series,
    output_dir: Path,
) -> pd.Series:
    valid_mask = candidate_metrics["candidate_valid"].fillna(False)
    if not valid_mask.any():
        raise RuntimeError("No shape candidate produced complete finite validation predictions")
    valid = _add_composite_selection_score(candidate_metrics[valid_mask], reference)
    for column in valid.columns:
        if column not in candidate_metrics.columns:
            candidate_metrics[column] = np.nan
    candidate_metrics.loc[valid.index, valid.columns] = valid

    numerical = valid[valid["gate_numerical_guardrail"]].copy()
    if numerical.empty:
        raise RuntimeError("All shape candidates failed the numerical guardrail")
    accepted = numerical[numerical["all_provisional_acceptance_gates_pass"]]
    pool = accepted if not accepted.empty else numerical
    best_score = float(pool["reference_relative_engineering_score"].min())
    best_rmse = float(pool["validation_macro_rmse"].min())
    competitive = pool[
        (pool["reference_relative_engineering_score"] <= 1.05 * best_score)
        & (pool["validation_macro_rmse"] <= 1.10 * best_rmse)
    ]
    selected = competitive.sort_values([
        "complexity",
        "reference_relative_engineering_score",
        "validation_macro_rmse",
        "candidate_index",
    ]).iloc[0]
    candidate_metrics["selected_candidate"] = candidate_metrics["candidate_index"].eq(
        selected["candidate_index"]
    )
    candidate_metrics["selection_pool_status"] = (
        "all_gates_pass" if not accepted.empty
        else "provisional_no_candidate_passed_all_gates"
    )
    candidate_metrics.to_csv(output_dir / "shape_candidate_validation_metrics.csv", index=False)
    pd.DataFrame([selected]).to_csv(output_dir / "selected_shape_formula.csv", index=False)
    return selected


def _composite_formula_record(
    iteration: int,
    mean_row: pd.Series,
    scale_row: pd.Series,
    shape_row: pd.Series,
) -> dict:
    mean_formula = str(mean_row["formula_original_variables"])
    log_scale_formula = str(scale_row["formula_original_variables"])
    shape_formula = str(shape_row["formula_original_variables"])
    composite = f"({mean_formula}) + exp({log_scale_formula}) * ({shape_formula})"
    return {
        "iteration": iteration,
        "decomposition": "stress = case_mean + exp(case_log_scale) * normalised_shape",
        "case_mean_formula": mean_formula,
        "case_log_scale_formula": log_scale_formula,
        "case_scale_formula": f"exp({log_scale_formula})",
        "shape_formula": shape_formula,
        "composite_formula": composite,
        "case_mean_complexity": int(mean_row["complexity"]),
        "case_log_scale_complexity": int(scale_row["complexity"]),
        "shape_complexity": int(shape_row["complexity"]),
        "total_component_complexity": int(
            mean_row["complexity"] + scale_row["complexity"] + shape_row["complexity"] + 3
        ),
        "mean_family_signature": mean_row["family_signature"],
        "scale_family_signature": scale_row["family_signature"],
        "shape_family_signature": shape_row["family_signature"],
    }


def _save_composite_text(record: dict, output_dir: Path) -> None:
    text = (
        "CT3 hierarchical symbolic stress formula\n"
        "=======================================\n\n"
        f"Iteration: {record['iteration']}\n\n"
        "1. Case mean stress\n"
        f"mu = {record['case_mean_formula']}\n\n"
        "2. Positive case stress scale\n"
        f"scale = {record['case_scale_formula']}\n\n"
        "3. Normalised element-level spatial shape\n"
        f"shape = {record['shape_formula']}\n\n"
        "4. Combined deployable expression\n"
        f"sigma = {record['composite_formula']}\n\n"
        "All case-context summaries and within-case z scores are calculated only "
        "from predictor fields. No stress summary is required at deployment.\n"
    )
    (output_dir / "selected_composite_formula.txt").write_text(text, encoding="utf-8")


def _save_plots(
    case_metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    reference_rows: pd.DataFrame,
    output_dir: Path,
) -> None:
    validation = case_metrics[case_metrics["split"] == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, tail in zip(axes, ["p95", "p99"]):
        actual = validation[f"actual_{tail}"]
        predicted = validation[f"predicted_{tail}"]
        axis.scatter(actual, predicted, color="#d97732", s=42)
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#555555")
        axis.set_xlabel(f"Actual case {tail.upper()} stress")
        axis.set_ylabel(f"Predicted case {tail.upper()} stress")
        axis.set_title(f"Validation {tail.upper()} adaptation")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_formula_validation_tail_adaptation.png", dpi=180)
    plt.close(fig)

    comparison = pd.concat([
        reference_rows.assign(model="hierarchical_feasibility_HGB"),
        split_metrics.assign(model="hierarchical_symbolic"),
    ], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric, title in [
        (axes[0], "macro_rmse", "Macro RMSE"),
        (axes[1], "mean_p95_relative_error", "Mean P95 relative error"),
        (axes[2], "mean_p99_relative_error", "Mean P99 relative error"),
    ]:
        pivot = comparison.pivot(index="split", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axis, color=["#2a9d6f", "#d97732"])
        axis.set_title(title)
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=0)
        if axis is not axes[0] and axis.get_legend() is not None:
            axis.get_legend().remove()
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "symbolic_vs_hierarchical_feasibility.png", dpi=180)
    plt.close(fig)


def _refresh_cross_iteration_summary(output_root: Path) -> None:
    records = []
    metrics = []
    for iteration in range(1, 5):
        directory = output_root / f"iteration_{iteration}"
        formula_path = directory / "selected_composite_formula.csv"
        metric_path = directory / "selected_formula_split_metrics.csv"
        if formula_path.exists():
            records.append(pd.read_csv(formula_path))
        if metric_path.exists():
            metrics.append(pd.read_csv(metric_path))
    if records:
        formulas = pd.concat(records, ignore_index=True)
        formulas.to_csv(output_root / "selected_composite_formulas_by_iteration.csv", index=False)
        stability = formulas.groupby("shape_family_signature", dropna=False).agg(
            n_iterations=("iteration", "nunique"),
            iterations=("iteration", lambda values: json.dumps(sorted(set(map(int, values))))),
            mean_total_complexity=("total_component_complexity", "mean"),
        ).reset_index().sort_values(["n_iterations", "mean_total_complexity"], ascending=[False, True])
        stability.to_csv(output_root / "shape_structure_stability.csv", index=False)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(
            output_root / "symbolic_split_metrics_all_iterations.csv", index=False
        )


def preflight_hierarchical_symbolic_iteration(
    package_root: Path,
    config: HierarchicalSymbolicConfig,
) -> dict:
    """Validate paths, split isolation and prerequisite outputs without reading FEM cases."""
    config.validate()
    feasibility_config = FeasibilityConfig(
        iteration=config.iteration,
        rows_per_case=config.rows_per_training_case,
        output_subdir=f"iteration_{config.iteration}_pilot",
        force_rebuild_sample=False,
    )
    inputs = load_inputs(Path(package_root), feasibility_config)
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )
    missing_features = sorted(
        set(CASE_FORMULA_FEATURES) - set(inputs["case_summary"].columns)
    )
    if missing_features:
        raise ValueError(f"Missing case-context features: {missing_features}")
    development_ids = (
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"]
    )
    development_summary = inputs["case_summary"].loc[development_ids]
    context_values = development_summary[CASE_FORMULA_FEATURES].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(context_values).all():
        raise ValueError("Non-finite case-context values were found in development cases")
    case_scale = (
        development_summary["stress_p95"] - development_summary["stress_mean"]
    )
    if not np.isfinite(case_scale).all() or (case_scale <= 0.0).any():
        raise ValueError(
            "Every development case requires a finite positive stress_p95 - stress_mean"
        )
    feasibility_reference = (
        inputs["paths"].output_root
        / "05_hierarchical_feasibility_v5"
        / f"iteration_{config.iteration}_pilot"
        / "split_metrics.csv"
    )
    if not feasibility_reference.exists():
        raise FileNotFoundError(
            f"Run hierarchical feasibility iteration {config.iteration} first: "
            f"{feasibility_reference}"
        )
    return {
        "iteration": config.iteration,
        "train_cases": len(inputs["train_ids"]),
        "validation_cases": len(inputs["validation_ids"]),
        "internal_test_cases": len(inputs["internal_ids"]),
        "sealed_final_cases": len(inputs["final_ids"]),
        "case_formula_features": len(CASE_FORMULA_FEATURES),
        "shape_formula_features": len(SHAPE_FORMULA_FEATURES),
        "training_rows": len(inputs["train_ids"]) * config.rows_per_training_case,
        "total_wall_hours": config.total_wall_seconds / 3600,
        "search_window_hours": (
            config.total_wall_seconds - config.evaluation_reserve_seconds
        ) / 3600,
        "shape_batching": config.shape_batching,
        "shape_batch_size": config.shape_batch_size,
        "minimum_development_case_scale": float(case_scale.min()),
        "maximum_development_case_scale": float(case_scale.max()),
        "final_test_read": False,
    }


def run_hierarchical_symbolic_iteration(
    package_root: Path,
    config: HierarchicalSymbolicConfig,
) -> dict:
    """Run one restartable 24-hour hierarchical symbolic-regression rotation."""
    config.validate()
    run_started = time.time()
    hard_deadline = run_started + config.total_wall_seconds
    package_root = Path(package_root).resolve()
    output_root = package_root / "outputs" / "06_hierarchical_symbolic_regression"
    output_dir = output_root / config.output_subdir
    run_root = output_dir / "pysr_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "round_complete.json"
    if completion_path.exists() and not config.force_rerun_completed_iteration:
        print(f"Iteration {config.iteration} already complete: {completion_path}")
        return json.loads(completion_path.read_text(encoding="utf-8"))

    preflight = preflight_hierarchical_symbolic_iteration(package_root, config)
    feasibility_config = FeasibilityConfig(
        iteration=config.iteration,
        rows_per_case=config.rows_per_training_case,
        output_subdir=f"iteration_{config.iteration}_pilot",
        force_rebuild_sample=config.force_rebuild_training_sample,
    )
    inputs = load_inputs(package_root, feasibility_config)
    inputs["output_dir"] = output_dir
    assert_no_final_cases(
        inputs["train_ids"] + inputs["validation_ids"] + inputs["internal_ids"],
        inputs["manifest"],
    )
    _atomic_json(
        {
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": asdict(config),
            "preflight": preflight,
        },
        output_dir / "run_status.json",
    )

    pd.DataFrame([
        {"feature": feature, "role": "case_formula_and_shape_context"}
        for feature in CASE_FORMULA_FEATURES
    ] + [
        {"feature": feature, "role": "shape_formula"}
        for feature in SHAPE_FORMULA_FEATURES
        if feature not in CASE_FORMULA_FEATURES
    ]).to_csv(output_dir / "feature_registry.csv", index=False)

    summary = inputs["case_summary"]
    train_ids = inputs["train_ids"]
    validation_ids = inputs["validation_ids"]
    internal_ids = inputs["internal_ids"]
    X_case_train = _case_context_matrix(summary, train_ids)
    y_mean_train = summary.loc[train_ids, "stress_mean"].to_numpy(dtype=np.float32)
    y_log_scale_train = np.log(
        summary.loc[train_ids, "stress_p95"].to_numpy(dtype=np.float64)
        - summary.loc[train_ids, "stress_mean"].to_numpy(dtype=np.float64)
    ).astype(np.float32)

    mean_scaling_path = output_dir / "case_mean_scaling.csv"
    if mean_scaling_path.exists() and not config.force_rerun_completed_iteration:
        mean_scaling = _load_scaling(mean_scaling_path)
    else:
        mean_scaling = _scaling_payload(
            X_case_train, y_mean_train, CASE_FORMULA_FEATURES
        )
        _save_scaling(mean_scaling, "case_stress_mean", mean_scaling_path)

    scale_scaling_path = output_dir / "case_log_scale_scaling.csv"
    if scale_scaling_path.exists() and not config.force_rerun_completed_iteration:
        scale_scaling = _load_scaling(scale_scaling_path)
    else:
        scale_scaling = _scaling_payload(
            X_case_train, y_log_scale_train, CASE_FORMULA_FEATURES
        )
        _save_scaling(scale_scaling, "log_case_p95_minus_mean", scale_scaling_path)

    search_deadline = run_started + (
        config.total_wall_seconds - config.evaluation_reserve_seconds
    )
    mean_frontier = _run_search_stage(
        config=config,
        stage="case_mean",
        X=X_case_train,
        y=y_mean_train,
        weights=None,
        features=CASE_FORMULA_FEATURES,
        scaling=mean_scaling,
        timeout_seconds=min(
            config.case_mean_timeout_seconds,
            max(60, int(search_deadline - time.time())),
        ),
        output_dir=output_dir,
        run_root=run_root,
    )
    selected_mean = _select_case_mean_formula(
        mean_frontier,
        mean_scaling,
        summary,
        validation_ids,
        output_dir,
    )

    scale_frontier = _run_search_stage(
        config=config,
        stage="case_log_scale",
        X=X_case_train,
        y=y_log_scale_train,
        weights=None,
        features=CASE_FORMULA_FEATURES,
        scaling=scale_scaling,
        timeout_seconds=min(
            config.case_log_scale_timeout_seconds,
            max(60, int(search_deadline - time.time())),
        ),
        output_dir=output_dir,
        run_root=run_root,
    )
    selected_scale = _select_case_log_scale_formula(
        scale_frontier,
        scale_scaling,
        selected_mean,
        mean_scaling,
        summary,
        validation_ids,
        output_dir,
    )

    sample = assemble_training_sample(inputs, feasibility_config)
    sample["audit"].to_csv(output_dir / "training_sample_audit.csv", index=False)
    sample["manifest"].to_csv(
        output_dir / "training_sample_manifest.csv.gz", index=False, compression="gzip"
    )
    shape_indices = [MODEL_FEATURES.index(feature) for feature in SHAPE_FORMULA_FEATURES]
    X_shape = np.ascontiguousarray(sample["X"][:, shape_indices], dtype=np.float32)
    y_shape = np.ascontiguousarray(sample["y_shape"], dtype=np.float32)
    weights = np.ascontiguousarray(sample["weights"], dtype=np.float32)
    shape_scaling_path = output_dir / "shape_scaling.csv"
    if shape_scaling_path.exists() and not config.force_rerun_completed_iteration:
        shape_scaling = _load_scaling(shape_scaling_path)
    else:
        shape_scaling = _scaling_payload(
            X_shape, y_shape, SHAPE_FORMULA_FEATURES, weights
        )
        _save_scaling(shape_scaling, "normalised_spatial_shape", shape_scaling_path)

    remaining_shape_seconds = int(search_deadline - time.time())
    if remaining_shape_seconds < config.minimum_shape_timeout_seconds:
        raise TimeoutError(
            "Less than the minimum shape-search budget remains. Rerun this notebook; "
            "completed case-level stages will be reused."
        )
    shape_frontier = _run_search_stage(
        config=config,
        stage="shape",
        X=X_shape,
        y=y_shape,
        weights=weights,
        features=SHAPE_FORMULA_FEATURES,
        scaling=shape_scaling,
        timeout_seconds=remaining_shape_seconds,
        output_dir=output_dir,
        run_root=run_root,
    )
    shortlist = _shortlist_shape_frontier(
        shape_frontier, config.max_shape_candidates_for_full_validation
    )
    shortlist.to_csv(output_dir / "shape_full_validation_shortlist.csv", index=False)
    del sample, X_shape, y_shape, weights
    gc.collect()

    validation_candidate_metrics, validation_case_metrics = _evaluate_shape_candidates(
        candidates=shortlist,
        shape_scaling=shape_scaling,
        mean_row=selected_mean,
        mean_scaling=mean_scaling,
        scale_row=selected_scale,
        scale_scaling=scale_scaling,
        inputs=inputs,
        case_ids=validation_ids,
        split="validation",
        iteration=config.iteration,
        hard_deadline=hard_deadline,
    )
    reference_validation = _reference_metrics(inputs, config.iteration, "validation")
    selected_shape = _select_shape_formula(
        validation_candidate_metrics,
        reference_validation,
        output_dir,
    )
    selected_index = int(selected_shape["candidate_index"])
    selected_validation_cases = validation_case_metrics[
        validation_case_metrics["candidate_index"] == selected_index
    ].copy()
    selected_validation_cases["iteration"] = config.iteration
    selected_validation_cases.to_csv(
        output_dir / "selected_formula_validation_case_metrics.csv.gz",
        index=False,
        compression="gzip",
    )

    selected_shape_table = pd.DataFrame([selected_shape])
    internal_candidate_metrics, internal_case_metrics = _evaluate_shape_candidates(
        candidates=selected_shape_table,
        shape_scaling=shape_scaling,
        mean_row=selected_mean,
        mean_scaling=mean_scaling,
        scale_row=selected_scale,
        scale_scaling=scale_scaling,
        inputs=inputs,
        case_ids=internal_ids,
        split="internal_test",
        iteration=config.iteration,
        hard_deadline=hard_deadline,
    )
    internal_case_metrics["iteration"] = config.iteration
    internal_case_metrics.to_csv(
        output_dir / "selected_formula_internal_test_case_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    selected_cases = pd.concat(
        [selected_validation_cases, internal_case_metrics], ignore_index=True
    )
    selected_cases.to_csv(
        output_dir / "selected_formula_case_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    split_metrics = aggregate_case_metrics(
        selected_cases, ["iteration", "split", "model"]
    )
    split_metrics.to_csv(output_dir / "selected_formula_split_metrics.csv", index=False)
    tail_rows = []
    for split, group in selected_cases.groupby("split"):
        tail_rows.append({
            "iteration": config.iteration,
            "split": split,
            **_tail_adaptation(group),
        })
    tail_adaptation = pd.DataFrame(tail_rows)
    tail_adaptation.to_csv(output_dir / "selected_formula_tail_adaptation.csv", index=False)

    composite_record = _composite_formula_record(
        config.iteration, selected_mean, selected_scale, selected_shape
    )
    pd.DataFrame([composite_record]).to_csv(
        output_dir / "selected_composite_formula.csv", index=False
    )
    _save_composite_text(composite_record, output_dir)

    reference_rows = pd.DataFrame([
        _reference_metrics(inputs, config.iteration, "validation"),
        _reference_metrics(inputs, config.iteration, "internal_test"),
    ])
    _save_plots(selected_cases, split_metrics, reference_rows, output_dir)

    payload = {
        "status": "complete",
        "iteration": config.iteration,
        "diagnostic_only": False,
        "final_test_read": False,
        "train_cases": len(train_ids),
        "validation_cases": len(validation_ids),
        "internal_test_cases": len(internal_ids),
        "training_sample_rows": len(train_ids) * config.rows_per_training_case,
        "decomposition": composite_record["decomposition"],
        "selected_mean_candidate": int(selected_mean["candidate_index"]),
        "selected_scale_candidate": int(selected_scale["candidate_index"]),
        "selected_shape_candidate": selected_index,
        "selection_pool_status": str(
            pd.read_csv(output_dir / "shape_candidate_validation_metrics.csv")[
                "selection_pool_status"
            ].dropna().iloc[0]
        ),
        "elapsed_seconds": time.time() - run_started,
        "requested_wall_limit_seconds": config.total_wall_seconds,
        "output_directory": str(output_dir),
    }
    _atomic_json(payload, completion_path)
    _atomic_json(payload, output_dir / "run_status.json")
    _refresh_cross_iteration_summary(output_root)
    return payload
