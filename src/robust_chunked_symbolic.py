"""Unchanged frontier-conversion helpers retained from the earlier recovery module.
The unused old chunk-training workflow is not part of this delivery.
"""
from pathlib import Path
from typing import Sequence
import re
import json
import numpy as np
import pandas as pd
import sympy as sp

def _formula_expression(text: str) -> sp.Expr:
    locals_map = {
        "square": lambda value: value**2,
        "cube": lambda value: value**3,
        "abs": sp.Abs,
        "Abs": sp.Abs,
    }
    return sp.sympify(str(text), locals=locals_map)


def _unscale_expression(expression: sp.Expr, features: Sequence[str], scaling: dict) -> sp.Expr:
    replacements = {
        sp.Symbol(f"{feature}_scaled"): (
            sp.Symbol(feature) - sp.Float(scaling["x_mean"][index])
        ) / sp.Float(scaling["x_std"][index])
        for index, feature in enumerate(features)
    }
    return (
        sp.Float(scaling["y_mean"])
        + sp.Float(scaling["y_std"]) * expression.xreplace(replacements)
    )


def frontier_from_hall_of_fame(
    hall_path: Path,
    features: Sequence[str],
    scaling: dict,
    stage: str,
    run_id: str,
) -> pd.DataFrame:
    """Convert a raw PySR Hall of Fame into the project's canonical frontier."""

    raw = pd.read_csv(hall_path)
    # PySR writes raw Hall-of-Fame headers as Complexity/Loss/Equation, while
    # model.equations_ uses lower-case names. Recovery accepts both forms.
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    required = {"complexity", "loss", "equation"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{hall_path} is missing PySR columns: {missing}")
    records = []
    for candidate_index, row in raw.reset_index(drop=True).iterrows():
        expression = _formula_expression(row["equation"])
        original = _unscale_expression(expression, features, scaling)
        constant_free = re.sub(r"Float\([^\)]*\)", "CONST", sp.srepr(sp.factor(expression)))
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
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
            "score": float(row.get("score", np.nan)),
            "equation": str(row["equation"]),
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
        raise RuntimeError(f"No candidates were recovered from {hall_path}")
    return pd.DataFrame(records).sort_values(["complexity", "loss"]).reset_index(drop=True)
