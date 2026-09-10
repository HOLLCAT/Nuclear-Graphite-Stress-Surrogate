"""Read-only checks for the relocated research archive; no training or final rerun."""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "graphite-delivery-matplotlib"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def notebook_digest(notebook, *, exclude_markdown_source=False):
    content = copy.deepcopy(notebook)
    if exclude_markdown_source:
        for cell in content["cells"]:
            if cell["cell_type"] == "markdown":
                cell.pop("source", None)
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_annotation_edits():
    path = ROOT / "provenance/notebook_annotation_edits.json"
    audit = json.loads(path.read_text())
    check(audit["schema_version"] == 1, "Unknown notebook annotation audit schema")
    records = {}
    for item in audit["records"]:
        name = item["file"]
        target = (ROOT / name).resolve()
        check(target.is_relative_to(ROOT), "Annotation audit path leaves the package")
        check(name not in records, f"Duplicate annotation record: {name}")
        check(target.suffix == ".ipynb", "Only notebook annotations can be exempted")
        kind = item["change_kind"]
        check(kind in {"markdown_translation", "json_serialization_only"}, "Unknown annotation change")
        allowed = ROOT / ("provenance" if kind == "markdown_translation" else "notebooks")
        check(target.is_relative_to(allowed), "Annotation exception outside its allowed directory")
        check(sha(target) == item["current_sha256"], f"Reviewed annotation file changed: {name}")
        digest = notebook_digest(json.loads(target.read_text()),
                                 exclude_markdown_source=kind == "markdown_translation")
        check(digest == item["unchanged_content_sha256"],
              f"Notebook code, outputs or other protected content changed: {name}")
        records[name] = item
    return records


def validate_static():
    modules = {p.stem for p in (ROOT / "src").glob("*.py")}
    imports = set()
    py_count = 0
    notebook_count = 0
    for folder in ["src", "scripts", "formula_appendix"]:
        for path in (ROOT / folder).glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            py_count += 1
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
    # Import project modules only. PySR is lazy-loaded inside search workers.
    for name in sorted(modules):
        __import__(name)
    for path in (ROOT / "notebooks").rglob("*.ipynb"):
        nb = json.loads(path.read_text())
        check(nb.get("nbformat") == 4, f"Invalid notebook: {path}")
        ids = [c.get("id") for c in nb["cells"]]
        check(None not in ids and len(ids) == len(set(ids)), f"Invalid cell IDs: {path}")
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                ast.parse(code, filename=f"{path}:cell-{i}")
                check("/Users/novwin/" not in code, f"Machine-specific notebook path: {path}:{i}")
                check("/scratch/j96317yn/" not in code, f"Machine-specific notebook path: {path}:{i}")
        notebook_count += 1
    shell_count = 0
    for path in (ROOT / "cluster").iterdir():
        if path.suffix in {".slurm", ".sh"}:
            subprocess.run(["bash", "-n", str(path)], check=True)
            check("/scratch/${USER}/NotebookCT3" not in path.read_text(), f"Old Slurm root: {path}")
            shell_count += 1
    from ct3_common import build_paths
    for directory in [ROOT, ROOT / "notebooks", ROOT / "notebooks/19_One_Time_Final_50Case_Locked_149", ROOT / "scripts"]:
        paths = build_paths(directory)
        check(paths.package_root == ROOT, f"Root discovery failed: {directory}")
        check(paths.case_dir == ROOT / "FE_Results_Cases_All", "Unexpected default data path")
    return {"python_files_parsed": py_count, "project_modules_imported": len(modules),
            "notebooks_parsed": notebook_count, "shell_files_parsed": shell_count,
            "root_discovery_locations_tested": 4}


def validate_copies(full):
    with (ROOT / "provenance/source_file_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    annotations = validate_annotation_edits()
    verified = 0
    translated_snapshots = []
    for row in rows:
        target = ROOT / row["destination"]
        check(target.is_file(), f"Missing copied artifact: {target}")
        if full or row["role"] != "original_fem_data":
            check(sha(target) == row["delivery_sha256"], f"Delivery hash mismatch: {target}")
            verified += 1
        original = row.get("original_copy")
        if original:
            if original in annotations:
                item = annotations[original]
                check(item["change_kind"] == "markdown_translation", "Invalid source snapshot exception")
                check(item["original_sha256"] == row["source_sha256"], "Original source hash was replaced")
                translated_snapshots.append(original)
            else:
                check(sha(ROOT / original) == row["source_sha256"], f"Original source snapshot changed: {original}")
    return {"source_records": len(rows), "delivery_hashes_checked": verified,
            "raw_sha256_rechecked": full, "reviewed_annotation_records": len(annotations),
            "translated_source_snapshots_not_byte_identical": translated_snapshots}


def validate_frozen(smoke):
    import numpy as np
    import pandas as pd
    from ct3_common import evaluate_prediction_arrays, read_complete_case
    from export_three_spatial_comparison_cases import _load_final_evidence, _post_evaluation_preflight
    from final_locked_149_evaluation import PRIMARY_MODEL, SECONDARY_MODEL, predict_frozen_models, predictor_only_case_summary

    # The export preflight writes scratch checks to a TemporaryDirectory, not the frozen outputs.
    evidence = _load_final_evidence(ROOT)
    state, audit = _post_evaluation_preflight(ROOT, evidence["signature"])
    changed = audit.loc[~audit["unchanged"], "artifact"].tolist()
    check(set(changed) <= {"rotation_1_formal_completion"}, "Predictive frozen artifact changed")
    record = json.loads((ROOT / "outputs/17_v11_four_rotation_formal/iteration_1/pipeline_complete.json").read_text())
    for key, value in {"status": "complete", "iteration": 1, "selected_gain": 0.9,
                       "train_cases": 119, "validation_cases": 15, "internal_test_cases": 15,
                       "final_test_cases_read": 0}.items():
        check(record.get(key) == value, f"Unexpected registration metadata: {key}")
    raw = state["inventory"]
    check(len(raw) == 199, "Raw case count differs from 199")
    check(raw["n_elements_from_line_count"].eq(400360).all(), "Incorrect raw element count")
    manifest = state["manifest"]
    splits = {}
    for rotation, group in manifest.groupby("iteration"):
        counts = group["split"].value_counts().to_dict()
        check(counts == {"train": 119, "validation": 15, "internal_test": 15, "final_test": 50}, "Frozen split changed")
        check(group.groupby("similarity_group")["split"].nunique().max() == 1, "Similarity group crosses splits")
        splits[str(rotation)] = counts
    groupmap = pd.read_csv(ROOT / "shared/frozen_similarity_group_map_199cases.csv")
    sizes = groupmap.groupby("similarity_group").size()
    result = {"raw_cases": 199, "elements_per_case": 400360,
              "total_elements": int(raw["n_elements_from_line_count"].sum()),
              "splits": splits, "similarity_groups": int(len(sizes)),
              "singleton_groups": int((sizes == 1).sum()), "largest_group": int(sizes.max()),
              "frozen_artifacts_checked": len(audit), "non_predictive_metadata_differences": changed,
              "new_final_test_predictions": 0, "new_fits_or_searches": 0}
    if smoke:
        frame, _ = read_complete_case(state["path_by_case"]["case_01"])
        summary = predictor_only_case_summary(frame, "case_01", 1)
        predictions = predict_frozen_models(frame, summary, state["states"], state["mean_model"],
                                           state["active_iterations"], state["tail_gain"])
        old = pd.read_csv(ROOT / "outputs/18_149case_development_integration/formal_149case_integration/development_candidate_case_metrics.csv.gz")
        comparisons = []
        for model in [PRIMARY_MODEL, SECONDARY_MODEL]:
            current = evaluate_prediction_arrays(frame["sigma_max_principal"].to_numpy(dtype=float), predictions[model])
            prior = old[old["case_id"].eq("case_01") & old["model"].eq(model)].iloc[0]
            metrics = ["rmse", "mae", "r2", "predicted_mean", "predicted_p95", "predicted_p99", "top1pct_hotspot_overlap"]
            differences = {key: abs(float(current[key]) - float(prior[key])) for key in metrics}
            check(max(differences.values()) < 2e-4, f"Relocated prediction mismatch: {model}: {differences}")
            comparisons.append({"model": model, "case_id": "case_01", "elements": len(frame), "absolute_differences": differences})
        result["development_prediction_reproduction"] = comparisons
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-hashes", action="store_true", help="Also SHA-256 all 199 raw FEM files")
    parser.add_argument("--smoke", action="store_true", help="Reproduce frozen predictions on development case_01")
    parser.add_argument("--report", type=Path, help="Write a separate delivery audit, never an experimental result")
    args = parser.parse_args()
    print("[1/3] Syntax, imports and relocated root checks", flush=True)
    report = {"audit_kind": "delivery_validation_not_new_research", "static": validate_static()}
    print("[2/3] Copied file hashes", flush=True)
    report["copies"] = validate_copies(args.full_hashes)
    print("[3/3] Frozen contract, complete raw inventory and optional development prediction", flush=True)
    report["frozen"] = validate_frozen(args.smoke)
    report["current_validation_environment_not_historical_runtime"] = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "sympy", "matplotlib"]}}
    report["status"] = "passed"
    if args.report:
        output = args.report.resolve()
        check(not output.is_relative_to(ROOT / "outputs"), "Do not write delivery reports into research outputs")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
