"""Inspect stored final results; no model execution or output writes."""
import argparse
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", action="store_true")
    args = parser.parse_args()
    final = ROOT / "outputs/19_one_time_final_50case_locked_149/formal_final_50case"
    print(json.dumps(json.loads((final / "final_evaluation_complete.json").read_text()), indent=2))
    for name in ["final_50case_scope_metrics.csv", "final_primary_research_reporting_gates.csv", "final_spatial_case_selection.csv"]:
        print(f"\n{name}\n")
        print(pd.read_csv(final / name).to_string(index=False))
    print("\nThesis PNG figures:")
    for path in sorted((final / "figures_for_thesis").glob("*.png")):
        print(path.relative_to(ROOT))
    if args.formula:
        formula = ROOT / "outputs/18_149case_development_integration/formal_149case_integration/locked_model_formula.txt"
        print("\n" + formula.read_text())


if __name__ == "__main__":
    main()
