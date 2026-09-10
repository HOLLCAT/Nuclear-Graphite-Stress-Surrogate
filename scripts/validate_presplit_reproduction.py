"""Replay historical grouping cells into temporary outputs, never the frozen split."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = json.loads((ROOT / "notebooks/PreSplit_Grouping_and_Frozen_Manifest.ipynb").read_text())
    original_cwd = Path.cwd()
    captured = io.StringIO()
    try:
        os.chdir(ROOT)
        with tempfile.TemporaryDirectory(prefix="graphite-presplit-check-") as temporary:
            namespace = {"__name__": "__main__"}
            with contextlib.redirect_stdout(captured):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    tree = ast.parse("".join(cell["source"]))
                    for node in tree.body:
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OUTPUT_DIR" for t in node.targets):
                            node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(temporary)], keywords=[])
                    ast.fix_missing_locations(tree)
                    exec(compile(tree, f"presplit:cell-{index}", "exec"), namespace)
                expected = pd.read_csv(ROOT / "shared/frozen_case_split_manifest_199cases.csv")
                actual = pd.read_csv(Path(temporary) / "repeated_case_split_manifest_199cases.csv")
                columns = ["iteration", "case_id", "split", "similarity_group"]
                keys = ["iteration", "case_id"]
                pd.testing.assert_frame_equal(expected[columns].sort_values(keys).reset_index(drop=True),
                                              actual[columns].sort_values(keys).reset_index(drop=True))
                groups = pd.read_csv(Path(temporary) / "pre_split_similarity_group_map.csv")
                frozen_groups = pd.read_csv(ROOT / "shared/frozen_similarity_group_map_199cases.csv")
                cols = ["case_id", "similarity_group"]
                pd.testing.assert_frame_equal(groups[cols].sort_values("case_id").reset_index(drop=True),
                                              frozen_groups[cols].sort_values("case_id").reset_index(drop=True))
            print(json.dumps({"status": "passed", "group_rows": len(groups),
                              "split_rows": len(actual), "groups_and_four_splits_reproduced": True,
                              "output_location": "temporary_directory_removed",
                              "training_run": False, "final_predictions_run": False}, indent=2))
    except Exception:
        print(captured.getvalue()[-5000:])
        raise
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
