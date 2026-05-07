"""Regenerate tests/optimization/data/reference_linear_model.lp.

Run this script whenever a change to casadi_to_lp.py or the LinearModel
intentionally alters the LP output format:

    uv run python -m tests.optimization.regenerate_lp_reference

The script runs LinearModel.optimize() with export_lp=True, captures the
produced file, and overwrites the reference. Commit the updated reference
together with the code change.
"""

import os
import tempfile

from tests.optimization.test_export_lp import LinearModel

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_REFERENCE_PATH = os.path.join(_DATA_DIR, "reference_linear_model.lp")


def regenerate():
    with tempfile.TemporaryDirectory() as output_folder:
        problem = LinearModel(output_folder=output_folder)
        problem.optimize()
        lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
        assert len(lp_files) == 1, f"Expected 1 LP file, got {lp_files}"
        with open(os.path.join(output_folder, lp_files[0])) as f:
            content = f.read()

    with open(_REFERENCE_PATH, "w", newline="\n") as f:
        f.write(content)

    print(f"Regenerated {_REFERENCE_PATH} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    regenerate()
