import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# =============================================================================
# Experiment configuration
# =============================================================================

PARAMETER_NAME = "length"
PARAMETER_VALUES = [20000]
# PARAMETER_VALUES = [800, 1000, 5000, 10000, 20000, 30000, 50000, 70000, 100000]

# PARAMETER_NAME = "width"
# PARAMETER_VALUES = [10, 20, 30]

# PARAMETER_NAME = "friction_coefficient"
# PARAMETER_VALUES = [0.01, 0.03, 0.045, 0.08, 0.1]

LEVEL_VALUES = [0.0]

MODEL_FOLDER = Path("../model")
INPUT_FILE = Path("../input/timeseries_import.csv")

MODELS = [
    "ExampleIDZ.mo",
    "ExampleLinear.mo",
    "Example.mo",
    "ExampleFullSV.mo",
]

MODEL_RUNNER = "example.py"
RESULTS_PLOTTER = "channel_pulse_results.py"


def update_modelica_parameter(
    modelica_file,
    parameter_name,
    new_value,
):
    """
    Update a parameter value in a Modelica file.

    Args:
        modelica_file (str or Path): Path to .mo file
        parameter_name (str): Parameter name (e.g. 'length')
        new_value (str or number): New value (e.g. 'my_length' or 5000)
    """
    path = Path(modelica_file)
    text = path.read_text()

    # Regex to match: parameter_name = something ,
    pattern = rf"(\b{parameter_name}\s*=\s*)([^,;]+)([,;]?)"

    old_value = None

    def replacer(match):
        nonlocal old_value
        old_value = match.group(2)
        return f"{match.group(1)}{new_value}{match.group(3)}"

    new_text, count = re.subn(pattern, replacer, text, count=1)

    if count == 0:
        raise ValueError(f"Keyword '{parameter_name}' not found in {modelica_file}")

    path.write_text(new_text)
    ratio = float(new_value) / float(old_value)
    # ratio = 1

    print(f"Updated '{parameter_name}' in {modelica_file} to '{new_value}', ratio: '{ratio}'")
    return ratio


def scale_modelica_parameter(modelica_file, parameter_name, ratio):
    """
    Update a parameter value in a Modelica file.

    Args:
        modelica_file (str or Path): Path to .mo file
        keyword (str): Parameter name (e.g. 'length')
        new_value (str or number): New value (e.g. 'my_length' or 5000)
    """
    path = Path(modelica_file)
    text = path.read_text()

    # Regex to match: keyword = something ,
    pattern = rf"(\b{parameter_name}\s*=\s*)([^,)]+)([,)]?)"

    old_value = None

    def replacer(match):
        nonlocal old_value
        old_value = match.group(2)
        new_value = float(old_value) * ratio
        return f"{match.group(1)}{new_value}{match.group(3)}"

    new_text, count = re.subn(pattern, replacer, text, count=1)

    if count == 0:
        raise ValueError(f"Keyword '{parameter_name}' not found in {modelica_file}")

    path.write_text(new_text)

    print(f"Updated '{parameter_name}' in {modelica_file} ")


for parameter_value in PARAMETER_VALUES:
    print(f"Running experiment: {PARAMETER_NAME}={parameter_value}")

    for model in MODELS:
        ratio = update_modelica_parameter(
            MODEL_FOLDER / model,
            PARAMETER_NAME,
            parameter_value,
        )

    scale_modelica_parameter(
        MODEL_FOLDER / "ExampleID.mo",
        "Ad",
        ratio,
    )

    if PARAMETER_NAME == "length":
        scale_modelica_parameter(
            MODEL_FOLDER / "ExampleID.mo",
            "Delay_in_hour",
            ratio,
        )

    for level_value in LEVEL_VALUES:
        FILE1 = "example.py"
        FILE2 = "channel_pulse_results.py"
        FILE2_ARG = f"{level_value}_{parameter_value}"

        # -----------------------------
        # Step 1: Read CSV
        # -----------------------------
        df = pd.read_csv(INPUT_FILE)

        # Safety check
        if "Level_H" not in df.columns:
            raise ValueError("Column 'Level_H' not found in CSV")

        # -----------------------------
        # Step 2: Populate Level_H
        # -----------------------------
        df["Level_H"] = level_value

        # Save back to CSV
        df.to_csv(INPUT_FILE, index=False)

        print(f"Set all Level_H values to {level_value}")

        # -----------------------------
        # Step 3: Run file1.py
        # -----------------------------
        subprocess.run([sys.executable, FILE1], check=True)

        print(f"{FILE1} executed successfully")

        # -----------------------------
        # Step 4: Run file2.py with argument
        # -----------------------------
        subprocess.run([sys.executable, FILE2, FILE2_ARG], check=True)

        print(f"{FILE2} executed with argument: {FILE2_ARG}")

        print("=" * 80)
        print(f"Parameter study: {PARAMETER_NAME} = {parameter_value}")
        print("=" * 80)
