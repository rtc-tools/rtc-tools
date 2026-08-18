import ast
import re

import numpy as np


def parse_channel_block(text):
    """
    Finds the first occurrence of '<model_name> Channel(...)'
    and returns:
        - model_name (word before Channel)
        - parameter dictionary
        - start index
        - end index
    """

    # Capture word before Channel + parameters
    pattern = r"(\w+)\s+Channel\s*\((.*?)\)\s*annotation"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        raise ValueError("No Channel block found.")

    model_name = match.group(1)
    param_block = match.group(2)

    start = match.start(2)
    end = match.end(2)

    param_dict = {}

    lines = param_block.split(",")

    for line in lines:
        line = line.strip()

        # Remove inline comments
        line = re.sub(r"//.*", "", line).strip()

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass

            param_dict[key] = value

    return model_name, param_dict, start, end


def rebuild_channel_block(original_text, param_dict, start, end):
    """
    Replaces parameter block in original text
    """

    new_params = []
    for key, value in param_dict.items():
        if isinstance(value, str) and not value.startswith('"'):
            new_value = value
        else:
            new_value = repr(value)
        new_params.append(f"    {key} = {new_value}")

    new_param_block = ",\n".join(new_params)

    return original_text[:start] + "\n" + new_param_block + "\n  " + original_text[end:]


def main():
    width = 14.0
    length = 30000.0
    depth = 1.0
    discharge = 20.0
    slope = 0.002875

    input_file_list = ["ExampleIDZ", "ExampleID", "ExampleLinear", "Example", "ExampleFullSV"]

    for model_name in input_file_list:
        INPUT_FILE = "..\\model\\" + model_name + ".mo"
        OUTPUT_FILE = INPUT_FILE

        H_b_up = length * slope
        with open(INPUT_FILE) as f:
            content = f.read()

        model_name, param_dict, start, end = parse_channel_block(content)
        if "IDZ" in model_name:
            param_dict["width"] = width
            param_dict["length"] = length
            param_dict["H_b_up"] = H_b_up
            param_dict["H_b_down"] = 0.0
            param_dict["H_nominal"] = depth
            param_dict["Q_nominal"] = discharge
        elif "LinearisedSV" in model_name:
            param_dict["width"] = width
            param_dict["length"] = length
            param_dict["H_b_up"] = H_b_up
            param_dict["H_b_down"] = 0.0
            param_dict["H_nominal"] = depth
            param_dict["H_nominal_down"] = depth
            param_dict["Q_nominal"] = discharge
        elif "HomotopicLinear" in model_name:
            param_dict["width_up"] = width
            param_dict["width_down"] = width
            param_dict["length"] = length
            param_dict["H_b_up"] = H_b_up
            param_dict["H_b_down"] = 0.0
            param_dict["uniform_nominal_depth"] = depth
            param_dict["Q_nominal"] = discharge
        elif "Linear" in model_name:
            param_dict["width_up"] = width
            param_dict["width_down"] = width
            param_dict["length"] = length
            param_dict["H_b_up"] = H_b_up
            param_dict["H_b_down"] = 0.0
            param_dict["uniform_nominal_depth"] = depth
            param_dict["Q_nominal"] = discharge
        elif "ID" in model_name:
            param_dict["Ad"] = width * length
            param_dict["Delay_in_hour"] = length / (
                discharge / (width * depth) + np.sqrt(9.81 * depth)
            )

        print(model_name)

        print("\nFound Channel parameters:\n")
        for k, v in param_dict.items():
            print(f"{k} = {v}")

        updated_content = rebuild_channel_block(content, param_dict, start, end)

        with open(OUTPUT_FILE, "w") as f:
            f.write(updated_content)

        print(f"\nUpdated file written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
