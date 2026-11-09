import fnmatch
import inspect
import os
import subprocess
import sys
from unittest import TestCase


class ExamplesCollection:
    def __init__(self):
        self.errors_detected = {}
        for example_folder in self.examples_folders:
            for example in self.subexamples(example_folder):
                # initialize with failures
                example_path = os.path.join(self.examples_path, example_folder, "src", example)
                self.errors_detected[example_path] = True

    def local_function(self):
        pass

    @property
    def examples_path(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(inspect.getsourcefile(self.local_function))),
            "..",
            "..",
            "examples",
        )

    @property
    def examples_folders(self):
        return [f.name for f in os.scandir(self.examples_path) if f.is_dir()]

    def subexamples(self, example_folder):
        example_folder = os.path.join(self.examples_path, example_folder, "src")
        return [f for f in os.listdir(example_folder) if fnmatch.fnmatch(f, "*example*.py")]


class TestExamples(TestCase):
    """
    src subfolders of examples in the example folder are searched for files
    containing 'example' in their filename.
    """

    def run_examples(self, ec):
        env = sys.executable
        for example_path in ec.errors_detected.keys():
            try:
                output = subprocess.check_output([env, example_path], text=True)
            except Exception:
                ec.errors_detected[example_path] = True
            else:
                # IPOPT solves a problem with more equality constraints than free variables
                # only through a fallback, after reporting this.
                ec.errors_detected[example_path] = "Too few degrees of freedom" in output

    def test_examples(self):
        ec = ExamplesCollection()

        self.run_examples(ec)

        for example_path, error_detected in ec.errors_detected.items():
            path = os.path.normpath(example_path)
            file = os.path.basename(path)
            folder = os.path.relpath(example_path, ec.examples_path).split(os.sep)[0]
            if error_detected:
                print(f"An error occured while running '{file}' in example folder '{folder}'.")
            else:
                print(f"No errors occured while running '{file}' in example folder '{folder}'.")

        self.assertFalse(any(ec.errors_detected.values()))
