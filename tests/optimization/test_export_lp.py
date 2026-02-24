import logging
import os
import re
import tempfile
import unittest
from unittest import mock

import casadi as ca
import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from .data_path import data_path

# Penalty weight for the linear objective: keeps the objective linear (required for LP export)
# while giving the optimizer a non-trivial cost to minimise.
TERMINAL_STATE_PENALTY = 2.0

# Coefficients below this threshold are treated as zero during normalization,
# absorbing floating-point noise from CasADi's symbolic differentiation.
FP_NOISE_THRESHOLD = 1e-10

# Number of decimal places to round coefficients to during normalization.
FP_ROUND_DECIMALS = 6


def _normalize_lp(text: str) -> str:
    """Normalize an LP file for comparison by rounding coefficients and removing zero terms.

    1. Round all numeric literals to ``FP_ROUND_DECIMALS`` decimals.
    2. Collapse values below ``FP_NOISE_THRESHOLD`` to exact zero.
    3. Use integer representation for whole numbers (e.g. ``10.0`` -> ``10``).
    4. Remove ``+ 0 <var>`` / ``- 0 <var>`` terms produced by FP noise.
    5. Collapse runs of whitespace to single spaces and strip trailing whitespace.
    """

    def _round_match(m: re.Match) -> str:
        val = float(m.group(0))
        if abs(val) < FP_NOISE_THRESHOLD:
            val = 0.0
        val = round(val, FP_ROUND_DECIMALS)
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return str(val)

    text = re.sub(r"-?\d+\.?\d*(?:e[+-]?\d+)?", _round_match, text)
    # Remove terms with zero coefficients: "- 0 <varname>" or "+ 0 <varname>"
    text = re.sub(r"[+-]\s*0\s+\S+", "", text)
    # Collapse multiple spaces into one and strip trailing whitespace per line
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip() + "\n"


class _BaseTestModel(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    """Base class for LP export test models with common behavior.

    Subclasses override class attributes to customize behavior:
    - MODEL_NAME: Name of Modelica model to load (default: "ModelWithInitialLinear")
    - HAS_PARAMETERS: Whether to set u_max=2.0 (default: True)
    - OBJECTIVE_FACTORY: Callable(xf) -> objective expression (default: linear penalty)
    """

    MODEL_NAME = "ModelWithInitialLinear"
    HAS_PARAMETERS = True
    OBJECTIVE_FACTORY = None

    def __init__(self, output_folder):
        super().__init__(
            input_folder=data_path(),
            output_folder=output_folder,
            model_name=self.MODEL_NAME,
            model_folder=data_path(),
        )

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 11)

    def parameters(self, ensemble_member):
        if not self.HAS_PARAMETERS:
            return super().parameters(ensemble_member)
        parameters = super().parameters(ensemble_member)
        parameters["u_max"] = 2.0
        return parameters

    def pre(self):
        pass

    def constant_inputs(self, ensemble_member):
        return {"constant_input": Timeseries(self.times(), np.ones(11))}

    def seed(self, ensemble_member):
        return {}

    def objective(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        if self.OBJECTIVE_FACTORY:
            return self.OBJECTIVE_FACTORY(xf)
        return TERMINAL_STATE_PENALTY * xf

    def constraints(self, ensemble_member):
        return []

    def compiler_options(self):
        options = super().compiler_options()
        options["cache"] = False
        options["library_folders"] = []
        return options

    def solver_options(self):
        options = super().solver_options()
        options["export_model"] = True
        return options


class LinearModel(_BaseTestModel):
    """Linear problem (ModelWithInitialLinear) — LP export should succeed."""

    pass


class ConstantConstraintModel(_BaseTestModel):
    """Linear problem with a constant constraint (no variable terms) — LP export should succeed.

    Tests the edge case where constraint expression is a constant (c_str is empty and needs
    to be set to "0" in the LP file).
    """

    def constraints(self, ensemble_member):
        return [(5.0, 0.0, np.inf)]


class ConstantObjectiveModel(_BaseTestModel):
    """Linear problem with a constant term in the objective — LP export should succeed.

    Tests that constant terms in the objective are correctly exported to the LP file.
    """

    OBJECTIVE_FACTORY = staticmethod(lambda xf: 2.0 * xf + 10.0)


class NonLinearModel(_BaseTestModel):
    """Non-linear DAE model — LP export should raise (linear_collocation=False)."""

    MODEL_NAME = "Model"
    HAS_PARAMETERS = False
    OBJECTIVE_FACTORY = staticmethod(lambda xf: xf**2)


class LinearModelNonAffineObjective(_BaseTestModel):
    """Linear DAE with non-affine objective — LP export should raise."""

    OBJECTIVE_FACTORY = staticmethod(lambda xf: xf**2)


class NonAffineConstraintModel(_BaseTestModel):
    """Linear collocation problem with a non-affine (quadratic) constraint.

    LP export should raise.
    """

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [(xf**2, -np.inf, 1.0)]


class TestExportLP(unittest.TestCase):
    def setUp(self) -> None:
        """Save the current logger state and suppress rtctools warnings during tests."""
        self.logger = logging.getLogger("rtctools")
        self.original_level = self.logger.level
        # Suppress INFO/DEBUG logs during optimization to keep test output clean
        self.logger.setLevel(logging.WARNING)

    def tearDown(self) -> None:
        """Restore the original logger state after tests."""
        self.logger.setLevel(self.original_level)

    def _assert_no_lp_files_created(self, output_folder, reason=""):
        """Helper method: verify no LP files were created (for failure test cases).

        Args:
            output_folder: Directory to check for .lp files
            reason: Brief description of why no file should exist (for assertion message)
        """
        lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
        self.assertEqual(
            len(lp_files),
            0,
            f"No .lp file should be created when export fails. Reason: {reason}",
        )

    def test_lp_file_creation_and_naming(self):
        """
        When export_model=True on a linear problem, exactly one timestamped .lp file
        named after the problem class should be created in the output folder (not the
        current working directory). File naming should follow the pattern
        <ClassName>_<unix_timestamp>.lp.
        """
        with tempfile.TemporaryDirectory() as output_folder:
            problem = LinearModel(output_folder=output_folder)
            problem.optimize()

            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 1, "Expected exactly one .lp file in output folder")

            self.assertTrue(
                lp_files[0].startswith("LinearModel_"),
                f"Expected filename to start with class name 'LinearModel_', got: {lp_files[0]}",
            )

            # Verify the file was written to the output folder, not the working directory
            self.assertFalse(
                os.path.exists(os.path.join(os.getcwd(), lp_files[0])),
                "LP file should not be written to the current working directory",
            )

    def test_lp_content_matches_reference(self):
        """
        Compare the exported LP file against a known-good reference, with all numeric
        coefficients rounded to absorb floating-point noise from CasADi's symbolic
        differentiation. This catches regressions in variable naming, constraint
        structure, bounds, and objective formulation that the structural test above
        would miss.
        """
        reference_path = os.path.join(data_path(), "reference_linear_model.lp")
        with open(reference_path) as f:
            reference = _normalize_lp(f.read())

        with tempfile.TemporaryDirectory() as output_folder:
            problem = LinearModel(output_folder=output_folder)
            problem.optimize()

            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            lp_path = os.path.join(output_folder, lp_files[0])
            with open(lp_path) as f:
                actual = _normalize_lp(f.read())

        self.assertEqual(
            actual,
            reference,
            "Normalized LP output does not match reference file "
            f"({reference_path}). If the change is intentional, regenerate the "
            "reference file.",
        )

    def test_lp_export_raises_for_nonlinear_problem(self):
        """
        When export_model=True on a non-linear problem (linear_collocation is False),
        optimize() must raise a ValueError with a descriptive message rather than silently
        returning without solving. No .lp file should be written to the output folder.
        """
        with tempfile.TemporaryDirectory() as output_folder:
            problem = NonLinearModel(output_folder=output_folder)
            with self.assertRaises(ValueError) as ctx:
                problem.optimize()

            self.assertIn(
                "linear collocation",
                str(ctx.exception).lower(),
                "ValueError message should mention 'linear collocation'",
            )

            self._assert_no_lp_files_created(output_folder, "non-linear problem")

    def test_lp_export_raises_for_non_collocation_problem(self):
        """
        When export_model=True on a problem that does not inherit from
        CollocatedIntegratedOptimizationProblem (lacks linear_collocation attribute),
        _export_lp_file() must raise a ValueError indicating LP export requires a
        CollocatedIntegratedOptimizationProblem subclass.
        """
        # Create a minimal mock object that lacks the linear_collocation attribute
        # Configure it to return None for getattr(self, "linear_collocation", None)
        mock_problem = mock.Mock(spec=["_output_folder", "ensemble_size"])
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_problem._output_folder = temp_dir
            mock_problem.ensemble_size = 1  # Pass ensemble size check

            # Import the function to test it directly
            from rtctools.optimization.optimization_problem import OptimizationProblem

            # Create a dummy expand_f_g function
            X = ca.SX.sym("X", 1)
            expand_f_g = ca.Function("f_g", [X], [X, X]).expand()

            # Call _export_lp_file on the mock problem with no linear_collocation attribute
            # The getattr call will return None since linear_collocation is not in the spec
            with self.assertRaises(ValueError) as ctx:
                OptimizationProblem._export_lp_file(
                    mock_problem, expand_f_g, [0], [0], [0], [0], [False]
                )

            error_msg = str(ctx.exception)
            self.assertIn(
                "CollocatedIntegratedOptimizationProblem",
                error_msg,
                "ValueError should mention CollocatedIntegratedOptimizationProblem",
            )

    def test_lp_export_raises_for_non_affine_objective(self):
        """
        When export_model=True on a problem with a non-affine objective function,
        optimize() must raise a ValueError with a descriptive message about affinity
        requirement. Even though linear_collocation=True, a non-affine user-provided
        objective makes LP export impossible.
        """
        self._test_lp_export_raises_for_non_affine("objective", LinearModelNonAffineObjective)

    def test_lp_export_raises_for_non_affine_constraint(self):
        """
        When export_model=True on a problem with a non-affine constraint,
        optimize() must raise a ValueError with a descriptive message about affinity
        requirement. Even though linear_collocation=True, a non-affine user-provided
        constraint makes LP export impossible.
        """
        self._test_lp_export_raises_for_non_affine("constraint", NonAffineConstraintModel)

    def _test_lp_export_raises_for_non_affine(self, violation_type, model_class):
        """Helper method to test non-affine problem rejection.

        Args:
            violation_type: "objective" or "constraint"
            model_class: Model class with the violation
        """
        with tempfile.TemporaryDirectory() as output_folder:
            problem = model_class(output_folder=output_folder)
            with self.assertRaises(ValueError) as ctx:
                problem.optimize()

            error_msg = str(ctx.exception).lower()
            self.assertIn(
                "affine",
                error_msg,
                "ValueError message should mention 'affine'",
            )
            self.assertIn(
                violation_type,
                error_msg,
                f"ValueError message should mention '{violation_type}'",
            )

            self._assert_no_lp_files_created(output_folder, f"non-affine {violation_type}")

    def test_lp_export_prevents_file_collision(self):
        """
        When export_model=True and an LP file with the same name already exists in the
        output folder, the second export attempt must raise FileExistsError to prevent
        silent overwrites. This can occur in rapid-fire exports or parallel execution.
        """
        with tempfile.TemporaryDirectory() as output_folder:
            # First export
            problem1 = LinearModel(output_folder=output_folder)
            problem1.optimize()

            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 1, "First optimization should produce one LP file")

            # Extract timestamp from first filename using robust regex pattern
            first_lp_name = lp_files[0]
            match = re.match(r"(\w+)_(\d+)\.lp$", first_lp_name)
            self.assertIsNotNone(
                match,
                f"LP filename doesn't match expected pattern "
                f"'<ClassName>_<timestamp>.lp': {first_lp_name}",
            )

            class_name, timestamp_str = match.groups()
            self.assertEqual(
                class_name,
                "LinearModel",
                f"Expected LinearModel in filename, got: {class_name}",
            )
            first_timestamp_ns = int(timestamp_str)

            # Second export with same timestamp should raise FileExistsError
            problem2 = LinearModel(output_folder=output_folder)
            with mock.patch("rtctools.optimization.optimization_problem.time_ns") as mock_time_ns:
                mock_time_ns.return_value = first_timestamp_ns
                with self.assertRaises(FileExistsError) as ctx:
                    problem2.optimize()

                self.assertIn(
                    "already exists",
                    str(ctx.exception).lower(),
                    "FileExistsError message should mention that the file already exists",
                )

    def test_lp_export_handles_constant_constraint(self):
        """Verify that constant constraints (with no variable terms) are properly exported.

        This test specifically validates the code path where a constraint has only
        a constant expression. In the LP file, this means the constraint string c_str
        should be set to "0" (representing the left-hand side of the constraint).
        """
        with tempfile.TemporaryDirectory() as output_folder:
            problem = ConstantConstraintModel(output_folder=output_folder)
            problem.optimize()

            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 1, "Expected exactly one .lp file in output folder")

            # Verify the file was written and has content
            lp_path = os.path.join(output_folder, lp_files[0])
            with open(lp_path) as f:
                lp_content = f.read()
            self.assertGreater(len(lp_content), 0, "LP file should have content")

    def test_lp_export_includes_constant_objective_term(self):
        """Verify that constant terms in the objective are included in the LP export.

        Constant terms don't affect the optimal solution but do affect the objective
        value. For debugging and validation, the exported LP should match what's
        actually passed to the solver, including any constant terms.
        """
        with tempfile.TemporaryDirectory() as output_folder:
            problem = ConstantObjectiveModel(output_folder=output_folder)
            problem.optimize()

            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 1, "Expected exactly one .lp file in output folder")

            lp_path = os.path.join(output_folder, lp_files[0])
            with open(lp_path) as f:
                lp_content = f.read()

            # Extract the objective line (between "Minimize" and "Subject To")
            minimize_idx = lp_content.find("Minimize")
            subject_to_idx = lp_content.find("Subject To")
            self.assertGreater(minimize_idx, -1, "LP file should contain 'Minimize'")
            self.assertGreater(subject_to_idx, minimize_idx, "LP file should contain 'Subject To'")

            objective_section = lp_content[minimize_idx:subject_to_idx].strip()

            # The objective should contain the constant term "10" or "10.0"
            self.assertIn("10", objective_section, "Objective should include constant term 10")
