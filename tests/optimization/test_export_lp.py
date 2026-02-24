import logging
import os
import re
import tempfile
import unittest
from unittest import mock

import casadi as ca
import numpy as np

from rtctools._internal.casadi_to_lp import (
    build_constraints,
    build_objective,
    sanitize_var_names,
    write_lp_file,
)
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.optimization_problem import OptimizationProblem
from rtctools.optimization.timeseries import Timeseries

from .data_path import data_path

TERMINAL_STATE_PENALTY = 2.0

# Parameters for floating-point normalization in LP file comparison.
FP_NOISE_THRESHOLD = 1e-10
FP_ROUND_DECIMALS = 6


def _normalize_lp(text: str) -> str:
    """Normalize an LP file so that two files representing the same problem compare equal,
    even when floating-point noise differs across platforms or CasADi versions.

    Steps:
    1. Round every numeric literal to ``FP_ROUND_DECIMALS`` decimals and
       collapse values below ``FP_NOISE_THRESHOLD`` to zero.
    2. Format whole numbers as integers (``10.0`` -> ``10``). Values >= 1e15
       are kept as floats because float precision (~15 digits) can make
       non-integers appear whole at that scale.
    3. Remove zero-coefficient terms (``+ 0 <var>`` / ``- 0 <var>``).
    4. Collapse whitespace and strip trailing spaces per line.
    """

    def _round_match(m: re.Match) -> str:
        """Regex callback: round a single numeric literal."""
        val = float(m.group(0))
        if abs(val) < FP_NOISE_THRESHOLD:
            val = 0.0
        val = round(val, FP_ROUND_DECIMALS)
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return str(val)

    text = re.sub(r"-?\d+\.?\d*(?:e[+-]?\d+)?", _round_match, text)
    text = re.sub(r"[+-]\s*0\s+\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip() + "\n"


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class _BaseTestModel(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    """Base test model. Subclasses override class attributes to customize."""

    model_name = "ModelWithInitialLinear"
    has_parameters = True
    objective_factory = None

    def __init__(self, output_folder):
        super().__init__(
            input_folder=data_path(),
            output_folder=output_folder,
            model_name=self.model_name,
            model_folder=data_path(),
        )

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 9)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        if self.has_parameters:
            parameters["u_max"] = 2.0
        return parameters

    def pre(self):
        pass

    def constant_inputs(self, ensemble_member):
        return {"constant_input": Timeseries(self.times(), np.ones(len(self.times())))}

    def seed(self, ensemble_member):
        return {}

    def objective(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        if self.objective_factory:
            return self.objective_factory(xf)
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
        options["export_lp"] = True
        return options


class LinearModel(_BaseTestModel):
    """Linear MILP — LP export should succeed with a General section."""

    def variable_is_discrete(self, variable):
        return variable == "u"


class EnsembleLinearModel(LinearModel):
    """Linear MILP with two ensemble members — state variables should get _m{i} suffixes."""

    @property
    def ensemble_size(self):
        return 2

    def constant_inputs(self, ensemble_member):
        values = np.ones(len(self.times())) if ensemble_member == 0 else np.zeros(len(self.times()))
        return {"constant_input": Timeseries(self.times(), values)}


class NonLinearModel(_BaseTestModel):
    """Non-linear DAE — LP export should raise."""

    model_name = "Model"
    has_parameters = False
    objective_factory = staticmethod(lambda xf: xf**2)


class LinearModelNonAffineObjective(_BaseTestModel):
    """Linear DAE with non-affine objective — LP export should raise."""

    objective_factory = staticmethod(lambda xf: xf**2)


class NonAffineConstraintModel(_BaseTestModel):
    """Linear DAE with non-affine constraint — LP export should raise."""

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [(xf**2, -np.inf, 1.0)]


# ---------------------------------------------------------------------------
# Unit tests for builder functions
# ---------------------------------------------------------------------------


class TestLPBuilders(unittest.TestCase):
    """Unit tests for the pure builder functions in casadi_to_lp."""

    @staticmethod
    def _make_fg(n_vars, f_expr_fn, g_expr_fn):
        """Return (f, g, X, var_names) for hand-crafted CasADi expressions."""
        X = ca.SX.sym("X", n_vars)
        var_names = [f"x{i}" for i in range(n_vars)]
        return f_expr_fn(X), g_expr_fn(X), X, var_names

    def test_build_objective(self):
        """Objective string must include coefficients, constants, and handle edge cases."""
        # Linear with positive constant
        f, _, X, names = self._make_fg(2, lambda X: 2.0 * X[0] + 10.0, lambda X: X[0])
        result = build_objective(f, X, names)
        self.assertIn("2.0", result)
        self.assertIn("x0", result)
        self.assertIn("10.0", result)

        # Negative constant
        f, _, X, names = self._make_fg(1, lambda X: 3.0 * X[0] - 5.0, lambda X: X[0])
        result = build_objective(f, X, names)
        self.assertIn("- 5.0", result)

        # Zero objective must emit explicit '0'
        f, _, X, names = self._make_fg(1, lambda X: ca.SX(0), lambda X: X[0])
        result = build_objective(f, X, names)
        self.assertIn("0", result)

    def test_build_constraints_operators(self):
        """Constraint operator must match the bound type: >=, =, or <=."""
        # Constant LHS (no variable terms) with lower bound → >=
        _, g, X, names = self._make_fg(2, lambda X: X[0], lambda X: ca.SX(5.0))
        result = build_constraints(g, X, [0.0], [np.inf], names)
        self.assertRegex(result.strip(), r"0\s*>=")

        # Equal bounds → =
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: 2.0 * X[0])
        result = build_constraints(g, X, [3.0], [3.0], names)
        self.assertIn("=", result)
        self.assertNotIn(">=", result)
        self.assertNotIn("<=", result)

        # Only upper bound finite → <=
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        result = build_constraints(g, X, [-np.inf], [10.0], names)
        self.assertIn("<=", result)

    def test_build_constraints_no_finite_bound_raises(self):
        """A constraint with neither finite bound must raise ValueError."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        with self.assertRaises(ValueError):
            build_constraints(g, X, [-np.inf], [np.inf], names)

    def test_sanitize_var_names_ensemble(self):
        """Controls shared across members get no suffix; per-member states get _m{i}."""
        # u is a control: same slice in all members (shared)
        # x is a state: different slice per member
        indices = [
            {"u": slice(0, 2), "x": slice(2, 4)},
            {"u": slice(0, 2), "x": slice(4, 6)},
        ]
        names = sanitize_var_names(indices, 6)

        # Controls: no member suffix
        self.assertEqual(names[0], "u__0")
        self.assertEqual(names[1], "u__1")

        # States: per-member suffix
        self.assertEqual(names[2], "x__2_m0")
        self.assertEqual(names[3], "x__3_m0")
        self.assertEqual(names[4], "x__4_m1")
        self.assertEqual(names[5], "x__5_m1")

    def test_write_lp_file_binary_and_general(self):
        """Binary variables (bounds [0,1]) go to Binary section; others go to General."""
        var_names = ["x__0", "y__1", "z__2"]
        discrete = [True, True, False]
        lbx = [0, -5, 0]
        ubx = [1, 10, 100]
        with tempfile.TemporaryDirectory() as output_folder:
            write_lp_file(
                "test.lp",
                "0",
                "  0 = 0",
                "  -inf <= x__0 <= +inf",
                var_names,
                discrete,
                lbx,
                ubx,
                output_folder,
            )
            with open(os.path.join(output_folder, "test.lp")) as f:
                content = f.read()

        self.assertIn("Binary", content)
        self.assertIn("x__0", content.split("Binary")[1].split("End")[0])
        self.assertIn("General", content)
        self.assertIn("y__1", content.split("General")[1].split("Binary")[0])
        self.assertNotIn("z__2", content.split("General")[1])


# ---------------------------------------------------------------------------
# Functional export tests
# ---------------------------------------------------------------------------


class TestExportLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run the linear MILP model once and share results across all tests."""
        logger = logging.getLogger("rtctools")
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            with tempfile.TemporaryDirectory() as output_folder:
                problem = LinearModel(output_folder=output_folder)
                problem.optimize()
                cls._lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
                with open(os.path.join(output_folder, cls._lp_files[0])) as f:
                    cls._lp_content = f.read()
        finally:
            logger.setLevel(original_level)

    def test_lp_export_success(self):
        """Successful export must produce one correctly named file matching the reference."""
        # File naming and location
        self.assertEqual(len(self._lp_files), 1)
        self.assertTrue(self._lp_files[0].startswith("LinearModel_"))
        self.assertFalse(os.path.exists(os.path.join(os.getcwd(), self._lp_files[0])))

        # MILP: General section must list the discrete variable
        self.assertIn("General", self._lp_content)
        general_section = self._lp_content.split("General")[1].split("End")[0]
        self.assertIn("u__", general_section)

        # Content must match reference after FP normalization
        reference_path = os.path.join(data_path(), "reference_linear_model.lp")
        with open(reference_path) as f:
            reference = _normalize_lp(f.read())

        self.assertEqual(
            _normalize_lp(self._lp_content),
            reference,
            f"Normalized LP output does not match {reference_path}. "
            "If the change is intentional, regenerate the reference file.",
        )

    def test_lp_export_ensemble(self):
        """Ensemble export: one LP file, states suffixed _m{i}, controls unsuffixed."""
        logger = logging.getLogger("rtctools")
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            with tempfile.TemporaryDirectory() as output_folder:
                problem = EnsembleLinearModel(output_folder=output_folder)
                problem.optimize()
                lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
                self.assertEqual(len(lp_files), 1)
                with open(os.path.join(output_folder, lp_files[0])) as f:
                    content = f.read()
        finally:
            logger.setLevel(original_level)

        # Controls are shared — no member suffix
        self.assertIn("u__0", content)
        self.assertNotIn("u__0_m0", content)

        # States are per-member — both suffixes must appear
        self.assertIn("_m0", content)
        self.assertIn("_m1", content)

    def test_lp_export_rejects_invalid_problems(self):
        """Export must raise ValueError for non-affine or unsupported problems.

        Each case runs its own optimization (not shared via setUpClass) because
        the error is expected to occur during optimize(), not after.
        """
        self._assert_raises_on_optimize(NonLinearModel, "linear_collocation")
        self._assert_raises_on_optimize(LinearModelNonAffineObjective, "affine", "objective")
        self._assert_raises_on_optimize(NonAffineConstraintModel, "affine", "constraint")

        # Problem without _collint_variable_indices (not a CollocatedIntegratedOptimizationProblem)
        mock_problem = mock.Mock(spec=["_output_folder"])
        x = ca.SX.sym("x", 1)
        with self.assertRaises(ValueError) as ctx:
            OptimizationProblem._export_lp_file(mock_problem, x, x, x, [0], [0], [0], [0], [False])
        self.assertIn("CollocatedIntegratedOptimizationProblem", str(ctx.exception))

    def _assert_raises_on_optimize(self, model_class, *expected_words):
        """Run optimize() on model_class and assert it raises ValueError with expected_words."""
        with tempfile.TemporaryDirectory() as output_folder:
            problem = model_class(output_folder=output_folder)
            with self.assertRaises(ValueError) as ctx:
                problem.optimize()

            error_msg = str(ctx.exception).lower()
            for word in expected_words:
                self.assertIn(word, error_msg)

            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 0)
