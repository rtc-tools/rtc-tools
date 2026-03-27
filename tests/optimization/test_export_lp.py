import contextlib
import logging
import os
import re
import tempfile
import unittest
from unittest import mock

import casadi as ca
import numpy as np
from pymoca.backends.casadi.alias_relation import AliasRelation

from rtctools._internal.casadi_to_lp import (
    build_bounds,
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


@contextlib.contextmanager
def _suppress_info_logging():
    """Suppress INFO/DEBUG rtctools output during optimize() calls in tests.

    Sets the level to ERROR so that INFO and DEBUG solver chatter is hidden
    while still allowing test assertions on WARNING-level messages.
    """
    logger = logging.getLogger("rtctools")
    original_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(original_level)


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


# Test models


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


class _BaseNoModelicaModel(CollocatedIntegratedOptimizationProblem):
    """Base for models with no DAE equations and a single discrete path variable."""

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 5)

    def pre(self):
        pass

    def seed(self, ensemble_member):
        return {}

    def solver_options(self):
        options = super().solver_options()
        options["export_lp"] = True
        return options

    @property
    def path_variables(self):
        return super().path_variables + [ca.MX.sym("pump_on")]

    def variable_is_discrete(self, variable):
        return variable == "pump_on" or super().variable_is_discrete(variable)

    def bounds(self):
        bounds = super().bounds()
        bounds["pump_on"] = (0.0, 1.0)
        return bounds

    def objective(self, ensemble_member):
        pump_on = self.state_at("pump_on", self.times()[-1], ensemble_member=ensemble_member)
        return pump_on

    def constraints(self, ensemble_member):
        return []

    def path_constraints(self, ensemble_member):
        return []


class EmptyModelicaModel(ModelicaMixin, _BaseNoModelicaModel):
    """Empty Modelica file — linear_collocation is never set by the DAE linearity check."""

    def __init__(self, output_folder):
        super().__init__(
            input_folder=data_path(),
            output_folder=output_folder,
            model_name="EmptyModel",
            model_folder=data_path(),
        )

    def compiler_options(self):
        options = super().compiler_options()
        options["cache"] = False
        options["library_folders"] = []
        return options

    def variable_is_discrete(self, variable):
        return variable == "pump_on" or super().variable_is_discrete(variable)


class NoModelicaModel(_BaseNoModelicaModel):
    """No Modelica file — DAE abstract methods implemented as empty stubs."""

    def __init__(self, output_folder):
        self._dae_variables = {
            "states": [],
            "derivatives": [],
            "algebraics": [],
            "control_inputs": [],
            "constant_inputs": [],
            "lookup_tables": [],
            "parameters": [],
            "time": [ca.MX.sym("time")],
        }
        self._alias_relation = AliasRelation()
        super().__init__(input_folder=data_path(), output_folder=output_folder)

    @property
    def alias_relation(self):
        return self._alias_relation

    @property
    def dae_residual(self):
        return ca.MX()

    @property
    def dae_variables(self):
        return self._dae_variables


# Unit tests for builder functions


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
        # Linear with positive constant — must not start with '+'
        f, _, X, names = self._make_fg(2, lambda X: 2.0 * X[0] + 10.0, lambda X: X[0])
        result = build_objective(f, X, names)
        self.assertNotRegex(result, r"^\s*\+")
        self.assertRegex(result, r"2\b")  # coefficient 2 (:.15g strips trailing zeros)
        self.assertIn("x0", result)
        self.assertRegex(result, r"\b10\b")

        # Negative constant
        f, _, X, names = self._make_fg(1, lambda X: 3.0 * X[0] - 5.0, lambda X: X[0])
        result = build_objective(f, X, names)
        self.assertRegex(result, r"-\s*5\b")

        # Zero objective must emit explicit '0'
        f, _, X, names = self._make_fg(1, lambda X: ca.SX(0), lambda X: X[0])
        result = build_objective(f, X, names)
        self.assertIn("0", result)

        # Positive constant-only objective must not emit a leading '+' (invalid LP syntax)
        f, _, X, names = self._make_fg(1, lambda X: ca.SX(5.0), lambda X: X[0])
        result = build_objective(f, X, names)
        self.assertNotRegex(result, r"^\s*\+")
        self.assertRegex(result, r"\b5\b")

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

    def test_build_constraints_range(self):
        """A constraint with both bounds finite and lb < ub must emit two lines (>= and <=)."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        result = build_constraints(g, X, [1.0], [3.0], names)
        self.assertIn(">=", result)
        self.assertIn("<=", result)
        # Both lines must appear with correct RHS (b_i=0 here, so RHS = bound - 0)
        self.assertRegex(result, r">=\s*1")
        self.assertRegex(result, r"<=\s*3")

    def test_build_constraints_no_finite_bound_warns_and_emits(self):
        """A vacuous constraint (no finite bound) must emit a warning and write '<= +Inf'."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        with self.assertLogs("rtctools", level="WARNING") as log_ctx:
            result = build_constraints(g, X, [-np.inf], [np.inf], names)
        self.assertTrue(
            any(
                "vacuous" in msg.lower() or "no finite bound" in msg.lower()
                for msg in log_ctx.output
            )
        )
        self.assertIn("+Inf", result)

    def test_sanitize_var_names_ensemble(self):
        """Controls shared across members get no suffix; per-member states get __m{i}.
        Time-indexed variables (>1 slot) use __t{local_index}, where local_index is the
        0-based position within the variable's own slot sequence (i.e. the time step).
        Scalar variables (exactly 1 slot) receive no __t suffix."""
        # u is a control: same slots in all members (shared, 2 time steps)
        # x is a state: different slots per member (2 time steps each)
        indices = [
            {"u": [0, 1], "x": [2, 3]},
            {"u": [0, 1], "x": [4, 5]},
        ]
        names = sanitize_var_names(indices, 6)

        # Controls: time-indexed, no member suffix
        self.assertEqual(names[0], "u__t0")
        self.assertEqual(names[1], "u__t1")

        # States: time-indexed, per-member suffix
        self.assertEqual(names[2], "x__t0__m0")
        self.assertEqual(names[3], "x__t1__m0")
        self.assertEqual(names[4], "x__t0__m1")
        self.assertEqual(names[5], "x__t1__m1")

        # p occupies 1 slot (scalar, shared): no __t suffix, just the name
        # Verify slot 1 is still correctly named (not affected by p's scalar handling)
        indices_scalar = [{"p": [0], "x": [1, 2]}, {"p": [0], "x": [3, 4]}]
        names_scalar = sanitize_var_names(indices_scalar, 5)
        self.assertEqual(names_scalar[0], "p")  # scalar: no __t suffix
        self.assertNotIn("__t", names_scalar[0])  # double-check: no time index
        self.assertIn("__t", names_scalar[1])  # x is time-indexed

    def test_write_lp_file_no_overwrite(self):
        """write_lp_file() must not overwrite an existing file; it appends a counter instead."""
        var_names = ["x__0"]
        discrete = [False]
        lbx = [0.0]
        ubx = [1.0]
        bounds_str, binary_vars, general_vars = build_bounds(var_names, lbx, ubx, discrete)
        with tempfile.TemporaryDirectory() as output_folder:
            for _ in range(2):
                write_lp_file(
                    "test.lp",
                    "0",
                    "  0 = 0",
                    bounds_str,
                    binary_vars,
                    general_vars,
                    output_folder,
                )
            files = {f for f in os.listdir(output_folder) if f.endswith(".lp")}
            self.assertEqual(files, {"test.lp", "test_1.lp"})

    def test_write_lp_file_binary_and_general(self):
        """Binary variables (bounds [0,1]) go to Binary section; others go to General."""
        var_names = ["x__0", "y__1", "z__2"]
        discrete = [True, True, False]
        lbx = [0.0, -5.0, 0.0]
        ubx = [1.0, 10.0, 100.0]
        bounds_str, binary_vars, general_vars = build_bounds(var_names, lbx, ubx, discrete)
        with tempfile.TemporaryDirectory() as output_folder:
            write_lp_file(
                "test.lp",
                "0",
                "  0 = 0",
                bounds_str,
                binary_vars,
                general_vars,
                output_folder,
            )
            with open(os.path.join(output_folder, "test.lp")) as f:
                content = f.read()

        self.assertIn("Binary", content)
        self.assertIn("x__0", content.split("Binary")[1].split("End")[0])
        self.assertIn("General", content)
        self.assertIn("y__1", content.split("General")[1].split("Binary")[0])
        self.assertNotIn("z__2", content.split("General")[1])

    def test_write_lp_file_no_bounds_section_for_pure_binary(self):
        """write_lp_file must omit the Bounds section entirely when bounds_str is empty."""
        with tempfile.TemporaryDirectory() as output_folder:
            write_lp_file("test.lp", "0", "  0 = 0", "", ["b__0"], [], output_folder)
            with open(os.path.join(output_folder, "test.lp")) as f:
                content = f.read()
        self.assertNotIn("Bounds", content)
        self.assertIn("Binary", content)

    def test_build_bounds_returns_tuple_with_classified_vars(self):
        """build_bounds must classify binary/general vars and emit correct bound strings."""
        var_names = ["b__0", "g__1", "c__2"]
        discrete = [True, True, False]
        lbx = [0.0, -5.0, 0.0]
        ubx = [1.0, 10.0, 100.0]
        bounds_str, binary_vars, general_vars = build_bounds(var_names, lbx, ubx, discrete)

        # Binary: discrete with [0,1] bounds → no bounds entry, listed in binary_vars
        self.assertIn("b__0", binary_vars)
        self.assertNotIn("b__0", bounds_str)

        # General: discrete outside [0,1] bounds → bounds entry, listed in general_vars
        self.assertIn("g__1", general_vars)
        self.assertIn("g__1", bounds_str)

        # Continuous: not in either list, bounds entry present
        self.assertNotIn("c__2", binary_vars)
        self.assertNotIn("c__2", general_vars)
        self.assertIn("c__2", bounds_str)

        # Free variable: no finite bounds → emits "Free"
        bounds_str_free, _, _ = build_bounds(["f__0"], [-np.inf], [np.inf], [False])
        self.assertIn("Free", bounds_str_free)

    def test_sanitize_var_names_partial_sharing(self):
        """Index-level sharing: a variable covering different slot ranges per member
        must be treated as per-member, not shared. Inputs are pre-normalized to lists
        of ints by _collint_variable_indices_as_lists (numpy array normalization happens
        upstream in CIOP, not here)."""
        # Simulate ControlTreeMixin layout after normalization: u slots are shared,
        # x slots differ per member
        indices = [
            {"u": [0, 2], "x": [1, 3]},
            {"u": [0, 2], "x": [4, 5]},
        ]
        names = sanitize_var_names(indices, 6)

        # u occupies same slots in both members → shared, time-indexed
        self.assertEqual(names[0], "u__t0")
        self.assertEqual(names[2], "u__t1")

        # x occupies different slots per member → per-member suffix, time-indexed
        self.assertEqual(names[1], "x__t0__m0")
        self.assertEqual(names[3], "x__t1__m0")
        self.assertEqual(names[4], "x__t0__m1")
        self.assertEqual(names[5], "x__t1__m1")

    def test_write_lp_file_missing_output_folder(self):
        """write_lp_file must raise FileNotFoundError with a clear message
        when the folder is missing."""
        bounds_str, binary_vars, general_vars = build_bounds(["x__0"], [0.0], [1.0], [False])
        with self.assertRaises(FileNotFoundError) as ctx:
            write_lp_file(
                "test.lp",
                "0",
                "  0 = 0",
                bounds_str,
                binary_vars,
                general_vars,
                output_folder="/nonexistent/folder/that/does/not/exist",
            )
        self.assertIn("output folder", str(ctx.exception).lower())
        self.assertIn("/nonexistent/folder/that/does/not/exist", str(ctx.exception))


# Functional export tests


class TestExportLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run the linear MILP model once and share results across all tests."""
        with _suppress_info_logging(), tempfile.TemporaryDirectory() as output_folder:
            problem = LinearModel(output_folder=output_folder)
            problem.optimize()
            cls._lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            with open(os.path.join(output_folder, cls._lp_files[0])) as f:
                cls._lp_content = f.read()

    def test_lp_export_success(self):
        """Successful export must produce one correctly named file matching the reference."""
        # File naming and location
        self.assertEqual(len(self._lp_files), 1)
        self.assertTrue(self._lp_files[0].startswith("LinearModel_"))
        self.assertFalse(os.path.exists(os.path.join(os.getcwd(), self._lp_files[0])))

        # MILP: General section must list the discrete variable
        self.assertIn("General", self._lp_content)
        general_section = self._lp_content.split("General")[1].split("End")[0]
        self.assertRegex(general_section, r"\bu__t\d+")

        # Scalar variables (1 slot) must appear without __t suffix
        self.assertIn("initial_der(x)", self._lp_content)
        self.assertNotIn("initial_der(x)__t", self._lp_content)
        self.assertIn("initial_der(w)", self._lp_content)
        self.assertNotIn("initial_der(w)__t", self._lp_content)

        # Content must match reference after FP normalization
        reference_path = os.path.join(data_path(), "reference_linear_model.lp")
        with open(reference_path) as f:
            reference = _normalize_lp(f.read())

        self.assertEqual(
            _normalize_lp(self._lp_content),
            reference,
            f"Normalized LP output does not match {reference_path}. "
            "If the change is intentional, regenerate the reference file:\n"
            "  If using uv:      uv run python -m tests.optimization.regenerate_lp_reference\n"
            "  Otherwise use:    python -m tests.optimization.regenerate_lp_reference",
        )

    def test_lp_export_ensemble(self):
        """Ensemble export: one LP file, states suffixed _m{i}, controls unsuffixed."""
        with _suppress_info_logging(), tempfile.TemporaryDirectory() as output_folder:
            problem = EnsembleLinearModel(output_folder=output_folder)
            problem.optimize()
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 1)
            with open(os.path.join(output_folder, lp_files[0])) as f:
                content = f.read()

        # Controls are shared — no member suffix
        self.assertIn("u__t0", content)
        self.assertNotIn("u__t0__m0", content)

        # States are per-member — both suffixes must appear
        self.assertIn("__m0", content)
        self.assertIn("__m1", content)

    def test_lp_export_rejects_invalid_problems(self):
        """Export must raise ValueError for non-affine or unsupported problems.

        Each case runs its own optimization (not shared via setUpClass) because
        the error is expected to occur during optimize(), not after.
        """
        self._assert_raises_on_optimize(NonLinearModel, "modelica", "dae")
        self._assert_raises_on_optimize(LinearModelNonAffineObjective, "affine", "objective")
        self._assert_raises_on_optimize(NonAffineConstraintModel, "affine", "constraint")

        # Problem without _collint_variable_indices (not a CollocatedIntegratedOptimizationProblem)
        mock_problem = mock.Mock(spec=["_output_folder"])
        x = ca.SX.sym("x", 1)
        with self.assertRaises(ValueError) as ctx:
            OptimizationProblem._export_lp_file(mock_problem, x, x, x, [0], [0], [0], [0], [False])
        self.assertIn("CollocatedIntegratedOptimizationProblem", str(ctx.exception))

    def test_lp_export_no_dae_equations(self):
        """LP export must succeed when there are no DAE equations
        (linear_collocation stays None)."""
        for model_class in (EmptyModelicaModel, NoModelicaModel):
            with _suppress_info_logging(), tempfile.TemporaryDirectory() as output_folder:
                problem = model_class(output_folder=output_folder)
                problem.optimize()
                lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
                self.assertEqual(len(lp_files), 1, f"{model_class.__name__} produced no LP file")
                with open(os.path.join(output_folder, lp_files[0])) as f:
                    content = f.read()
                self.assertIn(
                    "Binary",
                    content,
                    f"{model_class.__name__}: pump_on should be in Binary section",
                )
                self.assertIn("pump_on__", content)

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
