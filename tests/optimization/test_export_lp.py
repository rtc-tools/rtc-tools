import contextlib
import importlib.metadata
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
    LP_RESERVED_NAME_PREFIXES,
    _build_bounds,
    _build_constraints,
    _build_objective,
    _build_user_constraint_base_names,
    _deduplicate_constraint_names,
    _deduplicate_nonempty_names,
    _format_lp_bound,
    _list_names,
    _sanitize_constraint_name,
    _sanitize_var_names,
    _write_lp_file,
    check_bound_conflicts,
    warn_crossed_bounds,
)
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.control_tree_mixin import ControlTreeMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.min_abs_goal_programming_mixin import (
    MinAbsGoal,
    MinAbsGoalProgrammingMixin,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.optimization_problem import OptimizationProblem
from rtctools.optimization.single_pass_goal_programming_mixin import (
    SinglePassGoalProgrammingMixin,
    SinglePassMethod,
)
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

    # Match numeric literals that are NOT part of an identifier.
    # Negative lookbehind/lookahead on word characters ensures we don't corrupt
    # variable names that happen to contain digit sequences (e.g. "q1e3", "x10").
    text = re.sub(r"(?<!\w)-?\d+\.?\d*(?:e[+-]?\d+)?(?!\w)", _round_match, text)
    text = re.sub(r"[+-]\s*0\s+\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip() + "\n"


@contextlib.contextmanager
def _silence_rtctools_logger():
    """Suppress rtctools INFO/DEBUG output during model runs in tests."""
    logger = logging.getLogger("rtctools")
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def _optimize_and_read_lp(model_class):
    """Optimize ``model_class`` in a temporary folder; return ``(problem, lp_file_content)``.

    Asserts that exactly one LP file was written.
    """
    with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
        problem = model_class(output_folder=output_folder)
        problem.optimize()
        lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
        assert len(lp_files) == 1, f"Expected exactly one LP file, got {lp_files}"
        with open(os.path.join(output_folder, lp_files[0])) as f:
            return problem, f.read()


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
        """No pre-processing needed for this test model."""

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

    def export_options(self):
        options = super().export_options()
        options["export_lp"] = True
        return options


class LinearModel(_BaseTestModel):
    """Linear MILP — LP export should succeed with a General section."""

    def variable_is_discrete(self, variable):
        return variable == "u"


class EnsembleLinearModel(LinearModel):
    """Linear MILP with two ensemble members — state variables should get __m{i} suffixes."""

    @property
    def ensemble_size(self):
        return 2

    def constant_inputs(self, ensemble_member):
        values = np.ones(len(self.times())) if ensemble_member == 0 else np.zeros(len(self.times()))
        return {"constant_input": Timeseries(self.times(), values)}


class EnsembleNamedConstraintModel(EnsembleLinearModel):
    """Ensemble model where each member returns two constraints with the same name.

    Cross-member uniqueness comes from _m{i} suffixes. Within-member duplicates
    (both rows named "dup_bound") must be resolved by deduplication, producing
    4 distinct logical constraint names. Each range constraint emits two LP lines
    with _lb/_ub suffixes, giving 8 LP lines with 8 distinct labels.
    """

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [
            (xf, -2.0, 2.0, "dup_bound"),
            (xf, -3.0, 3.0, "dup_bound"),
        ]


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
        """No pre-processing needed for this test model."""

    def seed(self, ensemble_member):
        return {}

    def export_options(self):
        options = super().export_options()
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

    def test__build_objective(self):
        """Objective string must include coefficients, constants, and handle edge cases."""
        # Linear with positive constant — must not start with '+'
        f, _, X, names = self._make_fg(2, lambda X: 2.0 * X[0] + 10.0, lambda X: X[0])
        result = _build_objective(f, X, names)
        self.assertNotRegex(result, r"^\s*\+")
        self.assertRegex(result, r"(?<![\w.])2(?![\w.])")  # coefficient 2 (:.15g strips zeros)
        self.assertIn("x0", result)
        self.assertRegex(result, r"(?<![\w.])10(?![\w.])")
        # x1 has zero coefficient — must not appear
        self.assertNotIn("x1", result)

        # Negative constant: correct sign token
        f, _, X, names = self._make_fg(1, lambda X: 3.0 * X[0] - 5.0, lambda X: X[0])
        result = _build_objective(f, X, names)
        self.assertRegex(result, r"3\b.*x0")
        self.assertRegex(result, r"-\s*5\b")

        # Zero objective must emit explicit '0' (some LP parsers reject a blank line)
        f, _, X, names = self._make_fg(1, lambda X: ca.SX(0), lambda X: X[0])
        result = _build_objective(f, X, names)
        self.assertEqual(result.strip(), "0")

        # Positive constant-only objective must not emit a leading '+' (invalid LP syntax)
        f, _, X, names = self._make_fg(1, lambda X: ca.SX(5.0), lambda X: X[0])
        result = _build_objective(f, X, names)
        self.assertNotRegex(result, r"^\s*\+")
        self.assertRegex(result, r"\b5\b")

    def test_build_constraints_operators(self):
        """Constraint operator and RHS value must match the bound type and magnitude."""
        # Equal bounds: must emit = not >= or <=
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        result, _ = _build_constraints(g, X, [3.0], [3.0], names)
        self.assertRegex(result.strip(), r"x0\s*=\s*3\b")
        self.assertNotIn(">=", result)
        self.assertNotIn("<=", result)

        # Only upper bound finite → <= with correct RHS
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        result, _ = _build_constraints(g, X, [-np.inf], [10.0], names)
        self.assertRegex(result.strip(), r"x0\s*<=\s*10\b")

    def test_build_constraints_names(self):
        """When constraint_names are provided each row must be prefixed 'name: expr op rhs'."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        # Upper bound only → <=
        result, _ = _build_constraints(g, X, [-np.inf], [5.0], names, constraint_names=["my_upper"])
        self.assertRegex(result.strip(), r"my_upper:\s*1\b.*x0\s*<=\s*5\b")
        # Lower bound only → >=
        result, _ = _build_constraints(g, X, [2.0], [np.inf], names, constraint_names=["my_lower"])
        self.assertRegex(result.strip(), r"my_lower:\s*1\b.*x0\s*>=\s*2\b")

    def test_build_constraints_sanitizes_forbidden_chars_and_logs(self):
        """Constraint names with forbidden characters must be sanitized and a debug log emitted."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        with self.assertLogs("rtctools", level="DEBUG") as cm:
            result, _ = _build_constraints(
                g, X, [-np.inf], [5.0], names, constraint_names=["has space+and+plus"]
            )
        self.assertRegex(result, r"has_space_and_plus:")
        self.assertNotIn("has space+and+plus", result)
        self.assertTrue(any("has space+and+plus" in line for line in cm.output))

    def test_build_constraints_range_both_lines_named(self):
        """Range constraints (finite lb and ub) must emit two lines with _lb/_ub suffixes."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        result, _ = _build_constraints(g, X, [2.0], [8.0], names, constraint_names=["my_range"])
        lines = [line.strip() for line in result.strip().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"my_range_lb:\s*1\b.*x0\s*>=\s*2\b")
        self.assertRegex(lines[1], r"my_range_ub:\s*1\b.*x0\s*<=\s*8\b")

    def test_build_constraints_vacuous_warns_and_skips(self):
        """Symbolic or constant constraints with both bounds infinite must warn and emit nothing."""
        for case_name, g_fn in [("symbolic", lambda X: X[0]), ("constant", lambda X: ca.SX(5.0))]:
            with self.subTest(case=case_name):
                _, g, X, names = self._make_fg(1, lambda X: X[0], g_fn)
                with self.assertLogs("rtctools", level="WARNING") as cm:
                    result, extra = _build_constraints(g, X, [-np.inf], [np.inf], names)
                self.assertTrue(any("both bounds infinite" in msg for msg in cm.output))
                self.assertEqual(result.strip(), "")
                self.assertEqual(extra, "")

    def test_build_constraints_vacuous_single_aggregated_warning(self):
        """Many vacuous rows must produce ONE warning listing (a capped number of) names."""
        X = ca.SX.sym("X", 1)
        n = 25
        g = ca.vertcat(*[X[0] * (k + 1) for k in range(n)], X[0])
        lbg = [-np.inf] * n + [0.0]
        ubg = [np.inf] * n + [1.0]
        names = [f"vac_{k}" for k in range(n)] + ["kept"]
        with self.assertLogs("rtctools", level="WARNING") as cm:
            result, _ = _build_constraints(g, X, lbg, ubg, ["x0"], constraint_names=names)
        vacuous_msgs = [m for m in cm.output if "both bounds infinite" in m]
        self.assertEqual(len(vacuous_msgs), 1)
        self.assertIn(f"skipped {n} constraint row(s)", vacuous_msgs[0])
        self.assertIn("'vac_0'", vacuous_msgs[0])
        self.assertIn("more", vacuous_msgs[0])
        self.assertNotIn("'vac_24'", vacuous_msgs[0])
        lines = [ln.strip() for ln in result.strip().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(ln.startswith("kept_") for ln in lines))

    def test_build_constraints_impossible_infinite_bounds_raise(self):
        """lb=+inf or ub=-inf is infeasible and cannot be written without an Inf RHS."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        cases = [
            ("lb_posinf", np.inf, np.inf),
            ("ub_neginf", -np.inf, -np.inf),
            ("lb_posinf_ub_finite", np.inf, 5.0),
            ("lb_finite_ub_neginf", 0.0, -np.inf),
        ]
        for case_name, lb, ub in cases:
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(ValueError, "my_row"):
                    _build_constraints(g, X, [lb], [ub], names, constraint_names=["my_row"])

    def test_build_constraints_constant_row_with_narrowly_crossed_bounds_is_kept(self):
        """lb > ub within the feasibility tolerance is still infeasible: emit the dummy row."""
        X = ca.SX.sym("X", 1)
        g = ca.vertcat(X[0], ca.SX(0.0))
        with self.assertLogs("rtctools", level="WARNING"):
            result, extra = _build_constraints(
                g, X, [0.0, 5e-10], [1.0, -5e-10], ["x0"], constraint_names=["box", "c"]
            )
        self.assertRegex(extra, r"0 <= _constant_c <= 0")
        self.assertRegex(result, r"(?m)^\s*c_lb: _constant_c >= 5e-10\s*$")
        self.assertRegex(result, r"(?m)^\s*c_ub: _constant_c <= -5e-10\s*$")

    def test_build_constraints_conflict_names_only_conflicting_rows(self):
        """The error must name the conflicting row (index > 0) and not the valid ones."""
        X = ca.SX.sym("X", 1)
        g = ca.vertcat(X[0], 2 * X[0], 3 * X[0])
        with self.assertRaises(ValueError) as ctx:
            _build_constraints(
                g,
                X,
                [0.0, 0.0, np.inf],
                [1.0, 1.0, np.inf],
                ["x0"],
                constraint_names=["a", "b", "c"],
            )
        msg = str(ctx.exception)
        self.assertIn("1 constraint(s)", msg)
        self.assertIn("c (lower=+Inf, upper=+Inf)", msg)
        self.assertNotIn("a (", msg)
        self.assertNotIn("b (", msg)

    def test_build_constraints_conflict_on_constant_or_next_to_vacuous_row_raises(self):
        """A conflict must raise even on a constant row or next to a vacuous row."""
        X = ca.SX.sym("X", 1)
        cases = [
            ("constant_row", ca.vertcat(X[0], ca.SX(3.0)), [0.0, np.inf], [1.0, np.inf]),
            ("next_to_vacuous", ca.vertcat(X[0], X[0]), [-np.inf, np.inf], [np.inf, np.inf]),
        ]
        for case_name, g, lbg, ubg in cases:
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(ValueError, r"1 constraint\(s\)"):
                    _build_constraints(g, X, lbg, ubg, ["x0"], constraint_names=["ok", "bad"])

    def test_build_constraints_crossed_bounds_written_as_given(self):
        """lb > ub (both finite) is infeasible but expressible: warn, and write both sides."""
        _, g, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0])
        with self.assertLogs("rtctools", level="WARNING") as cm:
            result, _ = _build_constraints(g, X, [2.0], [1.0], names, constraint_names=["row"])
        self.assertTrue(any("1 constraint(s)" in m and "row" in m for m in cm.output))
        lines = [line.strip() for line in result.strip().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"row_lb:\s*1 x0\s*>=\s*2$")
        self.assertRegex(lines[1], r"row_ub:\s*1 x0\s*<=\s*1$")

    def test_build_bounds_crossed_bounds_written_as_given(self):
        with self.assertLogs("rtctools", level="WARNING") as cm:
            bounds_str, _, _ = _build_bounds(["x", "y"], [2.0, 0.0], [1.0, 1.0], [False, False])
        self.assertTrue(any("1 variable(s)" in m and "x" in m for m in cm.output))
        self.assertRegex(bounds_str, r"(?m)^\s*2 <= x <= 1\s*$")
        self.assertRegex(bounds_str, r"(?m)^\s*0 <= y <= 1\s*$")

    def test_nan_bounds_error_is_capped(self):
        n = 25
        X = ca.SX.sym("X", 1)
        g = ca.vertcat(*[X[0] * (k + 1) for k in range(n)])
        names = [f"row_{k}" for k in range(n)]
        with self.assertRaises(ValueError) as ctx:
            _build_constraints(g, X, [np.nan] * n, [1.0] * n, ["x0"], constraint_names=names)
        msg = str(ctx.exception)
        self.assertIn("row_0 (lower)", msg)
        self.assertIn("(15 more)", msg)
        self.assertNotIn("row_24", msg)

    def test_build_constraints_overflowing_right_hand_side_raises(self):
        """A finite bound whose right-hand side overflows to Inf must not be written."""
        X = ca.SX.sym("X", 1)
        cases = [
            ("equality", X[0] - 1e308, 1e308, 1e308),
            ("lower", X[0] - 1e308, 1e308, np.inf),
            ("upper", X[0] + 1e308, -np.inf, -1e308),
        ]
        for case_name, g, lb, ub in cases:
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(ValueError, "non-finite right-hand side"):
                    _build_constraints(g, X, [lb], [ub], ["x0"], constraint_names=["big"])

    def test_build_constraints_never_writes_inf(self):
        """No Inf token may appear in the constraint section, for any mix of bounds."""
        X = ca.SX.sym("X", 2)
        g = ca.vertcat(X[0], X[1], X[0] + X[1], X[0] - X[1], ca.SX(3.0))
        lbg = [-np.inf, 0.0, -np.inf, 1.0, -np.inf]
        ubg = [5.0, np.inf, np.inf, 1.0, np.inf]
        with self.assertLogs("rtctools", level="WARNING"):
            result, extra = _build_constraints(g, X, lbg, ubg, ["x0", "x1"])
        self.assertNotRegex(result + extra, r"(?i)inf")
        # Rows 0, 1 and 3 are not vacuous and must be emitted (rows 2 and 4 are skipped).
        self.assertRegex(result, r"x0\s*<=\s*5(?![\w.])")
        self.assertRegex(result, r"x1\s*>=\s*0(?![\w.])")
        self.assertRegex(result, r"x0\s*-\s*1 x1\s*=\s*1(?![\w.])")

    def test_build_constraints_nonfinite_constant_raises(self):
        """A non-finite constant term would produce an Inf/NaN RHS and must raise."""
        X = ca.SX.sym("X", 1)
        g = X[0] + ca.SX(np.inf)
        with self.assertRaisesRegex(ValueError, "non-finite constant"):
            _build_constraints(g, X, [0.0], [1.0], ["x0"], constraint_names=["bad"])

    def test_build_constraints_labels_unique_after_range_expansion(self):
        """A range row "foo" emits "foo_lb", which must not collide with a row named "foo_lb"."""
        X = ca.SX.sym("X", 1)
        g = ca.vertcat(X[0], 2 * X[0])
        result, _ = _build_constraints(
            g, X, [0.0, 1.0], [10.0, np.inf], ["x0"], constraint_names=["foo", "foo_lb"]
        )
        labels = re.findall(r"^\s*(\S+):", result, re.MULTILINE)
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(set(labels)), 3, f"Duplicate labels: {labels}")

    def test_build_constraints_labels_unique_after_sanitization(self):
        """Distinct raw names that sanitize to the same label must be deduplicated, and
        constant rows must get distinct dummy variables."""
        X = ca.SX.sym("X", 1)
        g = ca.vertcat(X[0], 2 * X[0], ca.SX(5.0), ca.SX(6.0))
        with self.assertLogs("rtctools", level="WARNING"):
            result, extra = _build_constraints(
                g,
                X,
                [0.0, 0.0, 10.0, 10.0],
                [np.inf, np.inf, np.inf, np.inf],
                ["x0"],
                constraint_names=["a b", "a+b", "c d", "c+d"],
            )
        labels = re.findall(r"^\s*(\S+):", result, re.MULTILINE)
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(set(labels)), 4, f"Duplicate labels: {labels}")
        dummies = re.findall(r"0 <= (\S+) <= 0", extra)
        self.assertEqual(len(dummies), 2)
        self.assertEqual(len(set(dummies)), 2, f"Shared dummy variable: {dummies}")

    def test_build_constraints_dummy_does_not_collide_with_variable(self):
        """The dummy variable for a constant row must not reuse a decision-variable name."""
        X = ca.SX.sym("X", 1)
        g = ca.vertcat(ca.SX(5.0))
        _, extra = _build_constraints(
            g, X, [10.0], [np.inf], ["_constant_c"], constraint_names=["c"]
        )
        dummies = re.findall(r"0 <= (\S+) <= 0", extra)
        self.assertEqual(len(dummies), 1)
        self.assertNotEqual(dummies[0], "_constant_c")

    def test_build_bounds_returns_tuple_with_classified_vars(self):
        """_build_bounds must return (bounds_str, binary_vars, general_vars)."""
        var_names = ["b__0", "g__1", "c__2"]
        lbx = [0, -5, 0]
        ubx = [1, 10, 100]
        discrete = [True, True, False]
        bounds_str, binary_vars, general_vars = _build_bounds(var_names, lbx, ubx, discrete)
        self.assertEqual(binary_vars, ["b__0"])
        self.assertEqual(general_vars, ["g__1"])
        # Binary vars are excluded from Bounds section; general and continuous are included
        self.assertNotIn("b__0", bounds_str)
        self.assertRegex(bounds_str, r"-5 <= g__1 <= 10(?![\w.])")
        self.assertRegex(bounds_str, r"0 <= c__2 <= 100(?![\w.])")

    def test_build_constraints_constant_feasible_skipped(self):
        """A constant expression within bounds must be skipped silently."""
        # b_i = 5, lb = 0, ub = 10 → feasible, no output
        _, g, X, names = self._make_fg(2, lambda X: X[0], lambda X: ca.SX(5.0))
        result, extra = _build_constraints(g, X, [0.0], [10.0], names)
        self.assertEqual(result.strip(), "")
        self.assertEqual(extra, "")

    def test_build_constraints_constant_feasible_symbolic_skipped(self):
        """A symbolic expression that simplifies to zero must also be skipped when feasible."""
        # x - x = 0, lb = -1, ub = 1 → feasible
        _, _, X, names = self._make_fg(1, lambda X: X[0], lambda X: X[0] - X[0])  # NOSONAR
        g = X[0] - X[0]  # NOSONAR
        result, extra = _build_constraints(g, X, [-1.0], [1.0], names)
        self.assertEqual(result.strip(), "")
        self.assertEqual(extra, "")

    # Parametrized table for infeasible constant constraint tests.
    # Each row: (b_i_expr, lb, ub, name, constraint_regex, extra_regex)
    _INFEASIBLE_CONSTANT_CASES = [
        (
            "lower_bound",
            ca.SX(5.0),
            10.0,
            np.inf,
            "my_const",
            r"my_const:\s*_constant_my_const\s*>=\s*5",
            r"0\s*<=\s*_constant_my_const\s*<=\s*0",
        ),
        (
            "upper_bound",
            ca.SX(5.0),
            -np.inf,
            2.0,
            "my_const",
            r"my_const:\s*_constant_my_const\s*<=\s*-3",
            r"0\s*<=\s*_constant_my_const\s*<=\s*0",
        ),
        (
            "equality",
            ca.SX(5.0),
            3.0,
            3.0,
            "my_eq",
            r"my_eq:\s*_constant_my_eq\s*=\s*-2",
            r"0\s*<=\s*_constant_my_eq\s*<=\s*0",
        ),
        (
            # b_i = 0, lb = 2: dummy >= 2 - 0 = 2; but dummy == 0 → infeasible.
            "zero_bi",
            None,
            2.0,
            np.inf,
            "zero_const",
            r"zero_const:\s*_constant_zero_const\s*>=\s*2",
            r"0\s*<=\s*_constant_zero_const\s*<=\s*0",
        ),
        (
            "unnamed",
            ca.SX(5.0),
            10.0,
            np.inf,
            None,
            r"(?m)^\s*_constant_0\s*>=\s*5\s*$",
            r"0\s*<=\s*_constant_0\s*<=\s*0",
        ),
        (
            # Violation smaller than 1: b_i = 0, lb = 0.5. With the dummy fixed to 1 this
            # was wrongly exported as feasible (1 >= 0.5).
            "sub_unit_violation",
            None,
            0.5,
            np.inf,
            "small_viol",
            r"small_viol:\s*_constant_small_viol\s*>=\s*0\.5",
            r"0\s*<=\s*_constant_small_viol\s*<=\s*0",
        ),
    ]

    def test_build_constraints_constant_infeasible(self):
        """Infeasible constant expressions must emit a dummy variable constraint and extra bounds.

        The RHS is lb - b_i (not just lb), so solvers detect the infeasibility regardless of
        whether b_i is zero. The dummy variable is fixed to 0 in Bounds, so the row reads
        ``0 >= lb - b_i`` (i.e. ``b_i >= lb``) and presolve detects infeasibility for any
        violation size. Covers: lower bound, upper bound, equality, zero b_i, unnamed
        (auto-index), and sub-unit violation cases.
        """
        for (
            case_name,
            b_i_expr,
            lb,
            ub,
            name,
            c_regex,
            extra_regex,
        ) in self._INFEASIBLE_CONSTANT_CASES:
            with self.subTest(case=case_name):
                if b_i_expr is None:
                    # zero_bi: use x - x which CasADi simplifies to 0
                    X = ca.SX.sym("X", 1)
                    g = X[0] - X[0]  # NOSONAR
                    var_names = ["x0"]
                else:
                    expr = b_i_expr
                    _, g, X, var_names = self._make_fg(2, lambda X: X[0], lambda X, e=expr: e)
                constraint_names = [name] if name is not None else None
                result, extra = _build_constraints(
                    g, X, [lb], [ub], var_names, constraint_names=constraint_names
                )
                self.assertRegex(result, c_regex, msg=f"case={case_name}: constraint line wrong")
                self.assertRegex(extra, extra_regex, msg=f"case={case_name}: extra bounds wrong")

    def test_build_constraints_constant_named_label_sync(self):
        """Skipping a feasible constant row must not corrupt labels for subsequent rows.

        Regression guard: an earlier implementation used a pre-built label iterator that
        consumed a slot even for skipped rows, shifting all subsequent labels by one.
        """
        X = ca.SX.sym("X", 1)
        g = ca.vertcat(ca.SX(3.0), X[0])
        var_names = ["x0"]
        result, extra = _build_constraints(
            g, X, [0.0, 1.0], [5.0, np.inf], var_names, constraint_names=["skipped", "kept"]
        )
        self.assertNotIn("skipped", result)
        # Exactly one constraint line emitted, with the correct label and content
        lines = [ln for ln in result.strip().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], r"kept:\s*1\b.*x0\s*>=\s*1\b")
        self.assertNotRegex(lines[0], r"\b(c0|c1|0:|1:)")
        self.assertEqual(extra, "")

    def test__sanitize_var_names(self):
        """_sanitize_var_names must assign __m{i} suffixes to per-member variables and omit
        them for shared variables. Sharing is determined at variable level: if any member has
        a different slot list, all slots of that variable get member suffixes."""
        cases = [
            (
                "ensemble_shared_control",
                # u is shared (same slots in all members); x is per-member
                [{"u": [0, 1], "x": [2, 3]}, {"u": [0, 1], "x": [4, 5]}],
                6,
                {
                    0: "u__t0",
                    1: "u__t1",
                    2: "x__t0__m0",
                    3: "x__t1__m0",
                    4: "x__t0__m1",
                    5: "x__t1__m1",
                },
            ),
            (
                "partial_sharing_all_per_member",
                # u has non-overlapping slot lists per member → all per-member
                [{"u": [0, 1]}, {"u": [2, 3]}, {"u": [4, 5]}],
                6,
                {
                    0: "u__t0__m0",
                    1: "u__t1__m0",
                    2: "u__t0__m1",
                    3: "u__t1__m1",
                    4: "u__t0__m2",
                    5: "u__t1__m2",
                },
            ),
        ]
        for case_name, indices, n_vars, expected in cases:
            with self.subTest(case=case_name):
                names = _sanitize_var_names(indices, n_vars)
                for slot, expected_name in expected.items():
                    self.assertEqual(
                        names[slot], expected_name, msg=f"case={case_name}, slot={slot}"
                    )

    def test_write_lp_file_binary_and_general(self):
        """Binary variables (bounds [0,1]) go to Binary section; others go to General."""
        var_names = ["x__0", "y__1", "z__2"]
        discrete = [True, True, False]
        lbx = [0, -5, 0]
        ubx = [1, 10, 100]
        bounds_str, binary_vars, general_vars = _build_bounds(var_names, lbx, ubx, discrete)
        with tempfile.TemporaryDirectory() as output_folder:
            _write_lp_file(
                "test.lp", "0", "  0 = 0", bounds_str, binary_vars, general_vars, output_folder
            )
            with open(os.path.join(output_folder, "test.lp")) as f:
                content = f.read()

        self.assertIn("Binary", content)
        binary_section = content.split("Binary")[1].split("End")[0]
        self.assertIn("x__0", binary_section)
        self.assertNotIn("y__1", binary_section)
        self.assertNotIn("z__2", binary_section)

        self.assertIn("General", content)
        general_section = content.split("General")[1].split("Binary")[0]
        self.assertIn("y__1", general_section)
        self.assertNotIn("x__0", general_section)
        self.assertNotIn("z__2", general_section)

    def test_write_lp_file_no_bounds_section_for_pure_binary(self):
        """_write_lp_file must omit the Bounds section entirely when bounds_str is empty."""
        with tempfile.TemporaryDirectory() as output_folder:
            _write_lp_file("test.lp", "0", "  0 = 0", "", ["b__0"], [], output_folder)
            with open(os.path.join(output_folder, "test.lp")) as f:
                content = f.read()
        self.assertNotIn("Bounds", content)
        self.assertIn("Binary", content)

    def test_write_lp_file_no_subject_to_section_when_no_constraints(self):
        """_write_lp_file must omit the Subject To section when constraints_str is empty
        (or whitespace), to avoid emitting a blank constraint line some readers reject."""
        with tempfile.TemporaryDirectory() as output_folder:
            _write_lp_file("test.lp", "0", "", "0 <= x__0 <= 1", [], [], output_folder)
            with open(os.path.join(output_folder, "test.lp")) as f:
                content = f.read()
        self.assertNotIn("Subject To", content)
        self.assertIn("Bounds", content)

    def test_write_lp_file_missing_output_folder(self):
        """_write_lp_file raises FileNotFoundError when folder does not exist."""
        var_names = ["x__0"]
        bounds_str, binary_vars, general_vars = _build_bounds(var_names, [0], [1], [False])
        with self.assertRaises(FileNotFoundError) as ctx:
            _write_lp_file(
                "test.lp",
                "0",
                "  0 = 0",
                bounds_str,
                binary_vars,
                general_vars,
                "/nonexistent/folder/xyz",
            )
        self.assertIn("output folder", str(ctx.exception).lower())
        self.assertIn("/nonexistent/folder/xyz", str(ctx.exception))

    def test_constraint_name_deduplication(self):
        """Duplicate constraint names in LP export must be deduplicated with _d0, _d1, ... suffixes,
        and must not collide with pre-existing names that already use that suffix pattern."""
        # Basic deduplication: three occurrences of "foo"
        result = _deduplicate_constraint_names(["foo", "foo", "foo"])
        self.assertEqual(result, ["foo_d0", "foo_d1", "foo_d2"])

        # Collision-safe: "foo_d0" is taken; first duplicate renames to "foo_d1".
        result2 = _deduplicate_constraint_names(["foo", "foo_d0", "foo"])
        self.assertEqual(len(set(result2)), 3, f"Duplicate labels after deduplication: {result2}")
        self.assertNotIn("foo", result2)  # original "foo" entries must all be renamed
        self.assertEqual(result2, ["foo_d1", "foo_d0", "foo_d2"])  # pre-existing "foo_d0" is kept

    def test_deduplicate_warns_on_reserved_prefix_collision(self):
        """A duplicate name starting with a reserved prefix must emit a warning."""
        reserved = next(iter(LP_RESERVED_NAME_PREFIXES))
        name = f"{reserved}foo"
        with self.assertLogs("rtctools", level="WARNING") as cm:
            result = _deduplicate_constraint_names([name, name])
        self.assertTrue(
            any("Internal constraint name" in line and name in line for line in cm.output)
        )
        self.assertEqual(len(set(result)), 2)

    def test_build_user_constraint_base_names_reserved_suffix_renamed(self):
        """Names ending with a reserved LP suffix must be renamed with _ren and emit a warning."""
        for reserved_suffix in ("_lb", "_ub", "_t0", "_m1", "_d3"):
            raw_name = f"foo{reserved_suffix}"
            tuples = [(None, 0.0, 1.0, raw_name)]
            with self.assertLogs("rtctools", level="WARNING") as cm:
                result = _build_user_constraint_base_names(tuples, "constraint")
            self.assertEqual(result, [f"foo{reserved_suffix}_ren"])
            self.assertTrue(any(raw_name in line for line in cm.output))

        # A name introduced by sanitization that ends with a reserved suffix is also renamed.
        # "foo lb" → sanitized to "foo_lb" → renamed to "foo_lb_ren"
        tuples = [(None, 0.0, 1.0, "foo lb")]
        with self.assertLogs("rtctools", level="WARNING"):
            result = _build_user_constraint_base_names(tuples, "constraint")
        self.assertEqual(result, ["foo_lb_ren"])

    def test_build_user_constraint_base_names_reserved_prefix_renamed(self):
        """User names starting with a reserved prefix must be renamed with _ren, so they can
        never collide with auto-generated rows (e.g. "initial_residual_0")."""
        for prefix in LP_RESERVED_NAME_PREFIXES:
            raw_name = f"{prefix}0"
            tuples = [(None, 0.0, 1.0, raw_name)]
            with self.assertLogs("rtctools", level="WARNING") as cm:
                result = _build_user_constraint_base_names(tuples, "constraint")
            self.assertEqual(result, [f"{prefix}0_ren"])
            self.assertTrue(any("reserved for auto-generated" in line for line in cm.output))

        # Auto-generated fallback names use a reserved prefix but must not be renamed.
        result = _build_user_constraint_base_names([(None, 0.0, 1.0)], "constraint")
        self.assertEqual(result, ["constraint_0"])

    def test_build_bounds_upper_only_has_explicit_neg_inf_lower(self):
        """lb=-inf with finite ub must emit an explicit -Inf lower bound; "x <= ub" alone
        would make LP readers apply their default lower bound of 0."""
        bounds_str, _, _ = _build_bounds(["x"], [-np.inf], [5.0], [False])
        self.assertEqual(bounds_str.strip(), "-Inf <= x <= 5")

    def test_build_bounds_impossible_infinite_bounds_raise(self):
        """Variable bounds lbx=+inf or ubx=-inf are infeasible and must raise."""
        for case_name, lb, ub in [("lb_posinf", np.inf, np.inf), ("ub_neginf", -np.inf, -np.inf)]:
            with self.subTest(case=case_name):
                with self.assertRaisesRegex(ValueError, "my_var"):
                    _build_bounds(["my_var"], [lb], [ub], [False])

    def test_constraint_tuple_backward_compat(self):
        """_ConstraintTuple must unpack as 3 elements; name accessible via .name property.

        _ConstraintTuple is the container through which user-provided constraint names
        reach the LP export pipeline, so its contract is tested here alongside the other
        LP builder unit tests.
        """
        from rtctools.optimization.goal_programming_mixin_base import _ConstraintTuple

        expr = ca.SX.sym("x")
        t = _ConstraintTuple((expr, 0.0, 1.0, "my_name"))

        # Unpacking must yield exactly 3 elements (backward-compatible)
        f, lb, ub = t
        self.assertIs(f, expr)
        self.assertEqual(lb, 0.0)
        self.assertEqual(ub, 1.0)

        # .name property returns the 4th element
        self.assertEqual(t.name, "my_name")
        self.assertEqual(len(t), 4)

        # 3-tuple _ConstraintTuple (no name): .name returns None
        t3 = _ConstraintTuple((expr, 0.0, 1.0))
        f3, lb3, ub3 = t3
        self.assertIs(f3, expr)
        self.assertIsNone(t3.name)

    def test_write_lp_file_no_overwrite(self):
        """_write_lp_file() must not overwrite an existing file; it appends a counter instead."""
        var_names = ["x__0"]
        discrete = [False]
        lbx = [0]
        ubx = [1]
        bounds_str, binary_vars, general_vars = _build_bounds(var_names, lbx, ubx, discrete)
        with tempfile.TemporaryDirectory() as output_folder:
            for _ in range(2):
                _write_lp_file(
                    "test.lp", "0", "  0 = 0", bounds_str, binary_vars, general_vars, output_folder
                )
            files = {f for f in os.listdir(output_folder) if f.endswith(".lp")}
            self.assertEqual(files, {"test.lp", "test_1.lp"})

    def test_sanitize_var_names_sharing(self):
        """Slots shared by all members carry no member suffix, the rest are per-member."""
        cases = [
            (
                "fully_shared",
                [{"a": [0, 1], "p": [2]}, {"a": [0, 1], "p": [2]}],
                ["a__t0", "a__t1", "p"],
            ),
            (
                "per_member",
                [{"x": [0, 1], "q": [4]}, {"x": [2, 3], "q": [5]}],
                ["x__t0__m0", "x__t1__m0", "x__t0__m1", "x__t1__m1", "q__m0", "q__m1"],
            ),
            (
                "shared_before_branching",
                [{"u": [0, 1]}, {"u": [0, 2]}, {"u": [0, 3]}],
                ["u__t0", "u__t1__m0", "u__t1__m1", "u__t1__m2"],
            ),
            (
                "subset_of_members_uses_lowest",
                [{"u": [0, 1]}, {"u": [0, 1]}, {"u": [0, 2]}],
                ["u__t0", "u__t1__m0", "u__t1__m2"],
            ),
            ("single_member", [{"a": [0, 1], "p": [2]}], ["a__t0", "a__t1", "p"]),
            ("brackets_replaced", [{"v[1]": [0]}], ["v_I1I_"]),
        ]
        for case_name, indices_per_member, expected in cases:
            with self.subTest(case=case_name):
                with mock.patch("rtctools._internal.casadi_to_lp.logger") as logger:
                    names = _sanitize_var_names(indices_per_member, len(expected))
                logger.warning.assert_not_called()
                self.assertEqual(sorted(names), sorted(expected))

    def test_sanitize_var_names_slot_conflict_and_unassigned(self):
        """Different variables on one slot warn; unfilled slots are marked and warned about."""
        with mock.patch("rtctools._internal.casadi_to_lp.logger") as logger:
            names = _sanitize_var_names([{"a": [0]}, {"b": [0]}], 2)
        self.assertEqual(logger.warning.call_count, 2)
        self.assertIn("different variable names", logger.warning.call_args_list[0].args[0])
        self.assertIn("0 -> 'b__m1'", logger.warning.call_args_list[0].args[1])
        self.assertIn("not claimed", logger.warning.call_args_list[1].args[0])
        self.assertEqual(names[1], "__unassigned_1")

    def test_build_user_constraint_base_names_non_str_name(self):
        """A non-string constraint name raises a TypeError naming the constraint index."""
        with self.assertRaisesRegex(TypeError, "index 1 must be a str, got int"):
            _build_user_constraint_base_names([(0, 0, 1, "ok"), (0, 0, 1, 5)], "constraint")

    def test_sanitize_constraint_name(self):
        cases = [
            ("a b+c", "a_b_c"),
            ("1st_bound", "_1st_bound"),
            (".dot", "_.dot"),
            ("\\critical", "_critical"),
            ("a1.b", "a1.b"),
            ("", ""),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_sanitize_constraint_name(raw), expected)

    def test_build_user_constraint_base_names_edge_cases(self):
        """None names fall back, names reserved only after sanitising are renamed, and
        renamed names that collide are deduplicated."""
        cases = [
            ("none_and_short", [(0, 0, 1, None), (0, 0, 1)], ["constraint_0", "constraint_1"]),
            (
                "reserved_after_sanitising",
                [(0, 0, 1, "path constraint_x")],
                ["path_constraint_x_ren"],
            ),
            (
                "rename_collision",
                [(0, 0, 1, "foo_ub"), (0, 0, 1, "foo_ub_ren")],
                ["foo_ub_ren_d0", "foo_ub_ren_d1"],
            ),
            ("already_ending_in_ren", [(0, 0, 1, "x_ren")], ["x_ren"]),
            ("leading_digit", [(0, 0, 1, "1st")], ["_1st"]),
        ]
        for case_name, tuples, expected in cases:
            with self.subTest(case=case_name), _silence_rtctools_logger():
                self.assertEqual(_build_user_constraint_base_names(tuples, "constraint"), expected)

    def test_deduplicate_helpers_edge_cases(self):
        """Empty and single-name lists are untouched; empty names are ignored when dedup'ing."""
        self.assertEqual(_deduplicate_constraint_names([]), [])
        self.assertEqual(_deduplicate_constraint_names(["a"]), ["a"])
        self.assertEqual(_deduplicate_nonempty_names([]), [])
        self.assertEqual(_deduplicate_nonempty_names(["", ""]), ["", ""])
        self.assertEqual(
            _deduplicate_nonempty_names(["", "a", "", "a", ""]), ["", "a_d0", "", "a_d1", ""]
        )

    def test_format_lp_bound(self):
        cases = [
            (np.inf, "+Inf"),
            (-np.inf, "-Inf"),
            (-0.0, "-0"),
            (0.1, "0.1"),
            (1e300, "1e+300"),
            (1e-300, "1e-300"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(_format_lp_bound(value), expected)

    def test_write_lp_file_counter_limit(self):
        """_write_lp_file() must raise once 100 counter-suffixed files already exist."""
        with tempfile.TemporaryDirectory() as output_folder:
            names = ["test.lp"] + [f"test_{i}.lp" for i in range(1, 101)]
            for name in names:
                open(os.path.join(output_folder, name), "w").close()
            with self.assertRaisesRegex(RuntimeError, "Could not write LP file"):
                _write_lp_file("test.lp", "0", "  0 = 0", "", [], [], output_folder)
            self.assertEqual(len(os.listdir(output_folder)), len(names))

    def test_export_lp_raises_not_implemented_for_non_collocated(self):
        """OptimizationProblem._export_lp_file must raise NotImplementedError.

        Only CollocatedIntegratedOptimizationProblem implements LP export; calling the
        base-class method directly must raise with a clear message naming the required class.
        """
        mock_problem = mock.Mock(spec=["_output_folder"])
        x = ca.SX.sym("x", 1)
        with self.assertRaises(NotImplementedError) as ctx:
            OptimizationProblem._export_lp_file(mock_problem, x, x, x, [0], [0], [0], [0], [False])
        self.assertIn("CollocatedIntegratedOptimizationProblem", str(ctx.exception))

    def test_export_lp_not_in_solver_options(self):
        """``export_lp`` must live in export_options(), never in solver_options().

        Keeping it out of solver_options() means direct consumers of that dict (e.g.
        ``PumpingStationMixin.__solve_hq_subproblem``) need not know about or defensively
        strip ``export_lp`` to avoid CasADi rejecting it as an unknown option.
        """
        # Base default: export disabled, and the directives live in export_options().
        self.assertEqual(
            OptimizationProblem.export_options(mock.Mock()),
            {"export_lp": False, "track_constraint_names": False},
        )

        # The directives must never appear in the solver options dict.
        with tempfile.TemporaryDirectory() as output_folder:
            problem = LinearModel(output_folder=output_folder)
            self.assertNotIn("export_lp", problem.solver_options())
            self.assertNotIn("track_constraint_names", problem.solver_options())

    def test_export_lp_in_solver_options_warns(self):
        """export_lp set in solver_options() (a natural mistake) must warn."""

        class MisplacedExportLP(LinearModel):
            def solver_options(self):
                options = super().solver_options()
                options["export_lp"] = True  # wrong place on purpose
                return options

            def export_options(self):
                # Disable actual LP export; this test only checks the misplacement warning.
                return {"export_lp": False}

        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = MisplacedExportLP(output_folder=output_folder)
            with self.assertWarnsRegex(UserWarning, "export_options"):
                problem.optimize()


# Functional export tests (structure, naming, error handling — no solver needed)


class _NamedConstraintModel(_BaseTestModel):
    """Model with user-named constraints and path constraints."""

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [(xf, -2.0, 2.0, "terminal_x_bound")]

    def path_constraints(self, ensemble_member):
        x = self.state("x")
        return [(x, -3.0, 3.0, "state_bound")]


class _EmptyNameConstraintModel(_BaseTestModel):
    """Model that returns empty-string names in constraint tuples.

    An empty name is treated as absent — the constraint gets an auto-generated name.
    This guards against a backward-compatibility regression where passing "" raised
    ValueError.
    """

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [(xf, -2.0, 2.0, "")]

    def path_constraints(self, ensemble_member):
        x = self.state("x")
        return [(x, -3.0, 3.0, "")]


class _SingleMemberDuplicateNameModel(LinearModel):
    """Single-member model whose constraints() returns two rows with the same name.

    Used to verify that targeted (slice-only) deduplication resolves user-name
    collisions without touching internal names.
    """

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [
            (xf, -2.0, 2.0, "dup_name"),
            (xf, -3.0, 3.0, "dup_name"),
        ]


class _ScalarVectorNameCollisionModel(LinearModel):
    """Path constraints whose names collide only after vector expansion.

    A scalar constraint named "foo_0" and a size-2 vector constraint named "foo"
    both exist before expansion (no collision, dedup fast-path would skip). After
    the vector expands to scalar rows "foo_0"/"foo_1", the scalar "foo_0" and the
    vector's first expanded row collide. Deduplication must run after expansion to
    catch this.
    """

    def path_constraints(self, ensemble_member):
        x = self.state("x")
        w = self.state("w")
        return [
            (x, -3.0, 3.0, "foo_0"),
            (ca.vertcat(x, w), -3.0, 3.0, "foo"),
        ]


class _ConstantFeasibleConstraintModel(_BaseTestModel):
    """Model with a constraint that reduces to a constant feasible expression after expansion.

    ``ca.MX(3.0)`` has no symbolic dependency on any variable. After CasADi expansion
    the row becomes a constant — it must be silently skipped in the LP file and must
    not affect solver feasibility.
    """

    def constraints(self, ensemble_member):
        return [(ca.MX(3.0), 0.0, 10.0, "const_feasible")]


class _ReservedPrefixNameModel(LinearModel):
    """Single-member model whose user constraint name equals an auto-generated row name."""

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [(xf, -2.0, 2.0, "initial_residual_0")]


class _VacuousConstraintModel(LinearModel):
    """Model with a user constraint whose bounds are both infinite."""

    def __init__(self, *args, export_lp=True, **kwargs):
        self._export_lp_flag = export_lp
        super().__init__(*args, **kwargs)

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [(xf, -np.inf, np.inf, "no_bounds")]

    def export_options(self):
        options = super().export_options()
        options["export_lp"] = self._export_lp_flag
        return options


class _LinearModelHiGHS(LinearModel):
    """LinearModel solved via HiGHS through CasADi, so both solver sides match."""

    def solver_options(self):
        options = super().solver_options()
        options["casadi_solver"] = ca.qpsol
        options["solver"] = "highs"
        options["highs"] = {"output_flag": False}
        return options


class TestExportLP(unittest.TestCase):
    """LP export: file structure, naming, constraint names, and error handling."""

    @classmethod
    def setUpClass(cls):
        """Run the linear MILP model once and share the LP content across tests."""
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = LinearModel(output_folder=output_folder)
            problem.optimize()
            cls._lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            with open(os.path.join(output_folder, cls._lp_files[0])) as f:
                cls._lp_content = f.read()

    def test_lp_export_success(self):
        """Successful export must produce one correctly named file matching the reference."""
        self.assertEqual(len(self._lp_files), 1)
        self.assertTrue(self._lp_files[0].startswith("LinearModel_"))
        self.assertFalse(os.path.exists(os.path.join(os.getcwd(), self._lp_files[0])))

        # MILP: General section must list the discrete variable but not continuous ones
        self.assertIn("General", self._lp_content)
        general_section = self._lp_content.split("General")[1].split("End")[0]
        self.assertIn("u__", general_section)
        # Continuous state and algebraic variables must not appear in General
        self.assertNotIn("x__", general_section)
        self.assertNotIn("w__", general_section)

        # Collocation constraint names must be ordered time-outer, equation-inner
        # (matching ca.vec column-major flattening). Regression test: the loop was
        # previously equation-outer, producing wrong labels for multi-equation DAEs.
        collocation_lines = [
            line for line in self._lp_content.splitlines() if "collocation_eq" in line
        ]
        # Extract (t, eq) from each label and verify time is non-decreasing,
        # with equation index cycling 0..N within each time step.
        parsed = [
            (int(m.group(2)), int(m.group(1)))
            for line in collocation_lines
            if (m := re.search(r"collocation_eq(\d+)_t(\d+)", line))
        ]
        self.assertTrue(parsed, "No collocation constraint names found in LP output")
        self.assertEqual(
            parsed, sorted(parsed), "Collocation constraints are not ordered time-outer"
        )
        # The model has 3 collocation equations: each time step lists equations 0, 1, 2.
        eqs_by_time = {}
        for t, eq in parsed:
            eqs_by_time.setdefault(t, []).append(eq)
        for t, eqs in eqs_by_time.items():
            self.assertEqual(eqs, [0, 1, 2], f"Equation order wrong at t={t}")

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
        """Ensemble export: one LP file, states suffixed __m{i}, controls unsuffixed."""
        problem, content = _optimize_and_read_lp(EnsembleLinearModel)

        self.assertIn("u__t0", content)
        self.assertNotIn("u__t0__m0", content)
        # Member suffixes must appear on state variable names (check in Bounds section)
        bounds_section = content.split("Bounds")[1].split("End")[0]
        self.assertRegex(bounds_section, r"x__t\d+__m0")
        self.assertRegex(bounds_section, r"x__t\d+__m1")
        # All constraint names generated inside the ensemble loop must carry _m{i} suffixes
        self.assertRegex(content, r"collocation_eq\d+_t\d+_m0:")
        self.assertRegex(content, r"collocation_eq\d+_t\d+_m1:")
        self.assertRegex(content, r"delay_.*_t\d+_m0:")
        self.assertRegex(content, r"delay_.*_t\d+_m1:")
        # initial_residual rows carry a member suffix too
        self.assertRegex(content, r"initial_residual_\d+_m0:")
        self.assertRegex(content, r"initial_residual_\d+_m1:")
        self.assertNotRegex(content, r"initial_residual_\d+:")

    def test_collint_constraint_names_match_lp_labels(self):
        """_collint_constraint_names must be sanitized like the LP labels, e.g. delay names
        derived from pymoca variables containing "[" / "]"."""
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = LinearModel(output_folder=output_folder)
            problem.optimize()
            names = problem._collint_constraint_names
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            with open(os.path.join(output_folder, lp_files[0])) as f:
                content = f.read()
        delay_names = [n for n in names if n.startswith("delay_")]
        self.assertTrue(delay_names)
        self.assertFalse(any(c in n for n in names for c in "[]"))
        labels = set(re.findall(r"^\s*(\S+):", content, re.MULTILINE))
        for name in delay_names:
            self.assertIn(name, labels)

    def test_lp_export_ensemble_initial_residual_name_order(self):
        """initial_residual names must follow ca.vec's member-major order."""
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = EnsembleLinearModel(output_folder=output_folder)
            problem.optimize()
            names = problem._collint_constraint_names
        init_names = [n for n in names if n.startswith("initial_residual_")]
        # The model has 4 initial residual equations per member (2 derivative + 2 initial).
        n_eq = 4
        self.assertEqual(len(init_names), n_eq * problem.ensemble_size)
        expected = [
            f"initial_residual_{eq}_m{m}"
            for m in range(problem.ensemble_size)
            for eq in range(n_eq)
        ]
        self.assertEqual(init_names, expected)
        # They are the first rows of g
        self.assertEqual(names[: len(expected)], expected)

    def test_lp_export_user_name_with_reserved_prefix(self):
        """A user name equal to an internal name must be renamed, leaving the internal row
        intact and all labels globally unique."""
        with tempfile.TemporaryDirectory() as output_folder:
            problem = _ReservedPrefixNameModel(output_folder=output_folder)
            with self.assertLogs("rtctools", level="WARNING") as cm:
                problem.optimize()
            names = problem._collint_constraint_names
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            with open(os.path.join(output_folder, lp_files[0])) as f:
                content = f.read()
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("initial_residual_0", names)
        self.assertIn("initial_residual_0_ren", names)
        self.assertFalse(any("indicates a bug" in line for line in cm.output))
        subject_to = content.split("Subject To")[1].split("Bounds")[0]
        labels = re.findall(r"^\s*(\S+):", subject_to, re.MULTILINE)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn("initial_residual_0", labels)
        self.assertIn("initial_residual_0_ren_lb", labels)

    def test_vacuous_constraint_warns_only_on_lp_export(self):
        """The both-bounds-infinite warning is an LP export diagnostic: it must be logged
        when the LP file is written, and not otherwise."""
        logger = logging.getLogger("rtctools")
        for export_lp in (True, False):
            with self.subTest(export_lp=export_lp):
                with tempfile.TemporaryDirectory() as output_folder:
                    problem = _VacuousConstraintModel(
                        output_folder=output_folder, export_lp=export_lp
                    )
                    with mock.patch.object(logger, "warning") as warn:
                        problem.optimize()
                    vacuous_calls = [
                        c for c in warn.call_args_list if "both bounds infinite" in str(c.args[0])
                    ]
                    self.assertEqual(len(vacuous_calls), 1 if export_lp else 0)
                    if export_lp:
                        self.assertIn("no_bounds", str(vacuous_calls[0].args))
                        lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
                        with open(os.path.join(output_folder, lp_files[0])) as f:
                            content = f.read()
                        self.assertNotIn("no_bounds", content)

    def test_lp_export_deduplication_single_member(self):
        """Duplicate user constraint names in a single-member problem must be deduplicated.

        Verifies that only the user-provided slice of constraint_names is processed,
        leaving internal names (collocation, delay, etc.) untouched.
        """
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = _SingleMemberDuplicateNameModel(output_folder=output_folder)
            problem.optimize()
            # Later dedup passes in the LP builder would mask a missing upstream dedup in the
            # LP labels, so check the tracked names too.
            user_names = [n for n in problem._collint_constraint_names if n.startswith("dup_name")]
            self.assertEqual(user_names, ["dup_name_d0", "dup_name_d1"])
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            with open(os.path.join(output_folder, lp_files[0])) as f:
                content = f.read()
        # Both duplicate rows must appear with distinct suffixed labels
        self.assertIn("dup_name_d0_lb:", content)
        self.assertIn("dup_name_d0_ub:", content)
        self.assertIn("dup_name_d1_lb:", content)
        self.assertIn("dup_name_d1_ub:", content)
        self.assertNotIn("dup_name:", content)
        # Internal names must survive intact — the targeted slice must not touch them
        self.assertRegex(content, r"collocation_eq\d+_t\d+:")

    def test_lp_export_deduplication_after_vector_expansion(self):
        """A scalar name and a vector constraint's expanded name must not collide.

        A scalar path constraint named "foo_0" and a size-2 vector path constraint
        named "foo" produce distinct base names before expansion (dedup fast-path is a
        no-op), but the vector's expansion to "foo_0"/"foo_1" collides with the scalar
        "foo_0" row. All emitted labels for a given time step must be pairwise distinct.
        """
        problem, content = _optimize_and_read_lp(_ScalarVectorNameCollisionModel)

        # Collect every constraint label at t0 that starts with "foo" (both _lb/_ub
        # range sides), across all three expanded rows (scalar foo_0, vector foo_0/foo_1).
        labels_t0 = re.findall(r"^\s*(foo\S*_t0(?:_lb|_ub)):", content, re.MULTILINE)
        self.assertEqual(
            sorted(labels_t0),
            [
                "foo_0_d0_t0_lb",
                "foo_0_d0_t0_ub",
                "foo_0_d1_t0_lb",
                "foo_0_d1_t0_ub",
                "foo_1_t0_lb",
                "foo_1_t0_ub",
            ],
        )

    def test_lp_export_ensemble_named_constraints(self):
        """User constraint names in ensemble problems get _m{i} suffixes for cross-member
        uniqueness; within-member duplicates are resolved by deduplication.

        Each member returns two rows named "dup_bound" — deduplication produces
        "dup_bound_d0" and "dup_bound_d1" (4 distinct logical names total, one set per
        member). Each range constraint emits two LP lines with _lb/_ub suffixes, giving
        8 LP lines with 8 distinct labels.
        """
        problem, content = _optimize_and_read_lp(EnsembleNamedConstraintModel)
        # 2 members × 2 rows = 4 logical constraint names (deduplicated within each member).
        # Each range constraint emits 2 LP lines with _lb/_ub suffixes, so 8 distinct labels.
        constraint_labels = re.findall(r"dup_bound[^:]*:", content)
        self.assertEqual(len(constraint_labels), 8)
        self.assertEqual(len(set(constraint_labels)), 8, f"Duplicate labels: {constraint_labels}")
        # Verify dedup indices and member suffixes both present.
        # Dedup runs on base names before member suffix expansion, so the dedup index
        # precedes the _m{i} suffix: dup_bound_d0_m0, dup_bound_d1_m0, etc.
        self.assertIn("dup_bound_d0_m0_lb:", content)
        self.assertIn("dup_bound_d0_m0_ub:", content)
        self.assertIn("dup_bound_d1_m0_lb:", content)
        self.assertIn("dup_bound_d1_m0_ub:", content)
        self.assertIn("dup_bound_d0_m1_lb:", content)
        self.assertIn("dup_bound_d0_m1_ub:", content)
        self.assertIn("dup_bound_d1_m1_lb:", content)
        self.assertIn("dup_bound_d1_m1_ub:", content)

    def test_lp_export_rejects_invalid_problems(self):
        """Export must raise ValueError for non-affine or unsupported problems."""
        self._assert_raises_on_optimize(NonLinearModel, "modelica", "dae")
        self._assert_raises_on_optimize(LinearModelNonAffineObjective, "affine", "objective")
        self._assert_raises_on_optimize(NonAffineConstraintModel, "affine", "constraint")

    def test_lp_export_no_dae_equations_empty_modelica(self):
        """LP export must succeed for a model with an empty Modelica file (no DAE equations).

        EmptyModelicaModel uses ModelicaMixin but the Modelica file declares no equations,
        so linear_collocation is never set by the DAE linearity check. The LP file must
        still be produced with pump_on classified as a binary variable.
        """
        problem, content = _optimize_and_read_lp(EmptyModelicaModel)
        self.assertIn("Binary", content)
        binary_section = content.split("Binary")[1].split("End")[0]
        self.assertIn("pump_on__", binary_section)
        self.assertNotIn("General", content)

    def test_lp_export_no_dae_equations_no_modelica(self):
        """LP export must succeed for a model with no Modelica file at all (pure path variables).

        NoModelicaModel stubs out all DAE abstract methods as empty. The LP file must still
        be produced with pump_on classified as a binary variable.
        """
        problem, content = _optimize_and_read_lp(NoModelicaModel)
        self.assertIn("Binary", content)
        binary_section = content.split("Binary")[1].split("End")[0]
        self.assertIn("pump_on__", binary_section)
        self.assertNotIn("General", content)

    def test_user_constraint_names_in_lp(self):
        """User constraint names appear verbatim; path constraints get _t{i} suffix."""
        problem, content = _optimize_and_read_lp(_NamedConstraintModel)

        # terminal_x_bound is a range constraint — emits _lb and _ub lines
        self.assertRegex(content, r"terminal_x_bound_lb:.*x__")
        self.assertRegex(content, r"terminal_x_bound_ub:.*x__")
        n_times = len(np.linspace(0.0, 1.0, 9))
        for t in range(n_times):
            # Each path constraint row is a range — emits _lb and _ub lines
            self.assertRegex(content, rf"state_bound_t{t}_lb:.*x__")
            self.assertRegex(content, rf"state_bound_t{t}_ub:.*x__")

    def test_empty_constraint_name_falls_back_to_auto_generated(self):
        """An empty string name must be treated as absent, not raise ValueError."""
        problem, content = _optimize_and_read_lp(_EmptyNameConstraintModel)
        # Auto-generated fallback names must appear, not the empty string
        self.assertRegex(content, r"(?m)^\s*constraint_0_lb:")
        self.assertRegex(content, r"(?m)^\s*path_constraint_0_t0_lb:")

    def test_export_lp_raises_on_constraint_name_count_mismatch(self):
        """_export_lp_file must raise ValueError when _constraint_names length != g row count.

        This guards against transcribe() overrides that add constraints without extending
        _constraint_names, which would otherwise produce silently wrong LP label assignments.
        """
        from unittest.mock import patch

        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = LinearModel(output_folder=output_folder)
            # Patch _collint_constraint_names to return a list with the wrong length
            with patch.object(
                type(problem),
                "_collint_constraint_names",
                new_callable=lambda: property(lambda self: ["only_one_name"]),
            ):
                with self.assertRaises(ValueError) as ctx:
                    problem.optimize()
        self.assertIn("constraint_names length", str(ctx.exception))

    def test_track_constraint_names_option_populates_collint_names_without_lp(self):
        """track_constraint_names=True must populate _collint_constraint_names without export_lp.

        This is the programmatic-access use case: inspecting constraint names for debugging
        or downstream processing without writing an LP file to disk.
        """
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:

            class _EmitNamesModel(_BaseTestModel):
                def export_options(self):
                    options = super().export_options()
                    options["export_lp"] = False
                    options["track_constraint_names"] = True
                    return options

            problem = _EmitNamesModel(output_folder=output_folder)
            problem.optimize()
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]

        self.assertEqual(lp_files, [], "No LP file expected when export_lp=False")
        names = problem._collint_constraint_names
        self.assertIsNotNone(names, "_collint_constraint_names must be populated")
        # One name per row of g: the same 46 rows that the LP export of this model labels.
        self.assertEqual(len(names), 46)
        self.assertTrue(
            all(isinstance(n, str) for n in names), "All constraint names must be strings"
        )
        self.assertTrue(names[0].startswith("initial_residual_"))
        self.assertTrue(any(n.startswith("collocation_eq") for n in names))

    def test_crossed_variable_bounds_are_exported_before_the_solve_fails(self):
        """Crossed finite bounds make the solve fail, but the LP file is written first (so it
        can be inspected in an external solver) and the crossing is reported."""

        class _CrossedBoundsModel(_LinearModelHiGHS):
            def bounds(self):
                return {**super().bounds(), "x": (1.0, 0.0)}

        with tempfile.TemporaryDirectory() as output_folder:
            problem = _CrossedBoundsModel(output_folder=output_folder)
            with (
                self.assertLogs("rtctools", level="WARNING") as cm,
                self.assertRaises(RuntimeError),
            ):
                problem.optimize()
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 1)
            with open(os.path.join(output_folder, lp_files[0])) as f:
                content = f.read()
        self.assertTrue(any("lower bound above the upper bound" in m for m in cm.output))
        self.assertRegex(content, r"(?m)^\s*1 <= x__t\d+ <= 0\s*$")

    def test_constant_feasible_constraint_skipped_in_lp(self):
        """A constraint that reduces to a feasible constant must be silently skipped in the LP
        file and must not affect solver feasibility."""
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = _ConstantFeasibleConstraintModel(output_folder=output_folder)
            success = problem.optimize()
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            with open(os.path.join(output_folder, lp_files[0])) as f:
                content = f.read()

        # Solver must succeed — the constant constraint is feasible
        self.assertTrue(success)
        # The row exists in g (so the skip branch is what removes it) ...
        self.assertIn("const_feasible", problem._collint_constraint_names)
        # ... and the constant constraint must not appear in the LP file at all
        self.assertNotIn("const_feasible", content)
        self.assertNotIn("_constant_", content)
        # Positive control: the other rows are still written
        self.assertRegex(content, r"(?m)^\s*collocation_eq0_t0:")

    def test_single_pass_update_bounds_lp_export(self):
        """SinglePassGoalProgrammingMixin with UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS and 3 priorities
        must produce LP files without crashing (tests _constraint_names reset across passes)."""
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = _SinglePassUpdateBoundsModel(output_folder=output_folder)
            problem.optimize()
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
        self.assertEqual(
            len(lp_files), 3, f"Expected 3 LP files for 3-priority single-pass, got {lp_files}"
        )

    def test_single_pass_gp_lp_export(self):
        """SinglePassGoalProgrammingMixin with export_lp=True produces LP files.

        SinglePassGoalProgrammingMixin calls super().optimize() once per priority, so one LP
        file is produced per priority, with a _priority_N suffix matching GoalProgrammingMixin.
        The priority-2 LP must contain the priority-1 objective-tightening constraint.
        """
        with _silence_rtctools_logger(), tempfile.TemporaryDirectory() as output_folder:
            problem = _SinglePassLinearModel(output_folder=output_folder)
            problem.optimize()
            lp_files = sorted(f for f in os.listdir(output_folder) if f.endswith(".lp"))
            p2_file = next(f for f in lp_files if "_priority_2" in f)
            with open(os.path.join(output_folder, p2_file)) as fh:
                p2_content = fh.read()
        # Two priorities → two LP files (one per super().optimize() call)
        self.assertEqual(
            len(lp_files), 2, f"Expected 2 LP files for 2-priority single-pass, got {lp_files}"
        )
        for lp_file in lp_files:
            self.assertTrue(lp_file.startswith("_SinglePassLinearModel_"))
        # Priority suffixes must be present (priorities 1 and 2)
        suffixes = {f.split("_priority_")[1].split(".")[0] for f in lp_files if "_priority_" in f}
        self.assertEqual(suffixes, {"1", "2"})
        # The priority-2 LP must contain the priority-1 tightening constraint
        self.assertRegex(p2_content, r"single_pass_objective_p1__at_p2_0:")

    def _assert_raises_on_optimize(self, model_class, *expected_words):
        with tempfile.TemporaryDirectory() as output_folder:
            problem = model_class(output_folder=output_folder)
            with self.assertRaises(ValueError) as ctx:
                problem.optimize()
            error_msg = str(ctx.exception).lower()
            for word in expected_words:
                self.assertIn(word, error_msg)
            lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
            self.assertEqual(len(lp_files), 0)


# LP solve round-trip: RTC-Tools (HiGHS) vs standalone HiGHS on exported LP


class TestBoundConflicts(unittest.TestCase):
    """check_bound_conflicts rejects bounds an LP file cannot express (+inf lower, -inf upper)."""

    def test_no_conflict(self):
        lb = np.array([-np.inf, 0.0, 1.0, -np.inf, np.nan, 2.0])
        ub = np.array([np.inf, 0.0, 2.0, -1.0, 1.0, 1.0])  # equal, crossed finite and NaN pass
        check_bound_conflicts(lb, ub, "variable", str)

    def test_empty(self):
        check_bound_conflicts(np.array([]), np.array([]), "variable", str)

    def test_conflicts_are_named(self):
        cases = [
            ("lower_posinf", np.inf, np.inf, "lower=+Inf, upper=+Inf"),
            ("lower_posinf_upper_finite", np.inf, 5.0, "lower=+Inf, upper=5"),
            ("upper_neginf", -np.inf, -np.inf, "lower=-Inf, upper=-Inf"),
            ("lower_finite_upper_neginf", 0.0, -np.inf, "lower=0, upper=-Inf"),
            ("posinf_and_neginf", np.inf, -np.inf, "lower=+Inf, upper=-Inf"),
        ]
        for case_name, lb, ub, expected in cases:
            for kind in ["variable", "constraint"]:
                with self.subTest(case=case_name, kind=kind):
                    with self.assertRaises(ValueError) as ctx:
                        check_bound_conflicts(
                            np.array([0.0, lb]), np.array([1.0, ub]), kind, lambda i: f"v{i}"
                        )
                    msg = str(ctx.exception)
                    self.assertIn(f"1 {kind}(s)", msg)
                    self.assertIn(f"v1 ({expected})", msg)
                    self.assertNotIn("v0", msg)

    def test_names_map_to_indices_of_conflicting_entries(self):
        lb = np.array([0.0, 0.0, np.inf, 0.0, 5.0])
        ub = np.array([1.0, 1.0, 2.0, -np.inf, 6.0])
        with self.assertRaises(ValueError) as ctx:
            check_bound_conflicts(lb, ub, "variable", lambda i: f"v{i}")
        msg = str(ctx.exception)
        self.assertIn("2 variable(s)", msg)
        self.assertIn("v2 (lower=+Inf, upper=2), v3 (lower=0, upper=-Inf)", msg)

    def test_many_conflicts_are_capped(self):
        # Exactly at the cap there is no "more" suffix, one above it there is.
        for n, expected_suffix in [(10, None), (11, "(1 more)"), (25, "(15 more)")]:
            with self.subTest(n=n):
                name_of = mock.Mock(side_effect=lambda i: f"v{i}")
                with self.assertRaises(ValueError) as ctx:
                    check_bound_conflicts(np.full(n, np.inf), np.zeros(n), "variable", name_of)
                msg = str(ctx.exception)
                self.assertIn(f"{n} variable(s)", msg)
                if expected_suffix is None:
                    self.assertNotIn("more)", msg)
                else:
                    self.assertIn(expected_suffix, msg)
                # Names are only computed for the listed entries
                self.assertEqual(name_of.call_count, 10)


class TestWarnCrossedBounds(unittest.TestCase):
    """Crossed finite bounds are written as given; a warning names the affected entries."""

    def test_warns_and_names_crossed_entries(self):
        lb = np.array([0.0, 2.0, 1.0, np.nan, 3.0])
        ub = np.array([1.0, 1.0, 1.0, 0.0, 2.0])
        with self.assertLogs("rtctools", level="WARNING") as cm:
            warn_crossed_bounds(lb, ub, "constraint", lambda i: f"c{i}")
        self.assertEqual(len(cm.output), 1)
        self.assertIn("2 constraint(s)", cm.output[0])
        self.assertIn("c1, c4", cm.output[0])
        self.assertNotIn("c0", cm.output[0])

    def test_one_ulp_crossing_warns_and_equal_bounds_do_not(self):
        with mock.patch("rtctools._internal.casadi_to_lp.logger") as logger:
            warn_crossed_bounds(np.array([1.0]), np.array([1.0]), "variable", str)
            logger.warning.assert_not_called()
            warn_crossed_bounds(
                np.array([1.0]), np.array([np.nextafter(1.0, 0.0)]), "variable", str
            )
            logger.warning.assert_called_once()

    def test_empty_and_no_crossing_do_not_warn(self):
        with mock.patch("rtctools._internal.casadi_to_lp.logger") as logger:
            warn_crossed_bounds(np.array([]), np.array([]), "variable", str)
            warn_crossed_bounds(np.array([0.0, -np.inf]), np.array([1.0, np.inf]), "v", str)
            logger.warning.assert_not_called()


class TestListNames(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_list_names([]), "")

    def test_cap_boundary(self):
        names = [f"n{i}" for i in range(11)]
        self.assertNotIn("more", _list_names(names[:10]))
        self.assertTrue(_list_names(names).endswith(", ... (1 more)"))
        self.assertNotIn("'n10'", _list_names(names))

    def test_fmt_only_called_for_listed_items(self):
        fmt = mock.Mock(side_effect=str)
        expected = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... (2 more)"
        self.assertEqual(_list_names(list(range(12)), fmt=fmt), expected)
        self.assertEqual(fmt.call_count, 10)

    def test_default_quotes(self):
        self.assertEqual(_list_names(["a", "b"]), "'a', 'b'")


class TestLPSolveRoundTrip(unittest.TestCase):
    """Exported LP files must be re-solvable by HiGHS with objectives matching RTC-Tools.

    Both sides use HiGHS (CasADi/qpsol for RTC-Tools, standalone highspy for re-solve),
    so any discrepancy points to an incorrect LP export rather than inter-solver differences.

    Three scenarios are tested:
    - Plain LP: single optimize() call, one LP file.
    - Goal programming: two priorities, two LP files. The priority-2 LP must carry the
      epsilon-tightening constraint from priority 1; a missing constraint would let HiGHS
      find a lower objective than RTC-Tools reported.
    - Control tree: non-anticipativity encoded via shared pre-branching control indices.
      If variable sharing is broken in the export, HiGHS finds a different objective.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import highspy  # noqa: F401

            cls._skip_reason = None
        except ImportError as e:
            try:
                importlib.metadata.version("highspy")
            except importlib.metadata.PackageNotFoundError:
                cls._skip_reason = f"highspy not installed: {e}"
                return
            # Installed but not importable: do not hide the round-trip coverage behind a skip.
            raise

        import shutil

        cls._tmp_dir = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls._tmp_dir, ignore_errors=True)
        with _silence_rtctools_logger():
            # --- Plain LP scenario ---
            with tempfile.TemporaryDirectory() as output_folder:
                problem = _LinearModelHiGHS(output_folder=output_folder)
                problem.optimize()
                cls._plain_rtctools_obj = problem.objective_value
                lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
                cls._plain_lp_path = os.path.join(cls._tmp_dir, lp_files[0])
                shutil.copy(os.path.join(output_folder, lp_files[0]), cls._plain_lp_path)

            # --- Goal programming scenario ---
            with tempfile.TemporaryDirectory() as output_folder:
                problem = _GPLinearModel(output_folder=output_folder)
                problem.optimize()
                cls._gp_rtctools_final_obj = problem.objective_value
                lp_files = sorted(f for f in os.listdir(output_folder) if f.endswith(".lp"))
                if len(lp_files) != 2:
                    raise AssertionError(
                        f"Expected 2 LP files for goal programming scenario, got {lp_files}"
                    )
                cls._gp_lp_files = []
                for lp_f in lp_files:
                    dst = os.path.join(cls._tmp_dir, lp_f)
                    shutil.copy(os.path.join(output_folder, lp_f), dst)
                    cls._gp_lp_files.append(dst)
                # Check: priority-1 goal is binding for priority-2 comparison
                p1_obj = cls._highs_solve(cls._gp_lp_files[0])
                if not p1_obj > 1e-4:
                    raise AssertionError(
                        f"Priority-1 LP objective {p1_obj} is ~0; goal is not binding. "
                        "The test setup is invalid — tighten target_max in _GPGoalPriority1."
                    )

            # --- Control tree scenario ---
            with tempfile.TemporaryDirectory() as output_folder:
                problem = _ControlTreeLinearModel(output_folder=output_folder)
                problem.optimize()
                cls._ct_rtctools_obj = problem.objective_value
                lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
                if len(lp_files) != 1:
                    raise AssertionError(
                        f"Expected 1 LP file for control tree scenario, got {lp_files}"
                    )
                cls._ct_lp_path = os.path.join(cls._tmp_dir, lp_files[0])
                shutil.copy(os.path.join(output_folder, lp_files[0]), cls._ct_lp_path)
                with open(cls._ct_lp_path) as f:
                    cls._ct_lp_content = f.read()

            # --- MinAbs goal programming scenario ---
            with tempfile.TemporaryDirectory() as output_folder:
                problem = _MinAbsLinearModel(output_folder=output_folder)
                problem.optimize()
                lp_files = [f for f in os.listdir(output_folder) if f.endswith(".lp")]
                if len(lp_files) != 1:
                    raise AssertionError(f"Expected 1 LP file for MinAbs scenario, got {lp_files}")
                with open(os.path.join(output_folder, lp_files[0])) as f:
                    cls._minabs_lp_content = f.read()

            # --- keep_soft_constraints scenario (pareto constraint) ---
            with tempfile.TemporaryDirectory() as output_folder:
                problem = _GPKeepSoftModel(output_folder=output_folder)
                problem.optimize()
                lp_files = sorted(f for f in os.listdir(output_folder) if f.endswith(".lp"))
                if len(lp_files) != 2:
                    raise AssertionError(
                        f"Expected 2 LP files for keep_soft_constraints scenario, got {lp_files}"
                    )
                with open(os.path.join(output_folder, lp_files[1])) as f:
                    cls._keep_soft_lp2_content = f.read()

    def setUp(self):
        if self._skip_reason:
            self.skipTest(self._skip_reason)

    @staticmethod
    def _highs_solve(lp_path):
        import highspy

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(lp_path)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise AssertionError(f"HiGHS did not reach optimality on {lp_path}")
        return h.getObjectiveValue()

    def _highs_status_of(self, obj_str, constr_str, bounds_str):
        import highspy

        with tempfile.TemporaryDirectory() as output_folder:
            _write_lp_file("t.lp", obj_str, constr_str, bounds_str, [], [], output_folder)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(os.path.join(output_folder, "t.lp"))
            h.run()
            return h.modelStatusToString(h.getModelStatus()), h.getObjectiveValue()

    def test_constant_infeasible_rows_round_trip(self):
        """Infeasible constant rows must stay infeasible after export, also for violations
        smaller than 1 (regression: the dummy variable used to be fixed to 1)."""
        X = ca.SX.sym("X", 1)
        cases = [
            ("lower", ca.SX(0.0), 0.5, np.inf),
            ("upper", ca.SX(0.0), -np.inf, -0.5),
            ("equality", ca.SX(0.0), 0.5, 0.5),
            ("range", ca.SX(0.0), 0.25, 0.75),
            ("crossed", ca.SX(0.0), 0.75, 0.25),
        ]
        for case_name, const, lb, ub in cases:
            with self.subTest(case=case_name):
                g = ca.vertcat(X[0], const)
                with _silence_rtctools_logger(), self.assertLogs("rtctools", "WARNING"):
                    constr_str, extra = _build_constraints(
                        g, X, [0.0, lb], [1.0, ub], ["x0"], constraint_names=["box", "c"]
                    )
                bounds_str, _, _ = _build_bounds(["x0"], [-10.0], [10.0], [False])
                status, _ = self._highs_status_of("x0", constr_str, bounds_str + "\n" + extra)
                self.assertEqual(status, "Infeasible")

    def test_upper_only_variable_bound_round_trip(self):
        """A variable with lb=-inf and finite ub must be free below in the exported LP."""
        X = ca.SX.sym("X", 1)
        constr_str, _ = _build_constraints(X[0], X, [-100.0], [np.inf], ["x0"])
        bounds_str, _, _ = _build_bounds(["x0"], [-np.inf], [5.0], [False])
        status, obj = self._highs_status_of("x0", constr_str, bounds_str)
        self.assertEqual(status, "Optimal")
        self.assertAlmostEqual(obj, -100.0)

    def test_names_starting_with_digit_or_period_round_trip(self):
        """Names that start with a digit or period are prefixed so HiGHS can read the file."""
        import highspy

        X = ca.SX.sym("X", 1)
        g = ca.vertcat(X[0], 2 * X[0])
        with _silence_rtctools_logger():
            constr_str, _ = _build_constraints(
                g, X, [1.0, 2.0], [np.inf, np.inf], ["x0"], constraint_names=["1st_bound", ".dot"]
            )
            bounds_str, _, _ = _build_bounds(["x0"], [0.0], [10.0], [False])
        self.assertRegex(constr_str, r"(?m)^\s*_1st_bound:")
        self.assertRegex(constr_str, r"(?m)^\s*_\.dot:")
        with tempfile.TemporaryDirectory() as output_folder:
            _write_lp_file("t.lp", "x0", constr_str, bounds_str, [], [], output_folder)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            self.assertEqual(
                h.readModel(os.path.join(output_folder, "t.lp")), highspy.HighsStatus.kOk
            )
            h.run()
            self.assertEqual(h.getModelStatus(), highspy.HighsModelStatus.kOptimal)

    def test_name_with_backslash_does_not_drop_row(self):
        """A backslash starts a comment in the LP format; it must not swallow the row."""
        import highspy

        X = ca.SX.sym("X", 1)
        g = ca.vertcat(X[0], 2 * X[0])
        with _silence_rtctools_logger():
            constr_str, _ = _build_constraints(
                g, X, [1.0, 2.0], [np.inf, np.inf], ["x0"], constraint_names=["\\critical", "ok"]
            )
            bounds_str, _, _ = _build_bounds(["x0"], [0.0], [10.0], [False])
        self.assertNotIn("\\", constr_str)
        with tempfile.TemporaryDirectory() as output_folder:
            _write_lp_file("t.lp", "x0", constr_str, bounds_str, [], [], output_folder)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(os.path.join(output_folder, "t.lp"))
            self.assertEqual(h.getNumRow(), 2)

    def test_crossed_bounds_round_trip(self):
        """Crossed finite bounds are written as given and stay infeasible in HiGHS."""
        X = ca.SX.sym("X", 1)
        with _silence_rtctools_logger():
            row_str, _ = _build_constraints(X[0], X, [2.0], [1.0], ["x0"])
            var_bounds, _, _ = _build_bounds(["x0"], [-10.0], [10.0], [False])
            crossed_var_bounds, _, _ = _build_bounds(["x0"], [2.0], [1.0], [False])
        status, _ = self._highs_status_of("x0", row_str, var_bounds)
        self.assertEqual(status, "Infeasible", "crossed constraint row")
        status, _ = self._highs_status_of("x0", "  1 x0 >= -100", crossed_var_bounds)
        self.assertEqual(status, "Infeasible", "crossed variable bounds")

    def test_plain_lp_round_trip(self):
        """HiGHS re-solving the plain LP must reproduce the RTC-Tools objective."""
        self.assertAlmostEqual(
            self._highs_solve(self._plain_lp_path),
            self._plain_rtctools_obj,
            places=5,
            msg="Plain LP: HiGHS objective differs from RTC-Tools. Export may be incorrect.",
        )

    def test_gp_epsilon_constraint_carried_over(self):
        """HiGHS on the priority-2 LP must reproduce the RTC-Tools final objective.

        The priority-2 goal conflicts with priority 1, so the carried-over priority-1 row is
        binding: removing it from the LP must change the optimum (checked below), which makes
        the objective comparison sensitive to that row being present.
        """
        with open(self._gp_lp_files[1]) as f:
            lp2 = f.read()
        self.assertRegex(lp2, r"(?m)^\s*_GPGoalPriority1_p1:\s*1 \S+ <= ")
        without_row = "\n".join(
            line for line in lp2.splitlines() if not line.strip().startswith("_GPGoalPriority1_p1:")
        )
        reduced_path = os.path.join(self._tmp_dir, "gp_priority_2_without_p1_row.lp")
        with open(reduced_path, "w") as f:
            f.write(without_row)
        self.assertGreater(
            abs(self._highs_solve(reduced_path) - self._gp_rtctools_final_obj),
            1e-3,
            "Test setup invalid: the carried-over priority-1 row does not bind at priority 2.",
        )
        highs_obj = self._highs_solve(self._gp_lp_files[1])
        self.assertAlmostEqual(
            highs_obj,
            self._gp_rtctools_final_obj,
            places=5,
            msg=(
                f"Goal programming priority-2 LP: HiGHS objective ({highs_obj}) differs from "
                f"RTC-Tools ({self._gp_rtctools_final_obj}). The priority-2 LP may be missing "
                "the epsilon-tightening constraint from priority 1."
            ),
        )

    def test_control_tree_lp_round_trip(self):
        """HiGHS re-solving the control tree LP must reproduce the RTC-Tools objective.

        The control tree encodes non-anticipativity by sharing control variable indices
        across ensemble members before the branching time. If the LP export incorrectly
        duplicates or splits those shared variables, HiGHS would find a different objective.
        """
        highs_obj = self._highs_solve(self._ct_lp_path)
        self.assertAlmostEqual(
            highs_obj,
            self._ct_rtctools_obj,
            places=5,
            msg=(
                f"Control tree LP: HiGHS objective ({highs_obj}) differs from "
                f"RTC-Tools ({self._ct_rtctools_obj}). The LP export may have broken "
                "non-anticipativity variable sharing."
            ),
        )

    def test_control_tree_lp_variable_sharing(self):
        """ControlTree LP export: pre-branching controls are shared, later ones per member.

        With branching at t=0.5 (time steps 0..8), the control slots at steps 0..3 are
        shared by all 3 members and carry no member suffix; from step 4 each member has its
        own slot. States are always per-member.
        """
        content = self._ct_lp_content

        shared = set(re.findall(r"\bu__t(\d+)(?!__m)\b", content))
        self.assertEqual(shared, {"0", "1", "2", "3"})
        per_member = set(re.findall(r"\bu__t(\d+)__m(\d+)\b", content))
        self.assertEqual(per_member, {(str(t), str(m)) for t in range(4, 9) for m in range(3)})

        # Each shared control occurs in the rows of every member
        for member in range(3):
            row = re.search(rf"(?m)^\s*initial_residual_0_m{member}:.*$", content)
            self.assertIsNotNone(row)
            self.assertRegex(row.group(0), r"\bu__t0\b")

        # States are always per-member
        bounds_section = content.split("Bounds")[1].split("End")[0]
        self.assertRegex(bounds_section, r"x__t\d+__m0")
        self.assertRegex(bounds_section, r"x__t\d+__m1")
        self.assertRegex(bounds_section, r"x__t\d+__m2")

    def test_goal_constraint_names_in_lp(self):
        """Goal constraints must appear with class-name_p{priority} labels and correct bodies."""
        with open(self._gp_lp_files[0]) as f:
            lp1 = f.read()
        with open(self._gp_lp_files[1]) as f:
            lp2 = f.read()
        # Priority-1 LP: goal constraint names the epsilon variable on its LHS
        self.assertRegex(lp1, r"(?m)^\s*_GPGoalPriority1_p1:.*eps_0_0 <= -2")
        # Priority-2 LP: tightened priority-1 constraint carried over (no epsilon — fixed bound)
        self.assertRegex(lp2, r"(?m)^\s*_GPGoalPriority1_p1:(?!.*eps).*<= ")
        # Priority-2 LP: new priority-2 goal constraint with its own epsilon
        self.assertRegex(lp2, r"(?m)^\s*_GPGoalPriority2_p2:.*eps_1_0 >= 3")

    def test_pareto_constraint_names_in_lp(self):
        """keep_soft_constraints=True must produce pareto_p{N} constraints in the LP."""
        lp2 = self._keep_soft_lp2_content
        # Priority-2 LP must carry a pareto_p1 constraint locking the priority-1 objective
        self.assertRegex(lp2, r"(?m)^\s*pareto_p1: 1 eps_0_0 <= ")

    def test_min_abs_constraint_names_in_lp(self):
        """MinAbsGoal auxiliary constraints must appear as {Class}_p{N}_abs_pos/neg."""
        content = self._minabs_lp_content
        self.assertRegex(content, r"(?m)^\s*_MinAbsStateGoal_p\d+_abs_pos:.* abs_\S+ >= 0")
        self.assertRegex(content, r"(?m)^\s*_MinAbsStateGoal_p\d+_abs_neg:\s*-1 .* abs_\S+ >= 0")


class _GPGoalPriority1(Goal):
    """Goal at priority 1 that is intentionally infeasible — forces a non-zero epsilon.

    target_max=-2.0 is unreachable: the model's state x is bounded from below by its
    dynamics and control bounds, so the optimizer cannot satisfy it exactly. This
    guarantees the epsilon variable is binding and the LP objective is non-trivial.
    """

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-5.0, 5.0)
    priority = 1
    target_max = -2.0
    order = 1  # linear penalty — keeps the transcribed LP affine


class _GPGoalPriority2(Goal):
    """Goal at priority 2 that conflicts with priority 1.

    Priority 1 pushes x(0.5) down (target_max=-2.0, unreachable); this goal wants x(0.5) up
    (target_min=3.0). The priority-1 result is carried over as a bound, so dropping that
    constraint from the priority-2 LP lets the optimizer lower the priority-2 objective.
    """

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-5.0, 5.0)
    priority = 2
    target_min = 3.0
    order = 1


class _GPLinearModel(GoalProgrammingMixin, _BaseTestModel):
    """Linear model with two-priority goals, both binding. Uses order=1 for affine LP export."""

    def goals(self):
        return [_GPGoalPriority1(), _GPGoalPriority2()]

    def solver_options(self):
        options = super().solver_options()
        # Use HiGHS via CasADi so both RTC-Tools and the standalone HiGHS re-solve
        # use the same solver, eliminating inter-solver numerical differences.
        options["casadi_solver"] = ca.qpsol
        options["solver"] = "highs"
        options["highs"] = {"output_flag": False}
        return options


class _ControlTreeLinearModel(ControlTreeMixin, _LinearModelHiGHS):
    """Linear model with a control tree: 3 members, branching at t=0.5.

    Before t=0.5 all members share the same control decisions (non-anticipativity).
    After t=0.5 each member has its own independent controls.
    The objective is linear so the exported LP is affine.
    """

    @property
    def ensemble_size(self):
        return 3

    def control_tree_options(self):
        return {
            "forecast_variables": ["constant_input"],
            "branching_times": [0.5],
            "k": 3,
        }

    def constant_inputs(self, ensemble_member):
        # Distinct forecasts so members are assigned to different post-branching paths
        start = 1.0 - 0.1 * ensemble_member
        stop = 0.5 + 0.1 * ensemble_member
        values = np.linspace(start, stop, len(self.times()))
        return {"constant_input": Timeseries(self.times(), values)}


class _MinAbsStateGoal(MinAbsGoal):
    """Minimize |x| at a mid-point time — produces abs_pos/abs_neg auxiliary constraints."""

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_nominal = 1.0
    priority = 1


class _MinAbsLinearModel(MinAbsGoalProgrammingMixin, GoalProgrammingMixin, _BaseTestModel):
    """Linear model with a MinAbsGoal — exercises abs_pos/abs_neg constraint naming."""

    def min_abs_goals(self):
        return [_MinAbsStateGoal()]


class _GPKeepSoftModel(GoalProgrammingMixin, _BaseTestModel):
    """Goal programming model with keep_soft_constraints=True — produces pareto_p{N} constraints."""

    def goals(self):
        return [_GPGoalPriority1(), _GPGoalPriority2()]

    def goal_programming_options(self):
        options = super().goal_programming_options()
        options["keep_soft_constraints"] = True
        options["fix_minimized_values"] = False
        return options


class _GPGoalPriority3(Goal):
    """Infeasible goal at priority 3 — used to exercise 3-priority single-pass LP export."""

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 1.0, ensemble_member=ensemble_member)

    function_range = (-5.0, 5.0)
    priority = 3
    target_max = -4.0  # also unreachable
    order = 1


class _SinglePassLinearModel(SinglePassGoalProgrammingMixin, _BaseTestModel):
    """Single-pass goal programming model with two priorities — smoke test for LP export."""

    def goals(self):
        return [_GPGoalPriority1(), _GPGoalPriority2()]

    def solver_options(self):
        options = super().solver_options()
        options["casadi_solver"] = ca.qpsol
        options["solver"] = "highs"
        options["highs"] = {"output_flag": False}
        return options


class _SinglePassUpdateBoundsModel(SinglePassGoalProgrammingMixin, _BaseTestModel):
    """Single-pass model with UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS and 3 priorities.

    Exercises the _constraint_names reset-per-pass fix: with UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS,
    all priority objective constraints are added to g from pass 1 onward. Without the reset,
    _constraint_names would grow unboundedly across passes, causing a ValueError on pass 3.
    """

    single_pass_method = SinglePassMethod.UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS

    def goals(self):
        return [_GPGoalPriority1(), _GPGoalPriority2(), _GPGoalPriority3()]

    def solver_options(self):
        options = super().solver_options()
        options["casadi_solver"] = ca.qpsol
        options["solver"] = "highs"
        options["highs"] = {"output_flag": False}
        return options
