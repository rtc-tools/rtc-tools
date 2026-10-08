import logging

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")


class SingleShootingBaseModel(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="SingleShootingModel",
            model_folder=data_path(),
        )

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    @property
    def integrate_states(self):
        return True

    def pre(self):
        # Do nothing
        pass

    def constant_inputs(self, ensemble_member):
        return {"constant_input": Timeseries(self.times(), np.linspace(1.0, 0.0, 21))}

    def bounds(self):
        # Variable bounds
        return {"u": (-2.0, 2.0)}

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class SingleShootingModel(SingleShootingBaseModel):
    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2


class TestSingleShooting(TestCase):
    def test_objective_value(self):
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, self.tolerance)

    def test_state(self):
        times = self.problem.times()
        parameters = self.problem.parameters(0)
        self.assertAlmostEqual(
            (self.results["x"][1:] - self.results["x"][:-1]) / (times[1:] - times[:-1]),
            parameters["k"] * self.results["x"][1:] + self.results["u"][1:],
            self.tolerance,
        )
        self.assertAlmostEqual(
            (self.results["w"][1:] - self.results["w"][:-1]) / (times[1:] - times[:-1]),
            self.results["x"][1:],
            self.tolerance,
        )

    def test_algebraic_variable(self):
        constant_input = self.problem.constant_inputs(0)["constant_input"]
        self.assertAlmostEqual(
            self.results["a"],
            self.results["x"] + self.results["w"] + constant_input.values,
            self.tolerance,
        )

    def setUp(self):
        self.problem = SingleShootingModel()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6


class SingleShootingGoalProgrammingModel(GoalProgrammingMixin, SingleShootingBaseModel):
    def goals(self):
        return [Goal1(), Goal2(), Goal3()]

    def path_goals(self):
        return [PathGoal1()]

    def set_timeseries(self, timeseries_id, timeseries, ensemble_member, **kwargs):
        # Do nothing
        pass


class PathGoal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 1
    target_min = -0.9e1
    target_max = 0.9e1


class Goal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 3
    target_min = 0.0


class Goal2(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.7, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 3
    target_min = 0.1


class Goal3(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("x", 0.1, 1.0, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 2
    target_max = 1.0


class TestGoalProgramming(TestCase):
    def setUp(self):
        self.problem = SingleShootingGoalProgrammingModel()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        self.assertAlmostGreaterThan(
            self.problem.interpolate(
                0.7, self.problem.times(), self.problem.extract_results()["x"]
            ),
            0.1,
            self.tolerance,
        )


class SingleShootingEnsembleModel(SingleShootingBaseModel):
    @property
    def ensemble_size(self):
        return 3

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        if ensemble_member == 0:
            return {"constant_input": Timeseries(self.times(), np.linspace(1.0, 0.0, 21))}
        elif ensemble_member == 1:
            return {"constant_input": Timeseries(self.times(), np.linspace(0.99, 0.5, 21))}
        else:
            return {"constant_input": Timeseries(self.times(), np.linspace(0.98, 1.0, 21))}


class TestEnsemble(TestCase):
    def setUp(self):
        self.problem = SingleShootingEnsembleModel()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_states(self):
        times = self.problem.times()
        for ensemble_member in range(self.problem.ensemble_size):
            parameters = self.problem.parameters(ensemble_member)
            results = self.problem.extract_results(ensemble_member)
            self.assertAlmostEqual(
                (results["x"][1:] - results["x"][:-1]) / (times[1:] - times[:-1]),
                parameters["k"] * results["x"][1:] + results["u"][1:],
                self.tolerance,
            )
            self.assertAlmostEqual(
                (results["w"][1:] - results["w"][:-1]) / (times[1:] - times[:-1]),
                results["x"][1:],
                self.tolerance,
            )

    def test_algebraic_variables(self):
        for ensemble_member in range(self.problem.ensemble_size):
            results = self.problem.extract_results(ensemble_member)
            constant_input = self.problem.constant_inputs(ensemble_member)["constant_input"]
            self.assertAlmostEqual(
                results["a"], results["x"] + results["w"] + constant_input.values, self.tolerance
            )


class SingleShootingEnsembleParametersModel(SingleShootingBaseModel):
    @property
    def ensemble_size(self):
        return 3

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["k"] = 1.0 + ensemble_member * 0.01
        return parameters

    def pre(self):
        super().pre()

        # NOTE: The first parameter (for ensemble_member=0) has to be 1 for it
        # to trigger the bug this test was introduced to cover.
        assert self.parameters(0)["k"] == 1.0


class TestEnsembleParameters(TestCase):
    def setUp(self):
        self.problem = SingleShootingEnsembleParametersModel()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_states(self):
        times = self.problem.times()
        for ensemble_member in range(self.problem.ensemble_size):
            parameters = self.problem.parameters(ensemble_member)
            results = self.problem.extract_results(ensemble_member)
            self.assertAlmostEqual(
                (results["x"][1:] - results["x"][:-1]) / (times[1:] - times[:-1]),
                parameters["k"] * results["x"][1:] + results["u"][1:],
                self.tolerance,
            )
            self.assertAlmostEqual(
                (results["w"][1:] - results["w"][:-1]) / (times[1:] - times[:-1]),
                results["x"][1:],
                self.tolerance,
            )

    def test_algebraic_variables(self):
        for ensemble_member in range(self.problem.ensemble_size):
            results = self.problem.extract_results(ensemble_member)
            constant_input = self.problem.constant_inputs(ensemble_member)["constant_input"]
            self.assertAlmostEqual(
                results["a"], results["x"] + results["w"] + constant_input.values, self.tolerance
            )


class OldApiErrorModel(SingleShootingBaseModel):
    """
    Test that an exception is raised when the old API is used, and that
    it points users to the new API.
    """

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2

    @property
    def integrated_states(self):
        return [*self.algebraic_states, *self.differentiated_states]

    @property
    def integrate_states(self):
        return False


class OldApiErrorTest(TestCase):
    def test_integration_exception_raised(self):
        with self.assertRaisesRegex(
            Exception,
            "The integrated_states property is no longer supported. Use integrate_states instead",
        ):
            self.problem.optimize()

    def setUp(self):
        self.problem = OldApiErrorModel()


class SingleShootingEnsembleBoundsModel(SingleShootingEnsembleModel):
    ensemble_specific_bounds = True

    def bounds(self, ensemble_member: int):
        if ensemble_member == 0:
            return {"u": (-1.0, 1.0)}  # Tighter bounds for ensemble member 0
        elif ensemble_member == 1:
            return {"u": (-1.5, 1.5)}  # Medium bounds for ensemble member 1
        else:
            return {"u": (-2.0, 2.0)}  # Default bounds for ensemble member 2


class TestEnsembleBounds(TestCase):
    def setUp(self):
        self.problem = SingleShootingEnsembleBoundsModel()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_ensemble_bounds_applied(self):
        # Test that different ensemble members have different effective bounds
        # by checking that the control inputs respect their individual bounds
        for ensemble_member in range(self.problem.ensemble_size):
            results = self.problem.extract_results(ensemble_member)
            bounds = self.problem.bounds(ensemble_member)
            u_bounds = bounds["u"]

            # Check that all control values are within the specified bounds
            self.assertTrue(np.all(results["u"] >= u_bounds[0] - self.tolerance))
            self.assertTrue(np.all(results["u"] <= u_bounds[1] + self.tolerance))


class SingleShootingConflictingBoundsModel(SingleShootingEnsembleModel):
    ensemble_specific_bounds = True

    def bounds(self, ensemble_member: int):
        if ensemble_member == 0:
            return {"u": (0.5, 1.0)}  # Lower bound 0.5, upper bound 1.0
        elif ensemble_member == 1:
            return {"u": (-1.0, 0.3)}  # Lower bound -1.0, upper bound 0.3
        else:
            return {"u": (-2.0, 2.0)}  # Default bounds


class TestConflictingBounds(TestCase):
    def test_conflicting_bounds_error(self):
        # Test that conflicting bounds (where max lower > min upper) raise an error
        problem = SingleShootingConflictingBoundsModel()
        with self.assertRaises(RuntimeError), self.assertLogs(level="ERROR") as cm:
            problem.optimize()

        self.assertIn("Lower bound 0.5 is higher than upper bound 0.3 for variable u", cm.output[0])
        self.assertRegex(cm.output[0], r"first at entry \d+ of \d+; \d+ entries conflict\)")
        self.assertIn("may be infeasible", cm.output[0])


class SingleShootingScalarConflictingBoundsModel(SingleShootingModel):
    """Integrated states have a single entry (int index) in the state vector."""

    def bounds(self):
        return {**super().bounds(), "x": (1.0, 0.0)}


class TestConflictingScalarBounds(TestCase):
    def test_conflicting_bounds_on_scalar_variable_are_logged(self):
        problem = SingleShootingScalarConflictingBoundsModel()
        with self.assertRaises(RuntimeError), self.assertLogs(level="ERROR") as cm:
            problem.optimize()

        self.assertTrue(
            any("is higher than upper bound" in msg and "variable x" in msg for msg in cm.output),
            cm.output,
        )


class SingleShootingEnsembleStatesBoundsModel(SingleShootingEnsembleModel):
    ensemble_specific_bounds = True

    def bounds(self, ensemble_member: int):
        if ensemble_member == 0:
            return {"x": (0.0, 2.0), "u": (-1.0, 1.0)}
        elif ensemble_member == 1:
            return {"x": (1.0, 3.0), "u": (-1.5, 1.5)}
        else:
            return {"x": (-1.0, 1.0), "u": (-2.0, 2.0)}

    def transcribe(self):
        discrete, lbx, ubx, lbg, ubg, x0, nlp = super().transcribe()

        self.test_lbx = lbx
        self.test_ubx = ubx

        return discrete, lbx, ubx, lbg, ubg, x0, nlp


class TestEnsembleBoundsStates(TestCase):
    def test_bounds_apply_per_member(self):
        problem = SingleShootingEnsembleStatesBoundsModel()
        problem.optimize()

        # TODO: Update this when we expose the indices as a public property
        indices = problem._CollocatedIntegratedOptimizationProblem__indices_as_lists

        bounds = [
            problem.bounds(ensemble_member) for ensemble_member in range(problem.ensemble_size)
        ]

        for ensemble_member in range(problem.ensemble_size):
            indices_variable = indices[ensemble_member]["x"]
            lb, ub = bounds[ensemble_member]["x"]
            np.testing.assert_array_equal(
                problem.test_lbx[indices_variable], np.full_like(indices_variable, lb)
            )
            np.testing.assert_array_equal(
                problem.test_ubx[indices_variable], np.full_like(indices_variable, ub)
            )


class SingleShootingEnsembleMissingBoundsModel(SingleShootingEnsembleModel):
    ensemble_specific_bounds = True

    def bounds(self, ensemble_member: int):
        if ensemble_member == 0:
            return {"x": (0.0, 2.0), "u": (-1.0, 1.0)}
        elif ensemble_member == 1:
            return {"x": (1.0, 3.0), "u": (-1.5, 1.5)}
        else:
            # u is missing bounds for member 2, will give a warning
            return {"x": (-1.0, 1.0)}


class TestEnsembleMissingBounds(TestCase):
    def test_ensemble_member_missing_bounds_control_input_warning(self):
        problem = SingleShootingEnsembleMissingBoundsModel()

        with self.assertLogs(level="WARNING") as cm:
            problem.optimize()

        self.assertIn(
            "OptimizationProblem: control input u has no bounds (ensemble_member=2)", cm.output[0]
        )


class EmptyStatesInIntervalModel(SingleShootingBaseModel):
    def _states_in_call(self, ensemble_member):
        """Separate method so we can also test it on its own outside of the
        objective function."""

        times = self.times()
        time_steps = np.diff(times)
        # Request an "empty" range of states; we expect this to return just t0
        # and tf.
        s = self.states_in(
            "x",
            t0=times[-2] + 0.1 * time_steps[-1],
            tf=times[-1] - 0.1 * time_steps[-1],
            ensemble_member=ensemble_member,
        )

        return s

    def objective(self, ensemble_member):
        return self._states_in_call(ensemble_member)[1] ** 2

    def interpolation_method(self, variable_name):
        if variable_name == "x":
            return self.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD
        else:
            return super().interpolation_method(variable_name)


class TestEmptyStatesInInterval(TestCase):
    def setUp(self):
        self.problem = EmptyStatesInIntervalModel()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_backwards_interpolated_empty_interval(self):
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, self.tolerance)

    def test_states_in_call_shape(self):
        s = self.problem._states_in_call(0)
        self.assertEqual(s.shape, (2, 1))
