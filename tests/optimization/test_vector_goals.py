import logging

import casadi as ca
import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin, StateGoal
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class Model(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitial",
            model_folder=data_path(),
        )

        self._objective_values = []

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["u_max"] = 2.0
        return parameters

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["constant_input"] = Timeseries(
            np.hstack(([self.initial_time, self.times()])),
            np.hstack(([1.0], np.linspace(1.0, 0.0, 21))),
        )
        return constant_inputs

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        bounds["x"] = (-5.0, 5.0)
        bounds["z"] = (-5.0, 5.0)
        return bounds

    def goal_programming_options(self):
        options = super().goal_programming_options()
        options["keep_soft_constraints"] = True
        return options

    def set_timeseries(self, timeseries_id, timeseries, ensemble_member, **kwargs):
        # Do nothing
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self._objective_values.append(self.objective_value)


class Goal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-5.0, 5.0)
    order = 1
    priority = 1
    target_min = 1.0


class Goal2(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.7, ensemble_member=ensemble_member)

    function_range = (-5.0, 5.0)
    order = 1
    priority = 1
    target_max = 0.8


class Goal3(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 1.0, ensemble_member=ensemble_member)

    function_range = (-5.0, 5.0)
    order = 1
    priority = 1
    target_max = 0.5


class Goal4(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("x", 0.1, 1.0, ensemble_member=ensemble_member)

    order = 2
    priority = 2


class ModelGoals(Model):
    def goals(self):
        return [Goal1(), Goal2(), Goal3(), Goal4()]


class Goal1_2_3(Goal):
    def function(self, optimization_problem, ensemble_member):
        return ca.vertcat(
            optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member),
            optimization_problem.state_at("x", 0.7, ensemble_member=ensemble_member),
            optimization_problem.state_at("x", 1.0, ensemble_member=ensemble_member),
        )

    function_range = (-5.0, 5.0)
    size = 3
    order = 1
    priority = 1
    target_min = np.array([1.0, -np.inf, -np.inf])
    target_max = np.array([np.inf, 0.8, 0.5])


class ModelGoalsVector(ModelGoals):
    def goals(self):
        return [Goal1_2_3(), Goal4()]


class PathGoal1(StateGoal):
    def __init__(self, optimization_problem):
        times = optimization_problem.times()
        n_times = len(times)

        self.target_min = Timeseries(times, np.full(n_times, 0.1))
        self.target_min.values[10] = np.nan

        self.target_max = np.nan

        super().__init__(optimization_problem)

    state = "x"
    order = 1
    priority = 1


class PathGoal2(StateGoal):
    state = "z"
    order = 1
    priority = 1
    target_min = 0.5
    target_max = 2.0


class PathGoal3(StateGoal):
    state = "x"
    order = 2
    priority = 2


class ModelPathGoals(Model):
    def path_goals(self):
        return [PathGoal1(self), PathGoal2(self), PathGoal3(self)]


class PathGoal1_2(Goal):
    def __init__(self, optimization_problem):
        bounds_x = optimization_problem.bounds()["x"]
        bounds_z = optimization_problem.bounds()["z"]
        lb = np.array([bounds_x[0], bounds_z[0]])
        ub = np.array([bounds_x[1], bounds_z[1]])
        self.function_range = (lb, ub)

        times = optimization_problem.times()
        n_times = len(times)

        self.target_min = Timeseries(
            times, np.stack((np.full(n_times, 0.1), np.full(n_times, 0.5)), axis=1)
        )
        self.target_min.values[10, 0] = np.nan

        self.target_max = np.array([np.nan, 2.0])

    def function(self, optimization_problem, ensemble_member):
        return ca.vertcat(optimization_problem.state("x"), optimization_problem.state("z"))

    size = 2
    order = 1
    priority = 1


class ModelPathGoalsVector(Model):
    def path_goals(self):
        return [PathGoal1_2(self), PathGoal3(self)]


class ModelDelay(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self, inline_delay_expressions=False):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelDelay",
            model_folder=data_path(),
        )
        self.inline_delay_expressions = inline_delay_expressions

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def goals(self):
        return [Goal1_2_3(), Goal2()]

    def goal_programming_options(self):
        goal_programming_options = super().goal_programming_options()
        goal_programming_options["keep_soft_constraints"] = True
        return goal_programming_options

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options

    def history(self, ensemble_member):
        history = super().history(ensemble_member)
        history["x"] = Timeseries(np.array([-0.2, -0.1, 0.0]), np.array([0.7, 0.9, 1.1]))
        history["w"] = Timeseries(np.array([-0.1, 0.0]), np.array([0.9, np.nan]))
        return history


class TestVectorGoals(TestCase):
    """
    NOTE: As long as the order of goals/constraints is the same, whether or not they are passed
    as a vector or not should not matter. Therefore we often check to see if two problems
    are _exactly_ equal.
    """

    def test_vector_goals(self):
        self.problem1 = ModelGoals()
        self.problem2 = ModelGoalsVector()
        self.problem1.optimize()
        self.problem2.optimize()

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        self.assertListEqual(self.problem1._objective_values, self.problem2._objective_values)
        self.assertTrue(np.array_equal(results1["x"], results2["x"]))

    def test_path_vector_goals_simple(self):
        self.problem1 = ModelPathGoals()
        self.problem2 = ModelPathGoalsVector()
        self.problem1.optimize()
        self.problem2.optimize()

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        self.assertListEqual(self.problem1._objective_values, self.problem2._objective_values)
        self.assertTrue(np.array_equal(results1["x"], results2["x"]))


class ScaleByProblemSizeMixin:
    def goal_programming_options(self):
        options = super().goal_programming_options()
        options["scale_by_problem_size"] = True
        return options


class ModelGoalsScale(ScaleByProblemSizeMixin, ModelGoals):
    pass


class ModelGoalsVectorScale(ScaleByProblemSizeMixin, ModelGoalsVector):
    pass


class ModelPathGoalsScale(ScaleByProblemSizeMixin, ModelPathGoals):
    pass


class ModelPathGoalsVectorScale(ScaleByProblemSizeMixin, ModelPathGoalsVector):
    pass


class TestVectorGoalsScaleProblemSize(TestCase):
    def test_vector_goals(self):
        self.problem1 = ModelGoalsScale()
        self.problem2 = ModelGoalsVectorScale()
        self.problem1.optimize()
        self.problem2.optimize()

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        self.assertListEqual(self.problem1._objective_values, self.problem2._objective_values)
        self.assertTrue(np.array_equal(results1["x"], results2["x"]))

    def test_path_vector_goals_scaled_vs_non_scaled(self):
        self.problem1 = ModelPathGoals()
        self.problem1_scaled = ModelPathGoalsScale()
        self.problem2 = ModelPathGoalsVector()
        self.problem2_scaled = ModelPathGoalsVectorScale()

        self.problem1.optimize()
        self.problem1_scaled.optimize()
        self.problem2.optimize()
        self.problem2_scaled.optimize()

        self.assertNotEqual(self.problem1._objective_values, self.problem1_scaled._objective_values)
        self.assertLess(self.problem1_scaled.objective_value, self.problem1.objective_value)

        self.assertNotEqual(self.problem2._objective_values, self.problem2_scaled._objective_values)
        self.assertLess(self.problem2_scaled.objective_value, self.problem2.objective_value)

    def test_path_vector_goals_simple(self):
        self.problem1 = ModelPathGoalsScale()
        self.problem2 = ModelPathGoalsVectorScale()
        self.problem1.optimize()
        self.problem2.optimize()

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        self.assertListEqual(self.problem1._objective_values, self.problem2._objective_values)
        self.assertTrue(np.array_equal(results1["x"], results2["x"]))


class TestVectorGoalsWithDelay(TestCase):
    """
    NOTE: As long as the order of goals/constraints is the same, whether or not they are passed
    as a vector or not should not matter. Therefore we often check to see if two problems
    are _exactly_ equal.
    """

    def test_vector_goals(self):
        self.problem1 = ModelDelay()
        self.problem1.optimize()
        assert True
