import logging

import casadi as ca
import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.min_abs_goal_programming_mixin import (
    MinAbsGoalProgrammingMixin,
    MinAbsStateGoal,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.single_pass_goal_programming_mixin import (
    CachingQPSol,
    SinglePassGoalProgrammingMixin,
    SinglePassMethod,
)
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class PathGoal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 1
    target_min = 0.0
    order = 1


class PathGoal2(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 2
    target_max = Timeseries(np.linspace(0.0, 1.0, 21), 21 * [1.0])


class VectorPathGoal(Goal):
    """2D vector path goal to test multi-dimensional epsilon seeding."""

    def function(self, optimization_problem, ensemble_member):
        return ca.vertcat(optimization_problem.state("x"), optimization_problem.state("x") * 2)

    size = 2
    function_range = (-1e1, 1e1)
    priority = 1
    target_max = np.array([1.0, 2.0])
    order = 1


class VectorGoal(Goal):
    """3D regular vector goal to test multi-dimensional epsilon seeding for non-path goals."""

    def function(self, optimization_problem, ensemble_member):
        x_at_05 = optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)
        return ca.vertcat(x_at_05, x_at_05 * 2, x_at_05 * 3)

    size = 3
    function_range = (-1e1, 1e1)
    priority = 1
    target_min = np.array([0.0, 0.0, 0.0])
    order = 1


class GoalMinU(MinAbsStateGoal):
    state = "u"
    priority = 3


class ModelPathGoals:
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitialLinear",
            model_folder=data_path(),
        )

        self._objective_values = []
        self._seeds = []
        self._state_vectors = []
        self._solver_outputs = []
        self._constraints = []
        self._initial_seed = None

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
            np.hstack([self.initial_time, self.times()]),
            np.hstack(([1.0], np.linspace(1.0, 0.0, 21))),
        )
        return constant_inputs

    def seed(self, ensemble_member):
        if self._initial_seed is None:
            self._initial_seed = super().seed(ensemble_member)
        return self._initial_seed

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options

    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "clp"
        options["casadi_solver"] = "qpsol"
        return options

    def goals(self):
        goals = super().goals().copy()
        goals.append(VectorGoal())
        return goals

    def path_goals(self):
        goals = super().path_goals().copy()

        goals.append(PathGoal1())
        goals.append(PathGoal2())
        goals.append(VectorPathGoal())

        return goals

    def min_abs_path_goals(self):
        return [GoalMinU(self)]

    def priority_started(self, priority):
        super().priority_started(priority)

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self._objective_values.append(self.objective_value)
        self._solver_outputs.append(self.solver_output.copy())

    def transcribe(self):
        discrete, lbx, ubx, lbg, ubg, x0, nlp = super().transcribe()

        self._seeds.append(x0.copy())
        self._state_vectors.append(nlp["x"])
        self._constraints.append(nlp["g"])

        return discrete, lbx, ubx, lbg, ubg, x0, nlp


class ModelGoalProgramming(
    ModelPathGoals,
    LinearizedOrderGoalProgrammingMixin,
    MinAbsGoalProgrammingMixin,
    GoalProgrammingMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def goal_programming_options(self):
        options = super().goal_programming_options()
        options["keep_soft_constraints"] = True
        return options


class ModelSinglePassGoalProgrammingAppend(
    ModelPathGoals,
    LinearizedOrderGoalProgrammingMixin,
    MinAbsGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    single_pass_method = SinglePassMethod.APPEND_CONSTRAINTS_OBJECTIVE


class ModelSinglePassGoalProgrammingUpdate(
    ModelPathGoals,
    LinearizedOrderGoalProgrammingMixin,
    MinAbsGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    single_pass_method = SinglePassMethod.UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS


class TestSinglePassGoalProgramming(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem_gp = ModelGoalProgramming()
        cls.problem_append = ModelSinglePassGoalProgrammingAppend()
        cls.problem_update = ModelSinglePassGoalProgrammingUpdate()
        cls.problem_gp.optimize()
        cls.problem_append.optimize()
        cls.problem_update.optimize()

    def test_objectives(self):
        self.assertAlmostEqual(
            np.array(self.problem_gp._objective_values),
            np.array(self.problem_append._objective_values),
            1e-5,
        )
        self.assertAlmostEqual(
            np.array(self.problem_gp._objective_values),
            np.array(self.problem_update._objective_values),
            1e-5,
        )

    def test_state_vector_instance(self):
        self.assertEqual(len({hash(x) for x in self.problem_append._state_vectors}), 1)
        self.assertEqual(len({hash(x) for x in self.problem_update._state_vectors}), 1)

    def test_state_vector_length(self):
        self.assertEqual(
            self.problem_gp._state_vectors[-1].size1(),
            self.problem_append._state_vectors[-1].size1(),
        )
        self.assertEqual(
            self.problem_gp._state_vectors[-1].size1(),
            self.problem_update._state_vectors[-1].size1(),
        )

    def test_constraints_length(self):
        diff_lengths_update = np.diff([x.size1() for x in self.problem_update._constraints])
        np.testing.assert_array_equal(diff_lengths_update, [0, 0])

        diff_lengths_append = np.diff([x.size1() for x in self.problem_append._constraints])
        np.testing.assert_array_equal(diff_lengths_append, [1, 1])

    def test_seed(self):
        np.testing.assert_array_equal(
            self.problem_append._seeds[-1], self.problem_append._solver_outputs[-2]
        )
        np.testing.assert_array_equal(
            self.problem_update._seeds[-1], self.problem_update._solver_outputs[-2]
        )

    def test_initial_seed(self):
        """
        Every goal's epsilon variable should be present in the initial seed, and
        should be set to one (scalar or vector of ones).
        """
        n_goals = len(self.problem_append.path_goals()) + len(self.problem_append.goals())
        self.assertEqual(len(self.problem_append._initial_seed.keys()), n_goals)

        for key, value in self.problem_append._initial_seed.items():
            # Handle both regular epsilons and path epsilons (Timeseries)
            if isinstance(value, Timeseries):
                # Path epsilon: check the .values attribute
                np.testing.assert_array_equal(
                    value.values, 1.0, err_msg=f"Path epsilon {key} should be seeded with ones"
                )
            else:
                # Regular epsilon: scalar or vector
                np.testing.assert_array_equal(
                    value, 1.0, err_msg=f"Regular epsilon {key} should be seeded with ones"
                )


class ModelSinglePassGoalProgrammingCachingQPSol(ModelSinglePassGoalProgrammingAppend):
    def pre(self):
        self._qpsol = CachingQPSol()
        super().pre()

    def solver_options(self):
        options = super().solver_options()
        options["casadi_solver"] = self._qpsol
        return options


class TestCachingQPSol(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = ModelSinglePassGoalProgrammingAppend()
        cls.problem_cache = ModelSinglePassGoalProgrammingCachingQPSol()
        cls.problem.optimize()
        cls.problem_cache.optimize()

    def test_objectives(self):
        np.testing.assert_array_equal(
            self.problem._objective_values, self.problem_cache._objective_values
        )

    def test_solver_outputs(self):
        self.assertEqual(len(self.problem_cache._solver_outputs), 3)
        self.assertEqual(len(self.problem._solver_outputs), 3)

        for a, b in zip(
            self.problem._solver_outputs, self.problem_cache._solver_outputs, strict=True
        ):
            np.testing.assert_array_equal(a, b)
