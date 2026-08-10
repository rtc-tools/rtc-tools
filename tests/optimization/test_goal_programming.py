import logging

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import (
    Goal,
    GoalProgrammingMixin,
    StateGoal,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class Model(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    n_priorities_completed = 0

    def __init__(self, test_skip_priority=False):
        self.test_skip_priority = test_skip_priority

        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitial",
            model_folder=data_path(),
        )

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

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def goals(self):
        return [Goal1(), Goal2(), Goal3()]

    def set_timeseries(self, timeseries_id, timeseries, ensemble_member, **kwargs):
        # Do nothing
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options

    def priority_started(self, priority: int) -> None:
        super().priority_started(priority)
        if self.test_skip_priority:
            if priority == 1:
                self.skip_priority = True

    def priority_completed(self, priority: int) -> None:
        super().priority_completed(priority)
        self.n_priorities_completed += 1


class Goal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 2
    target_min = 0.0
    violation_timeseries_id = "violation"
    function_value_timeseries_id = "function_value"


class Goal2(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.7, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 2
    target_min = 0.1


class Goal3(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("x", 0.1, 1.0, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 1
    target_max = 1.0


class TestGoalProgramming(TestCase):
    def setUp(self):
        self.problem = Model()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        objective_value_tol = 1e-6
        self.assertAlmostGreaterThan(
            self.problem.interpolate(
                0.7, self.problem.times(), self.problem.extract_results()["x"]
            ),
            0.1,
            objective_value_tol,
        )

    def test_number_of_completed_priorites(self):
        self.assertEqual(self.problem.n_priorities_completed, 2)


class TestGoalProgrammingSkipPriority(TestCase):
    def setUp(self):
        self.problem = Model(test_skip_priority=True)
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        objective_value_tol = 1e-6
        self.assertAlmostGreaterThan(
            self.problem.interpolate(
                0.7, self.problem.times(), self.problem.extract_results()["x"]
            ),
            0.1,
            objective_value_tol,
        )

    def test_number_of_completed_priorites(self):
        self.assertEqual(self.problem.n_priorities_completed, 1)


class GoalNoMinMax(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("x", ensemble_member=ensemble_member)

    function_nominal = 2e1
    priority = 1
    order = 1


class GoalLowMax(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("x", ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 1
    order = 1
    # TODO: Why this number? Is it a coincidence?
    target_max = function_range[0]


# Inherit from existing Model, as all properties are equal except the
# goals.
class ModelNoMinMax(Model):
    def goals(self):
        return [GoalNoMinMax()]


class ModelLowMax(Model):
    def goals(self):
        return [GoalLowMax()]


class TestGoalProgrammingNoMinMax(TestCase):
    def setUp(self):
        self.problem1 = ModelNoMinMax()
        self.problem2 = ModelLowMax()
        self.problem1.optimize()
        self.problem2.optimize()
        self.tolerance = 1e-6

    def test_nobounds_equal_lowmax(self):
        self.assertAlmostEqual(
            sum(self.problem1.extract_results()["x"]),
            sum(self.problem2.extract_results()["x"]),
            self.tolerance,
        )


class GoalMinimizeU(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("u", 0.5, ensemble_member=ensemble_member)

    priority = 1
    order = 1


class GoalMinimizeX(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-1e2, 1e2)
    priority = 2
    order = 1
    target_min = 2.0


class ModelMinimizeU(Model):
    def goals(self):
        return [GoalMinimizeU()]


class ModelMinimizeUandX(Model):
    def goals(self):
        return [GoalMinimizeU(), GoalMinimizeX()]


class TestGoalProgrammingHoldMinimization(TestCase):
    def setUp(self):
        self.problem1 = ModelMinimizeU()
        self.problem2 = ModelMinimizeUandX()
        self.problem1.optimize()
        self.problem2.optimize()
        self.tolerance = 1e-6

    def test_hold_minimization_goal(self):
        # Collocation point 0.5 is at index 10
        self.assertAlmostEqual(
            self.problem1.extract_results()["u"][10],
            self.problem2.extract_results()["u"][10],
            self.tolerance,
        )


class PathGoal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 1
    target_min = 0.0


class PathGoal2(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 2
    target_max = Timeseries(np.linspace(0.0, 1.0, 21), 21 * [1.0])


class PathGoal3(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("u")

    priority = 3


class PathGoal4(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("constant_input")

    priority = 4


class PathGoal5(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("k")

    priority = 5


class ModelPathGoals(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitial",
            model_folder=data_path(),
        )

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

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def path_goals(self):
        return [PathGoal1(), PathGoal2(), PathGoal3(), PathGoal4(), PathGoal5()]

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class TestGoalProgrammingPathGoals(TestCase):
    def setUp(self):
        self.problem = ModelPathGoals()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        value_tol = 1e-3
        for x in self.problem.extract_results()["x"]:
            self.assertAlmostGreaterThan(x, 0.0, value_tol)
            self.assertAlmostLessThan(x, 1.1, value_tol)


class PathGoal1Reversed(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 2
    target_min = 0.0


class PathGoal2Reversed(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 1
    target_max = Timeseries(np.linspace(0.0, 1.0, 21), 21 * [1.0])


class ModelPathGoalsReversed(ModelPathGoals):
    def path_goals(self):
        return [PathGoal1Reversed(), PathGoal2Reversed()]


class TestGoalProgrammingPathGoalsReversed(TestGoalProgrammingPathGoals):
    def setUp(self):
        self.problem = ModelPathGoalsReversed()
        self.problem.optimize()
        self.tolerance = 1e-6


class PathGoal1MaxEmpty(PathGoal1):
    def __init__(self, optimization_problem):
        times = optimization_problem.times()
        self.target_max = Timeseries(times, np.full(len(times), np.nan))
        super().__init__()


class ModelPathGoalsOnePriority(ModelPathGoals):
    def path_goals(self):
        return [PathGoal1()]


class ModelPathGoalsOnePriorityMaxEmpty(ModelPathGoalsOnePriority):
    def path_goals(self):
        return [PathGoal1MaxEmpty(self)]


class TestGoalProgrammingPathGoalsMaxEmpty(TestCase):
    def setUp(self):
        self.problem = ModelPathGoalsOnePriority()
        self.problem_max_empty = ModelPathGoalsOnePriorityMaxEmpty()
        self.problem.optimize()
        self.problem_max_empty.optimize()

    def test_objective_exactly_equal(self):
        self.assertEqual(self.problem.objective_value, self.problem_max_empty.objective_value)


class GoalMinU(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("u", ensemble_member=ensemble_member)

    priority = 3


class ModelPathGoalsMixed(ModelPathGoals):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._objective_values = []

    def solver_options(self):
        options = super().solver_options()

        # Relatively strict convergence and constraint criteria to be able to
        # compare. Errors propagate/compound from one priority to the next,
        # which is these are tighter than usual.
        options["ipopt"]["tol"] = 1e-9
        options["ipopt"]["acceptable_tol"] = 1e-8
        options["ipopt"]["constr_viol_tol"] = 1e-8
        options["ipopt"]["compl_inf_tol"] = 1e-8
        options["ipopt"]["acceptable_constr_viol_tol"] = 1e-7
        options["ipopt"]["acceptable_compl_inf_tol"] = 1e-7
        return options

    def path_goals(self):
        goals = [PathGoal1(), PathGoal2()]

        # These goals typically evaluate to a very small values making
        # comparisons on the objective value of a later priority goal
        # difficult. We add a weight to solve them more accurately.
        goals[0].weight = 10000
        goals[1].weight = 10000

        return goals

    def goals(self):
        return [GoalMinU()]

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self._objective_values.append(self.objective_value)


class PathGoal1Critical(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    priority = 1
    target_min = 0.0
    critical = True


class PathGoal1CriticalTimeseries(Goal):
    def __init__(self, optimization_problem):
        super().__init__()
        times = optimization_problem.times()
        self.target_min = Timeseries(times, np.full(len(times), 0.0))

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    priority = 1
    critical = True


class GoalLowerUCritical(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("u", ensemble_member=ensemble_member)

    priority = 3
    target_min = 1e-6
    critical = True


class ModelPathGoalsMixedCritical(ModelPathGoals):
    def path_goals(self):
        return [PathGoal1Critical(), PathGoal2()]

    def goals(self):
        return [GoalLowerUCritical()]


class ModelPathGoalsMixedCriticalTimeseries(ModelPathGoals):
    def path_goals(self):
        return [PathGoal1CriticalTimeseries(self), PathGoal2()]

    def goals(self):
        return [GoalLowerUCritical()]


class TestGoalProgrammingPathGoalsMixed(TestGoalProgrammingPathGoals):
    def setUp(self):
        self.problem = ModelPathGoalsMixed()
        self.problem.optimize()
        self.tolerance = 1e-6


class TestGoalProgrammingPathGoalsMixedCritical(TestGoalProgrammingPathGoals):
    def setUp(self):
        self.problem = ModelPathGoalsMixedCritical()
        self.problem.optimize()
        self.tolerance = 1e-6


class TestGoalProgrammingPathGoalsMixedCriticalTimeseries(TestGoalProgrammingPathGoals):
    def setUp(self):
        self.problem = ModelPathGoalsMixedCriticalTimeseries()
        self.problem.optimize()
        self.tolerance = 1e-6


class ModelPathGoalsMixedKeepSoft(ModelPathGoalsMixed):
    def goal_programming_options(self):
        options = super().goal_programming_options()
        options["keep_soft_constraints"] = True
        return options


class TestGoalProgrammingKeepSoftVariable(TestCase):
    def setUp(self):
        self.problem1 = ModelPathGoalsMixed()
        self.problem2 = ModelPathGoalsMixedKeepSoft()
        self.problem1.optimize()
        self.problem2.optimize()

    def test_keep_soft_constraints_objective(self):
        self.assertEqual(self.problem1._objective_values[0], self.problem2._objective_values[0])
        self.assertAlmostEqual(
            self.problem1._objective_values[1], self.problem2._objective_values[1], 1e-6
        )
        self.assertAlmostEqual(
            self.problem1._objective_values[2], self.problem2._objective_values[2], 1e-3
        )
        self.assertLess(self.problem2._objective_values[2], self.problem1._objective_values[2])


class ModelEnsemble(Model):
    @property
    def ensemble_size(self):
        return 2

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["constant_input"] = Timeseries(
            np.hstack([self.initial_time, self.times()]),
            np.hstack(([1.0], np.linspace(1.0, 0.0, 21))),
        )
        if ensemble_member == 0:
            constant_inputs["constant_input"] = Timeseries(self.times(), np.linspace(1.0, 0.0, 21))
        else:
            constant_inputs["constant_input"] = Timeseries(self.times(), np.linspace(1.0, 0.5, 21))
        return constant_inputs


class TestGoalProgrammingEnsemble(TestGoalProgramming):
    def setUp(self):
        self.problem = ModelEnsemble()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        objective_value_tol = 1e-6
        self.assertAlmostGreaterThan(
            self.problem.interpolate(
                0.7, self.problem.times(), self.problem.extract_results(0)["x"]
            ),
            0.1,
            objective_value_tol,
        )
        self.assertAlmostGreaterThan(
            self.problem.interpolate(
                0.7, self.problem.times(), self.problem.extract_results(1)["x"]
            ),
            0.1,
            objective_value_tol,
        )


class PathGoalSmoothing(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.der("u")

    priority = 3


class ModelPathGoalsSmoothing(
    GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem
):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitial",
            model_folder=data_path(),
        )

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

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def path_goals(self):
        return [PathGoal1(), PathGoal2(), PathGoalSmoothing()]

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class TestGoalProgrammingSmoothing(TestCase):
    def setUp(self):
        self.problem = ModelPathGoalsSmoothing()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        value_tol = 1e-3
        for x in self.problem.extract_results()["x"]:
            self.assertAlmostGreaterThan(x, 0.0, value_tol)
            self.assertAlmostLessThan(x, 1.1, value_tol)


class StateGoal1(StateGoal):
    state = "x"
    priority = 1
    target_min = 0.0
    violation_timeseries_id = "violation2"
    function_value_timeseries_id = "function_value2"


class StateGoal2(StateGoal):
    state = "x"
    priority = 2
    target_max = Timeseries(np.linspace(0.0, 1.0, 21), 21 * [1.0])


class StateGoal3(StateGoal):
    state = "u"
    priority = 3


class ModelStateGoals(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitial",
            model_folder=data_path(),
        )

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

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        bounds["x"] = (-10, 10)
        return bounds

    def path_goals(self):
        return [StateGoal1(self), StateGoal2(self), StateGoal3(self)]

    def set_timeseries(self, timeseries_id, timeseries, ensemble_member, **kwargs):
        # Do nothing
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class TestGoalProgrammingStateGoals(TestCase):
    def setUp(self):
        self.problem = ModelStateGoals()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        value_tol = 1e-3
        for x in self.problem.extract_results()["x"]:
            self.assertAlmostGreaterThan(x, 0.0, value_tol)
            self.assertAlmostLessThan(x, 1.1, value_tol)


class ModelMinimizeTwoGoals(ModelMinimizeUandX):
    def __init__(self, *args, scale_by_problem_size=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale_by_problem_size = scale_by_problem_size

    def goal_programming_options(self):
        options = super().goal_programming_options()

        if self.scale_by_problem_size:
            options["scale_by_problem_size"] = True

        return options

    def goals(self):
        goals = super().goals()
        for g in goals:
            g.priority = 1
        return goals


class PathGoalMinimizeU(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("u")

    priority = 1
    order = 1


class PathGoalMinimizeX(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    priority = 1
    order = 1


class ModelMinimizeTwoPathGoals(Model):
    def __init__(self, *args, scale_by_problem_size=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale_by_problem_size = scale_by_problem_size

    def goal_programming_options(self):
        options = super().goal_programming_options()

        if self.scale_by_problem_size:
            options["scale_by_problem_size"] = True

        return options

    def goals(self):
        return []

    def path_goals(self):
        return [PathGoalMinimizeU(), PathGoalMinimizeX()]


class ModelMinimizeTwoTargetPathGoals(ModelMinimizeTwoPathGoals):
    def path_goals(self):
        goals = super().path_goals()
        for g in goals:
            g.function_range = (-2.0, 10.0)
            g.target_min = 2.0
            # To make sure the objective contains enough significant digits to
            # compare, we make it a bit larger with the weight
            g.weight = 100
        return goals


class TestScaleByProblemSize(TestCase):
    tolerance = 1e-5

    def test_goals_scale_by_problem_size(self):
        self.problem1 = ModelMinimizeTwoGoals()
        self.problem2 = ModelMinimizeTwoGoals(scale_by_problem_size=True)
        self.problem1.optimize()
        self.problem2.optimize()

        obj_value_no_scale = self.problem1.objective_value
        obj_value_scale = self.problem2.objective_value

        self.assertAlmostEqual(1.0, 2 * obj_value_scale / obj_value_no_scale, self.tolerance)

    def test_path_minimization_goals_scale_by_problem_size(self):
        self.problem1 = ModelMinimizeTwoPathGoals()
        self.problem2 = ModelMinimizeTwoPathGoals(scale_by_problem_size=True)
        self.problem1.optimize()
        self.problem2.optimize()

        n_times = len(self.problem2.times())

        obj_value_no_scale = self.problem1.objective_value
        obj_value_scale = self.problem2.objective_value

        self.assertAlmostEqual(
            1.0, 2 * n_times * obj_value_scale / obj_value_no_scale, self.tolerance
        )

    def test_path_target_goals_scale_by_problem_size(self):
        self.problem1 = ModelMinimizeTwoTargetPathGoals()
        self.problem2 = ModelMinimizeTwoTargetPathGoals(scale_by_problem_size=True)
        self.problem1.optimize()
        self.problem2.optimize()

        n_times = len(self.problem2.times())

        obj_value_no_scale = self.problem1.objective_value
        obj_value_scale = self.problem2.objective_value

        self.assertAlmostEqual(
            1.0, 2 * n_times * obj_value_scale / obj_value_no_scale, self.tolerance
        )


class EmptyGoalOneTimeseries(Goal):
    target_min = np.nan
    target_max = Timeseries(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")


class EmptyGoalTwoTimeseries(Goal):
    target_min = Timeseries(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))
    target_max = Timeseries(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")


class NonEmptyGoalTimeseries(Goal):
    target_min = Timeseries(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    target_max = Timeseries(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")


class NonEmptyGoalMinimization(Goal):
    target_min = (np.nan,)
    target_max = np.nan

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")


class TestEmptyGoals(TestCase):
    def test_goal_empty(self):
        g = EmptyGoalOneTimeseries()
        self.assertTrue(g.is_empty)

        g = EmptyGoalTwoTimeseries()
        self.assertTrue(g.is_empty)

    def test_goal_non_empty(self):
        g = NonEmptyGoalTimeseries()
        self.assertFalse(g.is_empty)

        g = NonEmptyGoalMinimization()
        self.assertFalse(g.is_empty)


class ModelInvalidGoals(Model):
    _goals = []

    def goals(self):
        return self._goals


class InvalidGoal(Goal):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)


class TestGoalProgrammingInvalidGoals(TestCase):
    def setUp(self):
        self.problem = ModelInvalidGoals()

    def test_target_min_lt_function_range_lb(self):
        self.problem._goals = [InvalidGoal(function_range=(-2.0, 2.0), target_min=-3.0)]
        with self.assertRaisesRegex(Exception, "minimum should be greater than the lower"):
            self.problem.optimize()

    def test_target_min_eq_function_range_lb(self):
        self.problem._goals = [InvalidGoal(function_range=(-2.0, 2.0), target_min=-2.0)]
        with self.assertRaisesRegex(Exception, "minimum should be greater than the lower"):
            self.problem.optimize()

    def test_target_max_gt_function_range_ub(self):
        self.problem._goals = [InvalidGoal(function_range=(-2.0, 2.0), target_max=3.0)]
        with self.assertRaisesRegex(Exception, "maximum should be smaller than the upper"):
            self.problem.optimize()

    def test_target_max_eq_function_range_ub(self):
        self.problem._goals = [InvalidGoal(function_range=(-2.0, 2.0), target_max=2.0)]
        with self.assertRaisesRegex(Exception, "maximum should be smaller than the upper"):
            self.problem.optimize()

    def test_critical_minimization(self):
        self.problem._goals = [InvalidGoal(critical=True)]
        with self.assertRaisesRegex(Exception, "Minimization goals cannot be critical"):
            self.problem.optimize()

    def test_minimization_function_range(self):
        self.problem._goals = [InvalidGoal(function_range=(-2.0, 2.0))]
        with self.assertRaisesRegex(Exception, "Specifying function range not allowed"):
            self.problem.optimize()

    def test_function_range_present(self):
        self.problem._goals = [InvalidGoal(target_min=2.0)]
        with self.assertRaisesRegex(Exception, "Could not determine a finite function_range"):
            self.problem.optimize()

    def test_function_range_valid(self):
        self.problem._goals = [InvalidGoal(function_range=(2.0, -2.0), target_min=2.1)]
        with self.assertRaisesRegex(Exception, "Invalid function range"):
            self.problem.optimize()

    def test_function_nominal_positive(self):
        self.problem._goals = [InvalidGoal(function_nominal=-1.0)]
        with self.assertRaisesRegex(Exception, "Nonpositive nominal value"):
            self.problem.optimize()

        self.problem._goals = [InvalidGoal(function_nominal=0.0)]
        with self.assertRaisesRegex(Exception, "Nonpositive nominal value"):
            self.problem.optimize()

    def test_priority_not_cast_int(self):
        self.problem._goals = [InvalidGoal(priority="test")]
        with self.assertRaisesRegex(Exception, "castable to int"):
            self.problem.optimize()

    def test_target_min_timeseries(self):
        # Only path goals can have Timeseries as target min/max
        self.problem._goals = [
            InvalidGoal(
                function_range=(-2.0, 2.0), target_min=Timeseries(self.problem.times(), [1.0])
            )
        ]
        with self.assertRaisesRegex(Exception, "Target min cannot be a Timeseries"):
            self.problem.optimize()

    def test_target_max_timeseries(self):
        # Only path goals can have Timeseries as target min/max
        self.problem._goals = [
            InvalidGoal(
                function_range=(-2.0, 2.0), target_max=Timeseries(self.problem.times(), [1.0])
            )
        ]
        with self.assertRaisesRegex(Exception, "Target max cannot be a Timeseries"):
            self.problem.optimize()

    def test_goal_weight_minimization(self):
        self.problem._goals = [InvalidGoal(weight=-1.0)]

        # For minimization goals we can have a negative goal weight
        try:
            self.problem.optimize()
        except Exception:
            self.fail("test_goal_weight_minimization() failed unexpectedly.")

    def test_goal_weight_targets(self):
        self.problem._goals = [InvalidGoal(function_range=(-2.0, 2.0), target_max=1.0, weight=-1.0)]

        with self.assertRaisesRegex(Exception, "Goal weight should be positive"):
            self.problem.optimize()


class ModelPathGoalsSeed(ModelPathGoals):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._results = []
        self._x0 = []

    def transcribe(self):
        discrete, lbx, ubx, lbg, ubg, x0, nlp = super().transcribe()
        self._x0.append(x0)

        return discrete, lbx, ubx, lbg, ubg, x0, nlp

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self._results.append(self.solver_output)

    def path_goals(self):
        return [PathGoal3(), PathGoal4(), PathGoal5()]


class TestGoalProgrammingSeed(TestCase):
    def setUp(self):
        self.problem = ModelPathGoalsSeed()
        self.problem.optimize()

    def test_seed(self):
        self.assertTrue(np.allclose(self.problem._results[0], self.problem._x0[1]))
        self.assertTrue(np.allclose(self.problem._results[1], self.problem._x0[2]))


class EnsembleBoundsGoalProgrammingModel(ModelEnsemble):
    """
    Base model for testing ensemble-specific bounds in goal programming.
    """

    ensemble_specific_bounds = True

    def bounds(self, ensemble_member: int):
        # Do not call super().bounds() as that one does not accept
        # ensemble_member as an argument.
        bounds = {}
        bounds["u"] = (-2.0, 2.0)
        return bounds


class EnsembleBoundsStateGoal(StateGoal):
    """Generic StateGoal for state 'x' used in ensemble bounds tests."""

    state = "x"
    priority = 1
    target_max = 2.0
    # function_range is (np.nan, np.nan) by default, allowing auto-detection or
    # explicit override.


class ModelForUnequalBoundsDicts(EnsembleBoundsGoalProgrammingModel):
    """Model where bounds() returns different dictionaries per ensemble member."""

    def bounds(self, ensemble_member: int):
        bounds = super().bounds(ensemble_member)
        if ensemble_member == 0:
            bounds["x"] = (-5.0, 5.0)
        else:
            bounds["x"] = (-6.0, 6.0)
        return bounds

    def path_goals(self):
        return [EnsembleBoundsStateGoal(self)]


class TestGoalProgrammingEnsembleBoundsUnequalDictsError(TestCase):
    def test_exception_on_unequal_bounds_dictionaries(self):
        """
        Tests that an exception is raised if ensemble_specific_bounds=True,
        StateGoal.function_range is not set, and problem.bounds() dicts differ
        per ensemble member.
        """
        problem = ModelForUnequalBoundsDicts()
        with self.assertRaisesRegex(
            ValueError,
            "Bounds for state x are not the same for all ensemble members; "
            "please set the function_range explicitly",
        ):
            problem.optimize()


class ModelForTypeMismatchBoundsDicts(EnsembleBoundsGoalProgrammingModel):
    """
    Model where bounds() returns dictionaries of different types for different
    ensemble members.
    """

    def bounds(self, ensemble_member: int):
        bounds = super().bounds(
            ensemble_member
        )  # This provides a base dict e.g., {'u': (-2.0, 2.0)}
        if ensemble_member == 0:
            bounds["x"] = (-7.0, 7.0)
        else:
            bounds["x"] = Timeseries(times=np.array([0.0, 1.0]), values=np.array([0.0, 0.0]))
        return bounds

    def path_goals(self):
        # Reuse the same StateGoal which does not have an explicit function_range
        return [EnsembleBoundsStateGoal(self)]


class TestGoalProgrammingEnsembleBoundsTypeMismatchError(TestCase):
    def test_exception_on_type_mismatch_bounds_dictionaries(self):
        """
        Tests that a ValueError is raised if ensemble_specific_bounds=True,
        StateGoal.function_range is not set, and problem.bounds() returns
        dictionaries of different types for different ensemble members.
        """
        problem = ModelForTypeMismatchBoundsDicts()
        with self.assertRaisesRegex(
            ValueError,
            "Bounds for state x are not the same for all ensemble members; "
            "please set the function_range explicitly",
        ):
            problem.optimize()


class ExplicitFunctionRangeGoal(EnsembleBoundsStateGoal):
    """EnsembleBoundsStateGoal with an explicitly set function_range."""

    function_range = (-20.0, 20.0)


class ModelForUnequalBoundsDictsWithExplicitFunctionRange(ModelForUnequalBoundsDicts):
    """
    Model with unequal bounds dicts, but the goal has an explicit function_range.
    Inherits differing bounds() from ModelForUnequalBoundsDicts.
    """

    def path_goals(self):
        return [ExplicitFunctionRangeGoal(self)]


class TestGoalProgrammingEnsembleBoundsUnequalDictsWithExplicitFunctionRange(TestCase):
    def test_no_exception_if_function_range_is_explicit(self):
        """
        Tests that no "Bounds not same" ValueError is raised if StateGoal.function_range
        is explicitly set, even if problem.bounds() dicts differ per ensemble member.
        """
        problem = ModelForUnequalBoundsDictsWithExplicitFunctionRange()
        try:
            problem.optimize()
            self.assertIsNotNone(problem.objective_value, "Optimization did not run to completion.")
        except ValueError as e:
            if "Bounds for state x are not the same" in str(e):
                self.fail(
                    "ValueError for unequal bounds dicts should not be "
                    "raised when function_range is explicit."
                )
            else:
                raise


class ModelForEqualBoundsDicts(EnsembleBoundsGoalProgrammingModel):
    """Model where bounds() returns identical dictionaries for all ensemble members."""

    def bounds(self, ensemble_member: int):
        bounds = super().bounds(ensemble_member)
        bounds["x"] = (-7.7, 7.7)
        return bounds

    def path_goals(self):
        return [EnsembleBoundsStateGoal(self)]


class TestGoalProgrammingEnsembleBoundsAutoSetFunctionRange(TestCase):
    def test_function_range_auto_set_correctly_on_equal_bounds_dicts(self):
        """
        Tests that StateGoal.function_range is automatically and correctly set
        if ensemble_specific_bounds=True, function_range is not explicit,
        and problem.bounds() dicts are identical for all ensemble members.
        """
        problem = ModelForEqualBoundsDicts()
        problem.optimize()

        # The auto-set function_range should be from problem.bounds(0).get('x')
        function_range = problem.path_goals()[0].function_range

        self.assertEqual(
            function_range,
            problem.bounds(0).get("x"),
            f"Automatically set function_range is incorrect. "
            f"Expected {problem.bounds(0).get('x')}, got {function_range}",
        )


class ModelForInfiniteBounds(EnsembleBoundsGoalProgrammingModel):
    """Model whose bounds() yields (-inf, inf) for the goal's state, mimicking a
    model that did not provide finite min/max (e.g. the pymoca >= 0.11 regression
    that drops min/max modifiers)."""

    def bounds(self, ensemble_member: int):
        bounds = super().bounds(ensemble_member)
        bounds["x"] = (-np.inf, np.inf)
        return bounds

    def path_goals(self):
        return [EnsembleBoundsStateGoal(self)]


class CriticalEnsembleBoundsStateGoal(EnsembleBoundsStateGoal):
    """Critical variant: function_range is not used for critical goals, so an
    infinite derived range must NOT raise."""

    critical = True


class ModelForInfiniteBoundsCritical(ModelForInfiniteBounds):
    def path_goals(self):
        return [CriticalEnsembleBoundsStateGoal(self)]


class TestGoalProgrammingInfiniteDerivedFunctionRange(TestCase):
    def test_actionable_error_when_derived_function_range_is_infinite(self):
        """
        When function_range is auto-derived from bounds() but those bounds are not
        finite, the goal should fail with an actionable message pointing the user
        at function_range (raised in validate_goals()), rather than the previous
        generic "No function range specified" error.
        """
        problem = ModelForInfiniteBounds()
        with self.assertRaisesRegex(Exception, "Could not determine a finite function_range"):
            problem.optimize()

    def test_critical_goal_with_infinite_range_does_not_raise(self):
        """
        function_range is not used for critical goals, so an infinite derived
        range must not trigger the function_range error (backwards compatibility).
        """
        problem = ModelForInfiniteBoundsCritical()
        try:
            problem.optimize()
        except Exception as e:
            if "function_range" in str(e):
                self.fail(
                    "Critical goal with infinite derived function_range should not "
                    f"raise the function_range error, but got: {e}"
                )
            raise
        # Confirm the goal actually ran to completion with an infinite derived range,
        # so this test cannot pass vacuously if a fixture change makes bounds finite.
        self.assertIsNotNone(problem.objective_value)


class ModelForExplicitFunctionRangeWithInfiniteBounds(ModelForInfiniteBounds):
    """Model whose bounds() yields (-inf, inf) but the goal has an explicit function_range.
    Verifies that an explicit function_range is not overwritten by bounds()."""

    def path_goals(self):
        return [ExplicitFunctionRangeGoal(self)]


class TestGoalProgrammingExplicitFunctionRangeNotOverwritten(TestCase):
    def test_explicit_function_range_is_retained(self):
        """
        StateGoal.__init__ must not overwrite an explicitly-set function_range with
        the value from bounds(), even when bounds() returns (-inf, inf).
        """
        problem = ModelForExplicitFunctionRangeWithInfiniteBounds()
        problem.optimize()
        self.assertEqual(
            problem.path_goals()[0].function_range,
            (-20.0, 20.0),
            "Explicit function_range was overwritten by bounds() — guard missing or broken.",
        )
