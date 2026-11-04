from unittest.mock import patch

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path


class ModelSetup:
    def __init__(self, model_name="ModelHomotopyParameter", **homotopy_options):
        self._homotopy_options = homotopy_options
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_folder=data_path(),
            model_name=model_name,
        )

    def homotopy_options(self):
        return {**super().homotopy_options(), **self._homotopy_options}

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 11)

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class HomotopyModel(
    ModelSetup, HomotopyMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem
):
    def objective(self, ensemble_member):
        return self.integral("x", ensemble_member=ensemble_member)


class DelayedHistoryModel(HomotopyModel):
    def history(self, ensemble_member):
        history = super().history(ensemble_member)
        history["x"] = Timeseries(np.array([-0.4, -0.2, 0.0]), np.array([0.5, 0.8, 1.1]))
        return history


class StateAtGoal(Goal):
    function_range = (-10.0, 10.0)
    target_min = 0.0

    def __init__(self, t, priority):
        self.t = t
        self.priority = priority

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", self.t, ensemble_member)


class HomotopyGPModel(
    ModelSetup,
    HomotopyMixin,
    GoalProgrammingMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    skip_intermediate_priority_1 = False

    def __init__(self, **homotopy_options):
        self.solved = []
        super().__init__(delta_theta_0=0.5, **homotopy_options)

    def goals(self):
        return [StateAtGoal(0.5, priority=1), StateAtGoal(1.0, priority=2)]

    def priority_started(self, priority):
        super().priority_started(priority)
        theta = self.parameters(0)["theta"]
        self.skip_priority = self.skip_intermediate_priority_1 and theta < 1.0 and priority == 1

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self.solved.append((self.parameters(0)["theta"], priority))


class TestHomotopyMixin(TestCase):
    def solved_thetas(self, failing_solve=None, **homotopy_options):
        """Run the continuation with a fake solver and return the theta of each solve."""
        problem = HomotopyModel(**homotopy_options)
        thetas = []

        def solve(**kwargs):
            thetas.append(problem.parameters(0)["theta"])
            return len(thetas) != failing_solve

        with (
            patch.object(CollocatedIntegratedOptimizationProblem, "optimize", side_effect=solve),
            patch.object(problem, "extract_results", return_value={}),
        ):
            problem.optimize(postprocessing=False)
        return thetas

    def test_solves_nonlinear_problem(self):
        problem = HomotopyModel()
        self.assertTrue(problem.optimize())
        results = problem.extract_results()
        np.testing.assert_allclose(
            results["nonlinear_output"], results["x"] + results["x"] ** 2, atol=1e-6
        )

    def test_last_step_ends_at_theta_one(self):
        for delta_theta_0, expected in [
            (0.3, [0.0, 0.3, 0.6, 0.9, 1.0]),
            (0.1, np.linspace(0.0, 1.0, 11)),
        ]:
            with self.subTest(delta_theta_0=delta_theta_0):
                thetas = self.solved_thetas(delta_theta_0=delta_theta_0)
                np.testing.assert_allclose(thetas, expected)
                self.assertEqual(thetas[-1], 1.0)

    def test_failure_halves_the_shortened_step(self):
        thetas = self.solved_thetas(failing_solve=3, delta_theta_0=0.8)
        np.testing.assert_allclose(thetas, [0.0, 0.8, 1.0, 0.9, 1.0])

    def test_invalid_options(self):
        for options in [
            {"theta_start": 1.5},
            {"delta_theta_0": 0.0},
            {"delta_theta_min": 0.0},
            {"homotopy_parameter": "unknown"},
            {"skip_intermediate_priorities": True},
            {"non_linear_thresh_time_idx": 3},
            {"model_name": "ModelHomotopyInput", "non_linear_thresh_time_idx": -1},
        ]:
            with self.subTest(**options), self.assertRaises(ValueError):
                HomotopyModel(**options).optimize()

    def test_skip_intermediate_priorities(self):
        for skip_intermediate, skip_first, expected in [
            (False, False, [(0.0, 1), (0.0, 2), (0.5, 1), (0.5, 2), (1.0, 1), (1.0, 2)]),
            (True, False, [(0.0, 1), (0.5, 1), (1.0, 1), (1.0, 2)]),
            (True, True, [(0.0, 2), (0.5, 2), (1.0, 1), (1.0, 2)]),
        ]:
            with self.subTest(skip_intermediate=skip_intermediate, skip_first=skip_first):
                problem = HomotopyGPModel(skip_intermediate_priorities=skip_intermediate)
                problem.skip_intermediate_priority_1 = skip_first
                self.assertTrue(problem.optimize())
                self.assertEqual(problem.solved, expected)

    def test_time_dependent_theta(self):
        problem = DelayedHistoryModel("ModelHomotopyInput", non_linear_thresh_time_idx=3)
        self.assertTrue(problem.optimize())
        results = problem.extract_results()
        theta = np.where(np.arange(11) < 3, 1.0, 0.0)
        np.testing.assert_allclose(
            results["nonlinear_output"], results["x"] + theta * results["x"] ** 2
        )
        # At t = 0.2, the delay reads x = 0.8 and theta from the history at t = -0.2.
        np.testing.assert_allclose(results["delayed_nonlinear_output"][2], 0.8 + 0.8**2)

    def test_negative_alias_of_theta_input(self):
        problem = HomotopyModel(
            "ModelHomotopyInputAlias",
            homotopy_parameter="negative_theta",
            non_linear_thresh_time_idx=3,
        )
        self.assertTrue(problem.optimize())
        results = problem.extract_results()
        theta = np.where(np.arange(11) < 3, -1.0, 0.0)
        np.testing.assert_allclose(
            results["nonlinear_output"], results["x"] + theta * results["x"] ** 2
        )
