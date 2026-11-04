from unittest.mock import patch

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin

from ..test_case import TestCase
from .data_path import data_path


class ModelSetup:
    def __init__(self, **homotopy_options):
        self._homotopy_options = homotopy_options
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_folder=data_path(),
            model_name="ModelHomotopyParameter",
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
