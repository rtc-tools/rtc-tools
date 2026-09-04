import logging
import unittest

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.DEBUG)


class Model(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
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

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class ModelNoHistory(Model):
    def history(self, ensemble_member):
        return {}


class ModelPartialHistory(Model):
    def history(self, ensemble_member):
        history = super().history(ensemble_member)
        history["x"] = Timeseries(np.array([-0.2, -0.1, 0.0]), np.array([0.7, 0.9, 1.1]))
        return history


class ModelCompleteHistory(Model):
    def history(self, ensemble_member):
        history = super().history(ensemble_member)
        history["x"] = Timeseries(np.array([-0.2, -0.1, 0.0]), np.array([0.7, 0.9, 1.1]))
        history["w"] = Timeseries(np.array([-0.1, 0.0]), np.array([0.9, np.nan]))
        return history


class TestDelayHistoryWarnings(TestCase, unittest.TestCase):
    def test_no_history(self):
        # Test default mode
        problem = ModelNoHistory(inline_delay_expressions=False)
        with self.assertLogs(logger, level="WARN") as cm:
            problem.optimize()
            self.assertEqual(
                cm.output,
                [
                    "WARNING:rtctools:Incomplete history for delayed expression x. "
                    "Extrapolating t0 value backwards in time.",
                    "WARNING:rtctools:Incomplete history for delayed expression w. "
                    "Extrapolating t0 value backwards in time.",
                ],
            )
        results = problem.extract_results()

        # Test inline mode
        problem_inline = ModelNoHistory(inline_delay_expressions=True)
        with self.assertLogs(logger, level="WARN") as cm:
            problem_inline.optimize()
            self.assertEqual(
                cm.output,
                [
                    "WARNING:rtctools:Incomplete history for delayed expression x. "
                    "Extrapolating t0 value backwards in time.",
                    "WARNING:rtctools:Incomplete history for delayed expression w. "
                    "Extrapolating t0 value backwards in time.",
                ],
            )
        results_inline = problem_inline.extract_results()

        # Check that the results align
        self.assertAlmostEqual(results["x"], results_inline["x"], 1e-6)
        self.assertAlmostEqual(results["w"], results_inline["w"], 1e-6)

        # Check that the inline results has fewer constraints
        self.assertLess(
            len(problem_inline.transcribed_problem["lbg"]), len(problem.transcribed_problem["lbg"])
        )

    def test_partial_history(self):
        problem = ModelPartialHistory()
        with self.assertLogs(logger, level="WARN") as cm:
            problem.optimize()
            self.assertEqual(
                cm.output,
                [
                    "WARNING:rtctools:Incomplete history for delayed expression w. "
                    "Extrapolating t0 value backwards in time."
                ],
            )

    def test_complete_history(self):
        problem = ModelCompleteHistory()
        with self.assertLogs(logger, level="WARN") as cm:
            problem.optimize()
            self.assertEqual(cm.output, [])
            # if no log message occurs, assertLogs will throw an AssertionError
            logger.warning("All is well")


class ModelNestedDelay(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self, inline_delay_expressions):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelNestedDelay",
            model_folder=data_path(),
        )
        self.inline_delay_expressions = inline_delay_expressions

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 21)

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["u"] = Timeseries(self.times(), 1.0 + self.times())
        return constant_inputs

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class TestNestedDelay(TestCase, unittest.TestCase):
    def test_nested_delay(self):
        u = 1.0 + np.linspace(0.0, 1.0, 21)
        # Each delay shifts by two steps and extrapolates the t0 value backwards
        expected = 2 * np.concatenate([np.full(4, u[0]), u[:-4]])
        for inline_delay_expressions in (False, True):
            with self.subTest(inline_delay_expressions=inline_delay_expressions):
                problem = ModelNestedDelay(inline_delay_expressions)
                problem.optimize()
                self.assertAlmostEqual(problem.extract_results()["y"], expected, 1e-6)


class ModelDelayLoop(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self, inline_delay_expressions):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelDelayLoop",
            model_folder=data_path(),
        )
        self.inline_delay_expressions = inline_delay_expressions

    def times(self, variable=None):
        return np.linspace(0.0, 20.0, 21)

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["u"] = Timeseries(self.times(), 1.0 + self.times())
        return constant_inputs

    def history(self, ensemble_member):
        history = super().history(ensemble_member)
        history["a"] = Timeseries(np.array([-1.0, 0.0]), np.array([1.0, 2.0]))
        history["b"] = Timeseries(np.array([-1.0, 0.0]), np.array([2.0, 1.0]))
        return history

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class ModelDelayLoopNoHistory(ModelDelayLoop):
    def history(self, ensemble_member):
        return {}


class TestDelayLoop(TestCase, unittest.TestCase):
    def test_delay_loop(self):
        # a and b swap values every step, starting from the history
        a = np.tile([2.0, 1.0], 11)[:21]
        expected = 2 * a + 1.0 + np.linspace(0.0, 20.0, 21)
        for inline_delay_expressions in (False, True):
            with self.subTest(inline_delay_expressions=inline_delay_expressions):
                problem = ModelDelayLoop(inline_delay_expressions)
                problem.optimize()
                self.assertAlmostEqual(problem.extract_results()["y"], expected, 1e-6)

    def test_delay_loop_no_history(self):
        # Extrapolating the t0 values backwards makes a and b depend on each other at t0
        problem = ModelDelayLoopNoHistory(inline_delay_expressions=True)
        with self.assertRaisesRegex(Exception, "Circular dependency"):
            problem.optimize()
