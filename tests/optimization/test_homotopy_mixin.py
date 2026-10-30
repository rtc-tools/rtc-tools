import logging
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import (
    Goal,
    GoalProgrammingMixin,
)
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.io_mixin import IOMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class BasicHomotopyModel(HomotopyMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self, **kwargs):
        kwargs["input_folder"] = data_path()
        kwargs["output_folder"] = data_path()
        kwargs["model_folder"] = data_path()
        kwargs.setdefault("model_name", "ModelHomotopyParameter")
        super().__init__(**kwargs)

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 11)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["u_max"] = 2.0
        return parameters

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["constant_input"] = Timeseries(
            np.hstack([self.initial_time, self.times()]),
            np.hstack(([1.0], np.linspace(1.0, 0.0, 11))),
        )
        return constant_inputs

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def objective(self, ensemble_member):
        return self.integral("x", ensemble_member=ensemble_member)

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class DummyHomotopyIOMixin(IOMixin):
    def read(self):
        ref_datetime = datetime(2000, 1, 1)
        self.io.reference_datetime = ref_datetime
        times_sec = np.linspace(0.0, 1.0, 6)
        datetimes = [ref_datetime + timedelta(seconds=x) for x in times_sec]

        values = {
            "constant_input": [1.0] * 6,
            "u_max": [2.0] * 6,
        }
        for key, value in values.items():
            self.io.set_timeseries(key, datetimes, np.array(value))

        self.io.set_parameter("u_max", 2.0)

    def write(self):
        pass


class TimeseriesHomotopyModel(
    HomotopyMixin, DummyHomotopyIOMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem
):
    def __init__(self, thresh_idx=3, **kwargs):
        self.thresh_idx = thresh_idx
        self.thetas_seen = []
        kwargs["input_folder"] = data_path()
        kwargs["output_folder"] = data_path()
        kwargs["model_folder"] = data_path()
        kwargs.setdefault("model_name", "ModelHomotopyInput")
        super().__init__(**kwargs)

    def times(self, variable=None):
        return IOMixin.times(self, variable)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        if self.thresh_idx is not None:
            parameters["non_linear_thresh_time_idx"] = self.thresh_idx
        parameters["u_max"] = 2.0
        return parameters

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["constant_input"] = Timeseries(
            self.times(), np.linspace(1.0, 0.0, len(self.times()))
        )
        return constant_inputs

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def objective(self, ensemble_member):
        if self.homotopy_theta is not None:
            self.thetas_seen.append(np.array(self.homotopy_theta, copy=True))
        return self.integral("x", ensemble_member=ensemble_member)

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class OptionTimeseriesHomotopyModel(TimeseriesHomotopyModel):
    def __init__(self, **kwargs):
        super().__init__(thresh_idx=None, **kwargs)

    def homotopy_options(self):
        options = super().homotopy_options()
        options["non_linear_thresh_time_idx"] = 3
        options["delta_theta_0"] = 0.5
        return options


class HistoryHomotopyModel(OptionTimeseriesHomotopyModel):
    def read(self):
        super().read()
        self.io.reference_datetime += timedelta(seconds=0.4)
        self.io.set_timeseries(
            "x", self.io.datetimes, np.array([0.5, 0.8, 1.1, np.nan, np.nan, np.nan])
        )


class EnsembleHomotopyModel(OptionTimeseriesHomotopyModel):
    def read(self):
        super().read()
        self.io.set_parameter("u_max", 2.0, ensemble_member=1)


class GPGoalPriority1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("x", 0.1, 1.0, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 1
    target_max = 1.0


class GPGoalPriority2(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 2
    target_min = 0.0


class HomotopyGPModel(
    HomotopyMixin, GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem
):
    def __init__(self, prune_intermediate_subproblems=False, **kwargs):
        self._prune = prune_intermediate_subproblems
        self.completed_priorities = []
        kwargs["input_folder"] = data_path()
        kwargs["output_folder"] = data_path()
        kwargs["model_folder"] = data_path()
        kwargs.setdefault("model_name", "ModelHomotopyParameter")
        super().__init__(**kwargs)

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 11)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["u_max"] = 2.0
        return parameters

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["constant_input"] = Timeseries(
            np.hstack([self.initial_time, self.times()]),
            np.hstack(([1.0], np.linspace(1.0, 0.0, 11))),
        )
        return constant_inputs

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def goals(self):
        return [GPGoalPriority1(), GPGoalPriority2()]

    def homotopy_options(self):
        options = super().homotopy_options()
        options["delta_theta_0"] = 0.5
        options["prune_intermediate_subproblems"] = self._prune
        return options

    def priority_completed(self, priority: int) -> None:
        super().priority_completed(priority)
        self.completed_priorities.append((self.homotopy_theta, priority))

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class TestHomotopyMixin(TestCase):
    def test_basic_homotopy(self):
        problem = BasicHomotopyModel()
        success = problem.optimize()
        self.assertTrue(success)
        self.assertAlmostEqual(problem.homotopy_theta, 1.0, 1e-6)
        self.assertFalse(problem._homotopy_is_intermediate)
        results = problem.extract_results()
        np.testing.assert_allclose(
            results["nonlinear_output"], results["x"] + results["x"] ** 2, atol=1e-6
        )

    def test_fixed_input_alias_without_io(self):
        for alias, sign in [("theta_alias", 1), ("negative_theta_alias", -1)]:
            for threshold in [None, 3]:
                with self.subTest(alias=alias, threshold=threshold):
                    problem = BasicHomotopyModel(model_name="ModelHomotopyInputAlias")
                    self.assertEqual(
                        problem.alias_relation.canonical_signed(alias), ("theta", sign)
                    )
                    self.assertNotIn(alias, problem.constant_inputs(0))
                    options = problem.homotopy_options()
                    options.update(
                        homotopy_parameter=alias,
                        non_linear_thresh_time_idx=threshold,
                        delta_theta_0=0.5,
                    )
                    with patch.object(problem, "homotopy_options", return_value=options):
                        self.assertTrue(problem.optimize())
                        inputs = problem.constant_inputs(0)
                    expected = np.ones(len(problem.times()))
                    if threshold is not None:
                        expected[threshold:] = 0.0
                    np.testing.assert_array_equal(inputs[alias].values, expected)
                    np.testing.assert_array_equal(inputs["theta"].values, sign * expected)
                    np.testing.assert_array_equal(inputs["theta"].times, problem.times())
                    results = problem.extract_results()
                    np.testing.assert_allclose(
                        results["nonlinear_output"],
                        results["x"] + sign * expected * results["x"] ** 2,
                        atol=1e-6,
                    )

    def test_timeseries_homotopy(self):
        thresh = 3
        problem = TimeseriesHomotopyModel(thresh_idx=thresh)
        success = problem.optimize()
        self.assertTrue(success)

        # Datetimes has length 6
        n_times = len(problem.io.datetimes)
        self.assertEqual(n_times, 6)

        # Check recorded thetas
        self.assertGreater(len(problem.thetas_seen), 0)
        for theta_arr in problem.thetas_seen:
            self.assertEqual(len(theta_arr), n_times)
            # Before thresh: continuation theta
            # At or after thresh: 0.0
            np.testing.assert_array_equal(theta_arr[thresh:], np.zeros(n_times - thresh))

        # Check final theta is array with 1.0 before thresh and 0.0 after
        final_theta = problem.homotopy_theta
        self.assertIsInstance(final_theta, np.ndarray)
        np.testing.assert_array_almost_equal(final_theta[:thresh], np.ones(thresh))
        np.testing.assert_array_equal(final_theta[thresh:], np.zeros(n_times - thresh))

        # Check io timeseries for theta
        ts_times, ts_values = problem.io.get_timeseries("theta")
        np.testing.assert_array_almost_equal(ts_values[:thresh], np.ones(thresh))
        np.testing.assert_array_equal(ts_values[thresh:], np.zeros(n_times - thresh))
        self.assertFalse(problem._homotopy_is_intermediate)
        results = problem.extract_results()
        np.testing.assert_allclose(
            results["nonlinear_output"],
            results["x"] + final_theta * results["x"] ** 2,
            atol=1e-6,
        )

    def test_option_threshold_and_equations_at_each_step(self):
        problem = OptionTimeseriesHomotopyModel()
        original_extract = problem.extract_results
        seen = []

        def extract(ensemble_member=0):
            results = original_extract(ensemble_member)
            theta = problem.homotopy_theta
            np.testing.assert_allclose(
                results["nonlinear_output"],
                results["x"] + theta * results["x"] ** 2,
                atol=1e-6,
            )
            seen.append(theta[0])
            return results

        with patch.object(problem, "extract_results", side_effect=extract):
            self.assertTrue(problem.optimize())
        self.assertEqual(seen, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(problem.homotopy_theta, [1, 1, 1, 0, 0, 0])

    def test_parameter_threshold_overrides_option(self):
        problem = OptionTimeseriesHomotopyModel()
        problem.thresh_idx = 2
        self.assertTrue(problem.optimize())
        np.testing.assert_array_equal(problem.homotopy_theta, [1, 1, 0, 0, 0, 0])

    def test_history_does_not_count_towards_threshold(self):
        problem = HistoryHomotopyModel()
        self.assertTrue(problem.optimize())
        np.testing.assert_array_equal(problem.homotopy_theta, [1, 1, 1, 0])
        _, values = problem.io.get_timeseries("theta")
        np.testing.assert_array_equal(values, [1, 1, 1, 1, 1, 0])
        theta_ci = problem.constant_inputs(0)["theta"]
        np.testing.assert_allclose(theta_ci.times, [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
        np.testing.assert_array_equal(theta_ci.values, [1, 1, 1, 1, 1, 0])
        results = problem.extract_results()
        # At t=0.2 (index 1 of forecast), delay(..., 0.4) evaluates at t=-0.2 (history),
        # where x=0.8 and historical theta=1.0 -> 0.8 + 1.0 * 0.8^2 = 1.44.
        self.assertAlmostEqual(results["delayed_nonlinear_output"][1], 0.8 + 0.8**2, 1e-5)

    def test_history_with_forecast_only_parent_input(self):
        problem = HistoryHomotopyModel()
        problem.pre()
        inputs = {"theta": Timeseries(problem.times(), np.zeros(len(problem.times())))}
        with patch.object(IOMixin, "constant_inputs", return_value=inputs):
            self.assertTrue(problem.optimize(preprocessing=False))
        theta = problem.constant_inputs(0)["theta"]
        np.testing.assert_allclose(theta.times, problem.io.times_sec)
        self.assertAlmostEqual(
            problem.extract_results()["delayed_nonlinear_output"][1], 0.8 + 0.8**2, 1e-5
        )

    def test_delayed_history_at_each_continuation_step(self):
        for threshold in [0, 3, 6]:
            with self.subTest(threshold=threshold):
                problem = HistoryHomotopyModel()
                problem.thresh_idx = threshold
                original_extract = problem.extract_results
                seen = []

                def extract(
                    ensemble_member=0, original_extract=original_extract, problem=problem, seen=seen
                ):
                    results = original_extract(ensemble_member)
                    theta = float(np.asarray(problem.homotopy_theta).flat[0])
                    seen.append(theta)
                    self.assertAlmostEqual(
                        results["delayed_nonlinear_output"][1], 0.8 + theta * 0.8**2, 1e-5
                    )
                    return results

                with patch.object(problem, "extract_results", side_effect=extract):
                    self.assertTrue(problem.optimize())
                self.assertEqual(seen, [0.0] if threshold == 0 else [0.0, 0.5, 1.0])

    def test_zero_threshold_solves_once(self):
        problem = OptionTimeseriesHomotopyModel()
        problem.thresh_idx = 0
        with patch.object(
            CollocatedIntegratedOptimizationProblem,
            "optimize",
            autospec=True,
            side_effect=CollocatedIntegratedOptimizationProblem.optimize,
        ) as solve:
            self.assertTrue(problem.optimize())
        self.assertEqual(solve.call_count, 1)
        np.testing.assert_array_equal(problem.homotopy_theta, np.zeros(6))
        self.assertFalse(problem._homotopy_is_intermediate)

    def test_parameter_theta_is_not_written_to_io(self):
        problem = TimeseriesHomotopyModel(thresh_idx=None, model_name="ModelHomotopyParameter")
        self.assertTrue(problem.optimize())
        self.assertNotIn("theta", problem.io.get_timeseries_names())

    def test_pruning_without_goal_programming_warns(self):
        problem = BasicHomotopyModel()
        options = problem.homotopy_options()
        options["prune_intermediate_subproblems"] = True
        with patch.object(problem, "homotopy_options", return_value=options):
            with self.assertLogs("rtctools", level="WARNING") as logs:
                self.assertTrue(problem.optimize())
        self.assertTrue(any("prune_intermediate_subproblems" in m for m in logs.output))

    def test_pruning_support_depends_on_inheritance_order(self):
        class Problem(
            GoalProgrammingMixin,
            HomotopyMixin,
            ModelicaMixin,
            CollocatedIntegratedOptimizationProblem,
        ):
            def times(self, variable=None):
                return np.linspace(0.0, 1.0, 11)

        for cls, expected in [(Problem, False), (HomotopyGPModel, True)]:
            with self.subTest(cls=cls.__name__):
                self.assertEqual(
                    HomotopyMixin._HomotopyMixin__pruning_supported(cls.__new__(cls)), expected
                )

    def test_history_method_without_io_extends_theta(self):
        class Problem(BasicHomotopyModel):
            def history(self, ensemble_member):
                history = super().history(ensemble_member)
                history["x"] = Timeseries(np.array([-0.4, -0.2, 0.0]), np.array([0.5, 0.8, 1.1]))
                return history

        problem = Problem(model_name="ModelHomotopyInput")
        options = problem.homotopy_options()
        options["non_linear_thresh_time_idx"] = 3
        with patch.object(problem, "homotopy_options", return_value=options):
            self.assertTrue(problem.optimize())
            theta = problem.constant_inputs(0)["theta"]
        np.testing.assert_allclose(theta.times[:2], [-0.4, -0.2])
        np.testing.assert_array_equal(theta.values[:2], [1.0, 1.0])
        # delay(..., 0.4) at t=0.2 uses history at t=-0.2.
        self.assertAlmostEqual(
            problem.extract_results()["delayed_nonlinear_output"][2], 0.8 + 0.8**2, 1e-5
        )

    def test_unknown_homotopy_parameter_raises(self):
        problem = BasicHomotopyModel()
        options = problem.homotopy_options()
        options["homotopy_parameter"] = "thetaa"
        with patch.object(problem, "homotopy_options", return_value=options):
            with self.assertRaisesRegex(ValueError, "thetaa"):
                problem.optimize()

    def test_forecast_grid_different_from_io_grid(self):
        class Problem(OptionTimeseriesHomotopyModel):
            def times(self, variable=None):
                return np.linspace(0.0, 1.0, 11)

            def read(self):
                super().read()
                self.io.reference_datetime += timedelta(seconds=0.4)
                self.io.set_timeseries(
                    "x", self.io.datetimes, np.array([0.5, 0.8, 1.1, np.nan, np.nan, np.nan])
                )

        problem = Problem()
        self.assertTrue(problem.optimize())
        expected_theta = np.hstack([np.ones(3), np.zeros(8)])
        np.testing.assert_array_equal(problem.homotopy_theta, expected_theta)
        theta_ci = problem.constant_inputs(0)["theta"]
        np.testing.assert_allclose(
            theta_ci.times, np.hstack([[-0.4, -0.2], np.linspace(0.0, 1.0, 11)])
        )
        np.testing.assert_array_equal(theta_ci.values, np.hstack([[1.0, 1.0], expected_theta]))
        results = problem.extract_results()
        np.testing.assert_allclose(
            results["nonlinear_output"],
            results["x"] + expected_theta * results["x"] ** 2,
            atol=1e-6,
        )

    def test_io_cutoff_snaps_equivalent_timestamps(self):
        # times()[6] is 0.6000000000000001, while the IO timestamp is exactly 0.6.
        class Problem(OptionTimeseriesHomotopyModel):
            def times(self, variable=None):
                return np.linspace(0.0, 1.0, 11)

            def homotopy_options(self):
                options = super().homotopy_options()
                options["non_linear_thresh_time_idx"] = 6
                return options

        problem = Problem()
        self.assertTrue(problem.optimize())
        np.testing.assert_array_equal(problem.homotopy_theta, np.r_[np.ones(6), np.zeros(5)])
        _, values = problem.io.get_timeseries("theta")
        np.testing.assert_array_equal(values, [1, 1, 1, 0, 0, 0])
        np.testing.assert_allclose(problem.timeseries_at("theta", 0.6), 0.0, atol=1e-12)

    def test_timeseries_at_cutoff_absent_from_io_grid(self):
        # Forecast spacing 0.1, IO spacing 0.2: the cutoff at t=0.3 is not an IO timestamp.
        class Problem(OptionTimeseriesHomotopyModel):
            def times(self, variable=None):
                return np.linspace(0.0, 1.0, 11)

            def read(self):
                super().read()
                self.io.reference_datetime += timedelta(seconds=0.4)

        problem = Problem()
        self.assertTrue(problem.optimize())
        theta_ci = problem.constant_inputs(0)["theta"]
        for t in [-0.2, 0.0, 0.2, 0.25, 0.3, 0.4, 1.0]:
            with self.subTest(t=t):
                self.assertAlmostEqual(
                    float(problem.timeseries_at("theta", t)),
                    float(np.interp(t, theta_ci.times, theta_ci.values)),
                    1e-12,
                )
        self.assertAlmostEqual(float(problem.timeseries_at("theta", 0.3)), 0.0, 1e-12)
        # Other variables are still read from IO.
        self.assertAlmostEqual(float(problem.timeseries_at("constant_input", 0.3)), 1.0, 1e-12)

    def test_timeseries_at_uses_configured_interpolation_method(self):
        # Cutoff at times()[3] = 0.30000000000000004; t=0.3 must snap onto it.
        def make_problem(mode):
            class Problem(OptionTimeseriesHomotopyModel):
                def times(self, variable=None):
                    return np.linspace(0.0, 1.0, 11)

                def read(self):
                    super().read()
                    self.io.reference_datetime += timedelta(seconds=0.4)

                def interpolation_method(self, variable=None, mode=mode):
                    if variable == "theta" and mode is not None:
                        return mode
                    return super().interpolation_method(variable)

            return Problem()

        forward = OptionTimeseriesHomotopyModel.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD
        backward = OptionTimeseriesHomotopyModel.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD
        cases = [
            (None, {0.2: 1.0, 0.25: 0.5, 0.3: 0.0}),
            (forward, {0.2: 1.0, 0.25: 1.0, 0.3: 0.0, 0.35: 0.0}),
            (backward, {0.2: 1.0, 0.25: 0.0, 0.3: 0.0}),
        ]
        for mode, expected in cases:
            with self.subTest(mode=mode):
                problem = make_problem(mode)
                self.assertTrue(problem.optimize())
                for t, value in expected.items():
                    self.assertAlmostEqual(float(problem.timeseries_at("theta", t)), value, 1e-12)
                ts = np.array(list(expected))
                np.testing.assert_allclose(
                    problem.timeseries_at("theta", ts), list(expected.values()), atol=1e-12
                )

    def test_ensemble_theta_synchronization(self):
        problem = EnsembleHomotopyModel()
        self.assertTrue(problem.optimize())
        self.assertEqual(problem.ensemble_size, 2)
        for member in range(2):
            _, values = problem.io.get_timeseries("theta", member)
            np.testing.assert_array_equal(values, [1, 1, 1, 0, 0, 0])
            results = problem.extract_results(member)
            np.testing.assert_allclose(
                results["nonlinear_output"],
                results["x"] + values * results["x"] ** 2,
                atol=1e-6,
            )

    def test_vector_theta_requires_fixed_input(self):
        problem = BasicHomotopyModel()
        options = problem.homotopy_options()
        options["non_linear_thresh_time_idx"] = 3
        with patch.object(problem, "homotopy_options", return_value=options):
            with self.assertRaisesRegex(ValueError, "fixed input"):
                problem.optimize()

    def test_invalid_threshold(self):
        problem = TimeseriesHomotopyModel()
        for threshold in [-1, 1.5, 3.0000000001, np.nan, np.inf, "3", True, [3]]:
            with self.subTest(threshold=threshold):
                problem.thresh_idx = threshold
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    problem.optimize()

    def test_threshold_boundary_values(self):
        for threshold in [0, 6, 7, 6.0, np.float64(6), np.int64(6)]:
            with self.subTest(threshold=threshold):
                problem = TimeseriesHomotopyModel(thresh_idx=threshold)
                self.assertTrue(problem.optimize())
                self.assertFalse(problem._homotopy_is_intermediate)
                if threshold == 0:
                    np.testing.assert_array_equal(problem.homotopy_theta, np.zeros(6))
                else:
                    self.assertEqual(problem.homotopy_theta, 1.0)

    def test_clamped_endpoint_failure_bisects_actual_interval(self):
        problem = BasicHomotopyModel()
        options = problem.homotopy_options()
        options["delta_theta_0"] = 0.8
        seen = []

        def solve(**kwargs):
            seen.append(problem.homotopy_theta)
            return not (problem.homotopy_theta == 1.0 and len(seen) == 3)

        with (
            patch.object(problem, "homotopy_options", return_value=options),
            patch.object(CollocatedIntegratedOptimizationProblem, "optimize", side_effect=solve),
            patch.object(problem, "extract_results", return_value={}),
        ):
            self.assertTrue(problem.optimize(postprocessing=False))
        np.testing.assert_allclose(seen, [0.0, 0.8, 1.0, 0.9, 1.0])

    def test_pruning_flag_reset_on_exception(self):
        problem = HomotopyGPModel(prune_intermediate_subproblems=True)
        with patch.object(GoalProgrammingMixin, "optimize", side_effect=RuntimeError("solver")):
            with self.assertRaisesRegex(RuntimeError, "solver"):
                problem.optimize()
        self.assertFalse(problem._gp_prune_subproblems)

    def test_vector_homotopy_with_goal_programming(self):
        # Homotopy must wrap the complete goal programming loop.
        class Problem(HomotopyGPModel):
            def __init__(self, **kwargs):
                self.completed = []
                super().__init__(
                    model_name="ModelHomotopyInput", prune_intermediate_subproblems=True, **kwargs
                )

            def homotopy_options(self):
                options = super().homotopy_options()
                options["non_linear_thresh_time_idx"] = 3
                return options

            def priority_completed(self, priority):
                self.completed.append((self._homotopy_is_intermediate, priority))

        problem = Problem()
        self.assertTrue(problem.optimize())
        self.assertEqual(problem.completed, [(True, 1), (True, 1), (False, 1), (False, 2)])

    def test_intermediate_subproblem_pruning_disabled(self):
        # When prune_intermediate_subproblems is False:
        # Both priority 1 and priority 2 are solved at theta=0.0, theta=0.5, theta=1.0
        problem = HomotopyGPModel(prune_intermediate_subproblems=False)
        success = problem.optimize()
        self.assertTrue(success)

        # 3 homotopy steps (0.0, 0.5, 1.0) * 2 priorities = 6 priority executions
        self.assertEqual(len(problem.completed_priorities), 6)
        priorities = [p for _, p in problem.completed_priorities]
        self.assertEqual(priorities, [1, 2, 1, 2, 1, 2])

    def test_intermediate_subproblem_pruning_enabled(self):
        # When prune_intermediate_subproblems is True:
        # At theta=0.0: only priority 1 is solved
        # At theta=0.5: only priority 1 is solved
        # At theta=1.0: both priority 1 and priority 2 are solved
        problem = HomotopyGPModel(prune_intermediate_subproblems=True)
        success = problem.optimize()
        self.assertTrue(success)

        # intermediate steps (theta=0.0: [1], theta=0.5: [1]), final step (theta=1.0: [1, 2])
        # Total: 4 priority executions
        self.assertEqual(len(problem.completed_priorities), 4)
        priorities = [p for _, p in problem.completed_priorities]
        self.assertEqual(priorities, [1, 1, 1, 2])

    def test_intermediate_subproblem_pruning_respects_skip_priority(self):
        class SkipFirstPriorityModel(HomotopyGPModel):
            def priority_started(self, priority: int) -> None:
                super().priority_started(priority)
                self.skip_priority = priority == 1

        problem = SkipFirstPriorityModel(prune_intermediate_subproblems=True)
        self.assertTrue(problem.optimize())
        priorities = [p for _, p in problem.completed_priorities]
        self.assertEqual(priorities, [2, 2, 2])

    def test_pruning_skips_only_intermediate_first_priority(self):
        class Problem(HomotopyGPModel):
            def priority_started(self, priority):
                super().priority_started(priority)
                self.skip_priority = self._homotopy_is_intermediate and priority == 1

        problem = Problem(prune_intermediate_subproblems=True)
        self.assertTrue(problem.optimize())
        self.assertEqual(problem.completed_priorities, [(0.0, 2), (0.5, 2), (1.0, 1), (1.0, 2)])
