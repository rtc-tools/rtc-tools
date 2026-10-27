import logging

import casadi as ca
import numpy as np

from rtctools._internal.alias_tools import AliasDict
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
    def __init__(self):
        self._extra_variable = ca.MX.sym("extra")
        self._u_copy_mx = ca.MX.sym("u_copy")

        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitial",
            model_folder=data_path(),
        )
        self._interpolation_x = None
        self._interpolation_u = None

    def times(self, variable=None):
        times = np.linspace(0.0, 1.0, 21)
        if variable == "u":
            times = np.delete(times, 10)
        return times

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["u_max"] = 2.0
        return parameters

    def pre(self):
        # Do nothing
        pass

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        return AliasDict(
            self.alias_relation,
            {"constant_input": Timeseries(self.times(), 1 - self.times())},
        )

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    @property
    def path_variables(self):
        v = super().path_variables.copy()
        return [*v, self._u_copy_mx]

    @property
    def extra_variables(self):
        v = super().extra_variables
        return [*v, self._extra_variable]

    def bounds(self):
        b = super().bounds()
        b[self._extra_variable.name()] = [-1000, 1000]
        return b

    def path_constraints(self, ensemble_member):
        c = super().path_constraints(ensemble_member)[:]
        c.append((self.state("x") - self._extra_variable, -np.inf, 0.0))
        c.append((self._u_copy_mx - self.state("u"), 0.0, 0.0))
        return c

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options

    @property
    def equidistant(self):
        return self._equidistant

    def interpolation_method(self, variable):
        if variable == "x" and self._interpolation_x is not None:
            return self._interpolation_x
        elif variable == "u" and self._interpolation_u is not None:
            return self._interpolation_u
        else:
            return super().interpolation_method(variable)


class ModelSharedControlSlots(Model):
    def discretize_control(self, variable, ensemble_member, times, offset):
        if variable == "u":
            return offset + (np.arange(len(times)) + 1) // 2
        return super().discretize_control(variable, ensemble_member, times, offset)


class TestNumericalInterpolation(TestCase):
    def setUp(self):
        self.problem = Model()

        self.ts_equi = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.ts_non_equi = np.array([0.0, 1.0, 2.0, 3.0, 5.0])

        self.fs = np.array([0.0, 2.0, np.inf, 6.0, 8.0])


class TestNumericalLinearInterpolation(TestNumericalInterpolation):
    def test_linear(self):
        self.problem._equidistant = True

        np.testing.assert_array_equal(
            self.problem.interpolate(
                np.array([-1.0, 0.5, 2.0, 3.5, 6.0]),
                self.ts_equi,
                self.fs,
                -100.0,
                100.0,
                self.problem.INTERPOLATION_LINEAR,
            ),
            [-100.0, 1.0, np.inf, 7.0, 100.0],
        )

    def test_linear_nonequi(self):
        self.problem._equidistant = False

        np.testing.assert_array_equal(
            self.problem.interpolate(
                np.array([-1.0, 0.5, 2.0, 3.5, 6.0]),
                self.ts_non_equi,
                self.fs,
                -100.0,
                100.0,
                self.problem.INTERPOLATION_LINEAR,
            ),
            [-100.0, 1.0, np.inf, 6.5, 100.0],
        )

    def test_scalar(self):
        self.problem._equidistant = False

        t = np.array([-1.0, 0.5, 2.0, 3.5, 6.0])

        res_arr = self.problem.interpolate(
            t, self.ts_non_equi, self.fs, -100.0, 100.0, self.problem.INTERPOLATION_LINEAR
        )

        res_scalar = np.array(
            [
                self.problem.interpolate(
                    x, self.ts_non_equi, self.fs, -100.0, 100.0, self.problem.INTERPOLATION_LINEAR
                )
                for x in t
            ]
        )

        np.testing.assert_array_equal(res_scalar, res_arr)

    def test_2d_interpolate(self):
        self.problem._equidistant = False

        np.random.seed(42)

        # 3 timeseries of length 10
        shape = (10, 3)

        ts = np.cumsum(np.random.rand(shape[0]))
        fs_2d = np.random.rand(*shape)
        t = np.random.rand(shape[0])

        res_2d = self.problem.interpolate(t, ts, fs_2d, -np.inf, np.inf)

        res_1d = np.stack(
            [
                self.problem.interpolate(t, ts, fs_2d[:, i], -np.inf, np.inf)
                for i in range(shape[1])
            ],
            axis=1,
        )

        self.assertEqual(res_2d.shape, shape)
        self.assertEqual(res_1d.shape, shape)

        np.testing.assert_array_equal(res_2d, res_1d)


class TestNumericalBlockInterpolation(TestNumericalInterpolation):
    t = np.array([-1.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 6.0])

    f_left = -100.0
    f_right = 100.0
    y_forward = np.array([-100.0, 0.0, 2.0, 2.0, np.inf, np.inf, 6.0, 100.0])
    y_backward = np.array([-100.0, 2.0, 2.0, np.inf, np.inf, 6.0, 8.0, 100.0])

    def test_block_forward(self):
        self.problem._equidistant = True

        np.testing.assert_array_equal(
            self.problem.interpolate(
                self.t,
                self.ts_equi,
                self.fs,
                self.f_left,
                self.f_right,
                self.problem.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD,
            ),
            self.y_forward,
        )

    def test_block_forward_nonequi(self):
        self.problem._equidistant = False

        np.testing.assert_array_equal(
            self.problem.interpolate(
                self.t,
                self.ts_non_equi,
                self.fs,
                self.f_left,
                self.f_right,
                self.problem.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD,
            ),
            self.y_forward,
        )

    def test_block_backward(self):
        self.problem._equidistant = True

        np.testing.assert_array_equal(
            self.problem.interpolate(
                self.t,
                self.ts_equi,
                self.fs,
                self.f_left,
                self.f_right,
                self.problem.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD,
            ),
            self.y_backward,
        )

    def test_block_backward_nonequi(self):
        self.problem._equidistant = False

        np.testing.assert_array_equal(
            self.problem.interpolate(
                self.t,
                self.ts_non_equi,
                self.fs,
                self.f_left,
                self.f_right,
                self.problem.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD,
            ),
            self.y_backward,
        )


class TestNumericalInterpolationExceptions(TestNumericalInterpolation):
    def test_array_no_fleft(self):
        self.problem._equidistant = False

        with self.assertRaisesRegex(Exception, "left of range"):
            self.problem.interpolate(
                np.array([-1.0, 0.5, 2.0, 3.5, 6.0]), self.ts_non_equi, self.fs, None, np.nan
            )

    def test_array_no_fright(self):
        self.problem._equidistant = False

        with self.assertRaisesRegex(Exception, "right of range"):
            self.problem.interpolate(
                np.array([-1.0, 0.5, 2.0, 3.5, 6.0]), self.ts_non_equi, self.fs, np.nan, None
            )


class TestSymbolicInterpolation(TestCase):
    def setUp(self):
        self.problem = Model()
        self.problem._equidistant = False
        self.tolerance = 1e-8

    def _at_0_half_1(self, method, variable):
        times = self.problem.times(variable)

        t_1 = times[1]
        t_half = times[0] + (times[1] - times[0]) / 2.0
        t_0 = times[0]

        x_1 = method(variable, t_1)
        x_half = method(variable, t_half)
        x_0 = method(variable, t_0)

        f = ca.Function("f", [self.problem.solver_input], [x_0, x_half, x_1])
        res_state_at = np.array(f(self.problem.solver_output)).ravel()

        return res_state_at

    def test_state_at_linear(self):
        self.problem._interpolation_x = self.problem.INTERPOLATION_LINEAR
        self.problem.optimize()

        results = self.problem.extract_results()
        res_state_at = self._at_0_half_1(self.problem.state_at, "x")

        self.assertAlmostEqual(
            res_state_at,
            [results["x"][0], (results["x"][1] + results["x"][0]) / 2, results["x"][1]],
            self.tolerance,
        )

    def test_state_at_block_forward(self):
        self.problem._interpolation_x = self.problem.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD
        self.problem.optimize()

        results = self.problem.extract_results()
        res_state_at = self._at_0_half_1(self.problem.state_at, "x")

        self.assertAlmostEqual(
            res_state_at, [results["x"][0], results["x"][0], results["x"][1]], self.tolerance
        )

    def test_state_at_block_backward(self):
        self.problem._interpolation_x = self.problem.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD
        self.problem.optimize()

        results = self.problem.extract_results()
        res_state_at = self._at_0_half_1(self.problem.state_at, "x")

        self.assertAlmostEqual(
            res_state_at, [results["x"][0], results["x"][1], results["x"][1]], self.tolerance
        )

    def test_control_at_linear(self):
        self.problem._interpolation_u = self.problem.INTERPOLATION_LINEAR
        self.problem.optimize()

        results = self.problem.extract_results()
        res_control_at = self._at_0_half_1(self.problem.control_at, "u")

        self.assertAlmostEqual(
            res_control_at,
            [results["u"][0], (results["u"][1] + results["u"][0]) / 2, results["u"][1]],
            self.tolerance,
        )

    def test_control_at_block_forward(self):
        self.problem._interpolation_u = self.problem.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD
        self.problem.optimize()

        results = self.problem.extract_results()
        res_control_at = self._at_0_half_1(self.problem.control_at, "u")

        self.assertAlmostEqual(
            res_control_at, [results["u"][0], results["u"][0], results["u"][1]], self.tolerance
        )

    def test_control_at_block_backward(self):
        self.problem._interpolation_u = self.problem.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD
        self.problem.optimize()

        results = self.problem.extract_results()
        res_control_at = self._at_0_half_1(self.problem.control_at, "u")

        self.assertAlmostEqual(
            res_control_at, [results["u"][0], results["u"][1], results["u"][1]], self.tolerance
        )

    def test_map_linear(self):
        self.problem._interpolation_u = self.problem.INTERPOLATION_LINEAR
        self.problem.optimize()

        results = self.problem.extract_results()

        u = results["u"]
        u_copy = results["u_copy"]

        self.assertAlmostEqual(np.delete(u_copy, 10), u, self.tolerance)
        self.assertAlmostEqual(u_copy[10], (u[10] + u[9]) / 2.0, self.tolerance)

    def test_map_linear_shared_control_slots(self):
        # u[9] and u[10], on either side of t = 0.5, share one X slot
        self.problem = ModelSharedControlSlots()
        self.problem._equidistant = False
        self.problem._interpolation_u = self.problem.INTERPOLATION_LINEAR
        self.problem.optimize()

        results = self.problem.extract_results()

        u = results["u"]
        u_copy = results["u_copy"]

        self.assertAlmostEqual(u_copy[10], (u[10] + u[9]) / 2.0, self.tolerance)

    def test_map_block_forward(self):
        self.problem._interpolation_u = self.problem.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD
        self.problem.optimize()

        results = self.problem.extract_results()

        u = results["u"]
        u_copy = results["u_copy"]

        self.assertAlmostEqual(np.delete(u_copy, 10), u, self.tolerance)
        self.assertAlmostEqual(u_copy[10], u[9], self.tolerance)

    def test_map_block_backward(self):
        self.problem._interpolation_u = self.problem.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD
        self.problem.optimize()

        results = self.problem.extract_results()

        u = results["u"]
        u_copy = results["u_copy"]

        self.assertAlmostEqual(np.delete(u_copy, 10), u, self.tolerance)
        self.assertAlmostEqual(u_copy[10], u[10], self.tolerance)


class ModelShort(Model):
    """Model with just two collocation time steps, and 'u' only _one_ time step"""

    def times(self, variable=None):
        times = np.array([0.0, 1.0])
        if variable == "u":
            times = np.array([1.0])
        return times


class TestSymbolicInterpolationShort(TestCase):
    def setUp(self):
        self.problem = ModelShort()
        self.tolerance = 1e-8

    def test_single_timestep_u_backwards(self):
        """Test that CasADi interp1d works when there is only one time step to
        'interpolate' from"""
        self.problem._interpolation_x = self.problem.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD
        self.problem.optimize()

        results = self.problem.extract_results()

        self.assertEqual(len(results["u_copy"]), 2)
        self.assertEqual(len(results["u"]), 1)

        np.testing.assert_array_equal(results["u_copy"], np.broadcast_to(results["u"], (2,)))
