from unittest import TestCase

import numpy as np
from casadi import SX, Function, linspace

import rtctools.data.interpolation.bspline1d
import rtctools.data.interpolation.bspline2d


class TestBSpline1DFit(TestCase):
    # Only tests for defaut case, k = 3

    def setUp(self):
        start = 0
        end = np.pi * 4
        self.x = np.linspace(start, end, 30)
        self.y = np.sin(self.x)
        self.num_test_points = 100
        self.testpoints = linspace(start, end, self.num_test_points)

    def _y_list(self, monotonicity=0, curvature=0):
        y_list = np.empty(self.num_test_points - 1)
        tck = rtctools.data.interpolation.bspline1d.BSpline1D.fit(
            self.x, self.y, monotonicity=monotonicity, curvature=curvature
        )
        x = SX.sym("x")
        f = Function("f", [x], [rtctools.data.interpolation.bspline1d.BSpline1D(*tck)(x)])
        for xi in range(self.num_test_points - 1):
            y_list[xi] = f(self.testpoints[xi])[0]
        return y_list

    def test_monotonicity(self):
        y_list = self._y_list(monotonicity=1)
        self.assertTrue(np.all((y_list[1:] - y_list[:-1]) > 0))
        y_list = self._y_list(monotonicity=-1)
        self.assertTrue(np.all((y_list[1:] - y_list[:-1]) < 0))

    def test_curvature(self):
        y_list = self._y_list(curvature=1)
        y_slope_list = y_list[1:] - y_list[:-1]
        self.assertTrue(np.all((y_slope_list[1:] - y_slope_list[:-1]) > 0))

        y_list = self._y_list(curvature=-1)
        y_slope_list = y_list[1:] - y_list[:-1]
        self.assertTrue(np.all((y_slope_list[1:] - y_slope_list[:-1]) < 0))

    def test_fit(self):
        """
        Regression test for BSpline1D.fit.

        This checks that BSpline1D.fit succeeds for a particular case when using
        the nlp option nlp_scaling_method=none.
        """
        x = [4.91, 5.92, 10.83, 11.16, 11.51]
        y = [1.124038e08, 1.354540e08, 2.475718e08, 2.551113e08, 2.631086e08]
        t, c, k = rtctools.data.interpolation.bspline1d.BSpline1D.fit(
            x=x,
            y=y,
            k=3,
            monotonicity=1,
            curvature=0,
            ipopt_options={"nlp_scaling_method": "none"},
        )
        t_ref = np.array(
            [4.9099, 4.9099, 4.9099, 4.9099, 10.83, 11.5101, 11.5101, 11.5101, 11.5101]
        )
        c_ref = np.array(
            [
                1.12401518e08,
                1.57433811e08,
                2.07664058e08,
                2.57930573e08,
                2.63110885e08,
            ]
        )
        # 9 knots and k = 3 give 9 - k - 1 = 5 coefficients, one per data point.
        np.testing.assert_almost_equal(t, t_ref)
        np.testing.assert_almost_equal(c, c_ref, decimal=0)
        self.assertEqual(k, 3)

        # One point inside the support of each of the 5 basis functions fixes all 5
        # coefficients, so equal values here mean the same curve, not just similar ones.
        x_trial = [5.4, 8.0, 10.83, 11.0, 11.3]
        y_ref = [1.23586075e08, 1.82936781e08, 2.47571800e08, 2.51455702e08, 2.58310098e08]
        x_sym = SX.sym("x")
        f = Function(
            "f", [x_sym], [rtctools.data.interpolation.bspline1d.BSpline1D(t, c, k)(x_sym)]
        )
        y_trial = [float(f(xi)) for xi in x_trial]
        np.testing.assert_almost_equal(y_trial, y_ref, decimal=0)


# class TestBSpline2D(TestCase):
#     def setUp(self):
#         x = np.linspace(-1.0, 1.0, 10)
#         y = np.linspace(-1.0, 1.0, 10)
#         xx, yy = np.meshgrid(x, y)
#         x = xx.flatten()
#         y = yy.flatten()
#
#         z = (x + y) * np.exp(-(x**2 + y**2))
#         self.tck = scipy.interpolate.bisplrep(x, y, z)
#         self.bspline = rtctools.data.interpolation.bspline2d.BSpline2D(*self.tck)
#
#         points1 = [-1.0, -0.9999, 0.0, 0.5, 0.75, 0.99999]
#         # Test Cartesian product points1 x points1
#         self.times = [(x, y) for x in points1 for y in points1]
#         self.tolerance = 1e-10
#
#     def test_value(self):
#         x = SX.sym('x')
#         y = SX.sym('y')
#         f = SXFunction('f', [x, y], [self.bspline(x, y)])
#
#         for (x, y) in self.times:
#             [z] = f.call([x, y])
#             self.assertAlmostEqual(scipy.interpolate.bisplev(x, y, self.tck), z)
#
#     def test_derivative(self):
#         x = SX.sym('x')
#         y = SX.sym('y')
#         f = SXFunction('f', [x, y], [self.bspline(x, y)])
#
#         derx = SXFunction('derx', [x, y], [f.jac(0, 0)])
#
#         dery = SXFunction('dery', [x, y], [f.jac(1, 0)])
#
#         for (x, y) in self.times:
#             [dzdx] = derx.call([x, y])
#             self.assertAlmostEqual(scipy.interpolate.bisplev(x, y, self.tck, dx=1), dzdx)
#
#             [dzdy] = dery.call([x, y])
#             self.assertAlmostEqual(scipy.interpolate.bisplev(x, y, self.tck, dy=1), dzdy)
