from unittest import TestCase

import casadi as ca
import numpy as np

from rtctools._internal.casadi_helpers import interpolate


class TestInterpolate(TestCase):
    def test_piecewise_constant_matches_interp1d(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            ts = np.unique(rng.integers(0, 100, rng.integers(1, 10))).astype(float)
            # Points at, between and outside the knots
            t = np.concatenate([ts, rng.uniform(-10, 110, 20)])
            xs = rng.random((len(ts), 2))
            xs_sym = ca.SX.sym("x", len(ts), 2)
            for mode, mode_str in ((1, "floor"), (2, "ceil")):
                if len(ts) == 1:
                    expected = ca.DM(np.repeat(xs, len(t), axis=0))
                else:
                    expected = ca.interp1d(ts, ca.DM(xs), t, mode_str, False)
                np.testing.assert_array_equal(interpolate(ts, xs, t, False, mode), expected)

                symbolic = ca.Function("f", [xs_sym], [interpolate(ts, xs_sym, t, False, mode)])
                np.testing.assert_array_equal(symbolic(xs), expected)

    def test_piecewise_constant_rejects_non_increasing_times(self):
        for ts in ([0.0, 1.0, 1.0, 2.0], [0.0, 2.0, 1.0]):
            with self.assertRaises(ValueError):
                interpolate(np.array(ts), np.zeros((len(ts), 1)), np.array([0.5]), False, 1)
