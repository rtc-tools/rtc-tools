import numpy as np
from casadi import SX, Function, if_else, inf, jacobian, logic_and, nlpsol, sum2, vertcat

from .bspline import BSpline


class BSpline1D(BSpline):
    """
    Arbitrary order, one-dimensional, non-uniform B-Spline implementation using Cox-de Boor
    recursion.
    """

    def __init__(self, t, w, k=3):
        """
        Create a new 1D B-Spline object.

        :param t: Knot vector.
        :param w: Weight vector.
        :param k: Spline order.
        """

        # Store arguments
        self.__t = t
        self.__w = w
        self.__k = k

    def __call__(self, x):
        """
        Evaluate the B-Spline at point x.

        The support of this function is the half-open interval [t[0], t[-1]).

        :param x: The point at which to evaluate.

        :returns: The spline evaluated at the given point.
        """
        y = 0.0
        for i in range(len(self.__t) - self.__k - 1):
            y += if_else(
                logic_and(x >= self.__t[i], x <= self.__t[i + self.__k + 1]),
                self.__w[i] * self.basis(self.__t, x, self.__k, i),
                0.0,
            )
        return y

    @classmethod
    def fit(
        cls,
        x,
        y,
        k=3,
        monotonicity=0,
        curvature=0,
        num_test_points=100,
        epsilon=1e-7,
        delta=1e-4,
        interior_pts=None,
        ipopt_options=None,
    ):
        """
        fit() returns a tck tuple like scipy.interpolate.splrep, but adjusts
        the weights to meet the desired constraints to the curvature of the spline curve.

        :param monotonicity:
            - is an integer, magnitude is ignored
            - if positive, causes spline to be monotonically increasing
            - if negative, causes spline to be monotonically decreasing
            - if 0, leaves spline monotonicity unconstrained

        :param curvature:
            - is an integer, magnitude is ignored
            - if positive, causes spline curvature to be positive (convex)
            - if negative, causes spline curvature to be negative (concave)
            - if 0, leaves spline curvature unconstrained

        :param num_test_points:
            - sets the number of points that the constraints will be applied at across
              the range of the spline

        :param epsilon:
            - offset of monotonicity and curvature constraints from zero, ensuring strict
              monotonicity
            - if epsilon is set to less than the tolerance of the solver, errors will result

        :param delta:
            - amount the first and last knots are extended outside the range of the splined points
            - ensures that the spline evaluates correctly at the first and last nodes, as
              well as the distance delta beyond these nodes

        :param interior_pts:
            - optional list of interior knots to use

        :returns: A tuple of spline knots, weights, and order.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        N = len(x)

        if interior_pts is None:
            # Generate knots: This algorithm is based on the Fitpack algorithm by p.dierckx
            # The original code lives here: http://www.netlib.org/dierckx/
            if k % 2 == 1:
                interior_pts = x[k // 2 + 1 : -k // 2]
            else:
                interior_pts = (x[k // 2 + 1 : -k // 2] + x[k // 2 : -k // 2 - 1]) / 2
        t = np.concatenate(
            (np.full(k + 1, x[0] - delta), interior_pts, np.full(k + 1, x[-1] + delta))
        )
        num_coeffs = len(t) - k - 1

        # Casadi Variable Symbols
        c = SX.sym("c", num_coeffs)
        x_sym = SX.sym("x")

        # Casadi Representation of Spline Function & Derivatives
        expr = cls(t, c, k)(x_sym)
        free_vars = [c, x_sym]
        bspline = Function("bspline", free_vars, [expr])
        J = jacobian(expr, x_sym)
        # bspline_prime = Function('bspline_prime', free_vars, [J])
        H = jacobian(J, x_sym)
        bspline_prime_prime = Function("bspline_prime_prime", free_vars, [H])

        # Objective Function
        xpt = SX.sym("xpt")
        ypt = SX.sym("ypt")
        sq_diff = Function("sq_diff", [c, xpt, ypt], [(ypt - bspline(c, xpt)) ** 2])
        sq_diff = sq_diff.map(N, "serial")
        f = sum2(sq_diff(c, SX(x), SX(y)))

        # Setup Curvature Constraints
        delta_c_max = np.full(num_coeffs - 1, inf)
        delta_c_min = np.full(num_coeffs - 1, -inf)
        max_slope_slope = np.full(num_test_points, inf)
        min_slope_slope = np.full(num_test_points, -inf)
        if monotonicity != 0:
            if monotonicity < 0:
                delta_c_max = np.full(num_coeffs - 1, -epsilon)
            else:
                delta_c_min = np.full(num_coeffs - 1, epsilon)
        if curvature != 0:
            if curvature < 0:
                max_slope_slope = np.full(num_test_points, -epsilon)
            else:
                min_slope_slope = np.full(num_test_points, epsilon)
        monotonicity_constraints = vertcat(*[c[i + 1] - c[i] for i in range(num_coeffs - 1)])
        x_linspace = np.linspace(x[0], x[-1], num_test_points)
        curvature_constraints = vertcat(*[bspline_prime_prime(c, SX(x)) for x in x_linspace])
        g = vertcat(monotonicity_constraints, curvature_constraints)
        lbg = np.concatenate((delta_c_min, min_slope_slope))
        ubg = np.concatenate((delta_c_max, max_slope_slope))

        # Perform mini-optimization problem to calculate the the values of c
        nlp = {"x": c, "f": f, "g": g}
        my_solver = "ipopt"
        solver = nlpsol(
            "solver",
            my_solver,
            nlp,
            {"print_time": 0, "expand": True, "ipopt": ipopt_options},
        )
        sol = solver(lbg=lbg, ubg=ubg)
        stats = solver.stats()
        return_status = stats["return_status"]
        if return_status not in ["Solve_Succeeded", "Solved_To_Acceptable_Level", "SUCCESS"]:
            raise Exception(f"Spline fitting failed with status {return_status}")

        # Return the new tck tuple
        return (t, np.array(sol["x"]).ravel(), k)
