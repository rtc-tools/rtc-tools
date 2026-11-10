import casadi as ca
import numpy as np

from rtctools.optimization.goal_programming_mixin_base import (
    Goal,
    StateGoal,
    _GoalConstraint,
    _GoalProgrammingMixinBase,
)


class LinearizedOrderGoal(Goal):
    #: Override linearization of goal order. Related global goal programming
    #: option is ``linearize_goal_order``
    #: (see :py:meth:`LinearizedOrderGoalProgrammingMixin.goal_programming_options`).
    #: The default value of None defers to the global option, but the user can
    #: explicitly override it per goal by setting this value to True or False.
    linearize_order = None

    #: Coefficients to linearize a goal's order
    _linear_coefficients = {}

    @classmethod
    def _get_linear_coefficients(cls, order, eps=0.1, kind="balanced"):
        assert order > 1, "Order should be strictly larger than one"

        try:
            return cls._linear_coefficients[eps][order]
        except KeyError:
            pass

        x = ca.SX.sym("x")
        a = ca.SX.sym("a")
        b = ca.SX.sym("b")

        # Strike a balance between "absolute error < eps" and "relative error < eps" by
        # multiplying eps with x**(order-1)
        if kind == "balanced":
            f = x**order - eps * x ** (order - 1) - (a * x + b)
        elif kind == "abs":
            f = x**order - eps - (a * x + b)
        else:
            raise Exception(f"Unknown error approximation strategy '{kind}'")

        res_vals = ca.Function("res_vals", [x, ca.vertcat(a, b)], [f])

        do_step = ca.rootfinder("next_state", "fast_newton", res_vals)

        x = 0.0
        a = 0.0
        b = 0.0

        xs = [0.0]
        while x < 1.0:
            # Initial guess larger than 1.0 to always have the next point be
            # on the right (i.e. not left) side.
            x = float(do_step(2.0, [a, b]))
            a = order * x ** (order - 1)
            b = x**order - a * x
            xs.append(x)

        # Turn underestimate into an overestimate, such that we get rid of
        # horizontal line at origin.
        xs[-1] = 1.0
        xs = np.array(xs)
        ys = xs**order

        a = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        b = ys[1:] - a * xs[1:]
        lines = list(zip(a, b, strict=True))

        cls._linear_coefficients.setdefault(eps, {})[order] = lines

        return lines


class LinearizedOrderStateGoal(LinearizedOrderGoal, StateGoal):
    """
    Convenience class definition for linearized order state goals. Note that
    it is possible to just inherit from :py:class:`.LinearizedOrderGoal` to get the needed
    functionality for control of the linearization at goal level.
    """

    pass


class LinearizedOrderGoalProgrammingMixin(_GoalProgrammingMixinBase):
    """
    Adds support for linearization of the goal objective functions, i.e. the
    violation variables to a certain power. This can be used to keep a problem
    fully linear and/or make sure that no quadratic constraints appear when using
    the goal programming option ``keep_soft_constraints``.
    """

    def goal_programming_options(self):
        """
        If ``linearize_goal_order`` is set to ``True``, the goal's order will be
        approximated linearly for any goals where order > 1. Note that this option
        does not work with minimization goals of higher order. Instead, it is
        suggested to transform these minimization goals into goals with a target (and
        function range) when using this option. Note that this option can be overridden
        on the level of a goal by using a :py:class:`LinearizedOrderGoal` (see
        :py:attr:`LinearizedOrderGoal.linearize_order`).
        """
        options = super().goal_programming_options()
        options["linearize_goal_order"] = True
        return options

    def _gp_validate_goals(self, goals, is_path_goal):
        options = self.goal_programming_options()

        for goal in goals:
            goal_linearize = None
            if isinstance(goal, LinearizedOrderGoal):
                goal_linearize = goal.linearize_order

            if goal_linearize or (options["linearize_goal_order"] and goal_linearize is not False):
                if not goal.has_target_bounds and goal.order > 1:
                    raise Exception(
                        "Higher order minimization goals not allowed with "
                        f"`linearize_goal_order` for goal {goal}"
                    )

        super()._gp_validate_goals(goals, is_path_goal)

    def _gp_goal_constraints(self, goals, sym_index, options, is_path_goal):
        options = self.goal_programming_options()

        def _linearize_goal(goal):
            goal_linearize = None
            if isinstance(goal, LinearizedOrderGoal):
                goal_linearize = goal.linearize_order

            if goal_linearize or (options["linearize_goal_order"] and goal_linearize is not False):
                if goal.order > 1 and not goal.critical:
                    return True
                else:
                    return False
            else:
                return False

        lo_soft_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        lo_epsilons = []

        # For the linearized goals, we use all of the normal processing,
        # except for the objective. We can override the objective function by
        # setting a _objective_func function on the Goal object.
        for j, goal in enumerate(goals):
            if not _linearize_goal(goal):
                continue

            assert goal.has_target_bounds, "Cannot linearize minimization goals"

            # Make a linear epsilon, and constraints relating the linear
            # variable to the original objective function
            path_prefix = "path_" if is_path_goal else ""
            linear_variable = ca.MX.sym(path_prefix + f"lineps_{sym_index}_{j}", goal.size)

            lo_epsilons.append(linear_variable)

            if isinstance(goal, LinearizedOrderGoal):
                coeffs = goal._get_linear_coefficients(goal.order)
            else:
                coeffs = LinearizedOrderGoal._get_linear_coefficients(goal.order)

            epsilon_name = path_prefix + f"eps_{sym_index}_{j}"

            for a, b in coeffs:
                # We add to soft constraints, as these constraints are no longer valid when
                # having `keep_soft_constraints` = False. This is because the `epsilon` and
                # the `linear_variable` no longer exist in the next priority.
                for ensemble_member in range(self.ensemble_size):

                    def _f(
                        problem,
                        goal=goal,
                        epsilon_name=epsilon_name,
                        linear_variable=linear_variable,
                        a=a,
                        b=b,
                        ensemble_member=ensemble_member,
                        is_path_constraint=is_path_goal,
                    ):
                        if is_path_constraint:
                            eps = problem.variable(epsilon_name)
                            lin = problem.variable(linear_variable.name())
                        else:
                            eps = problem.extra_variable(epsilon_name, ensemble_member)
                            lin = problem.extra_variable(linear_variable.name(), ensemble_member)

                        return lin - a * eps - b

                    lo_soft_constraints[ensemble_member].append(
                        _GoalConstraint(goal, _f, 0.0, np.inf, False)
                    )

            if is_path_goal and options["scale_by_problem_size"]:
                goal_m, goal_M = self._gp_min_max_arrays(goal, target_shape=len(self.times()))
                goal_active = np.isfinite(goal_m) | np.isfinite(goal_M)
                n_active = np.sum(goal_active.astype(int), axis=0)
            else:
                n_active = 1

            def _objective_func(
                problem,
                ensemble_member,
                goal=goal,
                linear_variable=linear_variable,
                is_path_goal=is_path_goal,
                n_active=n_active,
            ):
                if is_path_goal:
                    lin = problem.variable(linear_variable.name())
                else:
                    lin = problem.extra_variable(linear_variable.name(), ensemble_member)

                return goal.weight * lin / n_active

            goal._objective_func = _objective_func

        (
            epsilons,
            objectives,
            soft_constraints,
            hard_constraints,
            extra_constants,
        ) = super()._gp_goal_constraints(goals, sym_index, options, is_path_goal)

        epsilons = epsilons + lo_epsilons
        for ensemble_member in range(self.ensemble_size):
            soft_constraints[ensemble_member].extend(lo_soft_constraints[ensemble_member])

        return epsilons, objectives, soft_constraints, hard_constraints, extra_constants
