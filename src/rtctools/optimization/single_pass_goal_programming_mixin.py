import itertools
import logging
from collections import OrderedDict
from enum import Enum

import casadi as ca
import numpy as np

from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check

from .goal_programming_mixin import GoalProgrammingMixin
from .goal_programming_mixin_base import (  # noqa: F401
    Goal,
    StateGoal,
    _EmptyEnsembleList,
    _EmptyEnsembleOrderedDict,
    _GoalProgrammingMixinBase,
)
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class SinglePassMethod(Enum):
    APPEND_CONSTRAINTS_OBJECTIVE = 1
    UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS = 2


class SinglePassGoalProgrammingMixin(_GoalProgrammingMixinBase):
    r"""
    Adds lexicographic goal programming to your optimization problem.

    Unlike :py:class:`.GoalProgrammingMixin`, this mixin will call
    :py:meth:`.transcribe` only once per call to :py:meth:`.optimize`, and not
    :math:`N` times for :math:`N` priorities. It works similar to how
    `keep_soft_constraints = True` works for :py:class:`.GoalProgrammingMixin`,
    while avoiding the repeated calls to transcribe the problem.

    This mixin can work in one of two ways. What is shared between them is
    that all violation variables of all goals are generated once at the
    beginning, such that the state vector is exactly the same for all
    priorities. They also share that all goal constraints are added from the
    start. How they differ is in how they handle/append the constraints on the
    objective of previous priorities:

    1. At priority :math:`i` the constraints are the same as the ones at
       priority :math:`i - 1` with the addition of the objective constraint
       related to priority :math:`i - 1`. This is the default method.

    2. All objective constraints are added at the start. The objective
       constraints will have bound of :math:`[-\inf, \inf]` at the start, to be
       updated after each priority finishes.

    There is a special `qpsol` alternative available :py:class:`CachingQPSol`,
    that will avoid recalculations on constraints that were already there in
    previous priorities. This works for both options outlined above, because
    the assumptions of :py:class:`CachingQPSol` are that:

    1. The state vector does not change
    2. Any new constraints are appended at the end

    .. note::

        Just like GoalProgrammingMixin, objective constraints are only added on
        the goal objectives, not on any custom user objective.
    """

    single_pass_method = SinglePassMethod.APPEND_CONSTRAINTS_OBJECTIVE

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        # Initialize instance variables, so that the overridden methods may be
        # called outside of the goal programming loop, for example in pre().
        self._gp_first_run = True
        self.__results_are_current = False

        self.__constraint_store = _EmptyEnsembleOrderedDict()
        self.__path_constraint_store = _EmptyEnsembleOrderedDict()

        self.__problem_constraints = _EmptyEnsembleList()
        self.__problem_path_constraints = _EmptyEnsembleList()
        self.__problem_epsilons = []
        self.__problem_path_epsilons = []
        self.__problem_path_timeseries = []
        self.__problem_parameters = []

        self.__current_priority = 0
        self.__original_constraints = None
        self.__previous_constraints = None

        self.__soft_constraints_per_priority = []
        self.__path_soft_constraints_per_priority = []

        self.__objectives_per_priority = []
        self.__path_objectives_per_priority = []

        if isinstance(self, GoalProgrammingMixin):
            raise Exception(
                "Cannot be an instance of both GoalProgrammingMixin "
                "and SinglePassGoalProgrammingMixin"
            )

    @property
    def extra_variables(self):
        return self.__problem_epsilons

    @property
    def path_variables(self):
        return self.__problem_path_epsilons

    @ensemble_bounds_check
    def bounds(self, ensemble_member: int | None = None):
        bounds = (
            super().bounds(ensemble_member) if self.ensemble_specific_bounds else super().bounds()
        )

        for epsilon in self.__problem_epsilons + self.__problem_path_epsilons:
            bounds[epsilon.name()] = (0.0, 1.0)
        return bounds

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)

        n_times = len(self.times())

        # Append min/max timeseries to the constant inputs. Note that min/max
        # timeseries are shared between all ensemble members.
        for variable, value in self.__problem_path_timeseries:
            if isinstance(value, np.ndarray):
                value = Timeseries(self.times(), np.broadcast_to(value, (n_times, len(value))))
            elif not isinstance(value, Timeseries):
                value = Timeseries(self.times(), np.full(n_times, value))

            constant_inputs[variable] = value
        return constant_inputs

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        # Append min/max values to the parameters. Note that min/max values
        # are shared between all ensemble members.
        for variable, value in self.__problem_parameters:
            parameters[variable] = value

        return parameters

    def seed(self, ensemble_member):
        assert self._gp_first_run

        seed = super().seed(ensemble_member)

        # Seed epsilons of current priority
        for epsilon in self.__problem_epsilons:
            eps_size = epsilon.size1()
            if eps_size > 1:
                seed[epsilon.name()] = np.ones(eps_size)
            else:
                seed[epsilon.name()] = 1.0

        times = self.times()
        for epsilon in self.__problem_path_epsilons:
            eps_size = epsilon.size1()
            if eps_size > 1:
                seed[epsilon.name()] = Timeseries(times, np.ones((eps_size, len(times))))
            else:
                seed[epsilon.name()] = Timeseries(times, np.ones(len(times)))

        return seed

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        additional_constraints = itertools.chain(
            self.__constraint_store[ensemble_member].values(),
            self.__problem_constraints[ensemble_member],
        )

        for constraint in additional_constraints:
            constraints.append((constraint.function(self), constraint.min, constraint.max))

        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)

        additional_path_constraints = itertools.chain(
            self.__path_constraint_store[ensemble_member].values(),
            self.__problem_path_constraints[ensemble_member],
        )

        for constraint in additional_path_constraints:
            path_constraints.append((constraint.function(self), constraint.min, constraint.max))

        return path_constraints

    def solver_options(self):
        # TODO: Split off into private

        # Call parent
        options = super().solver_options()

        solver = options["solver"]
        assert solver in ["bonmin", "ipopt"]

        # Make sure constant states, such as min/max timeseries for violation variables,
        # are turned into parameters for the final optimization problem.
        ipopt_options = options[solver]
        ipopt_options["fixed_variable_treatment"] = "make_parameter"

        # Define temporary variable to avoid infinite loop between
        # solver_options and goal_programming_options.
        self._loop_breaker_solver_options = True

        if not hasattr(self, "_loop_breaker_goal_programming_options"):
            if not self.goal_programming_options()["mu_reinit"]:
                ipopt_options["mu_strategy"] = "monotone"
                if not self._gp_first_run:
                    ipopt_options["mu_init"] = self.solver_stats["iterations"]["mu"][-1]

        delattr(self, "_loop_breaker_solver_options")

        return options

    def goal_programming_options(self) -> dict[str, float | bool]:
        """
        Returns a dictionary of options controlling the goal programming process.

        +---------------------------+-----------+---------------+
        | Option                    | Type      | Default value |
        +===========================+===========+===============+
        | ``constraint_relaxation`` | ``float`` | ``0.0``       |
        +---------------------------+-----------+---------------+
        | ``mu_reinit``             | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``fix_minimized_values``  | ``bool``  | ``True/False``|
        +---------------------------+-----------+---------------+
        | ``check_monotonicity``    | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``equality_threshold``    | ``float`` | ``1e-8``      |
        +---------------------------+-----------+---------------+
        | ``scale_by_problem_size`` | ``bool``  | ``False``     |
        +---------------------------+-----------+---------------+

        When a priority's objective is turned into a hard constraint,
        the constraint is relaxed with ``constraint_relaxation``. Use of this option is
        normally not required. Note that:

        When using the default solver (IPOPT), its barrier parameter ``mu`` is
        normally re-initialized at every iteration of the goal programming
        algorithm, unless mu_reinit is set to ``False``.  Use of this option
        is normally not required.

        If ``fix_minimized_values`` is set to ``True``, goal functions will be set to equal their
        optimized values in optimization problems generated during subsequent priorities. Otherwise,
        only an upper bound will be set. Use of this option is normally not required.
        Note that the use of this option may add non-convex constraints to the optimization
        problem. The default value for this parameter is ``True`` for the default solvers
        IPOPT/BONMIN. If any other solver is used, the default value is ``False``.

        If ``check_monotonicity`` is set to ``True``, then it will be checked whether goals with
        the same function key form a monotonically decreasing sequence with regards to the target
        interval.

        The option ``equality_threshold`` controls when a two-sided inequality constraint is folded
        into an equality constraint.

        If ``scale_by_problem_size`` is set to ``True``, the objective (i.e. the sum of the
        violation variables) will be divided by the number of goals, and the path objective will
        be divided by the number of path goals and the number of active time steps (per goal).
        This will make sure the objectives are always in the range [0, 1], at the cost of solving
        each goal/time step less accurately.

        :returns: A dictionary of goal programming options.
        """

        options = {}

        options["mu_reinit"] = True
        options["constraint_relaxation"] = 0.0  # Disable by default
        options["fix_minimized_values"] = False
        options["check_monotonicity"] = True
        options["equality_threshold"] = 1e-8
        options["scale_by_problem_size"] = False

        # Forced options to be able to re-use GoalProgrammingMixin's
        # GoalProgrammingMixin._gp_* functions. These are not relevant for
        # SinglePassGoalProgrammingMixin, or should be set to a certain value
        # for it to make sense.
        options["violation_relaxation"] = 0.0  # Disable by default
        options["violation_tolerance"] = np.inf  # Disable by default
        options["interior_distance"] = 1e-6
        options["keep_soft_constraints"] = True

        # Define temporary variable to avoid infinite loop between
        # solver_options and goal_programming_options.
        self._loop_breaker_goal_programming_options = True

        if not hasattr(self, "_loop_breaker_solver_options"):
            if self.solver_options()["solver"] in {"ipopt", "bonmin"}:
                options["fix_minimized_values"] = True

        delattr(self, "_loop_breaker_goal_programming_options")

        return options

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do pre-processing
        if preprocessing:
            self.pre()

        # Group goals into subproblems
        subproblems = []
        goals = self.goals()
        path_goals = self.path_goals()

        options = self.goal_programming_options()

        # Validate goal definitions
        self._gp_validate_goals(goals, is_path_goal=False)
        self._gp_validate_goals(path_goals, is_path_goal=True)

        priorities = sorted(
            {int(goal.priority) for goal in itertools.chain(goals, path_goals) if not goal.is_empty}
        )

        for priority in priorities:
            subproblems.append(
                (
                    priority,
                    [
                        goal
                        for goal in goals
                        if int(goal.priority) == priority and not goal.is_empty
                    ],
                    [
                        goal
                        for goal in path_goals
                        if int(goal.priority) == priority and not goal.is_empty
                    ],
                )
            )

        # Solve the subproblems one by one
        logger.info("Starting goal programming")

        success = False

        self.__constraint_store = [OrderedDict() for ensemble_member in range(self.ensemble_size)]
        self.__path_constraint_store = [
            OrderedDict() for ensemble_member in range(self.ensemble_size)
        ]

        self.__problem_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__problem_path_constraints = [[] for ensemble_member in range(self.ensemble_size)]

        self.__problem_epsilons = []
        self.__problem_parameters = []
        self.__problem_path_epsilons = []
        self.__problem_path_timeseries = []

        self._gp_first_run = True
        self.__results_are_current = False

        self.__current_priority = 0
        self.__original_constraints = None

        self.__objectives_per_priority = []
        self.__path_objectives_per_priority = []

        self.__additional_constraints = []
        self.__objectives = []

        for i, (_, goals, path_goals) in enumerate(subproblems):
            (
                subproblem_epsilons,
                subproblem_objectives,
                subproblem_soft_constraints,
                hard_constraints,
                subproblem_parameters,
            ) = self._gp_goal_constraints(goals, i, options, is_path_goal=False)

            (
                subproblem_path_epsilons,
                subproblem_path_objectives,
                subproblem_path_soft_constraints,
                path_hard_constraints,
                subproblem_path_timeseries,
            ) = self._gp_goal_constraints(path_goals, i, options, is_path_goal=True)

            # Put hard constraints in the constraint stores
            self._gp_update_constraint_store(self.__constraint_store, hard_constraints)
            self._gp_update_constraint_store(self.__path_constraint_store, path_hard_constraints)

            # Append new variables, parameters, timeseries and constraints to
            # their respective lists
            self.__problem_epsilons.extend(subproblem_epsilons)
            self.__problem_path_epsilons.extend(subproblem_path_epsilons)

            self.__problem_parameters.extend(subproblem_parameters)
            self.__problem_path_timeseries.extend(subproblem_path_timeseries)

            for ensemble_member in range(self.ensemble_size):
                self.__problem_constraints[ensemble_member].extend(
                    subproblem_soft_constraints[ensemble_member]
                )
                self.__problem_path_constraints[ensemble_member].extend(
                    subproblem_path_soft_constraints[ensemble_member]
                )

            self.__objectives_per_priority.append(subproblem_objectives)
            self.__path_objectives_per_priority.append(subproblem_path_objectives)

        for priority in priorities:
            logger.info(f"Solving goals at priority {priority}")

            # Call the pre priority hook
            self.priority_started(priority)

            # Solve subproblem
            success = super().optimize(
                preprocessing=False,
                postprocessing=False,
                log_solver_failure_as_error=log_solver_failure_as_error,
            )
            if not success:
                break

            self._gp_first_run = False

            # To match GoalProgrammingMixin's behavior of applying the
            # constraint_relaxation value at priority 2 on the objective of
            # priority 2 (and not that of priority 1), we have to store the
            # two relevant options here for later use.
            options = self.goal_programming_options()
            self.__objective_constraint_options = {
                k: v
                for k, v in options.items()
                if k in {"fix_minimized_values", "constraint_relaxation"}
            }

            # Store results.  Do this here, to make sure we have results even
            # if a subsequent priority fails.
            self.__results_are_current = False
            self.__results = [
                self.extract_results(ensemble_member)
                for ensemble_member in range(self.ensemble_size)
            ]
            self.__results_are_current = True

            # Call the post priority hook, so that intermediate results can be
            # logged/inspected.
            self.priority_completed(priority)

            self.__current_priority += 1

        logger.info("Done goal programming")

        # Do post-processing
        if postprocessing:
            self.post()

        # Done
        return success

    def transcribe(self):
        def _objective_func(subproblem_objectives, subproblem_path_objectives):
            val = 0.0
            for ensemble_member in range(self.ensemble_size):
                n_objectives = self._gp_n_objectives(
                    subproblem_objectives, subproblem_path_objectives, ensemble_member
                )
                expr = self._gp_objective(subproblem_objectives, n_objectives, ensemble_member)
                expr += ca.sum1(
                    self.map_path_expression(
                        self._gp_path_objective(
                            subproblem_path_objectives, n_objectives, ensemble_member
                        ),
                        ensemble_member,
                    )
                )
                val += self.ensemble_member_probability(ensemble_member) * expr

            return val

        if self._gp_first_run:
            discrete, lbx, ubx, lbg, ubg, x0, nlp = super().transcribe()
            self.__original_transcribe = (discrete, lbx, ubx, lbg, ubg, x0, nlp)

            self.__additional_constraints = []
            self.__objectives = []

            # Objectives
            for subproblem_objectives, subproblem_path_objectives in zip(
                self.__objectives_per_priority, self.__path_objectives_per_priority, strict=True
            ):
                self.__objectives.append(
                    _objective_func(subproblem_objectives, subproblem_path_objectives)
                )

            if self.single_pass_method == SinglePassMethod.UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS:
                # The objectives are also directly added as constraints
                constraints = [(objective, -np.inf, np.inf) for objective in self.__objectives]
                self.__additional_constraints.extend(constraints)

        # Add constraint on the objective of previous priority
        if self.__current_priority > 0:
            options = self.__objective_constraint_options

            previous_objective = self.__objectives[self.__current_priority - 1]
            f = ca.Function("tmp", [self.solver_input], [previous_objective])
            obj_val = float(f(self.solver_output))

            if options["fix_minimized_values"]:
                lb, ub = obj_val, obj_val
                self.linear_collocation = False  # Disable solver option jac_c_constant for IPOPT
            else:
                obj_val += options["constraint_relaxation"]
                lb, ub = -np.inf, obj_val

            if self.single_pass_method == SinglePassMethod.APPEND_CONSTRAINTS_OBJECTIVE:
                self.__additional_constraints.append(
                    (self.__objectives[self.__current_priority - 1], lb, ub)
                )
            elif self.single_pass_method == SinglePassMethod.UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS:
                ind = self.__current_priority - 1
                constraint = self.__additional_constraints[ind]
                self.__additional_constraints[ind] = (constraint[0], lb, ub)

        # Update the NLP
        discrete, lbx, ubx, lbg, ubg, x0, nlp = self.__original_transcribe
        nlp = nlp.copy()

        if self.__additional_constraints:
            g_extra, lbg_extra, ubg_extra = zip(*self.__additional_constraints, strict=True)

            g = ca.vertcat(nlp["g"], *g_extra)
            lbg = [*lbg.copy(), *lbg_extra]
            ubg = [*ubg.copy(), *ubg_extra]

            nlp["g"] = g

        nlp["f"] = self.__objectives[self.__current_priority]

        if not self._gp_first_run:
            x0 = self.solver_output.copy()

        return discrete, lbx, ubx, lbg, ubg, x0, nlp

    def extract_results(self, ensemble_member=0):
        if self.__results_are_current:
            logger.debug("Returning cached results")
            return self.__results[ensemble_member]

        # If self.__results is not up to date, do the super().extract_results
        # method
        return super().extract_results(ensemble_member)


class CachingQPSol:
    """
    Alternative to :py:func:`ca.qpsol` that caches the Jacobian between calls.

    Typical usage would be something like:

    .. code-block::

        def pre(self):
            self._qpsol = CachingQPSol()
            super().pre()

        def solver_options():
            options = super().solver_options()
            options['casadi_solver'] = self._qpsol
            return options
    """

    def __init__(self):
        self._tlcache = {}

    def __call__(self, name, solver_name, nlp, options):
        class Solver:
            def __init__(
                self, nlp=nlp, solver_name=solver_name, options=options, cache=self._tlcache
            ):
                x = nlp["x"]
                f = nlp["f"]
                g = nlp["g"]

                if isinstance(x, ca.MX):
                    # Can only convert SX to DM
                    x = ca.SX.sym("X", *x.shape)
                    x_mx = nlp["x"]
                    expand = True
                else:
                    x_mx = None
                    expand = False

                if expand:
                    expand_f = ca.Function("f", [x_mx], [f]).expand()
                    f = expand_f(x)

                # Gradient of the objective: gf == Hx + g
                gf = ca.gradient(f, x)

                # Identify the linear term in the objective
                c = ca.substitute(gf, x, ca.DM.zeros(x.sparsity()))

                # Identify the quadratic term in the objective
                H = 0.5 * ca.jacobian(gf, x, {"symmetric": True})

                if cache:
                    if not x.size1() == cache["A"].size2():
                        raise Exception(
                            "Number of variables {} does not match "
                            "cached constraint matrix dimensions {}".format(
                                x.size1(), cache["A"].shape
                            )
                        )

                    n_g_cache = cache["A"].size1()
                    n_g = g.size1()

                    if n_g_cache == n_g:
                        b = cache["b"]
                        A = cache["A"]
                    else:
                        g_new = g[n_g_cache:]

                        if expand:
                            expand_g_new = ca.Function("f", [x_mx], [g_new]).expand()
                            g_new = expand_g_new(x)

                        # Identify the constant term in the constraints
                        b = ca.vertcat(
                            cache["b"], ca.substitute(g_new, x, ca.DM.zeros(x.sparsity()))
                        )

                        # Identify the linear term in the constraints
                        A = ca.vertcat(cache["A"], ca.jacobian(g_new, x))
                else:
                    if expand:
                        expand_g = ca.Function("f", [x_mx], [g]).expand()
                        g = expand_g(x)

                    # Identify the constant term in the constraints
                    b = ca.substitute(g, x, ca.DM.zeros(x.sparsity()))

                    # Identify the linear term in the constraints
                    A = ca.jacobian(g, x)

                cache["A"] = A
                cache["b"] = b

                self._solver = ca.conic(
                    "mysolver", solver_name, {"h": H.sparsity(), "a": A.sparsity()}, options
                )
                self._solver_in = {}
                self._solver_in["h"] = ca.DM(H)
                self._solver_in["g"] = ca.DM(c)
                self._solver_in["a"] = ca.DM(A)
                self._b = ca.DM(b)

            def __call__(self, x0, lbx, ubx, lbg, ubg):
                self._solver_in["x0"] = x0
                self._solver_in["lbx"] = lbx
                self._solver_in["ubx"] = ubx
                self._solver_in["lba"] = lbg - self._b
                self._solver_in["uba"] = ubg - self._b

                solver_out = self._solver(**self._solver_in)

                solver_out["f"] = solver_out["cost"]

                return solver_out

            def stats(self):
                return self._solver.stats().copy()

        return Solver()
