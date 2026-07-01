import itertools
import logging
from collections import OrderedDict

import casadi as ca
import numpy as np

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check

from .goal_programming_mixin_base import (  # noqa: F401
    Goal,
    StateGoal,
    _EmptyEnsembleList,
    _EmptyEnsembleOrderedDict,
    _GoalConstraint,
    _GoalProgrammingMixinBase,
)
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class GoalProgrammingMixin(_GoalProgrammingMixinBase):
    """
    Adds lexicographic goal programming to your optimization problem.
    """

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        # Initialize instance variables, so that the overridden methods may be
        # called outside of the goal programming loop, for example in pre().
        self._gp_first_run = True
        self.__results_are_current = False
        self.__subproblem_epsilons = []
        self.__subproblem_objectives = []
        self.__subproblem_soft_constraints = _EmptyEnsembleList()
        self.__subproblem_parameters = []
        self.__constraint_store = _EmptyEnsembleOrderedDict()

        self.__subproblem_path_epsilons = []
        self.__subproblem_path_objectives = []
        self.__subproblem_path_soft_constraints = _EmptyEnsembleList()
        self.__subproblem_path_timeseries = []
        self.__path_constraint_store = _EmptyEnsembleOrderedDict()

        self.__original_parameter_keys = {}
        self.__original_constant_input_keys = {}

        # Lists that are only filled when 'keep_soft_constraints' is True
        self.__problem_constraints = _EmptyEnsembleList()
        self.__problem_path_constraints = _EmptyEnsembleList()
        self.__problem_epsilons = []
        self.__problem_path_epsilons = []
        self.__problem_path_timeseries = []
        self.__problem_parameters = []

    @property
    def extra_variables(self):
        return self.__problem_epsilons + self.__subproblem_epsilons

    @property
    def path_variables(self):
        return self.__problem_path_epsilons + self.__subproblem_path_epsilons

    @ensemble_bounds_check
    def bounds(self, ensemble_member: int | None = None):
        bounds = (
            super().bounds(ensemble_member) if self.ensemble_specific_bounds else super().bounds()
        )

        for epsilon in (
            self.__subproblem_epsilons
            + self.__subproblem_path_epsilons
            + self.__problem_epsilons
            + self.__problem_path_epsilons
        ):
            bounds[epsilon.name()] = (0.0, 1.0)
        return bounds

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)

        if ensemble_member not in self.__original_constant_input_keys:
            self.__original_constant_input_keys[ensemble_member] = set(constant_inputs.keys())

        # Remove min/max timeseries of previous priorities
        for k in set(constant_inputs.keys()):
            if k not in self.__original_constant_input_keys[ensemble_member]:
                del constant_inputs[k]

        n_times = len(self.times())

        # Append min/max timeseries to the constant inputs. Note that min/max
        # timeseries are shared between all ensemble members.
        for variable, value in self.__subproblem_path_timeseries + self.__problem_path_timeseries:
            if isinstance(value, np.ndarray):
                value = Timeseries(self.times(), np.broadcast_to(value, (n_times, len(value))))
            elif not isinstance(value, Timeseries):
                value = Timeseries(self.times(), np.full(n_times, value))

            constant_inputs[variable] = value
        return constant_inputs

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        if ensemble_member not in self.__original_parameter_keys:
            self.__original_parameter_keys[ensemble_member] = set(parameters.keys())

        # Remove min/max parameters of previous priorities
        for k in set(parameters.keys()):
            if k not in self.__original_parameter_keys[ensemble_member]:
                del parameters[k]

        # Append min/max values to the parameters. Note that min/max values
        # are shared between all ensemble members.
        for variable, value in self.__subproblem_parameters + self.__problem_parameters:
            parameters[variable] = value
        return parameters

    def seed(self, ensemble_member):
        if self._gp_first_run:
            seed = super().seed(ensemble_member)
        else:
            # Seed with previous results
            seed = AliasDict(self.alias_relation)
            for key, result in self.__results[ensemble_member].items():
                times = self.times(key)
                if (result.ndim == 1 and len(result) == len(times)) or (
                    result.ndim == 2 and result.shape[0] == len(times)
                ):
                    # Only include seed timeseries which are consistent
                    # with the specified time stamps.
                    seed[key] = Timeseries(times, result)
                elif (result.ndim == 1 and len(result) == 1) or (
                    result.ndim == 2 and result.shape[0] == 1
                ):
                    seed[key] = result

        # Seed epsilons of current priority
        for epsilon in self.__subproblem_epsilons:
            eps_size = epsilon.size1()
            if eps_size > 1:
                seed[epsilon.name()] = np.ones(eps_size)
            else:
                seed[epsilon.name()] = 1.0

        times = self.times()
        for epsilon in self.__subproblem_path_epsilons:
            eps_size = epsilon.size1()
            if eps_size > 1:
                seed[epsilon.name()] = Timeseries(times, np.ones((eps_size, len(times))))
            else:
                seed[epsilon.name()] = Timeseries(times, np.ones(len(times)))

        return seed

    def objective(self, ensemble_member):
        n_objectives = self._gp_n_objectives(
            self.__subproblem_objectives, self.__subproblem_path_objectives, ensemble_member
        )
        return self._gp_objective(self.__subproblem_objectives, n_objectives, ensemble_member)

    def path_objective(self, ensemble_member):
        n_objectives = self._gp_n_objectives(
            self.__subproblem_objectives, self.__subproblem_path_objectives, ensemble_member
        )
        return self._gp_path_objective(
            self.__subproblem_path_objectives, n_objectives, ensemble_member
        )

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        additional_constraints = itertools.chain(
            self.__constraint_store[ensemble_member].values(),
            self.__problem_constraints[ensemble_member],
            self.__subproblem_soft_constraints[ensemble_member],
        )

        for constraint in additional_constraints:
            constraints.append((constraint.function(self), constraint.min, constraint.max))

        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)

        additional_path_constraints = itertools.chain(
            self.__path_constraint_store[ensemble_member].values(),
            self.__problem_path_constraints[ensemble_member],
            self.__subproblem_path_soft_constraints[ensemble_member],
        )

        for constraint in additional_path_constraints:
            path_constraints.append((constraint.function(self), constraint.min, constraint.max))

        return path_constraints

    def solver_options(self):
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
        | ``violation_relaxation``  | ``float`` | ``0.0``       |
        +---------------------------+-----------+---------------+
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
        | ``interior_distance``     | ``float`` | ``1e-6``      |
        +---------------------------+-----------+---------------+
        | ``scale_by_problem_size`` | ``bool``  | ``False``     |
        +---------------------------+-----------+---------------+
        | ``keep_soft_constraints`` | ``bool``  | ``False``     |
        +---------------------------+-----------+---------------+

        Before turning a soft constraint of the goal programming algorithm into a hard constraint,
        the violation variable (also known as epsilon) of each goal is relaxed with the
        ``violation_relaxation``. Use of this option is normally not required.

        When turning a soft constraint of the goal programming algorithm into a hard constraint,
        the constraint is relaxed with ``constraint_relaxation``. Use of this option is
        normally not required. Note that:

        1. Minimization goals do not get ``constraint_relaxation`` applied when
           ``fix_minimized_values`` is True.

        2. Because of the constraints it generates, when ``keep_soft_constraints`` is True, the
            option ``fix_minimized_values`` needs to be set to False for the
            ``constraint_relaxation`` to be applied at all.

        A goal is considered to be violated if the violation, scaled between 0 and 1, is greater
        than the specified tolerance. Violated goals are fixed.  Use of this option is normally not
        required.

        When using the default solver (IPOPT), its barrier parameter ``mu`` is
        normally re-initialized at every iteration of the goal programming
        algorithm, unless mu_reinit is set to ``False``.  Use of this option
        is normally not required.

        If ``fix_minimized_values`` is set to ``True``, goal functions will be set to equal their
        optimized values in optimization problems generated during subsequent priorities. Otherwise,
        only an upper bound will be set. Use of this option is normally not required.
        Note that a non-zero goal relaxation overrules this option; a non-zero relaxation will
        always result in only an upper bound being set.
        Also note that the use of this option may add non-convex constraints to the optimization
        problem.
        The default value for this parameter is ``True`` for the default solvers IPOPT/BONMIN. If
        any other solver is used, the default value is ``False``.

        If ``check_monotonicity`` is set to ``True``, then it will be checked whether goals with
        the same function key form a monotonically decreasing sequence with regards to the target
        interval.

        The option ``equality_threshold`` controls when a two-sided inequality constraint is folded
        into an equality constraint.

        The option ``interior_distance`` controls the distance from the scaled target bounds,
        starting from which the function value is considered to lie in the interior of the target
        space.

        If ``scale_by_problem_size`` is set to ``True``, the objective (i.e. the sum of the
        violation variables) will be divided by the number of goals, and the path objective will
        be divided by the number of path goals and the number of active time steps (per goal).
        This will make sure the objectives are always in the range [0, 1], at the cost of solving
        each goal/time step less accurately.

        The option ``keep_soft_constraints`` controls how the epsilon variables introduced in the
        target goals are dealt with in subsequent priorities.
        If ``keep_soft_constraints`` is set to False, each epsilon is replaced by its computed
        value and those are used to derive a new set of constraints.
        If ``keep_soft_constraints`` is set to True, the epsilons are kept as variables and the
        constraints are not modified. To ensure the goal programming philosophy, i.e., Pareto
        optimality, a single constraint is added to enforce that the objective function must
        always be at most the objective value. This method allows for a larger solution space, at
        the cost of having a (possibly) more complex optimization problem. Indeed, more variables
        are kept around throughout the optimization and any objective function is turned into a
        constraint for the subsequent priorities (while in the False option this was the case only
        for the function of minimization goals).

        :returns: A dictionary of goal programming options.
        """

        options = {}

        options["mu_reinit"] = True
        options["violation_relaxation"] = 0.0  # Disable by default
        options["constraint_relaxation"] = 0.0  # Disable by default
        options["violation_tolerance"] = np.inf  # Disable by default
        options["fix_minimized_values"] = False
        options["check_monotonicity"] = True
        options["equality_threshold"] = 1e-8
        options["interior_distance"] = 1e-6
        options["scale_by_problem_size"] = False
        options["keep_soft_constraints"] = False

        # Define temporary variable to avoid infinite loop between
        # solver_options and goal_programming_options.
        self._loop_breaker_goal_programming_options = True

        if not hasattr(self, "_loop_breaker_solver_options"):
            if self.solver_options()["solver"] in {"ipopt", "bonmin"}:
                options["fix_minimized_values"] = True

        delattr(self, "_loop_breaker_goal_programming_options")

        return options

    def __goal_hard_constraint(
        self, goal, epsilon, existing_constraint, ensemble_member, options, is_path_goal
    ):
        if not is_path_goal:
            epsilon = epsilon[:1]

        goal_m, goal_M = self._gp_min_max_arrays(goal, target_shape=epsilon.shape[0])

        if goal.has_target_bounds:
            # We use a violation variable formulation, with the violation
            # variables epsilon bounded between 0 and 1.
            m, M = (
                np.full_like(epsilon, -np.inf, dtype=np.float64),
                np.full_like(epsilon, np.inf, dtype=np.float64),
            )

            # A function range does not have to be specified for critical
            # goals. Avoid multiplying with NaN in that case.
            if goal.has_target_min:
                m = (
                    epsilon * ((goal.function_range[0] - goal_m) if not goal.critical else 0.0)
                    + goal_m
                    - goal.relaxation
                ) / goal.function_nominal
            if goal.has_target_max:
                M = (
                    epsilon * ((goal.function_range[1] - goal_M) if not goal.critical else 0.0)
                    + goal_M
                    + goal.relaxation
                ) / goal.function_nominal

            if goal.has_target_min and goal.has_target_max:
                # Avoid comparing with NaN
                inds = ~(np.isnan(m) | np.isnan(M))
                inds[inds] &= np.abs(m[inds] - M[inds]) < options["equality_threshold"]
                if np.any(inds):
                    avg = 0.5 * (m + M)
                    m[inds] = M[inds] = avg[inds]

            m[~np.isfinite(goal_m)] = -np.inf
            M[~np.isfinite(goal_M)] = np.inf

            inds = epsilon > options["violation_tolerance"]
            if np.any(inds):
                if is_path_goal:
                    expr = self.map_path_expression(
                        goal.function(self, ensemble_member), ensemble_member
                    )
                else:
                    expr = goal.function(self, ensemble_member)

                function = ca.Function("f", [self.solver_input], [expr])
                value = np.array(function(self.solver_output))

                m[inds] = (value - goal.relaxation) / goal.function_nominal
                M[inds] = (value + goal.relaxation) / goal.function_nominal

            m -= options["constraint_relaxation"]
            M += options["constraint_relaxation"]
        else:
            # Epsilon encodes the position within the function range.
            if options["fix_minimized_values"] and goal.relaxation == 0.0:
                m = epsilon / goal.function_nominal
                M = epsilon / goal.function_nominal
                self.check_collocation_linearity = False
                self.linear_collocation = False
            else:
                m = -np.inf * np.ones(epsilon.shape)
                M = (epsilon + goal.relaxation) / goal.function_nominal + options[
                    "constraint_relaxation"
                ]

        if is_path_goal:
            m = Timeseries(self.times(), m)
            M = Timeseries(self.times(), M)
        else:
            m = m[0]
            M = M[0]

        constraint = _GoalConstraint(
            goal,
            lambda problem, ensemble_member=ensemble_member, goal=goal: (
                goal.function(problem, ensemble_member) / goal.function_nominal
            ),
            m,
            M,
            True,
        )

        # Epsilon is fixed. Override previous {min,max} constraints for this
        # state.
        if existing_constraint:
            constraint.update_bounds(existing_constraint, enforce="other")

        return constraint

    def __soft_to_hard_constraints(self, goals, sym_index, is_path_goal):
        if is_path_goal:
            constraint_store = self.__path_constraint_store
        else:
            constraint_store = self.__constraint_store

        times = self.times()
        options = self.goal_programming_options()

        eps_format = "eps_{}_{}"
        if is_path_goal:
            eps_format = "path_" + eps_format

        # Handle function evaluation in a grouped manner to save time with
        # the call map_path_expression(). Repeated calls will make
        # repeated CasADi Function objects, which can be slow.
        goal_function_values = [None] * self.ensemble_size

        for ensemble_member in range(self.ensemble_size):
            goal_functions = OrderedDict()

            for j, goal in enumerate(goals):
                if (
                    not goal.has_target_bounds
                    or goal.violation_timeseries_id is not None
                    or goal.function_value_timeseries_id is not None
                ):
                    goal_functions[j] = goal.function(self, ensemble_member)

            if is_path_goal:
                expr = self.map_path_expression(
                    ca.vertcat(*goal_functions.values()), ensemble_member
                )
            else:
                expr = ca.transpose(ca.vertcat(*goal_functions.values()))

            f = ca.Function("f", [self.solver_input], [expr])
            raw_function_values = np.array(f(self.solver_output))
            goal_function_values[ensemble_member] = {
                k: raw_function_values[:, j].ravel() for j, k in enumerate(goal_functions.keys())
            }

        # Re-add constraints, this time with epsilon values fixed
        for ensemble_member in range(self.ensemble_size):
            for j, goal in enumerate(goals):
                if j in goal_function_values[ensemble_member]:
                    function_value = goal_function_values[ensemble_member][j]

                    # Store results
                    if goal.function_value_timeseries_id is not None:
                        self.set_timeseries(
                            goal.function_value_timeseries_id,
                            Timeseries(times, function_value),
                            ensemble_member,
                        )

                if goal.critical:
                    continue

                if goal.has_target_bounds:
                    epsilon = self.__results[ensemble_member][eps_format.format(sym_index, j)]

                    # Store results
                    if goal.violation_timeseries_id is not None:
                        function_value = goal_function_values[ensemble_member][j]
                        epsilon_active = np.copy(epsilon)
                        m = goal.target_min
                        if isinstance(m, Timeseries):
                            m = self.interpolate(
                                times, goal.target_min.times, goal.target_min.values
                            )
                        M = goal.target_max
                        if isinstance(M, Timeseries):
                            M = self.interpolate(
                                times, goal.target_max.times, goal.target_max.values
                            )
                        w = np.ones_like(function_value)
                        if goal.has_target_min:
                            # Avoid comparing with NaN while making sure that
                            # w[i] is True when m[i] is not finite.
                            m = np.array(m)
                            m[~np.isfinite(m)] = -np.inf
                            w = np.logical_and(
                                w,
                                (
                                    function_value / goal.function_nominal
                                    > m / goal.function_nominal + options["interior_distance"]
                                ),
                            )
                        if goal.has_target_max:
                            # Avoid comparing with NaN while making sure that
                            # w[i] is True when M[i] is not finite.
                            M = np.array(M)
                            M[~np.isfinite(M)] = np.inf
                            w = np.logical_and(
                                w,
                                (
                                    function_value / goal.function_nominal
                                    < M / goal.function_nominal + options["interior_distance"]
                                ),
                            )
                        epsilon_active[w] = np.nan
                        self.set_timeseries(
                            goal.violation_timeseries_id,
                            Timeseries(times, epsilon_active),
                            ensemble_member,
                        )

                    # Add a relaxation to appease the barrier method.
                    epsilon += options["violation_relaxation"]
                else:
                    epsilon = function_value

                fk = goal.get_function_key(self, ensemble_member)
                existing_constraint = constraint_store[ensemble_member].get(fk, None)

                constraint_store[ensemble_member][fk] = self.__goal_hard_constraint(
                    goal, epsilon, existing_constraint, ensemble_member, options, is_path_goal
                )

    def __add_subproblem_objective_constraint(self):
        # We want to keep the additional variables/parameters we set around
        self.__problem_epsilons.extend(self.__subproblem_epsilons)
        self.__problem_path_epsilons.extend(self.__subproblem_path_epsilons)
        self.__problem_path_timeseries.extend(self.__subproblem_path_timeseries)
        self.__problem_parameters.extend(self.__subproblem_parameters)

        for ensemble_member in range(self.ensemble_size):
            self.__problem_constraints[ensemble_member].extend(
                self.__subproblem_soft_constraints[ensemble_member]
            )
            self.__problem_path_constraints[ensemble_member].extend(
                self.__subproblem_path_soft_constraints[ensemble_member]
            )

        # Extract information about the objective value, this is used for the Pareto optimality
        # constraint. We only retain information about the objective functions defined through the
        # goal framework as user define objective functions may relay on local variables.
        subproblem_objectives = self.__subproblem_objectives.copy()
        subproblem_path_objectives = self.__subproblem_path_objectives.copy()

        def _constraint_func(
            problem,
            subproblem_objectives=subproblem_objectives,
            subproblem_path_objectives=subproblem_path_objectives,
        ):
            val = 0.0
            for ensemble_member in range(problem.ensemble_size):
                # NOTE: Users might be overriding objective() and/or path_objective(). Use the
                # private methods that work only on the goals.
                n_objectives = problem._gp_n_objectives(
                    subproblem_objectives, subproblem_path_objectives, ensemble_member
                )
                expr = problem._gp_objective(subproblem_objectives, n_objectives, ensemble_member)
                expr += ca.sum1(
                    problem.map_path_expression(
                        problem._gp_path_objective(
                            subproblem_path_objectives, n_objectives, ensemble_member
                        ),
                        ensemble_member,
                    )
                )
                val += problem.ensemble_member_probability(ensemble_member) * expr

            return val

        f = ca.Function("tmp", [self.solver_input], [_constraint_func(self)])
        obj_val = float(f(self.solver_output))

        options = self.goal_programming_options()

        if options["fix_minimized_values"]:
            constraint = _GoalConstraint(None, _constraint_func, obj_val, obj_val, True)
            self.check_collocation_linearity = False
            self.linear_collocation = False
        else:
            obj_val += options["constraint_relaxation"]
            constraint = _GoalConstraint(None, _constraint_func, -np.inf, obj_val, True)

        # The goal works over all ensemble members, so we add it to the last
        # one, as at that point the inputs of all previous ensemble members
        # will have been discretized, mapped and stored.
        self.__problem_constraints[-1].append(constraint)

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do pre-processing
        if preprocessing:
            self.pre()

        # Group goals into subproblems
        subproblems = []
        goals = self.goals()
        path_goals = self.path_goals()

        options = self.goal_programming_options()

        # Validate (in)compatible options
        if options["keep_soft_constraints"] and options["violation_relaxation"]:
            raise Exception(
                "The option 'violation_relaxation' cannot be used "
                "when 'keep_soft_constraints' is set."
            )

        # Validate goal definitions
        self._gp_validate_goals(goals, is_path_goal=False)
        self._gp_validate_goals(path_goals, is_path_goal=True)

        priorities = {
            int(goal.priority) for goal in itertools.chain(goals, path_goals) if not goal.is_empty
        }

        for priority in sorted(priorities):
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
        self.skip_priority = False

        self.__constraint_store = [OrderedDict() for ensemble_member in range(self.ensemble_size)]
        self.__path_constraint_store = [
            OrderedDict() for ensemble_member in range(self.ensemble_size)
        ]

        # Lists for when `keep_soft_constraints` is True
        self.__problem_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__problem_epsilons = []
        self.__problem_parameters = []
        self.__problem_path_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__problem_path_epsilons = []
        self.__problem_path_timeseries = []

        self._gp_first_run = True
        self.__results_are_current = False
        self.__original_constant_input_keys = {}
        self.__original_parameter_keys = {}
        for i, (priority, goals, path_goals) in enumerate(subproblems):
            logger.info(f"Solving goals at priority {priority}")

            # Call the pre priority hook
            self.priority_started(priority)

            if self.skip_priority:
                logger.info(
                    f"priority {priority} was removed in priority_started. No optimization problem "
                    "is solved at this priority."
                )
                continue

            (
                self.__subproblem_epsilons,
                self.__subproblem_objectives,
                self.__subproblem_soft_constraints,
                hard_constraints,
                self.__subproblem_parameters,
            ) = self._gp_goal_constraints(goals, i, options, is_path_goal=False)

            (
                self.__subproblem_path_epsilons,
                self.__subproblem_path_objectives,
                self.__subproblem_path_soft_constraints,
                path_hard_constraints,
                self.__subproblem_path_timeseries,
            ) = self._gp_goal_constraints(path_goals, i, options, is_path_goal=True)

            # Put hard constraints in the constraint stores
            self._gp_update_constraint_store(self.__constraint_store, hard_constraints)
            self._gp_update_constraint_store(self.__path_constraint_store, path_hard_constraints)

            # Solve subproblem
            success = super().optimize(
                preprocessing=False,
                postprocessing=False,
                log_solver_failure_as_error=log_solver_failure_as_error,
            )
            if not success:
                break

            self._gp_first_run = False

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

            if options["keep_soft_constraints"]:
                self.__add_subproblem_objective_constraint()
            else:
                self.__soft_to_hard_constraints(goals, i, is_path_goal=False)
                self.__soft_to_hard_constraints(path_goals, i, is_path_goal=True)

        logger.info("Done goal programming")

        # Do post-processing
        if postprocessing:
            self.post()

        # Done
        return success

    def extract_results(self, ensemble_member=0):
        if self.__results_are_current:
            logger.debug("Returning cached results")
            return self.__results[ensemble_member]

        # If self.__results is not up to date, do the super().extract_results
        # method
        return super().extract_results(ensemble_member)
