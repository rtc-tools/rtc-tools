import functools
import logging
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Callable

import casadi as ca
import numpy as np

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class _EmptyEnsembleList(list):
    """
    An indexable object containing infinitely many empty lists.
    Only to be used as a placeholder.
    """

    def __getitem__(self, key):
        return []


class _EmptyEnsembleOrderedDict(OrderedDict):
    """
    An indexable object containing infinitely many empty OrderedDicts.
    Only to be used as a placeholder.
    """

    def __getitem__(self, key):
        return OrderedDict()


class Goal(metaclass=ABCMeta):
    r"""
    Base class for lexicographic goal programming goals.

    **Types of goals**

    There are 2 types of goals: minimization goals and target goals.

    *Minimization goals*

    Minimization goals are of the form:

    .. math::
        \text{minimize } f(x).

    *Target goals*

    Target goals weakly enforce a constraint of the form

    .. math::
        m_{target} \leq g(x) \leq M_{target},

    by turning it into a minimization problem of the form

    .. math::
        \text{minimize } & \epsilon^r, \\
        \text{subject to } &g_{low}(\epsilon) \leq g(x) \leq g_{up}(\epsilon), \\
        \text{and } &0 \leq \epsilon \leq 1,

    where

    .. math::
        g_{low}(\epsilon) &:= (1-\epsilon) m_{target} + \epsilon m, \\
        g_{up}(\epsilon) &:= (1-\epsilon) M_{target} + \epsilon M.

    Here, :math:`m` and :math:`M` are hard constraints for :math:`g(x)`,
    :math:`m_{target}` and :math:`M_{target}` are target bounds for :math:`g(x)`,
    :math:`\epsilon` is an auxiliary variable
    that indicates how strongly the target bounds are violated,
    and :math:`\epsilon^r` is a function that indicates the variation of :math:`\epsilon`,
    where the order :math:`r` is by default :math:`2`.
    We have

    .. math::
        m < m_{target} \leq M_{target} < M.

    Note that when :math:`\epsilon=0`,
    the constraint on :math:`g(x)` becomes

    .. math::
        m_{target} \leq g(x) \leq M_{target}

    and if :math:`\epsilon=1`, it becomes

    .. math::
        m \leq g(x) \leq M.


    **Scaling goals**

    Goals can be scaled by a nominal value :math:`c_{nom}`
    to improve the performance of the solvers.
    In case of a minimization goal, the scaled problem is given by

    .. math::
        \text{minimize } \hat{f}(x),

    where :math:`\hat{f}(x) := f(x) / c_{nom}`.
    In case of a target goal, the scaled problem is given by

    .. math::
        \text{minimize } & \epsilon^r, \\
        \text{subject to } &\hat{g}_{low}(\epsilon) \leq \hat{g}(x) \leq \hat{g}_{up}(\epsilon), \\
        \text{and } &0 \leq \epsilon \leq 1,

    where :math:`\hat{g}(x) := g(x) / c_{nom}`,
    :math:`\hat{g}_{low}(\epsilon) := {g}_{low}(\epsilon) / c_{nom}`,
    and :math:`\hat{g}_{up}(\epsilon) := {g}_{up}(\epsilon) / c_{nom}`.


    **Implementing goals**

    A goal class is created by inheriting from the :py:class:Goal class and
    overriding the :func:`function` method.
    This method defines the goal function :math:`f(x)` in case of a minimization goal,
    and the goal function :math:`g(x)` in case of a target goal.
    A goal becomes a target goal
    if either the class attribute ``target_min`` or ``target_max`` is set.

    To further define a goal, the following class attributes can also be set.

    :cvar function_range:   Range of goal function :math:`[m ,M]`.
                            Only applies to target goals.
                            Required for a target goal.
    :cvar function_nominal: Nominal value of a function :math:`c_{nom}`.
                            Used for scaling. Default is ``1``.
    :cvar target_min:       Desired lower bound for goal function :math:`m_{target}`.
                            Default is ``numpy.nan``.
    :cvar target_max:       Desired upper bound for goal function :math:`M_{target}`.
                            Default is ``numpy.nan``.
    :cvar priority:         Priority of a goal. Default is ``1``.
    :cvar weight:           Optional weighting applied to the goal. Default is ``1.0``.
    :cvar order:            Penalization order of goal violation :math:`r`. Default is ``2``.
    :cvar critical:         If ``True``, the algorithm will abort if this goal cannot be fully met.
                            Default is ``False``.
    :cvar relaxation:       Amount of slack added to the hard constraints related to the goal.
                            Must be a nonnegative value.
                            The unit is equal to that of the goal function.
                            Default is ``0.0``.

    When ``target_min`` is set, but not ``target_max``,
    the target goal becomes a lower bound target goal
    and the constraint on :math:`g(x)` becomes

    .. math::
        g_{low}(\epsilon) \leq g(x).

    Similary, if ``target_max`` is set, but not ``target_min``,
    the target goal becomes a upper bound target goal
    and the constraint on :math:`g(x)` becomes

    .. math::
        g(x) \leq g_{up}(\epsilon).

    Relaxation is used to loosen the constraints that are set
    after the optimization of the goal's priority.

    Notes:
        *   If one is unsure about the function range,
            it is recommended to overestimate this interval.
            However, this will negatively influence how accurately the target bounds are met.
        *   The function range should be strictly larger than the target range.
            In particular, :math:`m < m_{target}` and :math:`M_{target} < M`.
        *   In a path goal, the target can be a Timeseries.
        *   In case of multiple goals with the same priority,
            it is crucial that an accurate function nominal value is provided.
            This ensures that all goals are given similar importance.

    A goal can be written in vector form. In a vector goal:
        * The goal size determines how many goals there are.
        * The goal function has shape ``(goal size, 1)``.
        * The function is either minimized or has, possibly various, targets.
        * Function nominal can either be an array with as many entries as the goal size or have a
          single value.
        * Function ranges can either be an array with as many entries as the goal size or have a
          single value.
        * In a goal, the target can either be an array with as many entries as the goal size or
          have a single value.
        * In a path goal, the target can also be a Timeseries whose values are either a
          1-dimensional vector or have as many columns as the goal size.

    **Examples**

    Example definition of the point goal :math:`x(t) \geq 1.1` for :math:`t=1.0` at priority 1::

        class MyGoal(Goal):
            def function(self, optimization_problem, ensemble_member):
                # State 'x' at time t = 1.0
                t = 1.0
                return optimization_problem.state_at('x', t, ensemble_member)

            function_range = (1.0, 2.0)
            target_min = 1.1
            priority = 1

    Example definition of the path goal :math:`x(t) \geq 1.1` for all :math:`t` at priority 2::

        class MyPathGoal(Goal):
            def function(self, optimization_problem, ensemble_member):
                # State 'x' at any point in time
                return optimization_problem.state('x')

            function_range = (1.0, 2.0)
            target_min = 1.1
            priority = 2

    **Note path goals**

    Note that for path goals, the ensemble member index is not passed to the call
    to :func:`OptimizationProblem.state`.  This call returns a time-independent symbol
    that is also independent of the active ensemble member.  Path goals are
    applied to all times and all ensemble members simultaneously.

    """

    @abstractmethod
    def function(self, optimization_problem: OptimizationProblem, ensemble_member: int) -> ca.MX:
        """
        This method returns a CasADi :class:`MX` object describing the goal function.

        :returns: A CasADi :class:`MX` object.
        """
        pass

    #: Range of goal function
    function_range = (np.nan, np.nan)

    #: Nominal value of function (used for scaling)
    function_nominal = 1.0

    #: Desired lower bound for goal function
    target_min = np.nan

    #: Desired upper bound for goal function
    target_max = np.nan

    #: Lower priority goals take precedence over higher priority goals.
    priority = 1

    #: Goals with the same priority are weighted off against each other in a
    #: single objective function.
    weight = 1.0

    #: The goal violation value is taken to the order'th power in the objective
    #: function.
    order = 2

    #: The size of the goal if it's a vector goal.
    size = 1

    #: Critical goals must always be fully satisfied.
    critical = False

    #: Absolute relaxation applied to the optimized values of this goal
    relaxation = 0.0

    #: Timeseries ID for function value data (optional)
    function_value_timeseries_id = None

    #: Timeseries ID for goal violation data (optional)
    violation_timeseries_id = None

    @property
    def has_target_min(self) -> bool:
        """
        ``True`` if the user goal has min bounds.
        """
        if isinstance(self.target_min, Timeseries):
            return True
        else:
            return np.any(np.isfinite(self.target_min))

    @property
    def has_target_max(self) -> bool:
        """
        ``True`` if the user goal has max bounds.
        """
        if isinstance(self.target_max, Timeseries):
            return True
        else:
            return np.any(np.isfinite(self.target_max))

    @property
    def has_target_bounds(self) -> bool:
        """
        ``True`` if the user goal has min/max bounds.
        """
        return self.has_target_min or self.has_target_max

    @property
    def is_empty(self) -> bool:
        target_min_set = isinstance(self.target_min, Timeseries) or np.any(
            np.isfinite(self.target_min)
        )
        target_max_set = isinstance(self.target_max, Timeseries) or np.any(
            np.isfinite(self.target_max)
        )

        if not target_min_set and not target_max_set:
            # A minimization goal
            return False

        target_min = self.target_min
        if isinstance(target_min, Timeseries):
            target_min = target_min.values

        target_max = self.target_max
        if isinstance(target_max, Timeseries):
            target_max = target_max.values

        min_empty = not np.any(np.isfinite(target_min))
        max_empty = not np.any(np.isfinite(target_max))

        return min_empty and max_empty

    def get_function_key(
        self, optimization_problem: OptimizationProblem, ensemble_member: int
    ) -> str:
        """
        Returns a key string uniquely identifying the goal function.  This
        is used to eliminate linearly dependent constraints from the optimization problem.
        """
        if hasattr(self, "function_key"):
            return self.function_key

        # This must be deterministic.  See RTCTOOLS-485.
        if not hasattr(Goal, "_function_key_counter"):
            Goal._function_key_counter = 0
        self.function_key = f"{self.__class__.__name__}_{Goal._function_key_counter}"
        Goal._function_key_counter += 1

        return self.function_key

    def __repr__(self) -> str:
        return (
            f"{self.__class__}(priority={self.priority}, target_min={self.target_min}, "
            f"target_max={self.target_max}, function_range={self.function_range})"
        )


class StateGoal(Goal):
    r"""
    Base class for lexicographic goal programming path goals that act on a single model state.

    A state goal is defined by setting at least the ``state`` class variable.

    :cvar state:            State on which the goal acts.  *Required*.
    :cvar target_min:       Desired lower bound for goal function.  Default is ``numpy.nan``.
    :cvar target_max:       Desired upper bound for goal function.  Default is ``numpy.nan``.
    :cvar priority:         Integer priority of goal.  Default is ``1``.
    :cvar weight:           Optional weighting applied to the goal.  Default is ``1.0``.
    :cvar order:            Penalization order of goal violation.  Default is ``2``.
    :cvar critical:         If ``True``, the algorithm will abort if this goal cannot be fully met.
                            Default is ``False``.

    Example definition of the goal :math:`x(t) \geq 1.1` for all :math:`t` at priority 2::

        class MyStateGoal(StateGoal):
            state = 'x'
            target_min = 1.1
            priority = 2

    Contrary to ordinary ``Goal`` objects, ``PathGoal`` objects need to be initialized with an
    ``OptimizationProblem`` instance to allow extraction of state metadata, such as bounds and
    nominal values.  Consequently, state goals must be instantiated as follows::

        my_state_goal = MyStateGoal(optimization_problem)

    Note that ``StateGoal`` is a helper class.  State goals can also be defined using ``Goal`` as
    direct base class, by implementing the ``function`` method and providing the
    ``function_range`` and ``function_nominal`` class variables manually.

    """

    #: The state on which the goal acts.
    state = None

    def __init__(self, optimization_problem):
        """
        Initialize the state goal object.

        :param optimization_problem: ``OptimizationProblem`` instance.
        """

        # Check whether a state has been specified
        if self.state is None:
            raise Exception("Please specify a state.")

        # Extract state range from model
        if self.has_target_bounds:
            try:
                if optimization_problem.ensemble_specific_bounds:
                    bounds = optimization_problem.bounds(0)
                    bounds_state_ref = bounds[self.state]
                    if np.array_equal(self.function_range, (np.nan, np.nan), equal_nan=True):
                        # If the user has not set the function range themselves, we
                        # try and set it automatically. This is only possible if the
                        # bounds are the same for all ensemble members.
                        for ensemble_member in range(optimization_problem.ensemble_size):
                            bounds_state_ensemble = optimization_problem.bounds(ensemble_member)[
                                self.state
                            ]
                            # First, check if the types are equal, and then check if the values are
                            # equal. For Timeseries and floats, we can do `==` comparison, for
                            # arrays we need to use np.all. To simplify we wrap the `==` for floats
                            # in an `np.all` as well.
                            if type(bounds_state_ref) is not type(
                                bounds_state_ensemble
                            ) or not np.all(bounds_state_ref == bounds_state_ensemble):
                                raise ValueError(
                                    f"Bounds for state {self.state} are not the same for all "
                                    f"ensemble members; please set the function_range explicitly"
                                )
                else:
                    bounds = optimization_problem.bounds()
                self.function_range = bounds[self.state]
            except KeyError:
                raise Exception(f"State {self.state} has no bounds or does not exist in the model.")

            if self.function_range[0] is None:
                raise Exception(f"Please provide a lower bound for state {self.state}.")
            if self.function_range[1] is None:
                raise Exception(f"Please provide an upper bound for state {self.state}.")

        # Extract state nominal from model
        self.function_nominal = optimization_problem.variable_nominal(self.state)

        # Set function key
        canonical, sign = optimization_problem.alias_relation.canonical_signed(self.state)
        self.function_key = canonical if sign > 0.0 else "-" + canonical

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)

    def __repr__(self):
        return (
            f"{self.__class__}(priority={self.priority}, state={self.state}, "
            f"target_min={self.target_min}, target_max={self.target_max}, "
            f"function_range={self.function_range})"
        )


class _GoalConstraint:
    def __init__(
        self,
        goal: Goal,
        function: Callable[[OptimizationProblem], ca.MX],
        m: float | np.ndarray | Timeseries,
        M: float | np.ndarray | Timeseries,
        optimized: bool,
    ):
        assert isinstance(m, (float, np.ndarray, Timeseries))
        assert isinstance(M, (float, np.ndarray, Timeseries))
        assert type(m) is type(M)

        # NumPy arrays only allowed for vector goals
        if isinstance(m, np.ndarray):
            assert len(m) == goal.size
            assert len(M) == goal.size

        self.goal = goal
        self.function = function
        self.min = m
        self.max = M
        self.optimized = optimized

    def update_bounds(self, other, enforce="self"):
        # NOTE: a.update_bounds(b) is _not_ the same as  b.update_bounds(a).
        # See how the 'enforce' parameter is used.

        min_, max_ = self.min, self.max
        other_min, other_max = other.min, other.max

        if isinstance(min_, Timeseries):
            assert isinstance(max_, Timeseries)
            assert isinstance(other_min, Timeseries)
            assert isinstance(other_max, Timeseries)

            min_ = min_.values
            max_ = max_.values
            other_min = other_min.values
            other_max = other_max.values

        min_ = np.maximum(min_, other_min)
        max_ = np.minimum(max_, other_max)

        # Ensure new constraint bounds do not loosen or shift
        # previous bounds due to numerical errors.
        if enforce == "self":
            min_ = np.minimum(max_, other_min)
            max_ = np.maximum(min_, other_max)
        else:
            min_ = np.minimum(min_, other_max)
            max_ = np.maximum(max_, other_min)

        # Ensure consistency of bounds. Bounds may become inconsistent due to
        # small numerical computation errors.
        min_ = np.minimum(min_, max_)

        if isinstance(self.min, Timeseries):
            self.min = Timeseries(self.min.times, min_)
            self.max = Timeseries(self.max.times, max_)
        else:
            self.min = min_
            self.max = max_


class _GoalProgrammingMixinBase(OptimizationProblem, metaclass=ABCMeta):
    def _gp_n_objectives(self, subproblem_objectives, subproblem_path_objectives, ensemble_member):
        return (
            ca.vertcat(*[o(self, ensemble_member) for o in subproblem_objectives]).size1()
            + ca.vertcat(*[o(self, ensemble_member) for o in subproblem_path_objectives]).size1()
        )

    def _gp_objective(self, subproblem_objectives, n_objectives, ensemble_member):
        if len(subproblem_objectives) > 0:
            acc_objective = ca.sum1(
                ca.vertcat(*[o(self, ensemble_member) for o in subproblem_objectives])
            )

            if self.goal_programming_options()["scale_by_problem_size"]:
                acc_objective = acc_objective / n_objectives

            return acc_objective
        else:
            return ca.MX(0)

    def _gp_path_objective(self, subproblem_path_objectives, n_objectives, ensemble_member):
        if len(subproblem_path_objectives) > 0:
            acc_objective = ca.sum1(
                ca.vertcat(*[o(self, ensemble_member) for o in subproblem_path_objectives])
            )

            if self.goal_programming_options()["scale_by_problem_size"]:
                # Objective is already divided by number of active time steps
                # at this point when `scale_by_problem_size` is set.
                acc_objective = acc_objective / n_objectives

            return acc_objective
        else:
            return ca.MX(0)

    @abstractmethod
    def goal_programming_options(self) -> dict[str, float | bool]:
        raise NotImplementedError()

    def goals(self) -> list[Goal]:
        """
        User problem returns list of :class:`Goal` objects.

        :returns: A list of goals.
        """
        return []

    def path_goals(self) -> list[Goal]:
        """
        User problem returns list of path :class:`Goal` objects.

        :returns: A list of path goals.
        """
        return []

    def _gp_min_max_arrays(self, g, target_shape=None):
        """
        Broadcasts the goal target minimum and target maximum to arrays of a desired target shape.

        Depending on whether g is a vector goal or not, the output shape differs:

        - A 2-D array of size (goal.size, target_shape or 1) if the goal size
          is larger than one, i.e. a vector goal
        - A 1-D array of size (target_shape or 1, ) otherwise
        """

        times = self.times()

        m, M = None, None
        if isinstance(g.target_min, Timeseries):
            m = self.interpolate(times, g.target_min.times, g.target_min.values, -np.inf, -np.inf)
            if m.ndim > 1:
                m = m.transpose()
        elif isinstance(g.target_min, np.ndarray) and target_shape:
            m = np.broadcast_to(g.target_min, (target_shape, g.size)).transpose()
        elif target_shape:
            m = np.full(target_shape, g.target_min)
        else:
            m = np.array([g.target_min]).transpose()
        if isinstance(g.target_max, Timeseries):
            M = self.interpolate(times, g.target_max.times, g.target_max.values, np.inf, np.inf)
            if M.ndim > 1:
                M = M.transpose()
        elif isinstance(g.target_max, np.ndarray) and target_shape:
            M = np.broadcast_to(g.target_max, (target_shape, g.size)).transpose()
        elif target_shape:
            M = np.full(target_shape, g.target_max)
        else:
            M = np.array([g.target_max]).transpose()

        if g.size > 1 and m.ndim == 1:
            m = np.broadcast_to(m, (g.size, len(m)))
        if g.size > 1 and M.ndim == 1:
            M = np.broadcast_to(M, (g.size, len(M)))

        if g.size > 1:
            assert m.shape == (g.size, 1 if target_shape is None else target_shape)
        else:
            assert m.shape == (1 if target_shape is None else target_shape,)
        assert m.shape == M.shape

        return m, M

    def _gp_validate_goals(self, goals, is_path_goal):
        goals = sorted(goals, key=lambda x: x.priority)

        options = self.goal_programming_options()

        # Validate goal definitions
        for goal in goals:
            m, M = goal.function_range

            # The function range should not be a symbolic expression
            if isinstance(m, ca.MX):
                assert m.is_constant()
                if m.size1() == 1:
                    m = float(m)
                else:
                    m = np.array(m.to_DM())

            if isinstance(M, ca.MX):
                assert M.is_constant()
                if M.size1() == 1:
                    M = float(M)
                else:
                    M = np.array(M.to_DM())

            assert isinstance(m, (float, int, np.ndarray))
            assert isinstance(M, (float, int, np.ndarray))

            if np.any(goal.function_nominal <= 0):
                raise Exception(f"Nonpositive nominal value specified for goal {goal}")

            if goal.critical and not goal.has_target_bounds:
                raise Exception("Minimization goals cannot be critical")

            if goal.critical:
                # Allow a function range for backwards compatibility reasons.
                # Maybe raise a warning that its not actually used?
                pass
            elif goal.has_target_bounds:
                if not np.all(np.isfinite(m)) or not np.all(np.isfinite(M)):
                    raise Exception(f"No function range specified for goal {goal}")

                if np.any(m >= M):
                    raise Exception(f"Invalid function range for goal {goal}")

                if goal.weight <= 0:
                    raise Exception(f"Goal weight should be positive for goal {goal}")
            else:
                if goal.function_range != (np.nan, np.nan):
                    raise Exception(f"Specifying function range not allowed for goal {goal}")

            if not is_path_goal:
                if isinstance(goal.target_min, Timeseries):
                    raise Exception(f"Target min cannot be a Timeseries for goal {goal}")
                if isinstance(goal.target_max, Timeseries):
                    raise Exception(f"Target max cannot be a Timeseries for goal {goal}")

            try:
                int(goal.priority)
            except ValueError:
                raise Exception(f"Priority of not int or castable to int for goal {goal}")

            if options["keep_soft_constraints"]:
                if goal.relaxation != 0.0:
                    raise Exception(
                        f"Relaxation not allowed with `keep_soft_constraints` for goal {goal}"
                    )
                if goal.violation_timeseries_id is not None:
                    raise Exception(
                        "Violation timeseries id not allowed with "
                        f"`keep_soft_constraints` for goal {goal}"
                    )
            else:
                if goal.size > 1:
                    raise Exception(
                        f"Option `keep_soft_constraints` needs to be set for vector goal {goal}"
                    )

            if goal.critical and goal.size > 1:
                raise Exception(f"Vector goal cannot be critical for goal {goal}")

        if is_path_goal:
            target_shape = len(self.times())
        else:
            target_shape = None

        # Check consistency and monotonicity of goals. Scalar target min/max
        # of normal goals are also converted to arrays to unify checks with
        # path goals.
        if options["check_monotonicity"]:
            for e in range(self.ensemble_size):
                # Store the previous goal of a certain function key we
                # encountered, such that we can compare to it.
                fk_goal_map = {}

                for goal in goals:
                    fk = goal.get_function_key(self, e)
                    prev = fk_goal_map.get(fk)
                    fk_goal_map[fk] = goal

                    if prev is not None:
                        goal_m, goal_M = self._gp_min_max_arrays(goal, target_shape)
                        other_m, other_M = self._gp_min_max_arrays(prev, target_shape)

                        indices = np.where(
                            np.logical_not(np.logical_or(np.isnan(goal_m), np.isnan(other_m)))
                        )
                        if goal.has_target_min:
                            if np.any(goal_m[indices] < other_m[indices]):
                                raise Exception(
                                    f"Target minimum of goal {goal} must be greater or equal than "
                                    f"target minimum of goal {prev}."
                                )

                        indices = np.where(
                            np.logical_not(np.logical_or(np.isnan(goal_M), np.isnan(other_M)))
                        )
                        if goal.has_target_max:
                            if np.any(goal_M[indices] > other_M[indices]):
                                raise Exception(
                                    f"Target maximum of goal {goal} must be less or equal than "
                                    f"target maximum of goal {prev}"
                                )

        for goal in goals:
            goal_m, goal_M = self._gp_min_max_arrays(goal, target_shape)
            goal_lb = np.broadcast_to(goal.function_range[0], goal_m.shape[::-1]).transpose()
            goal_ub = np.broadcast_to(goal.function_range[1], goal_M.shape[::-1]).transpose()

            if goal.has_target_min and goal.has_target_max:
                indices = np.where(
                    np.logical_not(np.logical_or(np.isnan(goal_m), np.isnan(goal_M)))
                )

                if np.any(goal_m[indices] > goal_M[indices]):
                    raise Exception(f"Target minimum exceeds target maximum for goal {goal}")

            if goal.has_target_min and not goal.critical:
                indices = np.where(np.isfinite(goal_m))
                if np.any(goal_m[indices] <= goal_lb[indices]):
                    raise Exception(
                        "Target minimum should be greater than the lower bound "
                        f"of the function range for goal {goal}"
                    )
                if np.any(goal_m[indices] > goal_ub[indices]):
                    raise Exception(
                        "Target minimum should not be greater than the upper bound "
                        f"of the function range for goal {goal}"
                    )
            if goal.has_target_max and not goal.critical:
                indices = np.where(np.isfinite(goal_M))
                if np.any(goal_M[indices] >= goal_ub[indices]):
                    raise Exception(
                        "Target maximum should be smaller than the upper bound "
                        f"of the function range for goal {goal}"
                    )
                if np.any(goal_M[indices] < goal_lb[indices]):
                    raise Exception(
                        "Target maximum should not be smaller than the lower bound "
                        f"of the function range for goal {goal}"
                    )

            if goal.relaxation < 0.0:
                raise Exception(f"Relaxation of goal {goal} should be a nonnegative value")

    def _gp_goal_constraints(self, goals, sym_index, options, is_path_goal):
        """
        There are three ways in which a goal turns into objectives/constraints:

        1. A goal with target bounds results in a part for the objective (the
           violation variable), and 1 or 2 constraints (target min, max, or both).
        2. A goal without target bounds (i.e. minimization goal) results in just a
           part for the objective.
        3. A critical goal results in just a (pair of) constraint(s). These are hard
           constraints, which need to be put in the constraint store to guarantee
           linear independence.
        """

        epsilons = []
        objectives = []
        soft_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        hard_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        extra_constants = []

        eps_format = "eps_{}_{}"
        min_format = "min_{}_{}"
        max_format = "max_{}_{}"

        if is_path_goal:
            eps_format = "path_" + eps_format
            min_format = "path_" + min_format
            max_format = "path_" + max_format

        for j, goal in enumerate(goals):
            if goal.critical:
                assert goal.size == 1, "Critical goals cannot be vector goals"
                epsilon = np.zeros(len(self.times()) if is_path_goal else 1)
            elif goal.has_target_bounds:
                epsilon = ca.MX.sym(eps_format.format(sym_index, j), goal.size)
                epsilons.append(epsilon)

            # Make symbols for the target bounds (if set)
            if goal.has_target_min:
                min_variable = min_format.format(sym_index, j)

                # NOTE: When using a vector goal, we want to be sure that its constraints
                # and objective end up _exactly_ equal to its non-vector equivalent. We
                # therefore have to get rid of any superfluous/trivial constraints that
                # would otherwise be generated by the vector goal.
                target_min_slice_inds = np.full(goal.size, True)

                if isinstance(goal.target_min, Timeseries):
                    target_min = Timeseries(goal.target_min.times, goal.target_min.values)
                    inds = np.logical_or(
                        np.isnan(target_min.values), np.isneginf(target_min.values)
                    )
                    target_min.values[inds] = -sys.float_info.max
                    n_times = len(goal.target_min.times)
                    target_min_slice_inds = ~np.all(
                        np.broadcast_to(inds.transpose(), (goal.size, n_times)), axis=1
                    )
                elif isinstance(goal.target_min, np.ndarray):
                    target_min = goal.target_min.copy()
                    inds = np.logical_or(np.isnan(target_min), np.isneginf(target_min))
                    target_min[inds] = -sys.float_info.max
                    target_min_slice_inds = ~inds
                else:
                    target_min = goal.target_min

                extra_constants.append((min_variable, target_min))
            else:
                min_variable = None

            if goal.has_target_max:
                max_variable = max_format.format(sym_index, j)

                target_max_slice_inds = np.full(goal.size, True)

                if isinstance(goal.target_max, Timeseries):
                    target_max = Timeseries(goal.target_max.times, goal.target_max.values)
                    inds = np.logical_or(
                        np.isnan(target_max.values), np.isposinf(target_max.values)
                    )
                    target_max.values[inds] = sys.float_info.max
                    n_times = len(goal.target_max.times)
                    target_max_slice_inds = ~np.all(
                        np.broadcast_to(inds.transpose(), (goal.size, n_times)), axis=1
                    )
                elif isinstance(goal.target_max, np.ndarray):
                    target_max = goal.target_max.copy()
                    inds = np.logical_or(np.isnan(target_max), np.isposinf(target_max))
                    target_max[inds] = sys.float_info.max
                    target_max_slice_inds = ~inds
                else:
                    target_max = goal.target_max

                extra_constants.append((max_variable, target_max))
            else:
                max_variable = None

            # Make objective for soft constraints and minimization goals
            if not goal.critical:
                if hasattr(goal, "_objective_func"):
                    _objective_func = goal._objective_func
                elif goal.has_target_bounds:
                    if is_path_goal and options["scale_by_problem_size"]:
                        goal_m, goal_M = self._gp_min_max_arrays(
                            goal, target_shape=len(self.times())
                        )
                        goal_active = np.isfinite(goal_m) | np.isfinite(goal_M)
                        n_active = np.sum(goal_active.astype(int), axis=-1)
                        # Avoid possible division by zero if goal is inactive
                        n_active = np.maximum(n_active, 1)
                    else:
                        n_active = 1

                    def _objective_func(
                        problem,
                        ensemble_member,
                        goal=goal,
                        epsilon=epsilon,
                        is_path_goal=is_path_goal,
                        n_active=n_active,
                    ):
                        if is_path_goal:
                            epsilon = problem.variable(epsilon.name())
                        else:
                            epsilon = problem.extra_variable(epsilon.name(), ensemble_member)

                        return goal.weight * ca.constpow(epsilon, goal.order) / n_active

                else:
                    if is_path_goal and options["scale_by_problem_size"]:
                        n_active = len(self.times())
                    else:
                        n_active = 1

                    def _objective_func(
                        problem,
                        ensemble_member,
                        goal=goal,
                        is_path_goal=is_path_goal,
                        n_active=n_active,
                    ):
                        f = goal.function(problem, ensemble_member) / goal.function_nominal
                        return goal.weight * ca.constpow(f, goal.order) / n_active

                objectives.append(_objective_func)

            # Make constraints for goals with target bounds
            if goal.has_target_bounds:
                if goal.critical:
                    for ensemble_member in range(self.ensemble_size):
                        constraint = self._gp_goal_hard_constraint(
                            goal, epsilon, None, ensemble_member, options, is_path_goal
                        )
                        hard_constraints[ensemble_member].append(constraint)
                else:
                    for ensemble_member in range(self.ensemble_size):
                        # We use a violation variable formulation, with the violation
                        # variables epsilon bounded between 0 and 1.
                        def _soft_constraint_func(
                            problem,
                            target,
                            bound,
                            inds,
                            goal=goal,
                            epsilon=epsilon,
                            ensemble_member=ensemble_member,
                            is_path_constraint=is_path_goal,
                        ):
                            if is_path_constraint:
                                target = problem.variable(target)
                                eps = problem.variable(epsilon.name())
                            else:
                                target = problem.parameters(ensemble_member)[target]
                                eps = problem.extra_variable(epsilon.name(), ensemble_member)

                            inds = inds.nonzero()[0].astype(int).tolist()

                            f = goal.function(problem, ensemble_member)
                            nominal = goal.function_nominal

                            return ca.if_else(
                                ca.fabs(target) < sys.float_info.max,
                                (f - eps * (bound - target) - target) / nominal,
                                0.0,
                            )[inds]

                        if goal.has_target_min and np.any(target_min_slice_inds):
                            _f = functools.partial(
                                _soft_constraint_func,
                                target=min_variable,
                                bound=goal.function_range[0],
                                inds=target_min_slice_inds,
                            )
                            constraint = _GoalConstraint(goal, _f, 0.0, np.inf, False)
                            soft_constraints[ensemble_member].append(constraint)
                        if goal.has_target_max and np.any(target_max_slice_inds):
                            _f = functools.partial(
                                _soft_constraint_func,
                                target=max_variable,
                                bound=goal.function_range[1],
                                inds=target_max_slice_inds,
                            )
                            constraint = _GoalConstraint(goal, _f, -np.inf, 0.0, False)
                            soft_constraints[ensemble_member].append(constraint)

        return epsilons, objectives, soft_constraints, hard_constraints, extra_constants

    def _gp_goal_hard_constraint(
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

    def _gp_update_constraint_store(self, constraint_store, constraints):
        for ensemble_member in range(self.ensemble_size):
            for other in constraints[ensemble_member]:
                fk = other.goal.get_function_key(self, ensemble_member)
                try:
                    constraint_store[ensemble_member][fk].update_bounds(other)
                except KeyError:
                    constraint_store[ensemble_member][fk] = other

    def priority_started(self, priority: int) -> None:
        """
        Called when optimization for goals of certain priority is started.

        :param priority: The priority level that was started.
        """
        self.skip_priority = False
        pass

    def priority_completed(self, priority: int) -> None:
        """
        Called after optimization for goals of certain priority is completed.

        :param priority: The priority level that was completed.
        """
        pass
