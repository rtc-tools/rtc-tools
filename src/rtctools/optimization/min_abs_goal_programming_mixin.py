import functools
import itertools

import casadi as ca
import numpy as np

from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check

from .goal_programming_mixin import GoalProgrammingMixin
from .goal_programming_mixin_base import (
    Goal,
    StateGoal,
    _ConstraintTuple,
    _EmptyEnsembleList,
    _EmptyEnsembleOrderedDict,
    _goal_constraint_name,
    _GoalConstraint,
    _GoalProgrammingMixinBase,
)
from .single_pass_goal_programming_mixin import SinglePassGoalProgrammingMixin
from .timeseries import Timeseries


class MinAbsGoal(Goal):
    """
    Absolute minimization goal class which can be used to minimize the
    absolute value of the goal's (linear) goal function. Contrary to its super
    class, the default order is 1 as absolute minimization is typically
    desired for fully linear problems.
    """

    order = 1


class MinAbsStateGoal(StateGoal, MinAbsGoal):
    pass


class _ConvertedMinAbsGoal(Goal):
    order = 1

    def __init__(self, abs_variable, is_path_goal, orig_goal):
        self.abs_variable = abs_variable
        self.is_path_goal = is_path_goal
        self.orig_goal = orig_goal

        # Copy relevant properties
        self.size = orig_goal.size
        self.weight = orig_goal.weight
        self.relaxation = orig_goal.relaxation / orig_goal.function_nominal
        self.priority = orig_goal.priority

    def function(self, optimization_problem, ensemble_member):
        if self.is_path_goal:
            return optimization_problem.variable(self.abs_variable.name())
        else:
            return optimization_problem.extra_variable(self.abs_variable.name(), ensemble_member)


class MinAbsGoalProgrammingMixin(_GoalProgrammingMixinBase):
    """
    Similar behavior to :py:class:`.GoalProgrammingMixin`, but any
    :py:class:`MinAbsGoal` passed to :py:meth:`.min_abs_goals` or
    :py:meth:`.min_abs_path_goals` will be automatically converted to:

      1. An auxiliary minimization variable
      2. Two additional linear constraints relating the auxiliary variable to the goal function
      3. A new goal (of a different type) minimizing the auxiliary variable
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # List for any absolute minimization goals
        self.__problem_constraints = _EmptyEnsembleList()
        self.__problem_vars = []
        self.__problem_path_constraints = _EmptyEnsembleList()
        self.__problem_path_vars = []
        self.__seeds = _EmptyEnsembleOrderedDict()
        self.__path_seeds = _EmptyEnsembleOrderedDict()

        self.__first_run = True

    @property
    def extra_variables(self):
        return super().extra_variables + self.__problem_vars

    @property
    def path_variables(self):
        return super().path_variables + self.__problem_path_vars

    @ensemble_bounds_check
    def bounds(self, ensemble_member: int | None = None):
        bounds = (
            super().bounds(ensemble_member) if self.ensemble_specific_bounds else super().bounds()
        )

        for abs_var in self.__problem_vars + self.__problem_path_vars:
            bounds[abs_var.name()] = (0.0, np.inf)
        return bounds

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)

        # Seed minimization variables of current priority (not those of
        # previous priorities, as those are handled by GoalProgrammingMixin).
        for abs_var, val in self.__seeds[ensemble_member].items():
            seed[abs_var] = val

        times = self.times()
        for abs_var, val in self.__path_seeds[ensemble_member].items():
            seed[abs_var] = Timeseries(times, val)

        return seed

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        for constraint in self.__problem_constraints[ensemble_member]:
            name = _goal_constraint_name(constraint)
            constraints.append(
                _ConstraintTuple((constraint.function(self), constraint.min, constraint.max, name))
            )

        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)

        for constraint in self.__problem_path_constraints[ensemble_member]:
            name = _goal_constraint_name(constraint)
            path_constraints.append(
                _ConstraintTuple((constraint.function(self), constraint.min, constraint.max, name))
            )

        return path_constraints

    def __validate_goals(self, goals, is_path_goal):
        goals = sorted(goals, key=lambda x: x.priority)

        for goal in goals:
            if not isinstance(goal, MinAbsGoal):
                raise Exception(f"Absolute goal not an instance of MinAbsGoal for goal {goal}")

            if goal.function_range != (np.nan, np.nan):
                raise Exception(
                    f"Absolute goal function is only allowed for minimization for goal {goal}"
                )

            if goal.order != 1:
                raise Exception(
                    f"Absolute goal function is only allowed for order = 1 for goal {goal}"
                )

            if goal.weight <= 0:
                raise Exception(
                    f"Absolute goal function is only allowed for weight > 0 for goal {goal}"
                )

    @staticmethod
    def __convert_goals(goals, sym_index, ensemble_size, is_path_goal):
        # Replace absolute minimization goals with a new goal, and some
        # additional hard constraints.
        constraints = [[] for ensemble_member in range(ensemble_size)]
        variables = []

        # It is easier to modify goals in place, but we do not want to modify
        # the original input list of goals. Make a copy to work with and
        # return when we are done.
        goals = goals.copy()

        for j, goal in enumerate(goals):
            assert isinstance(goal, MinAbsGoal)

            abs_variable_name = f"abs_{sym_index}_{j}"
            if is_path_goal:
                abs_variable_name = "path_" + abs_variable_name

            abs_variable = ca.MX.sym(abs_variable_name, goal.size)
            variables.append(abs_variable)

            # Set constraints on how the additional variable relates to the
            # original goal function, such that it corresponds to its absolute
            # value when minimizing.
            for ensemble_member in range(ensemble_size):

                def _constraint_func(
                    problem,
                    sign,
                    abs_variable=abs_variable,
                    ensemble_member=ensemble_member,
                    goal=goal,
                    is_path_goal=is_path_goal,
                ):
                    if is_path_goal:
                        abs_variable = problem.variable(abs_variable.name())
                    else:
                        abs_variable = problem.extra_variable(abs_variable.name(), ensemble_member)

                    return (
                        abs_variable
                        + sign * goal.function(problem, ensemble_member) / goal.function_nominal
                    )

                _pos = functools.partial(_constraint_func, sign=1)
                _neg = functools.partial(_constraint_func, sign=-1)

                goal_constr_base_name = f"{goal.__class__.__name__}_p{goal.priority}"
                constraints[ensemble_member].append(
                    _GoalConstraint(
                        None, _pos, 0.0, np.inf, False, f"{goal_constr_base_name}_abs_pos"
                    )
                )
                constraints[ensemble_member].append(
                    _GoalConstraint(
                        None, _neg, 0.0, np.inf, False, f"{goal_constr_base_name}_abs_neg"
                    )
                )

            # Overwrite the original goal, such that it is just a minimization
            # of the additional variable.
            goals[j] = _ConvertedMinAbsGoal(abs_variable, is_path_goal, goal)

        return goals, constraints, variables

    def __calculate_seed(self, goals, is_path_goal):
        assert self.__first_run is False

        seed = [{} for ensemble_member in range(self.ensemble_size)]

        for goal in goals:
            assert isinstance(goal, _ConvertedMinAbsGoal)

            for ensemble_member in range(self.ensemble_size):
                if is_path_goal:
                    expr = self.map_path_expression(
                        goal.orig_goal.function(self, ensemble_member), ensemble_member
                    )
                else:
                    expr = goal.orig_goal.function(self, ensemble_member)

                function = ca.Function("f", [self.solver_input], [expr])
                value = np.array(function(self.solver_output))

                assert value.ndim == 2

                if goal.size == 1:
                    if is_path_goal:
                        value = value.ravel()
                    else:
                        value = value.item()

                seed[ensemble_member][goal.abs_variable.name()] = np.abs(value)

        return seed

    def optimize(self, preprocessing=True, **kwargs):
        # Do pre-processing
        if preprocessing:
            self.pre()

        goals = self.min_abs_goals()
        path_goals = self.min_abs_path_goals()

        # Validate goal definitions
        self.__validate_goals(goals, is_path_goal=False)
        self.__validate_goals(path_goals, is_path_goal=True)

        # List for absolute minimization goals. These will be incrementally
        # filled only just before we need them to.
        self.__problem_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__problem_vars = []
        self.__problem_path_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__problem_path_vars = []

        # Similar to the above, but these keep track of all auxiliary
        # variables and constraints of priorities we have had (and therefore
        # activated), and those yet to come (and have not yet activated).
        self.__subproblem_constraints = {}
        self.__subproblem_vars = {}
        self.__subproblem_abs_goals = {}
        self.__subproblem_path_constraints = {}
        self.__subproblem_path_vars = {}
        self.__subproblem_path_abs_goals = {}

        # We want to have consistent naming with GPMixin for our auxiliary
        # variables. We therefore need to loop over all priorities, regardless
        # of whether there are any MinAbsGoals in it or not.
        priorities = {
            int(goal.priority)
            for goal in itertools.chain(goals, path_goals, self.goals(), self.path_goals())
            if not goal.is_empty
        }

        subproblems = []
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

        # Rewrite absolute minimization goals.
        self.__converted_goals = []
        self.__converted_path_goals = []

        for i, (priority, goals, path_goals) in enumerate(subproblems):
            (
                goals,
                self.__subproblem_constraints[priority],
                self.__subproblem_vars[priority],
            ) = self.__convert_goals(goals, i, self.ensemble_size, False)

            self.__converted_goals.extend(goals)
            self.__subproblem_abs_goals[priority] = goals

            (
                path_goals,
                self.__subproblem_path_constraints[priority],
                self.__subproblem_path_vars[priority],
            ) = self.__convert_goals(path_goals, i, self.ensemble_size, True)

            self.__converted_path_goals.extend(path_goals)
            self.__subproblem_path_abs_goals[priority] = path_goals

        return super().optimize(**kwargs, preprocessing=False)

    def priority_started(self, priority):
        super().priority_started(priority)

        # Enable constraints and auxiliary variables that we need starting
        # from this priority when using GoalProgrammingMixin. When using
        # SinglePassGoalProgrammingMixin, we need to add all constraints from
        # the start.
        if isinstance(self, GoalProgrammingMixin):
            priorities = [priority]
        elif isinstance(self, SinglePassGoalProgrammingMixin):
            if self.__first_run:
                priorities = self.__subproblem_constraints.keys()
            else:
                priorities = []

        for p in priorities:
            for a, b in zip(
                self.__problem_constraints, self.__subproblem_constraints[p], strict=True
            ):
                a.extend(b)

            self.__problem_vars.extend(self.__subproblem_vars[p])

            for a, b in zip(
                self.__problem_path_constraints, self.__subproblem_path_constraints[p], strict=True
            ):
                a.extend(b)

            self.__problem_path_vars.extend(self.__subproblem_path_vars[p])

        # Calculate the seed needed for goals/variables introduced in this
        # priority. We can only calculate a seed if this is not the first
        # priority.
        if not self.__first_run and isinstance(self, GoalProgrammingMixin):
            self.__seeds = self.__calculate_seed(self.__subproblem_abs_goals[priority], False)
            self.__path_seeds = self.__calculate_seed(
                self.__subproblem_path_abs_goals[priority], True
            )

        self.__first_run = False

    def min_abs_goals(self) -> list[MinAbsGoal]:
        """
        User problem returns list of :py:class:`MinAbsGoal` objects.

        :returns: A list of goals.
        """
        return []

    def goals(self):
        goals = super().goals()
        try:
            return goals + self.__converted_goals
        except AttributeError:
            return goals

    def min_abs_path_goals(self) -> list[MinAbsGoal]:
        """
        User problem returns list of :py:class:`MinAbsGoal` objects.

        :returns: A list of goals.
        """
        return []

    def path_goals(self):
        goals = super().path_goals()
        try:
            return goals + self.__converted_path_goals
        except AttributeError:
            return goals
