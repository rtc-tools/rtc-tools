Debugging an Optimization Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overview
--------

This example will illustrate a few tools that can help debugging an optimization problem.
For example, it will demonstrate how to test an optimization problem by only solving
for a fixed number of time steps.

A Basic Optimization Problem
----------------------------

For this example, the model is kept very basic.
We consider a single reservoir with a few target and optimization goals.
The optimization problem is given below.

.. literalinclude:: ../../../examples/single_reservoir/src/single_reservoir.py
  :language: python
  :pyobject: SingleReservoir

Optimizing for a given number of time steps
-------------------------------------------

By overwriting the method ``times``, we can control the times for which the problem is optimized.
In this case, we optimize for all times
unless the class attribute ``only_check_initial_values`` is set to ``True``.
Optimizing for only the initial time can be useful
to check for infeasibilities due to incompatible initial conditions.

Exporting the problem as an LP file
-----------------------------------

A linear or mixed-integer linear problem can be written to an LP file, to inspect it or to
solve it with another solver (e.g. HiGHS or CPLEX)::

    def export_options(self):
        options = super().export_options()
        options["export_lp"] = True
        return options

The file is written to the output folder, e.g. ``SingleReservoir_20250101_120000.lp``. When goal
programming solves more than one priority, each priority gets its own file with a
``_priority_N`` suffix. Export requires an affine objective, affine constraints, and bounds and
constant terms an LP file can hold (no NaN, no lower bound of +inf, no upper bound of -inf, no
infinite constant). It raises a :exc:`ValueError` otherwise.

By default the constraint rows are unnamed. Set the class attribute
``track_constraint_names = True`` to label each row by its origin:

* ``initial_residual_{i}`` and ``initial_derivative_{i}``: model equations at the initial time
* ``collocation_eq{i}__t{t}``: model equation ``i`` at time index ``t``
* ``delay_{variable}__t{t}``: delayed feedback
* ``constraint_{i}`` and ``path_constraint_{i}__t{t}``: entries of
  :meth:`~rtctools.optimization.optimization_problem.OptimizationProblem.constraints` and
  :meth:`~rtctools.optimization.optimization_problem.OptimizationProblem.path_constraints`
* ``objective_priority_{p}``: objective of an earlier priority, added by
  :class:`~rtctools.optimization.single_pass_goal_programming_mixin.SinglePassGoalProgrammingMixin`

Elements of a vector constraint get ``__{k}``, and ensemble members ``__m{i}``. A row with a
finite lower and upper bound becomes two lines, labelled ``__lb`` and ``__ub``. Characters
that LP readers reject become ``_``, and repeated labels get ``__d{n}``. The labels are also
available without exporting, in ``transcribed_problem["g_names"]`` after :meth:`optimize`.

To name a constraint yourself, set ``named_constraints = True`` and add the name as 4th
element of the tuple. With this flag on, constraint lists may mix 3- and 4-tuples, so code
that reads them, like a mixin calling ``super().constraints()``, must accept both::

    named_constraints = True
    track_constraint_names = True

    def path_constraints(self, ensemble_member):
        return [(self.state("Q_release"), 0.0, 10.0, "release_capacity")]

This yields rows labelled ``release_capacity__t0__lb``, ``release_capacity__t0__ub``, and so on.

Goal programming can name the constraints it adds as well: with ``named_constraints = True``
and ``track_constraint_names = True``, set the goal programming option ``constraint_names``
to label them ``{GoalClass}_p{priority}``, ``{GoalClass}_p{priority}_abs_pos`` / ``_abs_neg``
for a :class:`~rtctools.optimization.min_abs_goal_programming_mixin.MinAbsGoal`, and
``pareto_p{priority}`` for the objective constraint kept with ``keep_soft_constraints``.

Rows with two infinite bounds are left out of the file. A row that reduces to a constant
outside its bounds (by more than 1e-9, or with a lower bound above its upper bound) is kept,
written against a variable ``__zero`` fixed to 0, so that an LP solver also finds the exported
problem infeasible.
