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

Exporting the Transcribed Problem as an LP File
-----------------------------------------------

For linear (or mixed-integer linear) problems, the full transcribed optimization problem can be
exported as an LP file by setting the ``export_lp`` solver option. The file follows the
`Gurobi LP format <https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html#lp-format>`_,
which is supported by most LP/MILP solvers (CPLEX, HiGHS, GLPK, etc.)::

    def solver_options(self):
        options = super().solver_options()
        options["export_lp"] = True
        return options

The file is written to the output folder with a timestamp in the name, e.g.
``MyModel_20250101_120000.lp``. When used with goal programming and more than one priority,
a ``_priority_N`` suffix is added for each priority.

The LP file includes auto-generated constraint names that identify each constraint by its origin,
making it easy to cross-reference the LP output with your model:

* ``initial_residual_{i}`` — initial DAE residual constraints
* ``initial_derivative_{i}`` — initial derivative constraints (present when history data is available)
* ``collocation_eq{eq}_t{t}`` — collocation (discretized DAE) equality constraints, one per
  equation per interior time step (the initial time point is covered by ``initial_residual_``
  and ``initial_derivative_``)
* ``delay_{name}_t{t}`` — delayed feedback constraints
* ``constraint_{i}`` — user constraints from :meth:`~rtctools.optimization.optimization_problem.OptimizationProblem.constraints`
* ``path_constraint_{i}_t{t}`` — user path constraints from :meth:`~rtctools.optimization.optimization_problem.OptimizationProblem.path_constraints`
* ``single_pass_objective_p{prev}__at_p{curr}_{i}`` — priority-tightening constraints added by :class:`~rtctools.optimization.single_pass_goal_programming_mixin.SinglePassGoalProgrammingMixin`

For ensemble problems (``ensemble_size > 1``), all constraint names receive a ``_m{i}``
suffix to ensure uniqueness across ensemble members. This applies to both auto-generated names
(e.g. ``collocation_eq0_t0_m0``) and user-provided names (e.g. ``terminal_state_m0``,
``terminal_state_m1``).

Range constraints (finite lower and upper bound) are expressed as two separate inequalities
in the LP file. The lower-bound line carries a ``_lb`` suffix and the upper-bound line a
``_ub`` suffix (e.g. ``state_bounds_t0_lb`` and ``state_bounds_t0_ub``), so the two lines
share a common base name for easy cross-referencing. Vacuous constraints (both bounds
infinite) are also emitted as two lines (``>= -Inf`` and ``<= +Inf``) to aid debugging,
using the same ``_lb``/``_ub`` suffix convention. They have no effect on the solver.

If two user constraints share the same name, they are automatically deduplicated with
``_d0``, ``_d1``, … suffixes. The ``_d`` prefix distinguishes deduplication indices from
time (``_t{n}``), ensemble-member (``_m{n}``), and range-side (``_lb``/``_ub``) suffixes.
User-provided names ending with any of these reserved suffixes (``_d{n}``, ``_m{n}``,
``_t{n}``, ``_lb``, ``_ub``) will trigger a warning and be renamed with a ``_ren`` suffix
to avoid collisions.

You can also provide a custom name as an optional 4th element in the constraint tuple.
The name will appear verbatim in the LP file; path constraints receive a ``_t{i}`` time
suffix. An empty string ``""`` is treated as absent and falls back to the auto-generated name::

    def constraints(self, ensemble_member):
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return [(xf, 0.0, 1.0, "terminal_state")]

    def path_constraints(self, ensemble_member):
        return [(self.state("x"), -2.0, 2.0, "state_bounds")]

This produces LP output like::

    terminal_state_lb: 1 x__17 >= 0
    terminal_state_ub: 1 x__17 <= 1
    state_bounds_t0_lb: 1 x__9 >= -2
    state_bounds_t0_ub: 1 x__9 <= 2
    state_bounds_t1_lb: 1 x__10 >= -2
    state_bounds_t1_ub: 1 x__10 <= 2
    ...

LP export is only supported for
:class:`~rtctools.optimization.collocated_integrated_optimization_problem.CollocatedIntegratedOptimizationProblem`
subclasses with ``linear_collocation=True`` (the default). A :exc:`ValueError` is raised in
three cases: the problem uses non-linear Modelica DAE equations (``linear_collocation=False``),
the objective is non-affine, or any constraint is non-affine. A :exc:`NotImplementedError`
is raised when ``export_lp=True`` is used on a class that does not inherit from
``CollocatedIntegratedOptimizationProblem``.
