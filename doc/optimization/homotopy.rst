Treatment of nonconvexities using homotopy
==========================================

Using homotopy, a convex optimization problem can be continuously deformed into a non-convex problem.

Time-dependent homotopy
----------------------

Set ``non_linear_thresh_time_idx`` in ``homotopy_options()`` to retain nonlinear
equations only in the near-term forecast. The zero-based index refers to
``times()``: values before the index follow continuation, while values at and
after the index remain zero. IO history does not count towards this index;
historical timestamps receive the first forecast theta value. An explicitly
provided parameter of the same name takes precedence over the option. ``None``
or an index at or beyond the forecast length retains scalar homotopy.

Time-dependent theta must be declared as a fixed Modelica input:

.. code-block:: modelica

    input Real theta(fixed=true);

Use this input in the equations that blend the linear approximation and the
nonlinear formulation. It cannot vary in time when declared as
``parameter Real theta``. Components that expose theta only as a parameter
must also be adapted before time-dependent homotopy can be used. Incompatible
models raise an error instead of silently applying scalar theta. Scalar
homotopy with existing parameter-based models remains supported.

The model, not the mixin, determines whether theta zero yields a linear
approximation. ``homotopy_theta`` exposes the scalar theta or the forecast-grid
vector. Continuation is complete when the scalar continuation progress reaches
one, even though the vector's linear tail remains zero.

Intermediate goal programming
-----------------------------

With ``HomotopyMixin`` before ``GoalProgrammingMixin`` in the inheritance order,
set ``prune_intermediate_subproblems`` to ``True`` to solve only the first
available priority during intermediate continuation steps. This need not be
numeric priority 1. At the final step, all priorities are solved as usual.
Hard model constraints remain enforced at every solve. Pruning is disabled
by default.

.. autoclass:: rtctools.optimization.homotopy_mixin.HomotopyMixin
    :members: homotopy_options, homotopy_theta
    :show-inheritance: