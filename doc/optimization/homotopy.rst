Treatment of nonconvexities using homotopy
==========================================

Using homotopy, a convex optimization problem can be continuously deformed into a non-convex problem.

Continuation and requirements
-----------------------------

Theta starts at ``theta_start`` and increases by ``delta_theta_0``. The last
step is shortened if needed, so the problem is always solved at exactly
theta one. For example, ``delta_theta_0 = 0.3`` solves at 0, 0.3, 0.6, 0.9
and 1.0.

.. note::

    Earlier versions stopped once theta would exceed one, so a step size that
    does not divide one (such as 0.3) never solved at theta one. Models that
    previously converged only up to such a value may now give different results
    or fail at theta one.

``HomotopyMixin`` must precede ``IOMixin`` and its subclasses (``CSVMixin``,
``PIMixin``, ``NetCDFMixin``) in the inheritance order, for example:

.. code-block:: python

    class Example(HomotopyMixin, GoalProgrammingMixin, CSVMixin, ModelicaMixin,
                  CollocatedIntegratedOptimizationProblem):
        ...

``IOMixin`` caches parameters and inputs, so in the opposite order theta would
not change between continuation steps.

A ``ValueError`` is raised before solving if:

* ``HomotopyMixin`` does not precede ``IOMixin``;
* ``theta_start`` is not finite or not in the range [0, 1];
* ``delta_theta_0`` or ``delta_theta_min`` is not finite and positive;
* the optimization time grid is empty;
* ``homotopy_parameter`` is not a parameter or fixed input of the model
  (aliases are allowed);
* ``non_linear_thresh_time_idx`` is not a non-negative integer, or is used while
  theta is a Modelica parameter (see below).


Time-dependent homotopy
-----------------------

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
one, even though the vector's linear tail remains zero. An index of ``0`` keeps
the whole horizon linear; the problem is then solved only once.

.. note::

    The linearity check of the collocation equations substitutes random values
    for fixed inputs. With time-dependent theta, the step at theta zero is
    therefore not recognized as linear. It is solved correctly, but without the
    constant-Jacobian solver setting and LP export, and a warning about
    non-affine equations is logged.

Intermediate goal programming
-----------------------------

With ``HomotopyMixin`` before ``GoalProgrammingMixin`` in the inheritance order,
set ``prune_intermediate_subproblems`` to ``True`` to solve only the first
available priority during intermediate continuation steps. This need not be
numeric priority 1. At the final step, all priorities are solved as usual.
Hard model constraints remain enforced at every solve. Pruning is disabled
by default. If the option has no effect, for example without
``GoalProgrammingMixin``, with the opposite inheritance order, or with
``SinglePassGoalProgrammingMixin``, a warning is logged and the option is ignored.

.. autoclass:: rtctools.optimization.homotopy_mixin.HomotopyMixin
    :members: homotopy_options, homotopy_theta
    :show-inheritance: