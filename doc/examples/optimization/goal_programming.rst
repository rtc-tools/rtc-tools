Goal Programming: Defining Multiple Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    This example focuses on how to implement multi-objective optimization in
    RTC-Tools using Goal Programming. It assumes basic exposure to
    RTC-Tools. If you are a first-time user of RTC-Tools, see
    :doc:`basic`.

Goal programming is a way to satisfy (sometimes conflicting) goals by ranking
the goals by priority. The optimization algorithm will attempt to optimize each
goal one at a time, starting with the goal with the highest priority and moving
down through the list. Even if a goal cannot be satisfied, the goal programming
algorithm will move on when it has found the best possible answer. Goals can be
roughly divided into two types:

* As long as we satisfy the goal, we do not care by how much. If we cannot
  satisfy a goal, any lower priority goals are not allowed to increase the
  amount by which we exceed (which is equivalent to not allowing any change at
  all to the exceedance).
* We try to achieve as low a value as possible. Any lower priority goals are not
  allowed to result in an increase of this value (which is equivalent to not
  allowing any change at all).

In this example, we will be specifying two goals, on for each type. The higher
priority goal will be to maintain the water level of the storage element between
two levels. The lower priority goal will be to minimize the total volume pumped.

The Model
---------

.. note::

    This example uses the same hydraulic model as the MILP example. For a
    detailed explanation of the hydraulic model, including how to to formulate
    mixed integers in your model, see :doc:`mixed_integer`.

For this example, the model represents a typical setup for the dewatering of
lowland areas. Water is routed from the hinterland (modeled as discharge
boundary condition, right side) through a canal (modeled as storage element)
towards the sea (modeled as water level boundary condition on the left side).
Keeping the lowland area dry requires that enough water is discharged  to the
sea. If the sea water level is lower than the water level in the canal, the
water can be discharged to the sea via gradient flow through the orifice (or a
weir). If the sea water level is higher than in the canal, water must be pumped.

In OpenModelica Connection Editor, the model looks like this:

.. image:: ../../images/orifice_vs_pump_openmodelica.png

In text mode, the Modelica model looks as follows (with annotation statements
removed):

.. literalinclude:: ../../_build/mo/goal_programming.mo
  :language: modelica
  :lineno-match:

The Optimization Problem
------------------------

When using goal programming, the python script consists of the following blocks:

* Import of packages
* Declaration of Goals
* Declaration of the optimization problem class

  * Constructor
  * Declaration of constraint methods
  * Specification of Goals
  * Declaration of a ``priority_completed()`` method
  * Declaration of a ``pre()`` method
  * Declaration of a ``post()`` method
  * Additional configuration of the solver

* A run statement

Importing Packages
''''''''''''''''''

For this example, the import block is as follows:

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :lines: 1-8
  :lineno-match:

.. _goal-programming-declaring-goals:

Declaring Goals
'''''''''''''''

Goals are defined as classes that inherit the ``Goal`` parent class. The
components of goals can be found in :doc:`../../optimization/multi_objective`. In
this example, we demonstrate three ways to define a goal in RTC-Tools.

First, we have a high priority goal to keep the water level within a minimum and
maximum. Since we are applying this goal to a specific state (model variable) in
our model at every time step, we can inherit a special helper class to define
this goal, called a ``StateGoal``:

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: WaterLevelRangeGoal
  :lineno-match:

We also want to save energy, so we define a goal to minimize the integral of
``Q_pump``. This goal has a lower priority than the water level range goal.
This goal does not use a helper class:

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: MinimizeQpumpGoal
  :lineno-match:

We add a third goal minimizing the changes in``Q_pump``, and give it the
least priority. This goal smooths out the operation of the pump so that it
changes state as few times as possible. To get an idea of what the pump would
have done without this goal, see Mixed Integer: :ref:`mixed-integer-results`.
The order of this goal must be 2, so that it penalizes both positive and
negative derivatives. Order of 2 is the default, but we include it here
explicitly for the sake of clarity.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: MinimizeChangeInQpumpGoal
  :lineno-match:

Optimization Problem
''''''''''''''''''''

Next, we construct the class by declaring it and inheriting the desired parent
classes.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example
  :lineno-match:
  :end-before: """

Constraints can be declared by declaring the ``path_constraints()`` method.
Path constraints are constraints that are applied every timestep. To set a
constraint at an individual timestep, define it inside the ``constraints()``
method.

The "orifice" requires special constraints to be set in order to work. They are
implemented below in the ``path_constraints()`` method. Other parent classes
also declare this method, so we call the ``super()`` method so that we don't
overwrite their behaviour.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.path_constraints
  :lineno-match:

Now we pass in the goals. There are path goals and normal goals, so we have to
pass them in using separate methods. A path goal is a specific kind of goal that
applies to a particular variable at an individual time step, but that we want to
set for all the timesteps.

Non-path goals are more general goals that are not iteratively applied at every
timestep. We use the ``goals()`` method to pass a list of these goals to the
optimizer.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.goals
  :lineno-match:

For the goals that want to apply our goals to every timestep, so we use the
``path_goals()`` method. This is a method that returns a list of the path goals
we defined above. Note that with path goals,  each timestep is implemented as an
independent goal- if we cannot satisfy our min/max on time step A, it will not
affect our desire to satisfy the goal at time step B. Goals that inherit
``StateGoal`` are always path goals and must always be initialized with the
parameter ``self``.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.path_goals
  :lineno-match:

If all we cared about were the results, we could end our class declaration here.
However, it is usually helpful to track how the solution changes after
optimizing each priority level. To track these changes, we need to add three
methods.

The method ``pre()`` is already defined in RTC-Tools, but we would like to add
a line to it to create a variable for storing intermediate results. To do this,
we declare a new ``pre()`` method, call ``super().pre()`` to ensure
that the original method runs unmodified, and add in a variable declaration to
store our list of intermediate results:

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.pre
  :lineno-match:

Next, we define the ``priority_completed()`` method to inspect and summarize the
results. These are appended to our intermediate results variable after each
priority is completed.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.priority_completed
  :lineno-match:

We want some way to output our intermediate results. This is accomplished using
the ``post()`` method. Again, we need to call the ``super()`` method to avoid
overwriting the internal method.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.post
  :lineno-match:

Finally, we want to apply some additional configuration, reducing the amount of
information the solver outputs:

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :pyobject: Example.solver_options
  :lineno-match:

Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_optimization_problem()`` method we imported on the optimization
problem class we just created.

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../../../examples/goal_programming/src/example.py
  :language: python
  :lineno-match:

Running the Optimization Problem
--------------------------------

Following the execution of the optimization problem, the  ``post()`` method
should print out the following lines::

    After finishing goals of priority 1:
    Level goal satisfied at 19 of 21 time steps
    Integral of Q_pump = 74.18
    Sum of Changes in Q_pump: 7.83

    After finishing goals of priority 2:
    Level goal satisfied at 19 of 21 time steps
    Integral of Q_pump = 60.10
    Sum of Changes in Q_pump: 11.70

    After finishing goals of priority 3:
    Level goal satisfied at 19 of 21 time steps
    Integral of Q_pump = 60.10
    Sum of Changes in Q_pump: 10.07

As the output indicates, while optimizing for the priority 1 goal, no attempt
was made to minimize the integral of ``Q_pump``. The only objective was to
minimize the number of states in violation of the water level goal.

After optimizing for the priority 2 goal, the solver was able to find a solution
that reduced the integral of ``Q_pump`` without increasing the number of
timesteps where the water level exceeded the limit. However, this solution
induced additional variation into the operation of ``Q_pump``

After optimizing the priority 3 goal, the integral of ``Q_pump`` is the same and
the level goal has not improved. Without hurting any higher priority goals,
RTC-Tools was able to smooth out the operation of the pump.

Extracting Results
------------------

The results from the run are found in ``output/timeseries_export.csv``. Any
CSV-reading software can import it, but this is how results can be plotted using
the python library matplotlib:

.. plot:: examples/pyplots/goal_programming_results.py
   :include-source:

Observe that the red pump curve is indeed more smooth than what we have seen before. 
You may compare this picture with the one in the previous example 
(Mixed Integer Optimization). The observant user may conclude that although the curve 
is indeed more smooth, the surface area under the red curve also seems to have increased 
(i.e., in total, more water is pumped!) This is due to the extra goal that we introduced: 
the WaterLevelRangeGoal that wasn't present in the Mixed Integer example. Experiment 
with commenting out this goal on line 109 to see how it affects the results.