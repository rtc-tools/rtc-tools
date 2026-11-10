Mixed Integer Optimization: Pumps and Orifices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: ../../images/Woudagemaal.jpg

.. :href: https://commons.wikimedia.org/wiki/File:Woudagemaal.jpg
.. content is released under a CC0 Public Domain licence - no attribution needed

.. note::

    This example focuses on how to incorporate mixed integer components into a
    hydraulic model, and assumes basic exposure to RTC-Tools. To start with
    basics, see :doc:`basic`.

.. note::

  By default, if you define any integer or boolean variables in the model, RTC-Tools
  will switch from IPOPT to BONMIN. You can modify solver options by overriding
  the
  :meth:`solver_options()<rtctools.optimization.optimization_problem.OptimizationProblem.solver_options>`
  method. Refer to CasADi's nlpsol interface for a list of supported solvers.


The Model
---------

For this example, the model represents a typical setup for the dewatering of
lowland areas. Water is routed from the hinterland (modeled as discharge
boundary condition, right side) through a canal (modeled as storage element)
towards the sea (modeled as water level boundary condition on the left side).
Keeping the lowland area dry requires that enough water is discharged  to the
sea. If the sea water level is lower than the water level in the canal, the
water can be discharged to the sea via gradient flow through the orifice (or a
weir). If the sea water level is higher than in the canal, water must be pumped.

To discharge water via gradient flow is free, while pumping costs money. The
control task is to keep the water level in the canal below a given flood warning
level at minimum costs. The expected result is that the model computes a control
pattern that makes use of gradient flow whenever possible and activates the pump
only when necessary.

The model can be viewed and edited using the OpenModelica Connection Editor
program. First load the Deltares library into OpenModelica Connection Editor,
and then load the example model, located at
``<examples directory>\mixed_integer\model\Example.mo``. The model ``Example.mo``
represents a simple water system with the following elements:

* a canal segment, modeled as storage element
  ``Deltares.ChannelFlow.Hydraulic.Storage.Linear``,
* a discharge boundary condition
  ``Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge``,
* a water level boundary condition
  ``Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level``,
* a pump
  ``Deltares.ChannelFlow.Hydraulic.Structures.Pump``
* an orifice modeled as a pump
  ``Deltares.ChannelFlow.Hydraulic.Structures.Pump``

.. image:: ../../images/orifice_vs_pump_openmodelica.png

In text mode, the Modelica model looks as follows (with annotation statements
removed):

.. literalinclude:: ../../_build/mo/mixed_integer.mo
  :language: modelica
  :lineno-match:

The five water system elements (storage, discharge boundary condition, water
level boundary condition, pump, and orifice) appear under the ``model Example``
statement. The ``equation`` part connects these five elements with the help of
connections. Note that ``Pump`` extends the partial model ``HQTwoPort`` which
inherits from the connector ``HQPort``. With ``HQTwoPort``, ``Pump`` can be
connected on two sides. ``level`` represents a model boundary condition (model
is meant in a hydraulic sense here), so it can be connected to one other
element only. It extends the ``HQOnePort`` which again inherits from the
connector ``HQPort``.

In addition to elements, the input variables ``Q_in``, ``H_sea``, ``Q_pump``,
and ``Q_orifice`` are also defined. Because we want to view the water levels in
the storage element in the output file, we also define output
variables ``storage_level`` and ``sea_level``. It is usually easiest to set input
and output variables equal to their corresponding model variable in the same line.

To maintain the linearity of the model, we input the Boolean ``is_downhill`` as
a way to keep track of whether water can flow by gravity to the sea. This
variable is not used directly in the hydraulics, but we use it later in the
constraints in the python file.

The Optimization Problem
------------------------

The python script consists of the following blocks:

* Import of packages
* Definition of the optimization problem class

  * Constructor
  * Objective function
  * Definition of constraints
  * Additional configuration of the solver

* A run statement

Importing Packages
''''''''''''''''''

For this example, the import block is as follows:

.. literalinclude:: ../../../examples/mixed_integer/src/example.py
  :language: python
  :lines: 1-6
  :lineno-match:

Note that we are also importing ``inf`` from ``numpy``. We will use this later
in the constraints.

Optimization Problem
''''''''''''''''''''

Next, we construct the class by declaring it and inheriting the desired parent
classes.

.. literalinclude:: ../../../examples/mixed_integer/src/example.py
  :language: python
  :pyobject: Example
  :lineno-match:
  :end-before: """

Now we define an objective function. This is a class method that returns the
value that needs to be minimized. Here we specify that we want to minimize the
volume pumped:

.. literalinclude:: ../../../examples/mixed_integer/src/example.py
  :language: python
  :pyobject: Example.objective
  :lineno-match:

Constraints can be declared by declaring the ``path_constraints()`` method.
Path constraints are constraints that are applied every timestep. To set a
constraint at an individual timestep, define it inside the ``constraints``
method.

The orifice ``BooleanSubmergedOrifice`` requires special constraints to be set
in order to work. They are implemented below in the ``path_constraints()``
method. their parent classes also declare this method, so we call the
``super()`` method so that we don't overwrite their behaviour.

.. literalinclude:: ../../../examples/mixed_integer/src/example.py
  :language: python
  :pyobject: Example.path_constraints
  :lineno-match:

Finally, we want to apply some additional configuration, reducing the amount of
information the solver outputs:

.. literalinclude:: ../../../examples/mixed_integer/src/example.py
  :language: python
  :pyobject: Example.solver_options
  :lineno-match:

Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_optimization_problem()`` method we imported on the optimization
problem class we just created.

.. literalinclude:: ../../../examples/mixed_integer/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../../../examples/mixed_integer/src/example.py
  :language: python
  :lineno-match:

Running the Optimization Problem
--------------------------------

.. note:: An explanation of bonmin behaviour and output goes here.



Extracting Results
------------------

The results from the run are found in ``output/timeseries_export.csv``. Any
CSV-reading software can import it, but this is how results can be plotted using
the python library matplotlib:


.. plot:: examples/pyplots/mixed_integer_results.py
   :include-source:


.. _mixed-integer-results:

Observations
------------

Note that in the results plotted above, the pump runs with a constantly varying
throughput. To smooth out the flow through the pump, consider using goal
programming to apply a path goal minimizing the derivative of the pump at each
timestep. For an example, see the third goal in
:ref:`goal-programming-declaring-goals`.
