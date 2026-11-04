Mixed Integer Optimization: Pumped Hydropower System (PHS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    This example focuses on how to incorporate mixed integer components into a
    hydraulic model with an example in hydropower, and assumes basic exposure to RTC-Tools. To start with
    basics, see :doc:`basic`.

.. note::

  By default, if you define any integer or boolean variables in the model, RTC-Tools
  will switch from IPOPT to BONMIN. You can modify solver options by overriding
  the
  :meth:`solver_options()<rtctools.optimization.optimization_problem.OptimizationProblem.solver_options>`
  method. Refer to CasADi's nlpsol interface for a list of supported solvers. In this case we choose HiGHS
  because the problem in a convex MIP and thus HiGHS can be used and tends to be faster than BONMIN.


The Model
---------

For this example, the model represents a simpled Pumped Hydropower System. 

The model can be viewed and edited using the OpenModelica Connection Editor
program. First load the Deltares library into OpenModelica Connection Editor,
and then load the example model, located at
``<examples directory>\pumped_hydropower_system\model\PumpedStoragePlant.mo``. The model ``PumpedStoragePlant.mo``
represents a simple water system with the following elements:

* an inflow boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow``,
* an upper reservoir modelled as a storage element
  ``Deltares.ChannelFlow.SimpleRouting.Storage.Storage``,
* an lower reservoir
  ``Deltares.ChannelFlow.SimpleRouting.Reservoir.Reservoir``,
* The PHS pump which pumps water from the lower to the upper reservoir
  ``Deltares.ChannelFlow.SimpleRouting.Structures.DischargeControlledStructure``,
* The PHS turbine which generates energy while water flows from the upper to lower reservoir 
  ``Deltares.ChannelFlow.SimpleRouting.Structures.DischargeControlledStructure``,
* an outflow boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal``,


.. image:: ../../images/PumpedStoragePlant_openmodelica.png

In text mode, the Modelica model looks as follows (with annotation statements
removed): 

.. literalinclude:: ../../../examples/pumped_hydropower_system/model/PumpedStoragePlant.mo
  :language: modelica
  :lineno-match:

The five water system elements (inflow boundary condition, storage, reservoir, Discharge
controlled structure and outflow boundary condition) appear under the ``model PumpedStoragePlant``
statement. The ``equation`` part connects these five elements with the help of
connections. 

In addition to elements, the input variables ``Inflow_Q``, ``PumpFlow``, ``TurbineFlow``, 
``ReservoirTurbineFlow``, ``ReservoirSpillFlow`` and ``cost_perP`` are also defined. Because 
we want to view the power output and costs in the output file, we also define output
variables ``PumpPower``, ``TurbinePower``, ``V_LowerBasin``,``V_UpperBasin``, ``ReservoirPower``, 
``TotalSystemPower``,``TotalGeneratingPower``, ``SystemGeneratingRevenue``, ``TotalSystemRevenue``, 
``PumpCost``.

To maintain represent the behaviour of a reversible PHS pump/turbine unit, we input the Boolean 
``Turbine_is_on`` as a way to ensure the PHS unit cannot be pumping and generating at the same time. 
This variable is not used directly in the hydraulics, but we use it later in the
constraints in the python file.

The Optimization Problem
------------------------

The python script consists of the following blocks:

* Import of packages
* Definition of goals
* Definition of the optimization problem class

  * Pre-processing
  * Setting of varaible nominals
  * Definition of constraints
  * Additional configuration of the solver
  * Post-processing

* A run statement

Importing Packages
''''''''''''''''''

For this example, the import block is as follows:

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :lines: 1-12
  :lineno-match:

Optimization Problem
''''''''''''''''''''

First we define any goals we want to use during the optimization. These classes extend the ``Goal``
class of RTC-Tools. 

The ``TargetGoal`` calculates the deviations from the state and the ``target_min``
and ``target_max``. It is of order 4, so it then minimizes the sum of these deviations raised to the 
power of 4. For this example the class is defined as follows.

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: TargetGoal
  :lineno-match:

The ``MaxRevenueGoal`` maximizes the state variable. To achieve this, we minimize the negative of the 
state. For this example the class is defined as follows.

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: MaxRevenueGoal
  :lineno-match:

The ``MinCostGoal`` minimizes the state which is passed to the goal. For this example the class is 
defined as follows.

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: MinCostGoal
  :lineno-match:

Next, we construct the class by declaring it and inheriting the desired parent
classes.

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :lines: 51-59
  :lineno-match:

The pre-processing is called before any goals are added. Here we define a power_nominal.
This is the average of the ``Target_Power`` timeseries and is used to improve the scaling 
of the model. 

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: PumpStorage.pre
  :lineno-match:

Constraints can be declared by the ``path_constraints()`` method.
Path constraints are constraints that are applied every timestep. To set a
constraint at an individual timestep, define it inside the ``constraints``
method. Four constraints are added using the Big-M formulation to ensure the pumps and Turbine 
joining the upper and lower reservoirs cannot be used at the same time. 

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: PumpStorage.path_constraints
  :lineno-match:

Variable nominals can be set in the OpenModelica model file, or via the variable_nominal definition. 
This can be useful when a variable nominal depends on a non-fixed value. Here the 
power_nominal caluclated in the ``pre`` is set to be the nominal for the states ``TotalGeneratingPower``
and ``TurbinePower``.

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: PumpStorage.variable_nominal
  :lineno-match:

The optimization objectives are added in ``path_goals``. There are four goal in order of priority:

* Priority 10

  * Goal to meet target power

* Priority 20

  * Goal to minimize the spill of the lower reservoir

* Priority 25

  * Minimize the cost associated with operating the pump

* Priority 30

  * Maximize the revenue generated from using turbines at the upper and lower reservoir

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: PumpStorage.path_goals
  :lineno-match:

We want to apply some additional configuration, choosing the HiGHS solver over the 
default choice of bonmin for solving the mixed integer optimization problem. :

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: PumpStorage.solver_options
  :lineno-match:

Finally, we want to print the Total revenue of the system. This can be done in the ``post``.

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :pyobject: PumpStorage.post
  :lineno-match:

Run the Optimization Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_optimization_problem()`` method we imported on the optimization
problem class we just created.

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../../../examples/pumped_hydropower_system/src/example.py
  :language: python
  :lineno-match:

Running the Optimization Problem
--------------------------------

To run the model, run the python file. ``/examples/pumped_hydropower_system/src/example.py``.

Extracting Results
------------------

The results from the run are found in ``output/timeseries_export.csv``. Any
CSV-reading software can import it, but this is how results can be plotted using
the python library matplotlib:


.. plot:: examples/pyplots/pumped_hydropower_system_results.py
   :include-source:


.. _pumped_hydropower_system_results:

Observations
------------

The target power is met and to do so, the upper reservoir (pumped storage) is used. 
As we want to minimize the cost of using the pump, the pump is used only when the cost is low.
