Using Delay Equations
~~~~~~~~~~~~~~~~~~~~~

RTC-Tools supports using delay equations.
This works for both optimization and simulation problems,
although for simulation,
this currently only works for a constant delay time and a constant timestep size.

To illustrate the use of delay equations,
we consider the following integrator-delay model.

.. image:: ../../images/integrator_delay_modelica.png

.. literalinclude:: ../../../examples/integrator_delay/model/Example.mo
  :language: modelica

This model describes two inflows that come together.
One of the inflows can be controlled
and has a delay of 24 hours before joining the other inflow.

For the optimization problem,
the goal is to keep the volume at a constant level of 400,000 cubic meter.
The problem is given below.

.. literalinclude:: ../../../examples/integrator_delay/src/example.py
  :language: python
  :pyobject: ExampleOpt

The desired outflow is set to be 1.0 cubic meter / second.
The fixed inflow is usually 0.0 except at a few moments.
To get the desired output, the ideal controllable inflow
is the outflow minus the fixed inflow shifted 24 hours in advance.
The results are illustrated in the plot below.

.. plot:: examples/pyplots/integrator_delay.py

Note that the control inflow is 0.0 in the last 24 hours.
The reason is that these values do not impact the outflow for the given time period anymore,
due to the delay time of 24 hours.

For the simulation problem, we use the same model,
but the control inflow is now given as input
and is the same as the optimized inflow given by optimization problem.
The simulation problem is given below.

.. literalinclude:: ../../../examples/integrator_delay/src/example.py
  :language: python
  :pyobject: ExampleSim

Note that for the simulation problem, we have to set the option fixed_dt.
This is because currently optimization problems with delay equations
only work for a fixed timestep size.

Since the model and inputs are the same,
the volume output is exactly the same for both the simulation and optimization problem.
