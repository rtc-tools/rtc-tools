.. _tracking a setpoint:

Tracking a Setpoint
~~~~~~~~~~~~~~~~~~~

.. image:: ../../images/graig-coch-2117306_640.jpg

.. :href: https://pixabay.com/en/graig-coch-dam-wales-reservoir-uk-2117306/
.. pixabay content is released under a CC0 Public Domain licence - no attribution needed

Overview
--------

The purpose of this example is to understand the technical setup of an RTC-
Tools simulation model, how to run the model, and how to access the
results.

The scenario is the following: A reservoir operator is trying to keep the
reservoir's volume close to a given target volume. They are given a six-day
forecast of inflows given in 12-hour increments. To keep things simple, we
ignore the waterlevel-storage relation of the reservoir and head-discharge
relationships in this example. To make things interesting, the reservoir
operator is only able to release water at a few discrete flow rates, and only
change the discrete flow rate every 12 hours. They have chosen to use the RTC-
Tools simulator to see if a simple proportional controller will be able to
keep the system close enough to the target water volume.

The folder ``<examples directory>\simulation``
contains a complete RTC-Tools simulation problem. An RTC-Tools
directory has the following structure:

* ``input``: This folder contains the model input data. These are several files
  in comma separated value format, ``csv``.
* ``model``: This folder contains the Modelica model. The Modelica model
  contains the physics of the RTC-Tools model.
* ``output``: The folder where the output is saved in the file
  ``timeseries_export.csv``.
* ``src``: This folder contains a Python file. This file contains the
  configuration of the model and is used to run the model .

The Model
---------

The first step is to develop a physical model of the system. The model can be
viewed and edited using the OpenModelica Connection Editor (OMEdit) program.
For how to download and start up OMEdit, see :ref:`getting-started-omedit`.

Make sure to load the Deltares library before loading the example:

#. Load the Deltares library into OMEdit

   * Using the menu bar: *File -> Open Model/Library File(s)*
   * Select ``<library directory>\Deltares\package.mo``

#. Load the example model into OMEdit

   * Using the menu bar: *File -> Open Model/Library File(s)*
   * Select ``<examples directory\simulation\model\Example.mo``

Once loaded, we have an OpenModelica Connection Editor window that looks like
this:

.. image:: ../../images/simple_storage_openmodelica.png


The model ``Example.mo`` represents a simple system with the following
elements:

* a reservoir, modeled as storage element
  ``Deltares.ChannelFlow.SimpleRouting.Storage.Storage``,
* an inflow boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow``,
* an outfall boundary condition
  ``Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal``,
* connectors (black lines) connecting the elements.

You can use the mouse-over feature help to identify the predefined models from
the Deltares library. You can also drag the elements around- the connectors will
move with the elements. Adding new elements is easy- just drag them in from the
Deltares Library on the sidebar. Connecting the elements is just as easy- click
and drag between the ports on the elements.

In text mode, the Modelica model looks as follows (with
annotation statements removed):

.. literalinclude:: ../../_build/mo/simulation.mo
  :language: modelica
  :lineno-match:

The three water system elements (storage, inflow, and outfall) appear under
the ``model Example`` statement. The ``equation`` part connects these three
elements with the help of connections. Note that ``storage`` extends the partial
model ``QSISO`` which contains the connectors ``QIn`` and ``QOut``.
With ``QSISO``, ``storage`` can be connected on two sides. The ``storage``
element also has a variable ``Q_release``, which is the decision variable the
operator controls.

OpenModelica Connection Editor will automatically generate the element and
connector entries in the text text file. Defining inputs and outputs requires
editing the text file directly and assigning the inputs to the appropriate
element variables. For example, ``inflow(Q = Q_in)`` sets the ``Q`` variable
of the ``inflow`` element equal to ``Q_in``.

In addition to elements, the input variables ``Q_in`` and ``P_control`` are
also defined. ``Q_in`` is determined by the forecast and the operator cannot
control it, so we set ``Q_in(fixed = true)``. The actual values of ``Q_in``
are stored in ``timeseries_import.csv``. ``P_control`` is not defined anywhere
in the model or inputs- we will dynamically assign its value every timestep in
the python script, ``\src\example.py``.

Because we want to view the water volume in the storage element in the output
file, we also define an output variable ``storage_V`` and set it equal to the
corresponding state variable ``storage.V``. Ditto for ``Q_release = P_control``.

The Simulation Problem
----------------------

The python script is created and edited in a text editor. In general, the python
script consists of the following blocks:

* Import of packages
* Definition of the simulation problem class

  * Any additional configuration (e.g. overriding methods)

* A run statement

Importing Packages
''''''''''''''''''

Packages are imported using ``from ... import ...`` at the top of the file. In
our script, we import the classes we want the class to inherit, the
package ``run_simulation_problem`` form the ``rtctools.util`` package, and
any extra packages we want to use. For this example, the import block looks
like:

.. literalinclude:: ../../../examples/simulation/src/example.py
  :language: python
  :lines: 1-8
  :lineno-match:

Simulation Problem
''''''''''''''''''

The next step is to define the simulation problem class. We construct the
class by declaring the class and inheriting the desired parent classes. The
parent classes each perform different tasks related to importing and exporting
data and running the simulation problem. Each imported class makes a set of
methods available to the our simulation class.

.. literalinclude:: ../../../examples/simulation/src/example.py
  :language: python
  :pyobject: Example
  :lineno-match:
  :end-before: """

The next, we override any methods where we want to specify non-default
behaviour. In our simulation problem, we want to define a proportional
controller. In its simplest form, we load the current values of the volume and
target volume variables, calculate their difference, and set ``P_control`` to be
as close as possible to eliminating that difference during the upcoming timestep.

.. literalinclude:: ../../../examples/simulation/src/example.py
  :language: python
  :pyobject: Example.update
  :lineno-match:

Run the Simulation Problem
''''''''''''''''''''''''''''

To make our script run, at the bottom of our file we just have to call
the ``run_simulation_problem()`` method we imported on the simulation
problem class we just created.

.. literalinclude:: ../../../examples/simulation/src/example.py
  :language: python
  :lineno-match:
  :start-after: # Run

The Whole Script
''''''''''''''''

All together, the whole example script is as follows:

.. literalinclude:: ../../../examples/simulation/src/example.py
  :language: python
  :lineno-match:

Running RTC-Tools
-----------------

To run this basic example in RTC-Tools, navigate to the basic example ``src``
directory in the RTC-Tools shell and run the example using ``python
example.py``. For more details about using RTC-Tools, see
:ref:`running-rtc-tools`.

Extracting Results
------------------

The results from the run are found in ``output\timeseries_export.csv``. Any
CSV-reading software can import it. Here we used matplotlib to generate this plot.

.. plot:: examples/pyplots/simulation_results.py


Observations
------------

This plot shows that the operator is not able to keep the water level within
the bounds over the entire time horizon. They may need to increase the
controller timestep, use a more complete PID controller, or use model
predictive control such as the RTC-Tools optimization library to get the
results they want.