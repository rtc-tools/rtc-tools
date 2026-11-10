.. _custom equations:

Custom equations
~~~~~~~~~~~~~~~~

.. note::
  This method is used to add equations to the governing system of equations,
  e.g., the system of equations describing the physics.
  For computing input variables such as control inputs in the Python code,
  see the example :ref:`tracking a setpoint`.

This example illustrates how to implement custom equations in Python.
This could be used to implement equations that are not supported by Modelica.

Consider the following model:

.. math::
    \frac{dx}{dt} &= y, \\
    y &= -\frac{2}{3600}x.

The equation for ``x`` is given in the Modelica file,
while the equation for ``y`` is implemented in the Python code.

The Modelica file is given below.

.. literalinclude:: ../../../examples/simulation_with_custom_equations/model/SimpleModel.mo
  :language: modelica

To add custom equations to the simulation problem,
the method ``extra_equations`` is overwritten.
This is illustrated in the optimization problem below.

.. literalinclude:: ../../../examples/simulation_with_custom_equations/src/simple_model.py
  :language: python
  :pyobject: SimpleModel

The method ``extra_equations`` returns a list of equations given in the form of residuals,
i.e. expressions that should equal zero.
For the equation :math:`y = -(2/3600) x`, for example,
the residual is of the form :math:`(y - (-(2/3600) x)) / \text{some_nominal}`.
