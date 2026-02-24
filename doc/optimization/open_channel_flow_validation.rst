Validation of Current Homotopy Example Model
============================================

This document compares the current implementation of homotopy with the full
high‑resolution non‑linear solution of the SV Equations. For the comparison,
the simple example available in RTC‑Tools is used.

The channel data is the following::

    H_b_down = 0
    H_b_up = 2
    Q_nominal = 100
    friction_coefficient = 0.02
    length = 10000
    width = 30

The test case is the same presented in :ref:`channel_pulse`: an upstream and
downstream wave of 150 m³/s is given and the water levels are compared.

Four homotopy scenarios are evaluated against the SOBEK model, combining full (10‑node) and sparse (2‑node, typically used operationally) spatial discretizations with different linearization starting points, where the solution begins at :math:\theta = 0. Because the full non‑linear equation is solved, the choice of the second linearization point should, in principle, have no influence on the final solution. Thus the four cases:

- 10 nodes, nominal level 3 m  
- 10 nodes, nominal level 4 m  
- 2 nodes, nominal level 3 m  
- 2 nodes, nominal level 4 m  

The downstream and upstream water levels are shown in the following figures.

   
.. _Upstream_150:

.. figure:: ../images/validation/Upstream_150.png
   :alt: Upstream_150

.. _Downstream_150:

.. figure:: ../images/validation/Downstream_150.png
   :alt: Downstream_150


-------------------------
Analysis
-------------------------


The upstream levels are represented well with good discretization. Sobek
calculates a 0.59 m water level change, while well‑discretized homotopy
calculates 0.60 m water level change. The sparse‑discretized homotopy
calculates 0.94 m. This shows the very **strong influence of the discretization**.

The different **linear starting point has negligible influence** when using two
discretization points, and only a slight influence when using 10 points. 
Downstream we see similar results. There is no influence of the starting point
of the discretization.

.. note::

    To avoid errors due to this phenomenon (which could
    also be investigated), we gave an initial state that is the same as the Sobek
    model.



-------------------------
Summary
-------------------------

The model is highly sensitive to spatial discretization, so it is important to consider 
the effects that discretization may have on the results.
The model is also sensitive to the choice of the linearization starting point, 
especially when it needs to compute large changes in discharge. The initial steady-state solution often differs from that of SOBEK and depends on the point at which the linearization begins.