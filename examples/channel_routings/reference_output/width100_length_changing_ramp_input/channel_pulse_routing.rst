Modeling Waves with different routing algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous section we saw that RTC-Tools is able to handle non-linear hydraulics
using non-linear optimization. However, it is also possible to approximate the non-linear
routing with different linearizations / approximations. There are two type approximations in RTC-Tools:
blocks direcly providing water level or blocks only providing discharge. This section deals with the first options.

In this example we compare 4 routing methods:
 * Full non-linear Saint-Venant (as reference)
 * Built in linear block for Saint-Venant (deprecated)
 * Linearized Saint-Venant
 * Integrator Delay Zero


In this section we explain how to set up a model with each method and we discuss
the advantages and disadvantages.

First of all, you should have an idea of the responde of your channel between the points you want to control and the points are taking action. In this example we want to control the upstream water level by changing the upstream flow. (Most of the time there is a distance between these two points). Plot how your channel reacts if you immediately raise the control (e.g. discharge). Probably the controlled variable (here water level) will start to move after some time, will move up to a point and either settles there or moves down again and settles after some wavy movements. If the time it settles to a final level is smaller then the time step of your controller, then use ID model. If not, then there is a point to look into the dynamical models.

Depending on the reaction of the response, the channel can have three different kind of respondes: wavy firt order, first order, second order, second order with delay depending on the "relative length" of the channel. Based on the publicaiton *, for the first two types you can use IDZ, and for the last two types you can use Linearized Saint-Venant. How to check which category your channel belongs? Either inspect the plots, or calculate the give constants in *.

Here should come a picture with the 4 different responses.

.. list-table::
   :widths: 33 33 33
   :align: center

   * - .. figure:: ../../images/routing/SVcomparison_routing_0.0_5000.png
          :width: 95%
          
          Caption for image 1
          
     - .. figure:: ../../images/routing/SVcomparison_routing_0.0_10000.png
          :width: 95%
          
          Caption for image 1
          
     - .. figure:: ../../images/routing/SVcomparison_routing_0.0_20000.png
          :width: 95%

          Caption for image 1
          
   * - .. figure:: ../../images/routing/SVcomparison_routing_0.0_30000.png
          :width: 95%
          
     - .. figure:: ../../images/routing/SVcomparison_routing_0.0_50000.png
          :width: 95%
          
          Caption
          
     - .. figure:: ../../images/routing/SVcomparison_routing_0.0_50000.png

   Comparison of the routing model responses to an upstream discharge increase





How are these models used in RTC-Tools? Make a picture explaining it, and the consequences of the modelling. Suppose that you increased the upstream discharge by 50m3/s and you have the resulting water level increase as shown in :ref:`comparison_routing`. Now we need to think backwards. As we want to control the water level, we ask, how much extra dicharge do w need to reaise the water level by 2.6 cm? The "reality" (the best approximation) is the Saint-Venant equations. This shows that if the discharge increases 50m36s hte water level first increases 3 cm and settles at an increaes around 2.6 cm. If we just care about the final water level and we do have some buffer to catch that wave, the answer is 50m/s. If we do not have any buffer and want to avoid that the water level in any moment goes over the desired level, probably it is somewhat less than 50m3/s. Suppose we do have that small buffer. What happens in our controller? He uses his estimation of water level to find the right discharge. Using Linearized Saint-Venant, he will ask slightly more discharge. Using ID, he will ask much more discharge and might results in too high water levels in the first step. (He will correct it at the next operation hour, but it might already be too late). Using IDZ, he will underestimate the discharge. He will correct for it in the next moment to act, but the controller might be too slow. As a summary, for a "wavy" channel, to be on the conservative side it is better to use Linearized Saint-Venant (maye IDZ), but not ID.

Let's see an example without waves.

.. _comparison_routing:

.. figure:: ../../images/comparison_routing.png
   :alt: Comparison of routing model responses to an upstream discharge increase
   
   Figure 1



.. image:: ../../images/water_level_response.png





 



Decision Tree
-------------

Decision flow for routing models::

    Start
    |
    +-- Is your model mixed-integer?
    |   |
    |   +-- Yes → Use one of the linear routings
    |   |
    |   +-- No
    |       |
    |       +-- Do you use ensembles?
    |           |
    |           +-- Yes → Use one of the linear routings
    |           |
    |           +-- No → Do you have concerns about runtime?
    |               |
    |               +-- Yes → Use one of the linear routings
    |               |
    |               +-- No → Use homotopy

The Model
---------

In this example, water is flowing through a single channel. There is an inflow
at the upstream end and a water level bound at the downstream end.

In OpenModelica Connection Editor, the model looks like this in plan view:

.. image:: ../../images/single_channel.png

In text mode, the Modelica model looks as follows (with annotation statements
removed):

.. literalinclude:: ../../_build/mo/channel_pulse.mo
  :language: modelica
  :lineno-match:

The plan view of the model looks like this in HEC-RAS:

.. image:: ../../images/single_channel_hec-ras.png

The channel cross-section is a simple trapezoidal shape. As rendered by HEC-RAS,
here is a cross-section view of the channel being modeled:

.. image:: ../../images/channel_impulse_xs.png

The model was built with HEC-RAS version 5.0.6. In case you wish to verify the
HEC-RAS model yourself, a zip of the HEC-RAS model used in this comparison is
available: :download:`HEC-RAS.zip <../../../examples/channel_pulse/HEC-RAS.zip>`


The Python File
---------------

To keep this example simple and to allow for a 1:1 comparison with HEC-RAS, we
will not have any decision variables in this model. 

.. literalinclude:: ../../../examples/channel_pulse/src/example.py
  :language: python
  :lineno-match:

As you can see, this model is as simple as it gets. We only add a constraint to
keep the initialization states consistent with the HEC-RAS initialization.


Comparison of Discretizations and Numerical Schemes
---------------------------------------------------

HEC-RAS and RTC-Tools use different discretizations and numerical schemes, but also
solve different equations.  RTC-Tools solves the original nonlinear equations, whereas
HEC-RAS `solves a linearized momentum equation <http://www.hec.usace.army.mil/software/hec-ras/documentation/HEC-RAS%205.0%20Reference%20Manual.pdf>`_.

+-----------------------------+----------------------------------------------------------------+--------------------------------+ 
|                             | RTC-Tools 2                                                    | HEC-RAS                        | 
+=============================+================================================================+================================+ 
| Momentum equation           | Saint-Venant / inertial wave (default)                         | Linearized Saint-Venant        | 
+-----------------------------+----------------------------------------------------------------+--------------------------------+ 
| Spatial discretization      | Staggered                                                      | Collocated                     | 
+-----------------------------+----------------------------------------------------------------+--------------------------------+ 
| Numerical scheme (temporal) | Semi-implicit / implicit (default)                             | Centered Preissmann box scheme | 
+-----------------------------+----------------------------------------------------------------+--------------------------------+ 
| Numerical scheme (spatial)  | Central differences, upwind convective acceleration (optional) | Centered Preissmann box scheme | 
+-----------------------------+----------------------------------------------------------------+--------------------------------+ 

.. note::

    For optimization, the recommended momentum equation and temporal scheme for RTC-Tools is 
    semi-implicit inertial wave.  Consult Baayen and Piovesan,
    *A continuation approach to nonlinear model predictive control of open channel systems*,
    2018, for details.  A preprint is available online as 
    `arXiv:1801.06507 <https://arxiv.org/abs/1801.06507>`_.


Comparison of Results
---------------------

The results from the RTC-Tools run are found in the output directory with the
name ``timeseries_export.csv``, and the results generated by HEC-RAS have been
exported into the same directory under the name ``HEC-RAS_results.csv``. We can
compare the results using the Python library ``matplotlib``:

.. plot:: examples/pyplots/channel_pulse_results.py

Both HEC-RAS and RTC-Tools were run with a spatial step size of 1000 m and a 
temporal step size of 15 min.