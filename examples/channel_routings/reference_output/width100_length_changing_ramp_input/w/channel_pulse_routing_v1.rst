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
          
          Caption for image 1
          
     - .. figure:: ../../images/routing/SVcomparison_routing_0.0_50000.png
          :width: 95%
          
          Caption for image 1

     - 

Comparison of the routing model responses to an upstream discharge increase


How are these models used in RTC-Tools? Make a picture explaining it, and the consequences of the modelling. Suppose that you increased the upstream discharge by 50m3/s and you have the resulting water level increase as shown in :ref:`comparison_routing`. Now we need to think backwards. As we want to control the water level, we ask, how much extra dicharge do w need to reaise the water level by 2.6 cm? The "reality" (the best approximation) is the Saint-Venant equations. This shows that if the discharge increases 50m36s hte water level first increases 3 cm and settles at an increaes around 2.6 cm. If we just care about the final water level and we do have some buffer to catch that wave, the answer is 50m/s. If we do not have any buffer and want to avoid that the water level in any moment goes over the desired level, probably it is somewhat less than 50m3/s. Suppose we do have that small buffer. What happens in our controller? He uses his estimation of water level to find the right discharge. Using Linearized Saint-Venant, he will ask slightly more discharge. Using ID, he will ask much more discharge and might results in too high water levels in the first step. (He will correct it at the next operation hour, but it might already be too late). Using IDZ, he will underestimate the discharge. He will correct for it in the next moment to act, but the controller might be too slow. As a summary, for a "wavy" channel, to be on the conservative side it is better to use Linearized Saint-Venant (maye IDZ), but not ID.

Let's see an example without waves.

.. _comparison_routing:

.. figure:: ../../images/comparison_routing.png
   :alt: Comparison of routing model responses to an upstream discharge increase
   
   Figure 1



.. image:: ../../images/water_level_response.png





 



