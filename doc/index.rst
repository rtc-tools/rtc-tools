.. RTC-Tools documentation master file, created by
   sphinx-quickstart on Wed Jul 13 15:02:02 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. highlight:: python

.. image:: images/rtc-tools-icon-color.png
	:align: right
	:scale: 50%

RTC-Tools documentation
=======================

RTC-Tools is an open-source Python package designed to model, simulate,
and optimize networks or portfolios of assets, such as 
reservoirs, pumps, renewables, and batteries. It is part 
of `LF Energy <https://lfenergy.org/>`_.

RTC-Tools provides a modular and extensible framework for operational
optimization and control of complex systems across multiple domains, with a
primary focus on water management and energy systems. Originally initiated at
`Deltares <https://www.deltares.nl/>`_ in 2015, RTC-Tools is deployed for
water and power trading applications globally, with active implementations in North and South
America, Europe, Asia, and Australia.

Application domains include:

- **Water Management**: Model-predictive control of canals, polders,
  reservoirs, hydropower scheduling, and pumped storage systems, including
  turbines, pumps, weirs, and other hydraulic structures.
- **Energy Systems**: Battery Energy Storage System (BESS) and portfolio optimization,
  heat network design, power trading, and multi-energy system planning.

Community Support
-----------------

For questions, issues, and discussions, please use:

- `GitHub Issues <https://github.com/rtc-tools/rtc-tools/issues>`_ for bug
  reports and feature requests
- `GitHub Discussions <https://github.com/rtc-tools/rtc-tools/discussions>`_
  for questions and community discussions

Commercial Support
------------------

For applications in water management and hydropower, `Deltares <https://www.deltares.nl/>`_
offers commercial support.

For applications in power trading and battery energy storage systems (BESS),
`PortfolioEnergy <https://www.portfolioenergy.com/>`_ offers commercial
support.

Governance & Roadmap
--------------------

RTC-Tools development follows an open governance model. To learn more about
the project roadmap and contribute to future development:

- Review the `roadmap discussion <https://github.com/rtc-tools/rtc-tools/discussions/1725>`_
- Check `project milestones <https://github.com/rtc-tools/rtc-tools/milestones>`_
- Watch the `feature overview video <https://www.youtube.com/watch?v=GAuQ5ft8vr4>`_

Documentation Overview
----------------------

The first chapter of this documentation covers getting the software running
on your computer. The subsequent chapters describe the RTC-Tools Python API
for optimization and simulation. The examples chapter illustrates several key
features, including multi-objective optimization using weighting and
lexicographic goal programming methods, as well as optimization under
uncertainty using ensemble forecasts.



Contents
========

.. toctree::
   :maxdepth: 2

   getting-started
   optimization
   simulation
   examples
   related-projects


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
