Using a custom solver version
=============================

Setting the solver path
-----------------------

CasADi always searches its own installation directory first for solver plugins, so
``CASADIPATH`` alone cannot override a bundled solver. Use ``RTCTOOLS_EXTRA_CASADIPATH``
instead — it is prepended to CasADi's search path before the solver is instantiated,
giving it the highest priority.

``RTCTOOLS_EXTRA_CASADIPATH`` is read by :func:`.run_optimization_problem` and
:func:`.run_simulation_problem`. If you instantiate a problem class directly, configure
the path yourself before solving via ``casadi.GlobalOptions.setCasadiPath()``.

.. code-block:: bash

   # Linux/macOS
   export RTCTOOLS_EXTRA_CASADIPATH="/path/to/custom/solver/lib"

   # Windows
   set RTCTOOLS_EXTRA_CASADIPATH=C:\path\to\custom\solver\lib

**Windows only:** the plugin wrapper depends on other DLLs from the CasADi installation
(e.g. ``libcasadi.dll``), which Windows resolves via ``PATH``. Add the CasADi
installation directory to ``PATH`` if the plugin fails to load:

.. code-block:: bat

   set PATH=C:\path\to\casadi;%PATH%
   set RTCTOOLS_EXTRA_CASADIPATH=C:\path\to\custom\solver\lib

Custom solver folder contents
-----------------------------

The directory must contain the solver library and the CasADi plugin wrapper.
For HiGHS as an example:

.. list-table::
   :header-rows: 1

   * - Linux
     - Windows
     - Role
   * - ``libhighs.so``
     - ``libhighs.dll``
     - Solver library
   * - ``libcasadi_conic_highs.so``
     - ``libcasadi_conic_highs.dll``
     - CasADi plugin wrapper

The bundled plugin wrapper can be reused if it is ABI-compatible with the new solver
version; otherwise build a new one with the same toolchain as your CasADi installation.

.. note::

   If the plugin directory exists but the wrapper fails to load (e.g. ABI mismatch or a
   missing transitive DLL on Windows), CasADi reports the error when the solver is
   instantiated, not when the path is configured. Check the CasADi error output in that
   case.
