Using a custom solver version
=============================

Setting the solver path
-----------------------

CasADi searches its own installation directory before ``CASADIPATH``, so
``CASADIPATH`` alone cannot reliably override a bundled solver. ``RTCTOOLS_EXTRA_CASADIPATH``
is used instead — it is prepended to CasADi's search path before the solver is
instantiated, giving it the highest priority.

``RTCTOOLS_EXTRA_CASADIPATH`` is read by :func:`.run_optimization_problem` and
:func:`.run_simulation_problem`. If you instantiate a problem class directly, configure
the path yourself before solving using ``casadi.GlobalOptions.getCasadiPath()`` and
``casadi.GlobalOptions.setCasadiPath()`` — read the current path first and prepend to it,
rather than replacing it outright, to preserve the CasADi install directory.

When the path is set, RTC-Tools logs the active CasADi plugin search path at ``INFO``
level so you can confirm the override is in effect.

.. code-block:: bash

   # Linux/macOS
   export RTCTOOLS_EXTRA_CASADIPATH="/path/to/custom/solver/lib"

   # Windows
   set RTCTOOLS_EXTRA_CASADIPATH=C:\path\to\custom\solver\lib

**Windows only:** the solver library (e.g. ``libhighs.dll``) is a transitive dependency
of the plugin wrapper and is resolved by the standard Windows DLL search order (``PATH``).
Set ``PATH`` to include the custom solver directory **before starting the Python process**
so that the correct version is found rather than any copy already present on the system:

.. code-block:: bat

   set PATH=C:\path\to\custom\solver\lib;%PATH%
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

The plugin wrapper is compiled against both the CasADi source (for internal headers)
and the HiGHS headers. The solver library must match the HiGHS version the wrapper was
compiled against — the headers encode vtable layouts and struct sizes that must match
exactly. Using mismatched versions causes silent data corruption or crashes at runtime.
The bundled CasADi wrapper cannot be reused for a different HiGHS version — always
build a new one.

.. note::

   If the plugin directory exists but the wrapper fails to load (e.g. ABI mismatch or a
   missing transitive DLL on Windows), CasADi reports the error when the solver is
   instantiated, not when the path is configured. Check the CasADi error output in that
   case.
