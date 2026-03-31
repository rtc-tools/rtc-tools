Solver Configuration
====================

CasADi searches for solver plugins (e.g. HiGHS, IPOPT) in the directories listed in
``CASADIPATH``, but only after its own installation directory. This means it is not
possible to override a bundled solver using ``CASADIPATH`` alone.

To prepend a custom directory to CasADi's plugin search path, set the
``RTCTOOLS_EXTRA_CASADIPATH`` environment variable:

.. code-block:: bash

   # Linux/macOS
   export RTCTOOLS_EXTRA_CASADIPATH="/path/to/custom/solver/lib"

   # Windows
   set RTCTOOLS_EXTRA_CASADIPATH=C:\path\to\custom\solver\lib

This is equivalent to ``CASADIPATH``, but with higher priority. The directory is
prepended to CasADi's search path by ``run_optimization_problem()`` and
``run_simulation_problem()`` before the solver is instantiated.

The directory must contain a **CasADi plugin wrapper** for the solver (e.g.
``libcasadi_conic_highs.so`` on Linux or ``libcasadi_conic_highs.dll`` on Windows).
This is the shared library that CasADi loads to interface with the solver — placing
only the raw solver library (e.g. ``libhighs.so``) in the directory is not sufficient.

To use a different solver version than the one bundled with CasADi, place the new solver
library alongside a compatible plugin wrapper in the directory. If the new solver version
is ABI-compatible with the bundled one, the existing wrapper may work as-is. Otherwise,
you need to build a new plugin wrapper using the same toolchain as your CasADi
installation, linked against the new solver version.
