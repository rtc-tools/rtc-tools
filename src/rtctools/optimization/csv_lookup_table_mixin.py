import configparser
import glob
import logging
import os
from collections.abc import Iterable

import casadi as ca
import numpy as np
from scipy.interpolate import bisplev, bisplrep, splev
from scipy.optimize import brentq

import rtctools.data.csv as csv
from rtctools._internal.caching import cached
from rtctools.data.interpolation.bspline1d import BSpline1D
from rtctools.data.interpolation.bspline2d import BSpline2D
from rtctools.optimization.timeseries import Timeseries

from .optimization_problem import LookupTable as LookupTableBase
from .optimization_problem import OptimizationProblem

logger = logging.getLogger("rtctools")


class LookupTable(LookupTableBase):
    """
    Lookup table.
    """

    def __init__(self, inputs: list[ca.MX], function: ca.Function, tck: tuple = None):
        """
        Create a new lookup table object.

        :param inputs: List of lookup table input variables.
        :param function: Lookup table CasADi :class:`Function`.
        """
        self.__inputs = inputs
        self.__function = function

        self.__t, self.__c, self.__k = [None] * 3

        if tck is not None:
            if len(tck) == 3:
                self.__t, self.__c, self.__k = tck
            elif len(tck) == 5:
                self.__t = tck[:2]
                self.__c = tck[2]
                self.__k = tck[3:]

    @property
    @cached
    def domain(self) -> tuple:
        t = self.__t
        if t is None:
            raise AttributeError(
                "This lookup table was not instantiated with tck metadata. \
                                  Domain/Range information is unavailable."
            )
        if isinstance(t, tuple) and len(t) == 2:
            raise NotImplementedError(
                "Domain/Range information is not yet implemented for 2D LookupTables"
            )

        return np.nextafter(t[0], np.inf), np.nextafter(t[-1], -np.inf)

    @property
    @cached
    def range(self) -> tuple:
        return self(self.domain[0]), self(self.domain[1])

    @property
    def inputs(self) -> list[ca.MX]:
        """
        List of lookup table input variables.
        """
        return self.__inputs

    @property
    def function(self) -> ca.Function:
        """
        Lookup table CasADi :class:`Function`.
        """
        return self.__function

    @property
    @cached
    def __numeric_function_evaluator(self):
        return np.vectorize(
            lambda *args: np.nan if np.any(np.isnan(args)) else float(self.function(*args))
        )

    def __call__(self, *args: float | Iterable | Timeseries) -> float | np.ndarray | Timeseries:
        """
        Evaluate the lookup table.

        :param args: Input values.
        :type args: Float, iterable of floats, or :class:`.Timeseries`
        :returns: Lookup table evaluated at input values.

        Example use::

            y = lookup_table(1.0)
            [y1, y2] = lookup_table([1.0, 2.0])

        """
        evaluator = self.__numeric_function_evaluator
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Timeseries):
                return Timeseries(arg.times, self(arg.values))
            else:
                if hasattr(arg, "__iter__"):
                    arg = np.fromiter(arg, dtype=float)
                    return evaluator(arg)
                else:
                    arg = float(arg)
                    return evaluator(arg).item()
        else:
            if any(isinstance(arg, Timeseries) for arg in args):
                raise TypeError(
                    "Higher-order LookupTable calls do not yet support Timeseries parameters"
                )
            elif any(hasattr(arg, "__iter__") for arg in args):
                raise TypeError(
                    "Higher-order LookupTable calls do not yet support vector parameters"
                )
            else:
                args = np.fromiter(args, dtype=float)
                return evaluator(*args)

    def reverse_call(
        self,
        y: float | Iterable | Timeseries,
        domain: tuple[float, float] = (None, None),
        detect_range_error: bool = True,
    ) -> float | np.ndarray | Timeseries:
        """Do an inverted call on this LookupTable

        Uses SciPy brentq optimizer to simulate a reversed call.
        Note: Method does not work with higher-order LookupTables
        """
        if isinstance(y, Timeseries):
            # Recurse and return
            return Timeseries(y.times, self.reverse_call(y.values))

        # Get domain information
        l_d, u_d = domain
        if l_d is None:
            l_d = self.domain[0]
        if u_d is None:
            u_d = self.domain[1]

        # Cast y to array of float
        if hasattr(y, "__iter__"):
            y_array = np.fromiter(y, dtype=float)
        else:
            y_array = np.array([y], dtype=float)

        # Find not np.nan
        is_not_nan = ~np.isnan(y_array)
        y_array_not_nan = y_array[is_not_nan]

        # Detect if there is a range violation
        if detect_range_error:
            l_r, u_r = self.range
            lb_viol = y_array_not_nan < l_r
            ub_viol = y_array_not_nan > u_r
            all_viol = y_array_not_nan[lb_viol | ub_viol]
            if all_viol.size > 0:
                raise ValueError(f"Values {all_viol} are not in lookup table range ({l_r}, {u_r})")

        # Construct function to do inverse evaluation
        evaluator = self.__numeric_function_evaluator

        def inv_evaluator(y_target):
            """inverse evaluator function"""
            return brentq(lambda x: evaluator(x) - y_target, l_d, u_d)

        inv_evaluator = np.vectorize(inv_evaluator)

        # Calculate x_array
        x_array = np.full_like(y_array, np.nan, dtype=float)
        if y_array_not_nan.size != 0:
            x_array[is_not_nan] = inv_evaluator(y_array_not_nan)

        # Return x
        if hasattr(y, "__iter__"):
            return x_array
        else:
            return x_array.item()


class CSVLookupTableMixin(OptimizationProblem):
    """
    Adds lookup tables to your optimization problem.

    During preprocessing, the CSV files located inside the ``lookup_tables`` subfolder are read. In
    every CSV file, the first column contains the output of the lookup table. Subsequent columns
    contain the input variables.

    Cubic B-Splines are used to turn the data points into continuous lookup tables.

    Optionally, a file ``curvefit_options.ini`` may be included inside the ``lookup_tables`` folder.
    This file contains, grouped per lookup table, the following options:

    * monotonicity:
        * is an integer, magnitude is ignored
        * if positive, causes spline to be monotonically increasing
        * if negative, causes spline to be monotonically decreasing
        * if 0, leaves spline monotonicity unconstrained

    * curvature:
        * is an integer, magnitude is ignored
        * if positive, causes spline curvature to be positive (convex)
        * if negative, causes spline curvature to be negative (concave)
        * if 0, leaves spline curvature unconstrained

    .. note::

        Currently only one-dimensional lookup tables are fully supported.  Support for two-
        dimensional lookup tables is experimental.

    :cvar csv_delimiter:  Column delimiter used in CSV files. Default is ``,``.
    :cvar csv_lookup_table_debug:  Whether to generate plots of the spline fits.
        Default is ``False``.
    :cvar csv_lookup_table_debug_points:  Number of evaluation points for plots.
        Default is ``100``.
    """

    #: Column delimiter used in CSV files
    csv_delimiter = ","

    #: Debug settings
    csv_lookup_table_debug = False
    csv_lookup_table_debug_points = 100

    def __init__(self, **kwargs):
        # Check arguments
        if "input_folder" in kwargs:
            assert "lookup_table_folder" not in kwargs

            self.__lookup_table_folder = os.path.join(kwargs["input_folder"], "lookup_tables")
        else:
            self.__lookup_table_folder = kwargs["lookup_table_folder"]

        # Call parent
        super().__init__(**kwargs)

    def pre(self):
        # Call parent class first for default behaviour.
        super().pre()

        # Get curve fitting options from curvefit_options.ini file
        ini_path = os.path.join(self.__lookup_table_folder, "curvefit_options.ini")
        try:
            ini_config = configparser.RawConfigParser()
            ini_config.read(ini_path)
            no_curvefit_options = False
        except OSError:
            logger.info(
                "CSVLookupTableMixin: No curvefit_options.ini file found. Using default values."
            )
            no_curvefit_options = True

        def get_curvefit_options(curve_name, no_curvefit_options=no_curvefit_options):
            if no_curvefit_options:
                return 0, 0, 0

            curvefit_options = []

            def get_property(prop_name):
                try:
                    prop = int(ini_config.get(curve_name, prop_name))
                except configparser.NoSectionError:
                    prop = 0
                except configparser.NoOptionError:
                    prop = 0
                except ValueError:
                    raise Exception(
                        f"CSVLookupTableMixin: Invalid {prop_name} constraint for {curve_name}. "
                        f"{prop_name} should be either -1, 0, or 1."
                    )
                return prop

            for prop_name in ["monotonicity", "monotonicity2", "curvature"]:
                curvefit_options.append(get_property(prop_name))

            logger.debug(
                "CSVLookupTableMixin: Curve fit option for {}:({},{},{})".format(
                    curve_name, *curvefit_options
                )
            )
            return tuple(curvefit_options)

        def check_lookup_table(lookup_table):
            if lookup_table in self.__lookup_tables:
                raise Exception(
                    f"Cannot add lookup table {lookup_table}, "
                    "since there is already one with this name."
                )

        # Read CSV files
        logger.info("CSVLookupTableMixin: Generating Splines from lookup table data.")
        self.__lookup_tables = {}
        for filename in glob.glob(os.path.join(self.__lookup_table_folder, "*.csv")):
            logger.debug(f"CSVLookupTableMixin: Reading lookup table from {filename}")

            csvinput = csv.load(filename, delimiter=self.csv_delimiter)
            output = csvinput.dtype.names[0]
            inputs = csvinput.dtype.names[1:]

            # Get monotonicity and curvature from ini file
            mono, mono2, curv = get_curvefit_options(output)

            logger.debug(f"CSVLookupTableMixin: Output is {output}, inputs are {inputs}.")

            tck = None

            # If tck file is newer than the csv file, first try to load the cached values from
            # the tck file. The CasADi function is always rebuilt from the tck values; serialized
            # CasADi functions are not loaded, as these can reference external shared libraries.
            tck_filename = filename.replace(".csv", ".npz")
            valid_cache = False
            if os.path.exists(tck_filename):
                if no_curvefit_options:
                    valid_cache = os.path.getmtime(filename) < os.path.getmtime(tck_filename)
                else:
                    valid_cache = (
                        os.path.getmtime(filename) < os.path.getmtime(tck_filename)
                    ) and (os.path.getmtime(ini_path) < os.path.getmtime(tck_filename))
                if valid_cache:
                    logger.debug(
                        f"CSVLookupTableMixin: Attempting to use cached tck values for {output}"
                    )
                    try:
                        with np.load(filename.replace(".csv", ".npz")) as data:
                            tck = (data["arr_0"], data["arr_1"], int(data["arr_2"]))
                    except Exception:
                        valid_cache = False

            if not valid_cache:
                logger.info(f"CSVLookupTableMixin: Recalculating tck values for {output}")

            if len(csvinput.dtype.names) == 2:
                if not valid_cache:
                    k = 3  # default value
                    # 1D spline fitting needs k+1 data points
                    if len(csvinput[output]) >= k + 1:
                        tck = BSpline1D.fit(
                            csvinput[inputs[0]],
                            csvinput[output],
                            k=k,
                            monotonicity=mono,
                            curvature=curv,
                            ipopt_options={"nlp_scaling_method": "none"},
                        )
                    else:
                        raise Exception(
                            f"CSVLookupTableMixin: Too few data points in {filename} "
                            f"to do spline fitting. Need at least {k + 1} points."
                        )

                if self.csv_lookup_table_debug:
                    import pylab

                    i = np.linspace(
                        csvinput[inputs[0]][0],
                        csvinput[inputs[0]][-1],
                        self.csv_lookup_table_debug_points,
                    )
                    o = splev(i, tck)
                    pylab.clf()
                    # TODO: Figure out why cross-section B0607 in NZV does not
                    # conform to constraints!
                    pylab.plot(i, o)
                    pylab.plot(
                        csvinput[inputs[0]],
                        csvinput[output],
                        linestyle="",
                        marker="x",
                        markersize=10,
                    )
                    figure_filename = filename.replace(".csv", ".png")
                    pylab.savefig(figure_filename)

                symbols = [ca.SX.sym(inputs[0])]
                function = ca.Function("f", symbols, [BSpline1D(*tck)(symbols[0])])
                check_lookup_table(output)
                self.__lookup_tables[output] = LookupTable(symbols, function, tck)

            elif len(csvinput.dtype.names) == 3:
                if tck is None:
                    kx = 3  # default value
                    ky = 3  # default value

                    # 2D spline fitting needs (kx+1)*(ky+1) data points
                    if len(csvinput[output]) >= (kx + 1) * (ky + 1):
                        # TODO: add curvature parameters from curvefit_options.ini
                        # once 2d fit is implemented
                        tck = bisplrep(
                            csvinput[inputs[0]], csvinput[inputs[1]], csvinput[output], kx=kx, ky=ky
                        )
                    else:
                        raise Exception(
                            f"CSVLookupTableMixin: Too few data points in {filename} "
                            f"to do spline fitting. Need at least {(kx + 1) * (ky + 1)} points."
                        )

                if self.csv_lookup_table_debug:
                    import pylab

                    i1 = np.linspace(
                        csvinput[inputs[0]][0],
                        csvinput[inputs[0]][-1],
                        self.csv_lookup_table_debug_points,
                    )
                    i2 = np.linspace(
                        csvinput[inputs[1]][0],
                        csvinput[inputs[1]][-1],
                        self.csv_lookup_table_debug_points,
                    )
                    i1, i2 = np.meshgrid(i1, i2)
                    i1 = i1.flatten()
                    i2 = i2.flatten()
                    o = bisplev(i1, i2, tck)
                    pylab.clf()
                    pylab.plot_surface(i1, i2, o)
                    figure_filename = filename.replace(".csv", ".png")
                    pylab.savefig(figure_filename)
                symbols = [ca.SX.sym(inputs[0]), ca.SX.sym(inputs[1])]
                function = ca.Function("f", symbols, [BSpline2D(*tck)(symbols[0], symbols[1])])
                check_lookup_table(output)
                self.__lookup_tables[output] = LookupTable(symbols, function, tck)

            else:
                raise Exception(
                    f"CSVLookupTableMixin: {len(csvinput.dtype.names)}-dimensional "
                    "lookup tables not implemented yet."
                )

            if not valid_cache:
                np.savez(filename.replace(".csv", ".npz"), *tck)

    def lookup_tables(self, ensemble_member):
        # Call parent class first for default values.
        lookup_tables = super().lookup_tables(ensemble_member)

        # Update lookup_tables with imported csv lookup tables
        lookup_tables.update(self.__lookup_tables)

        return lookup_tables
