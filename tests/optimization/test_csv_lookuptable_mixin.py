import logging
import os
from unittest import mock

import casadi as ca
import numpy as np

from rtctools.data.interpolation.bspline1d import BSpline1D
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_lookup_table_mixin import CSVLookupTableMixin
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class Model(
    CSVLookupTableMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    model_name = "ModelWithLookupTable"

    def __init__(self, **kwargs):
        kwargs["input_folder"] = data_path()
        kwargs["output_folder"] = data_path()
        kwargs["model_folder"] = data_path()
        super().__init__(**kwargs)

    def path_objective(self, ensemble_member):
        # Minimize X
        return self.state("x")

    def bounds(self):
        bounds = super().bounds()
        lookup_table_x_prime = self.lookup_tables(0)["x_prime"]
        bounds["x"] = lookup_table_x_prime.domain
        bounds["x_prime"] = 4.0, 10.0
        return bounds

    def path_constraints(self, ensemble_member):
        # Symbolically constrain x_prime to x
        lookup_table_x_prime = self.lookup_tables(0)["x_prime"].function
        return [(lookup_table_x_prime(self.state("x")) - self.state("x_prime"), 0.0, 0.0)]

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class TestCSVLookupMixin(TestCase):
    def setUp(self):
        self.problem = Model()
        self.problem.optimize()
        self.results = self.problem.extract_results()

    def test_numeric_call(self):
        lt_x_prime = self.problem.lookup_tables(0)["x_prime"]
        test_pairs = (
            (0.2, 4.0),
            (1, 100),
            (np.nan, np.nan),
            (np.array([0.2, 0.3]), np.array([4.0, 9.0])),
            (np.array([0.2, np.nan, 0.3]), np.array([4.0, np.nan, 9.0])),
            ([0.2, 0.3], [4.0, 9.0]),
        )
        for x, y in test_pairs:
            np.testing.assert_allclose(lt_x_prime(x), y, equal_nan=True)
            np.testing.assert_allclose(lt_x_prime.reverse_call(y), x, equal_nan=True)

        np.testing.assert_allclose(
            lt_x_prime(Timeseries([0, 1, 2, 3], [0.1, 0.2, np.nan, 0.4])).values,
            np.array([1.0, 4.0, np.nan, 16.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            lt_x_prime.reverse_call(Timeseries([0, 1, 2, 3], [1.0, 4.0, np.nan, 16.0])).values,
            np.array([0.1, 0.2, np.nan, 0.4]),
            equal_nan=True,
        )

    def test_symbolic_success(self):
        x_results = self.results["x"]
        x_prime_desired_results = self.problem.lookup_tables(0)["x_prime"](x_results)
        x_prime_results = self.results["x_prime"]
        np.testing.assert_allclose(x_prime_results, x_prime_desired_results, equal_nan=True)
        # x_prime min is 4.0, so x should be 0.2 when minimized
        np.testing.assert_allclose(x_results, 0.2, equal_nan=True)


class TestCSVLookupTableCache(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(data_path(), "lookup_tables", "x_prime.csv")
        self.npz_file = self.csv_file.replace(".csv", ".npz")
        self.ca_file = self.csv_file.replace(".csv", ".ca")
        for f in (self.npz_file, self.ca_file):
            if os.path.exists(f):
                os.remove(f)

    def tearDown(self):
        if os.path.exists(self.ca_file):
            os.remove(self.ca_file)

    def test_serialized_function_not_loaded_from_cache(self):
        # First run fits the spline and writes the tck cache, but no serialized function
        Model().optimize()
        self.assertTrue(os.path.exists(self.npz_file))
        self.assertFalse(os.path.exists(self.ca_file))

        # A (potentially malicious) serialized CasADi function next to the cache must be ignored
        with open(self.ca_file, "w") as f:
            f.write("not a casadi function")
        ini_file = os.path.join(os.path.dirname(self.csv_file), "curvefit_options.ini")
        mtime = max(os.path.getmtime(self.csv_file), os.path.getmtime(ini_file)) + 10
        os.utime(self.npz_file, (mtime, mtime))
        os.utime(self.ca_file, (mtime, mtime))

        with (
            mock.patch.object(
                ca.Function, "load", side_effect=AssertionError("Function.load called")
            ) as function_load,
            mock.patch.object(BSpline1D, "fit", wraps=BSpline1D.fit) as spline_fit,
        ):
            problem = Model()
            problem.optimize()
            function_load.assert_not_called()
            # The cached tck values must have been used, i.e. no refit
            spline_fit.assert_not_called()

        lt_x_prime = problem.lookup_tables(0)["x_prime"]
        np.testing.assert_allclose(lt_x_prime(0.2), 4.0)
        np.testing.assert_allclose(float(lt_x_prime.function(0.3)), 9.0)
