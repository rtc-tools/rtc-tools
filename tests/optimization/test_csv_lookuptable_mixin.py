import logging
import subprocess
import sys
import unittest

import numpy as np

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


@unittest.skipIf(sys.version_info < (3, 15), "__lazy_modules__ needs Python 3.15")
class TestLazyScipy(TestCase):
    def test_import_does_not_load_scipy(self):
        code = (
            "import sys, rtctools.optimization.csv_lookup_table_mixin; "
            "assert 'scipy' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
