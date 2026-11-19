import os
from unittest import TestCase

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset, chartostring

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.netcdf_mixin import NetCDFMixin

from .data_path import data_path


class NetCDFModel(NetCDFMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model",
            model_folder=data_path(),
        )

        self.netcdf_id_map = {
            "x_delayed": ("loc_a", "x_delayed"),
            "u": ("loc_b", "u"),
            "y": ("loc_c", "y"),
            "z": ("loc_a", "z"),
            "switched": ("loc_c", "switched"),
            "constant_output": ("loc_a", "constant_output"),
        }

    def read(self):
        self.check_missing_variable_names = False
        super().read()

        # Just add the parameters ourselves for now (values taken from test_pi_mixin)
        params = {
            "k": 1.01,
            "x": 1.02,
            "SV_V_y": 22.02,
            "j": 12.01,
            "b": 13.01,
            "y": 12.02,
            "SV_H_y": 22.02,
        }
        for key, value in params.items():
            self.io.set_parameter(key, value)

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times()[-1])
        f = xf**2
        return f

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options

    def netcdf_id_to_variable(self, station_id, parameter):
        # We strip the location, but save it such that we can use it for the
        # output again. The Exception is a couple parameters that are not used
        # in the model, which we use to make sure that not accidental aliasing
        # (e.g. ignoring station_id) occurs.
        unused_variables = [
            ("loc_a", "u_min"),
            ("loc_a", "u_max"),
            ("loc_b", "x"),
            ("loc_b", "w"),
            ("loc_b", "constant_input"),
        ]

        if (station_id, parameter) in unused_variables:
            variable_name = f"{station_id}__{parameter}"
        else:
            variable_name = parameter

        self.netcdf_id_map[variable_name] = (station_id, parameter)
        return variable_name

    def netcdf_id_from_variable(self, variable_name):
        return self.netcdf_id_map[variable_name]

    def min_timeseries_id(self, variable):
        return "_".join((variable, "min"))

    def max_timeseries_id(self, variable):
        return "_".join((variable, "max"))


class TestNetCDFMixin(TestCase):
    def setUp(self):
        self.problem = NetCDFModel()
        self.tolerance = 1e-5

    def test_read(self):
        self.problem.read()

        datastore = self.problem.io
        self.assertTrue(np.all(datastore.get_timeseries("loc_a__u_min")[-1] == -3.0))
        self.assertTrue(np.all(datastore.get_timeseries("u_min")[-1] == -2.0))
        self.assertTrue(np.all(datastore.get_timeseries("loc_a__u_max")[-1] == 3.0))
        self.assertTrue(np.all(datastore.get_timeseries("u_max")[-1] == 2.0))

        expected_values = np.zeros((22,), dtype=float)
        expected_values[0] = 1.02
        expected_values[2] = 0.03
        self.assertTrue(np.array_equal(datastore.get_timeseries("x")[-1], expected_values))
        self.assertTrue(np.all(np.isnan(datastore.get_timeseries("loc_b__x")[-1])))

        expected_values = np.zeros((22,), dtype=float)
        expected_values[2] = 0.03
        self.assertTrue(np.array_equal(datastore.get_timeseries("w")[-1], expected_values))
        self.assertTrue(np.all(np.isnan(datastore.get_timeseries("loc_b__w")[-1])))

        self.assertTrue(np.all(datastore.get_timeseries("constant_input")[-1] == 1.0))
        self.assertTrue(np.all(datastore.get_timeseries("loc_b__constant_input")[-1] == 1.5))

    def test_write(self):
        self.problem.optimize()
        self.results = [self.problem.extract_results(i) for i in range(self.problem.ensemble_size)]

        bounds = self.problem.bounds()
        self.assertTrue(
            np.array_equal(bounds["u"][0].values, self.problem.get_timeseries("u_min").values)
        )
        self.assertTrue(
            np.array_equal(bounds["u"][1].values, self.problem.get_timeseries("u_max").values)
        )

        # open the exported file
        filename = os.path.join(data_path(), self.problem.timeseries_export_basename + ".nc")
        dataset = Dataset(filename)

        written_variables = dataset.variables.keys()
        self.assertEqual(len(written_variables), 11)
        self.assertIn("time", written_variables)
        self.assertIn("station_id", written_variables)
        self.assertIn("station_names", written_variables)
        self.assertNotIn("realization", written_variables)
        self.assertIn("lon", written_variables)
        self.assertIn("lat", written_variables)
        self.assertIn("y", written_variables)
        self.assertIn("constant_output", written_variables)
        self.assertIn("u", written_variables)
        self.assertIn("z", written_variables)
        self.assertIn("switched", written_variables)
        self.assertIn("x_delayed", written_variables)

        ids_var = dataset.variables["station_id"]
        self.assertEqual(ids_var.shape, (3, 5))
        self.assertEqual(ids_var.cf_role, "timeseries_id")
        station_ids = []
        for i in range(3):
            station_ids.append(str(chartostring(ids_var[i])))

        self.assertIn("loc_a", station_ids)
        self.assertIn("loc_b", station_ids)
        self.assertIn("loc_c", station_ids)

        # Assert that the order is deterministic
        loc_a_index = station_ids.index("loc_a")
        loc_b_index = station_ids.index("loc_b")
        loc_c_index = station_ids.index("loc_c")
        self.assertEqual(loc_a_index, 1)
        self.assertEqual(loc_b_index, 2)
        self.assertEqual(loc_c_index, 0)

        self.assertEqual(loc_a_index, 1)
        self.assertEqual(loc_b_index, 2)
        self.assertEqual(loc_c_index, 0)

        station_names = dataset.variables["station_names"]
        self.assertEqual(station_names.shape, (3, 10))
        self.assertEqual(str(chartostring(station_names[loc_a_index])), "Location A")
        self.assertEqual(str(chartostring(station_names[loc_b_index])), "Location B")

        self.assertAlmostEqual(
            dataset.variables["lon"][loc_a_index], 4.3780269, delta=self.tolerance
        )

        y = dataset.variables["y"]
        self.assertEqual(y.shape, (22, 3))
        for i in range(3):
            data = ma.filled(y[:, i], np.nan)
            if i == loc_c_index:
                self.assertAlmostEqual(data[0], 1.98, delta=self.tolerance)
                np.testing.assert_allclose(
                    data, self.results[i]["y"], rtol=self.tolerance, atol=self.tolerance
                )
                self.assertAlmostEqual(data[-1], 3.0, delta=self.tolerance)
            else:
                self.assertTrue(np.all(np.isnan(data)))

        u = dataset.variables["u"]
        self.assertEqual(u.shape, (22, 3))
        for i in range(3):
            data = ma.filled(u[:, i], np.nan)
            if i == loc_b_index:
                self.assertTrue(np.all(~np.isnan(data)))
            else:
                self.assertTrue(np.all(np.isnan(data)))

        constant_output = dataset.variables["constant_output"]
        self.assertEqual(constant_output.shape, (22, 3))
        for i in range(3):
            data = ma.filled(constant_output[:, i], np.nan)
            if i == loc_a_index:
                self.assertTrue(np.all(data == 1.0))
            else:
                self.assertTrue(np.all(np.isnan(data)))

        time = dataset.variables["time"]
        self.assertEqual(time.units, "seconds since 2013-05-09 22:00:00")
        self.assertEqual(time.standard_name, "time")
        self.assertEqual(time.axis, "T")
        self.assertTrue(np.allclose(time[:], np.arange(0, 22 * 3600, 3600, dtype=float)))
