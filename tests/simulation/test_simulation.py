import os
import re

import casadi as ca
import numpy as np

from rtctools._internal.alias_tools import AliasDict
from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.simulation.simulation_problem import SimulationProblem, Variable

from ..test_case import TestCase
from .data_path import data_path


class SimulationModel(SimulationProblem):
    _force_zero_delay = True

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model",
            model_folder=data_path(),
        )

    def seed(self):
        seed = super().seed()
        seed["z"] = 1.0
        return seed

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class TestSimulation(TestCase):
    def setUp(self):
        self.problem = SimulationModel()

    def test_object(self):
        self.assertIsInstance(self.problem, SimulationModel)

    def test_get_variables(self):
        all_variables = self.problem.get_variables()
        self.assertIsInstance(all_variables, AliasDict)

        self.assertEqual(
            set(all_variables),
            {
                "time",
                "constant_input",
                "k",
                "switched",
                "u",
                "u_out",
                "w",
                "der(w)",
                "x",
                "der(x)",
                "x_start",
                "y",
                "z",
                "_pymoca_delay_0[1,1]",
                "_pymoca_delay_0[1,1]_expr",
                "x_delayed",
            },
        )
        self.assertEqual(set(self.problem.get_parameter_variables()), {"x_start", "k"})
        self.assertEqual(set(self.problem.get_input_variables()), {"constant_input", "u"})
        self.assertEqual(
            set(self.problem.get_output_variables()),
            {"constant_output", "switched", "u_out", "y", "z", "x_delayed"},
        )

    def test_get_set_var(self):
        val = self.problem.get_var("switched")
        self.assertTrue(np.isnan(val))
        self.problem.set_var("switched", 10.0)
        val_reset = self.problem.get_var("switched")
        self.assertNotEqual(val_reset, val)

    def test_get_var_name_and_type(self):
        t = self.problem.get_var_type("switched")
        self.assertTrue(t is float)
        all_variables = self.problem.get_variables()
        idx = 0
        for var in all_variables.items():
            varname = var[0]
            if re.match(varname, "switched"):
                break
            idx += 1

        varname = self.problem.get_var_name(idx)
        self.assertEqual(varname, "switched")

    def test_get_time(self):
        # test methods for get_time
        start = 0.0
        stop = 10.0
        dt = 0.5
        self.problem.setup_experiment(start, stop, dt)
        self.problem.set_var("x_start", 0.0)
        self.problem.set_var("constant_input", 0.0)
        self.problem.set_var("u", 0.0)
        self.problem.initialize()
        self.assertAlmostEqual(self.problem.get_start_time(), start, 1e-6)
        self.assertAlmostEqual(self.problem.get_end_time(), stop, 1e-6)
        curtime = self.problem.get_current_time()
        self.assertAlmostEqual(curtime, start, 1e-6)
        while curtime < stop:
            self.problem.update(dt)
            curtime = self.problem.get_current_time()
        self.assertAlmostEqual(curtime, stop, 1e-6)

    def test_set_input(self):
        # run FMU model
        expected_values = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        stop = 1.0
        dt = 0.1
        self.problem.setup_experiment(0.0, stop, dt)
        self.problem.set_var("x_start", 0.25)
        self.problem.set_var("constant_input", 0.0)
        self.problem.set_var("u", 0.0)
        self.problem.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem.set_var("u", 0.0)
            self.problem.update(dt)
            val = self.problem.get_var("switched")
            self.assertEqual(val, expected_values[i])

            # Test zero-delayed expression
            self.assertAlmostEqual(
                self.problem.get_var("x_delayed"), self.problem.get_var("x") * 3 + 1, 1e-6
            )
            i += 1

    def test_set_input2(self):
        # run FMU model
        expected_values = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        stop = 1.0
        dt = 0.1
        self.problem.setup_experiment(0.0, stop, dt)
        self.problem.set_var("x_start", 0.25)
        self.problem.set_var("constant_input", 0.0)
        self.problem.set_var("u", 0.0)
        self.problem.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem.set_var("u", i)
            self.problem.update(dt)
            val = self.problem.get_var("u_out")
            self.assertEqual(val, i + 1)
            val = self.problem.get_var("switched")
            self.assertEqual(val, expected_values[i])
            i += 1

    def test_seed(self):
        seed = self.problem.seed()
        self.assertEqual(seed["z"], 1.0)

    def test_nan_in_next_state_is_reported(self):
        """Test that a nan in the rootfinder result is logged with the variable it belongs to."""
        self.problem.setup_experiment(0.0, 1.0, 0.1)
        self.problem.set_var("x_start", 0.25)
        self.problem.set_var("constant_input", 0.0)
        self.problem.set_var("u", 0.0)
        self.problem.initialize()

        index, _ = self.problem._SimulationProblem__indices["w"]
        n_states = self.problem._SimulationProblem__n_states
        self.problem._SimulationProblem__do_step = NanStep(n_states, index)

        with self.assertLogs("rtctools", "ERROR") as context_manager:
            self.problem.update(0.1)

        self.assertTrue(
            contains_regex(
                r"(?s)Found nan\(s\) in the next_state vector.*'w'", context_manager.output
            )
        )


class NanStep:
    """Stand-in for the rootfinder, returning a state vector with a nan at ``nan_index``."""

    def __init__(self, n_states, nan_index):
        self.values = np.zeros(n_states)
        self.values[nan_index] = np.nan

    def __call__(self, *args):
        return ca.DM(self.values)

    def stats(self):
        return {"success": True}


class SimulationModelBase(SimulationProblem):
    _force_zero_delay = True

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model_base",
            model_folder=data_path(),
        )


class SimulationModelNominal(SimulationProblem):
    _force_zero_delay = True

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model_nominal",
            model_folder=data_path(),
        )


class TestSimulationNominal(TestCase):
    def setUp(self):
        self.problem_base = SimulationModelBase()
        self.problem_nominal = SimulationModelNominal()

    def test_model_nominal(self):
        z_base = []
        z_nominal = []

        start = 0.0
        dt = 0.1
        stop = 1.0
        self.problem_base.setup_experiment(start, stop, dt)
        self.problem_base.set_var("x", 0)
        self.problem_base.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem_base.set_var("x", i / 100)
            self.problem_base.update(dt)
            z_base.append(self.problem_base.get_var("z"))
            i += 1

        self.problem_nominal.setup_experiment(start, stop, dt)
        self.problem_nominal.set_var("x", 0)
        self.problem_nominal.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem_nominal.set_var("x", i / 100)
            self.problem_nominal.update(dt)
            z_nominal.append(self.problem_nominal.get_var("z"))
            i += 1

        self.assertAlmostEqual(np.array(z_base), np.array(z_nominal), 1e-5)


class SimulationModelDelay(SimulationProblem):
    def __init__(self, fixed_dt):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model_delay",
            model_folder=data_path(),
            fixed_dt=fixed_dt,
        )


class TestSimulationDelay(TestCase):
    def set_problem(self, dt=None):
        self.problem_delay = SimulationModelDelay(fixed_dt=dt)

    def test_model_delay(self):
        start = 0.0
        stop = 1.0
        dt = 0.1
        self.set_problem(dt)
        x_start = 1.0
        x = []
        z1 = []
        z2 = []
        z3 = []
        self.problem_delay.set_var("x_start", x_start)
        self.problem_delay.setup_experiment(start, stop, dt)
        self.problem_delay.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem_delay.update(dt)
            x.append(self.problem_delay.get_var("x"))
            z1.append(self.problem_delay.get_var("z1"))
            z2.append(self.problem_delay.get_var("z2"))
            z3.append(self.problem_delay.get_var("z3"))
            i += 1
        x_ref = 2.0
        z1_ref = 17.8
        z2_ref = 189.0
        z3_ref = 1670.0
        self.assertAlmostEqual(x[-1], x_ref, 1e-6)
        self.assertAlmostEqual(z1[-1], z1_ref, 1e-6)
        self.assertAlmostEqual(z2[-1], z2_ref, 1e-6)
        self.assertAlmostEqual(z3[-1], z3_ref, 1e-6)

    def test_delay_equation_without_fixed_dt_exception(self):
        with self.assertRaisesRegex(ValueError, "fixed_dt should be set"):
            self.set_problem()


class SimulationModelCustomEquation(SimulationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model_custom_equation",
            model_folder=data_path(),
        )

    def extra_equations(self):
        variables = self.get_variables()

        y = variables["y"]
        x = variables["x"]

        constraint_nominal = (
            self.get_variable_nominal("x") * self.get_variable_nominal("y")
        ) ** 0.5

        return [(y - (-2 * x)) / constraint_nominal]


class TestSimulationCustomEquation(TestCase):
    def setUp(self):
        self.problem = SimulationModelCustomEquation()

    def test_model_custom_equation(self):
        start = 0.0
        stop = 1.0
        dt = 0.5
        x = []
        y = []
        self.problem.setup_experiment(start, stop, dt)
        self.problem.initialize()
        x.append(self.problem.get_var("x"))
        y.append(self.problem.get_var("y"))
        i = 0
        while i < int(stop / dt):
            self.problem.update(dt)
            x.append(self.problem.get_var("x"))
            y.append(self.problem.get_var("y"))
            i += 1
        x_ref = [2.0, 1.0, 0.5]
        self.assertAlmostEqual(x[-1], x_ref[-1], 1e-6)


class SimpleModelCustomEquationLinearInterpolation(SimulationProblem):
    x_grid = np.linspace(-1, 1, 6)
    values = [-1, -1, -2, -3, 0, 2]

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model_custom_equation",
            model_folder=data_path(),
        )

    def extra_variables(self):
        return [Variable("z", nominal=10.0, min=-10.0, max=10.0)]

    def extra_equations(self):
        variables = self.get_variables()

        x = variables["x"]
        y = variables["y"]
        z = variables["z"]

        eq_1 = y - (-2 / 3600 * x)

        # Interpolation example
        lookup_table = ca.interpolant("lookup_table", "linear", [self.x_grid], self.values)
        eq_2 = lookup_table(y) - z

        return [eq_1, eq_2]


class TestSimulationLinearInterpolation(TestCase):
    def setUp(self):
        self.problem = SimpleModelCustomEquationLinearInterpolation()

    def test_model_custom_equation_linear_interpolation(self):
        start = 0.0
        stop = 1.0
        dt = 0.5
        y = []
        z = []
        self.problem.setup_experiment(start, stop, dt)
        self.problem.initialize()
        y.append(self.problem.get_var("y"))
        z.append(self.problem.get_var("z"))
        i = 0
        while i < int(stop / dt):
            self.problem.update(dt)
            y.append(self.problem.get_var("y"))
            z.append(self.problem.get_var("z"))
            i += 1

        np.testing.assert_allclose(
            z,
            np.interp(y, self.problem.x_grid, self.problem.values),
        )


class SimulationModelWithFastNewton(SimulationModelBase):
    def rootfinder_options(self):
        return {"solver": "fast_newton", "solver_options": {"error_on_fail": False}}


class TestRootFinderOption(TestCase):
    def setUp(self):
        self.problem = SimulationModelWithFastNewton()

    def test_root_finder_option(self):
        start = 0.0
        stop = 1.0
        dt = 0.1
        z = []
        z_ref = []
        self.problem.setup_experiment(start, stop, dt)
        self.problem.set_var("x", 1.1 * start)
        self.problem.initialize()
        i = 0
        while i < int(stop / dt):
            t = self.problem.get_var("time")
            self.problem.set_var("x", 1.1 * t)
            self.problem.update(dt)
            z_ref.append(1.1)
            z.append(self.problem.get_var("z"))
            i += 1
        self.assertAlmostEqual(np.array(z[1:]), np.array(z_ref[1:]), 1e-6)


class SimulationModelInfeasibleInitialValue(SimulationProblem):
    def __init__(self, model):
        data_folder = os.path.join(data_path(), "infeasible_initial_value")
        super().__init__(
            input_folder=data_folder,
            output_folder=data_folder,
            model_name=model,
            model_folder=data_folder,
        )


class SimulationModelInfeasibleInitialValueWithSeed(SimulationModelInfeasibleInitialValue):
    def seed(self):
        return {"x": 25.0}


class SimulationModelInfeasibleInitialValueCSV(CSVMixin, SimulationProblem):
    def __init__(self, model, input_series=None):
        data_folder = os.path.join(data_path(), "infeasible_initial_value")
        if input_series is not None:
            self.timeseries_import_basename = input_series
        super().__init__(
            input_folder=data_folder,
            output_folder=data_folder,
            model_name=model,
            model_folder=data_folder,
        )


def contains_regex(regex: re.Pattern, messages: list[str]):
    """
    Check that a list of messages contains a given regex

    :param regex: a regular expression
    :param messages: a list of strings
    :returns: True if any of the messages contains the given regex.
    """
    for message in messages:
        if re.search(regex, message):
            return True
    return False


class TestSimulationInfeasibleInitialValue(TestCase):
    def test_initial_value_out_of_bounds(self):
        """Test that the correct initial value is set, when the given value is out of bounds."""
        problem = SimulationModelInfeasibleInitialValue("ModelOutOfBounds")
        problem.setup_experiment(start=0.0, stop=1.0, dt=0.1)
        with self.assertLogs("rtctools", "WARNING") as context_manager:
            problem.initialize()
        warning_pattern = (
            "Initialize: start value x = 20.0 is not in between bounds -inf and 10.0"
            + " and will be adjusted."
        )
        self.assertTrue(contains_regex(warning_pattern, context_manager.output))
        # x = max, not start in .mo file.
        x = problem.get_var("x")
        self.assertAlmostEqual(x, 10.0, 1e-6)

    def test_initial_value_out_of_symbolic_bounds(self):
        """Test that the correct initial value is set, when the given value is out of bounds."""
        problem = SimulationModelInfeasibleInitialValue("ModelOutOfSymbolicBounds")
        problem.setup_experiment(start=0.0, stop=1.0, dt=0.1)
        # No warning is given, because the upper bound is a symbolic expression.
        problem.initialize()
        # x = max, not start in .mo file.
        x = problem.get_var("x")
        self.assertAlmostEqual(x, 10.0, 1e-6)

    def test_initial_value_csv_out_of_bounds(self):
        """Test that the correct initial value is set, when the given value is out of bounds."""
        problem = SimulationModelInfeasibleInitialValueCSV("ModelWithBounds")
        problem.read()
        with self.assertLogs("rtctools", "INFO") as context_manager:
            problem.initialize()
        warning_pattern = (
            "Initialize: bounds of x will be overwritten"
            + " by the start value given by initial_state method."
        )
        self.assertTrue(contains_regex(warning_pattern, context_manager.output))
        # x = value in initial_state.csv, not max in .mo file.
        x = problem.get_var("x")
        self.assertAlmostEqual(x, 30.0, 1e-6)

    def test_conflict_initial_values(self):
        """Test that the correct value is set when there is a conflict in initial values."""
        problem = SimulationModelInfeasibleInitialValueCSV("ModelWithStart")
        problem.read()
        with self.assertLogs("rtctools", "WARNING") as context_manager:
            problem.initialize()
        warning_pattern = (
            "Initialize: Multiple initial values for x are provided:"
            + " {'modelica': 20.0, 'initial_state': 30.0}."
            + " Value from modelica file will be used to continue."
        )
        self.assertTrue(contains_regex(warning_pattern, context_manager.output))
        # x = value in .mo file, not value in initial_state.csv.
        x = problem.get_var("x")
        self.assertAlmostEqual(x, 20.0, 1e-6)

    def test_conflict_initial_values_with_symbol_in_model(self):
        """Test that the correct value is set when there is a conflict in initial values."""
        problem = SimulationModelInfeasibleInitialValueCSV("ModelWithSymbolicStart")
        problem.read()
        with self.assertLogs("rtctools", "WARNING") as context_manager:
            problem.initialize()
        warning_pattern = (
            "Initialize: Multiple initial values for x are provided:"
            + " {'modelica': MX.*, 'initial_state': 30.0}."
            + " Value from modelica file will be used to continue."
        )
        self.assertTrue(contains_regex(warning_pattern, context_manager.output))
        # y = value in .mo file, not value in initial_state.csv.
        x = problem.get_var("x")
        self.assertAlmostEqual(x, 20.0, 1e-6)

    def test_conflict_initial_values_with_zero_in_model(self):
        """Test that the correct value is set when there is a conflict in initial values."""
        # If the initial value is set in the .mo file and is zero,
        # then the value in the initial_state.csv will be used.
        # No warning is given, since rtc-tools does not know if the zero value
        # is set by the user or is given as default value by pymoca.
        problem = SimulationModelInfeasibleInitialValueCSV("ModelWithZeroStart")
        problem.read()
        problem.initialize()
        # x = value in initial_state.csv, not value in .mo file.
        x = problem.get_var("x")
        self.assertAlmostEqual(x, 30.0, 1e-6)

    def test_conflict_initial_values_with_seed(self):
        """Test that the correct value is set when there is a conflict in initial values."""
        problem = SimulationModelInfeasibleInitialValueWithSeed("ModelWithStart")
        problem.setup_experiment(start=0.0, stop=1.0, dt=0.1)
        with self.assertLogs("rtctools", "WARNING") as context_manager:
            problem.initialize()
        warning_pattern = (
            "Initialize: Multiple initial values for x are provided:"
            + " {'modelica': 20.0, 'seed': 25.0}."
            + " Value from seed method will be used to continue."
        )
        self.assertTrue(contains_regex(warning_pattern, context_manager.output))
        # x = value in seed method, not value in .mo file.
        x = problem.get_var("x")
        self.assertAlmostEqual(x, 25.0, 1e-6)

    def test_conflict_initial_values_input_files(self):
        """Test that the correct value is set when there is a conflict in input files."""
        problem = SimulationModelInfeasibleInitialValueCSV("ModelWithInputSeries")
        problem.read()
        problem.initialize()
        x = problem.get_var("x")
        f_in = problem.get_var("f_in")
        # x = value in initial_state.csv, not value in timeseries_import.csv.
        self.assertAlmostEqual(x, 30.0, 1e-6)
        # f_in = value in timeseries_import.csv, not value in initial_state.csv.
        self.assertAlmostEqual(f_in, 40.0, 1e-6)

    def test_initial_value_infeasible(self):
        """Test that a correct exception is raised when no initial value can be found."""
        exception_pattern = "no initial state could be found"
        with self.assertRaisesRegex(Exception, exception_pattern):
            problem = SimulationModelInfeasibleInitialValue("ModelInfeasible")
            problem.setup_experiment(start=0.0, stop=1.0, dt=0.1)
            problem.initialize()
