import logging
import sys
import unittest

import numpy as np
from casadi import MX

from rtctools._internal.alias_tools import AliasDict
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.DEBUG)


class Model(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        print(data_path())
        self._extra_variable = MX.sym("extra")
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelWithInitial",
            model_folder=data_path(),
        )

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["u_max"] = 2.0
        return parameters

    def pre(self):
        # Do nothing
        pass

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        return AliasDict(
            self.alias_relation,
            {"constant_input": Timeseries(self.times(), 1 - self.times())},
        )

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    @property
    def extra_variables(self):
        v = super().extra_variables
        return [*v, self._extra_variable]

    def bounds(self):
        b = super().bounds()
        b[self._extra_variable.name()] = [-1000, 1000]
        return b

    def path_constraints(self, ensemble_member):
        c = super().path_constraints(ensemble_member)[:]
        c.append((self.state("x") - self._extra_variable, -np.inf, 0.0))
        return c

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class ModelNonConvex(Model):
    def __init__(self, u_seed):
        super().__init__()

        self.u_seed = u_seed

    def objective(self, ensemble_member):
        # Make two local optima, at xf=1.0 and at xf=-1.0.
        xf = self.state_at("x", self.times()[-1], ensemble_member=ensemble_member)
        return (xf**2 - 1.0) ** 2

    @property
    def initial_residual(self):
        # Set the initial state for 'x' to the neutral point.
        return self.state("x")

    def seed(self, ensemble_member):
        # Seed the controls.
        seed = super().seed(ensemble_member)
        seed["u"] = Timeseries(self.times(), self.u_seed)
        return seed


class ModelScaled(Model):
    def variable_nominal(self, variable):
        if variable.startswith("x"):
            return 0.5
        else:
            return super().variable_nominal(variable)


class ModelConstrained(Model):
    def constraints(self, ensemble_member):
        # Constrain x(t=1.9)^2 >= 0.1.
        x = self.state_at("x", self.times()[-1] - 0.1, ensemble_member=ensemble_member)
        f = x**2
        return [(f, 0.1, sys.float_info.max)]


class ModelThreaded(Model):
    def map_options(self):
        options = super().map_options()
        options["mode"] = "thread"
        options["n_threads"] = 2
        return options


class ModelTrapezoidal(Model):
    @property
    def theta(self):
        return 0.5


class ModelShort(Model):
    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 2)


class ModelAggregation(Model):
    def times(self, variable=None):
        if variable == "u":
            return np.linspace(0.0, 1.0, 11)
        else:
            return np.linspace(0.0, 1.0, 21)


class ModelEnsemble(Model):
    @property
    def ensemble_size(self):
        return 2

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        if ensemble_member == 0:
            return {"constant_input": Timeseries(self.times(), np.linspace(1.0, 0.0, 21))}
        else:
            return {"constant_input": Timeseries(self.times(), np.linspace(1.0, 0.5, 21))}


class ModelAlgebraic(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelAlgebraic",
            model_folder=data_path(),
        )

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def pre(self):
        # Do nothing
        pass

    def bounds(self):
        # Variable bounds
        return {"u": (-2.0, 2.0)}

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def objective(self, ensemble_member):
        return self.integral("u")

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class ModelMixedInteger(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelMixedInteger",
            model_folder=data_path(),
        )

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def pre(self):
        # Do nothing
        pass

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def objective(self, ensemble_member):
        return self.integral("y")

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class ModelHistory(Model):
    def history(self, ensemble_member):
        h = super().history(ensemble_member)
        # NOTE: We deliberately add histories of different lengths
        h["x"] = Timeseries(np.array([-0.2, -0.1, 0.0]), np.array([0.7, 0.9, 1.1]))
        h["w"] = Timeseries(np.array([-0.1, 0.0]), np.array([0.9, np.nan]))
        return h

    def variable_nominal(self, variable):
        if variable in {"x", "w"}:
            return 2.0
        else:
            return super().variable_nominal(variable)


class ModelSymbolicParameters(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self, resolve_parameters=True, override_u_min=True):
        self.__resolve_parameters = resolve_parameters
        self.__override_u_min = override_u_min

        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="ModelSymbolicParameters",
            model_folder=data_path(),
        )

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters["u_max"] = 2.0
        parameters["x_initial"] = 1.1
        parameters["w_seed"] = 0.2
        parameters["a"] = -2.0

        if self.__override_u_min:
            parameters["u_min"] = -1.5
        return parameters

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []

        if not self.__resolve_parameters:
            compiler_options["replace_parameter_expressions"] = False
            compiler_options["resolve_parameter_values"] = False

        return compiler_options


class TestModelicaMixin(TestCase, unittest.TestCase):
    def setUp(self):
        self.problem = Model()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)

    def test_ifelse(self):
        print(self.results["switched"])
        print(self.results["x"])
        self.assertEqual(self.results["switched"][0], 1.0)
        self.assertEqual(self.results["switched"][-1], 2.0)

    def test_output(self):
        self.assertAlmostEqual(
            self.results["x"] ** 2 + np.sin(self.problem.times()),
            self.results["z"],
            self.tolerance,
        )

    def test_algebraic(self):
        self.assertAlmostEqual(
            self.results["y"] + self.results["x"],
            np.ones(len(self.problem.times())) * 3.0,
            self.tolerance,
        )

    def test_bounds(self):
        self.assertAlmostGreaterThan(self.results["u"], -2, self.tolerance)
        self.assertAlmostLessThan(self.results["u"], 2, self.tolerance)

    def test_constant_input(self):
        verify = np.linspace(1.0, 0.0, 21)
        self.assertAlmostEqual(self.results["constant_output"], verify, self.tolerance)

    def test_delayed_feedback(self):
        self.assertAlmostEqual(
            self.results["x_delayed"][2:], self.results["x"][:-2], self.tolerance
        )

    def test_multiple_states(self):
        self.assertAlmostEqual(self.results["w"][0], 0.0, self.tolerance)
        self.assertAlmostEqual(self.results["w"][-1], 0.5917, 1e-3)

    def test_extra_variable(self):
        self.assertAlmostLessThan(
            np.max(self.results["x"]),
            self.results[self.problem._extra_variable.name()],
            self.tolerance,
        )

    def test_ode(self):
        times = self.problem.times()
        parameters = self.problem.parameters(0)
        self.assertAlmostEqual(
            (self.results["x"][1:] - self.results["x"][:-1]) / (times[1:] - times[:-1]),
            parameters["k"] * self.results["x"][1:] + self.results["u"][1:],
            self.tolerance,
        )
        self.assertAlmostEqual(
            (self.results["w"][1:] - self.results["w"][:-1]) / (times[1:] - times[:-1]),
            self.results["x"][1:],
            self.tolerance,
        )

    def test_algebraic_variables(self):
        self.assertAlmostEqual(
            self.results["x"] + self.results["y"],
            3.0,
            self.tolerance,
        )

    @unittest.skip
    def test_states_in(self):
        states = list(self.problem.states_in("x", 0.05, 0.95))
        verify = []
        for t in self.problem.times()[1:-1]:
            verify.append(self.problem.state_at("x", t))
        self.assertEqual(repr(states), repr(verify))

        states = list(self.problem.states_in("x", 0.051, 0.951))
        verify = [self.problem.state_at("x", 0.051)]
        for t in self.problem.times()[2:-1]:
            verify.append(self.problem.state_at("x", t))
        verify.append(self.problem.state_at("x", 0.951))
        self.assertEqual(repr(states), repr(verify))

        states = list(self.problem.states_in("x", 0.0, 0.951))
        verify = []
        for t in self.problem.times()[0:-1]:
            verify.append(self.problem.state_at("x", t))
        verify.append(self.problem.state_at("x", 0.951))
        self.assertEqual(repr(states), repr(verify))

    def test_der(self):
        der = self.problem.der_at("x", 0.05)
        verify = (self.problem.state_at("x", 0.05) - self.problem.state_at("x", 0.0)) / 0.05
        self.assertEqual(repr(der), repr(verify))

        der = self.problem.der_at("x", 0.051)
        verify = (self.problem.state_at("x", 0.1) - self.problem.state_at("x", 0.05)) / 0.05
        self.assertEqual(repr(der), repr(verify))

    @unittest.skip("This test fails, because we use CasADi sumRows() now.")
    def test_integral(self):
        integral = self.problem.integral("x", 0.05, 0.95)
        knots = self.problem.times()[1:-1]
        verify = MX(0.0)
        for i in range(len(knots) - 1):
            verify += (
                0.5
                * (self.problem.state_at("x", knots[i]) + self.problem.state_at("x", knots[i + 1]))
                * (knots[i + 1] - knots[i])
            )
        self.assertEqual(repr(integral), repr(verify))

        integral = self.problem.integral("x", 0.051, 0.951)
        knots = []
        knots.append(0.051)
        knots.extend(self.problem.times()[2:-1])
        knots.append(0.951)
        verify = MX(0.0)
        for i in range(len(knots) - 1):
            verify += (
                0.5
                * (self.problem.state_at("x", knots[i]) + self.problem.state_at("x", knots[i + 1]))
                * (knots[i + 1] - knots[i])
            )
        self.assertEqual(repr(integral), repr(verify))

        integral = self.problem.integral("x", 0.0, 0.951)
        knots = list(self.problem.times()[0:-1]) + [0.951]
        verify = MX(0.0)
        for i in range(len(knots) - 1):
            verify += (
                0.5
                * (self.problem.state_at("x", knots[i]) + self.problem.state_at("x", knots[i + 1]))
                * (knots[i + 1] - knots[i])
            )
        self.assertEqual(repr(integral), repr(verify))


class TestModelicaMixinScaled(TestModelicaMixin):
    def setUp(self):
        self.problem = ModelScaled()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6


class TestModelicaMixinNonConvex(TestCase):
    def setUp(self):
        self.tolerance = 1e-6

    def test_seeding(self):
        # Verify that both optima are found, depending on the seeding.
        self.problem = ModelNonConvex(np.ones(21) * 2.0)
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.assertAlmostEqual(self.results["x"][-1], 1.0, self.tolerance)

        self.problem = ModelNonConvex(np.ones(21) * -2.0)
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.assertAlmostEqual(self.results["x"][-1], -1.0, self.tolerance)


class TestModelicaMixinConstrained(TestCase):
    def setUp(self):
        self.problem = ModelConstrained()
        self.problem.optimize()
        self.results = self.problem.extract_results()

    def test_objective_value(self):
        # Make sure the constraint at t=1.9 has been applied.  With |u| <= 2, this ensures that
        # x(t=2.0)=0.0, the unconstrained optimum, can never be reached.
        self.assertAlmostGreaterThan(self.problem.objective_value, 1e-2, 0)
        self.assertAlmostEqual(self.results["u"][-1], -2, 1e-6)


class TestModelicaMixinThreaded(TestCase):
    def setUp(self):
        self.problem = ModelThreaded()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestModelicaMixinTrapezoidal(TestCase):
    def setUp(self):
        self.problem = ModelTrapezoidal()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestModelicaMixinShort(TestCase):
    def setUp(self):
        self.problem = ModelShort()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestModelicaMixinAggregation(TestCase):
    def setUp(self):
        self.problem = ModelAggregation()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)

    def test_result_length(self):
        self.assertEqual(len(self.results["u"]), 11)
        self.assertEqual(len(self.results["x"]), 21)


class TestModelicaMixinEnsemble(TestCase):
    def setUp(self):
        self.problem = ModelEnsemble()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)


class TestModelicaMixinAlgebraic(TestCase):
    def setUp(self):
        self.problem = ModelAlgebraic()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_algebraic(self):
        self.assertAlmostEqual(
            self.results["y"] + self.results["u"],
            np.ones(len(self.problem.times())) * 1.0,
            self.tolerance,
        )


class TestModelicaMixinMixedInteger(TestCase):
    def setUp(self):
        self.problem = ModelMixedInteger()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_booleans(self):
        self.assertAlmostEqual(self.results["choice"], np.zeros(21, dtype=bool), self.tolerance)
        self.assertAlmostEqual(
            self.results["other_choice"], np.ones(21, dtype=bool), self.tolerance
        )
        self.assertAlmostEqual(self.results["y"], -1 * np.ones(21, dtype=bool), self.tolerance)


class TestModelicaMixinHistory(TestCase, unittest.TestCase):
    def setUp(self):
        self.problem = ModelHistory()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_initial_der_x(self):
        self.assertNotEqual(self.problem.variable_nominal("x"), 1.0)
        self.assertNotEqual(self.problem.variable_nominal("w"), 1.0)

        self.assertAlmostEqual(self.results["initial_der(x)"], 2.0, self.tolerance)
        self.assertAlmostEqual(
            self.results["initial_der(w)"], (self.results["w"][0] - 0.9) / 0.1, self.tolerance
        )

    def test_delayed_feedback(self):
        np.testing.assert_allclose(
            self.results["x_delayed"],
            [0.9, 1.0, *self.results["x"][:-2]],
            rtol=self.tolerance,
            atol=self.tolerance,
        )
        np.testing.assert_allclose(
            self.results["x_delayed_extra"],
            [0.85, 0.95, 1.05, *((self.results["x"][1:-2] + self.results["x"][:-3]) / 2)],
            rtol=self.tolerance,
            atol=self.tolerance,
        )


class TestModelicaMixinSymbolicParameters(TestCase, unittest.TestCase):
    def setUp(self):
        self.problem = ModelSymbolicParameters()
        self.tolerance = 1e-6

    def test_symbolic_seed(self):
        self.assertTrue(np.all(self.problem.seed(0)["w"].values == 0.2))

    def test_symbolic_initial_state(self):
        history = self.problem.history(0)
        self.assertEqual(history["x"].values[-1], 1.1)
        self.problem.optimize()
        self.assertAlmostEqual(self.problem.extract_results()["x"][0], 1.1, self.tolerance)


class TestModelicaMixinSymbolicParametersResolve(TestCase, unittest.TestCase):
    def test_parameters_resolve_or_not(self):
        """
        We set a 'default' value of u_min = a * b in the Pymoca model. If we
        tell pymoca to replace parameter expressions (default behavior), we can
        no longer override 'u_min' via the parameters() method. If we tell
        Pymoca to _not_ inline, we expect to be able to override 'u_min', but
        for it to also still have the default value of a * b.
        """
        problem = ModelSymbolicParameters()
        problem.optimize()
        parameters = problem.parameters(0)
        bounds = problem.bounds()

        # Check that overriding "u_min" has no effect if parameters are resolved
        self.assertNotEqual(bounds["u"][0], parameters["u_min"])

        # Check that overriding "u_min" has effect if parameters are not resolved
        problem_no_resolved = ModelSymbolicParameters(resolve_parameters=False)
        problem_no_resolved.optimize()
        parameters_no_resolved = problem_no_resolved.parameters(0)
        bounds_no_resolved = problem_no_resolved.bounds()

        self.assertEqual(bounds_no_resolved["u"][0], parameters_no_resolved["u_min"])

        # Check that if we do not override "u_min", it takes the default value a
        # * b, and that overriding the value of 'a' works.
        problem_no_override = ModelSymbolicParameters(
            resolve_parameters=False, override_u_min=False
        )
        problem_no_override.optimize()
        parameters_no_override = problem_no_override.parameters(0)
        bounds_no_override = problem_no_override.bounds()

        self.assertEqual(
            bounds_no_override["u"][0], parameters_no_override["a"] * parameters_no_override["b"]
        )


class ModelAliasBounds(Model):
    def bounds(self):
        bounds = super().bounds()
        bounds["negative_alias"] = (-2.0, 1.0)
        return bounds


class TestAliasBounds(TestCase):
    def test_alias_bounds(self):
        problem = ModelAliasBounds()
        bounds = problem.bounds()
        self.assertEqual(bounds["x"], (-1.0, 2.0))
        self.assertEqual(bounds["negative_alias"], (-2.0, 1.0))
