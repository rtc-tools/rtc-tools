import logging
import unittest

import casadi as ca
import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class Model(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
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

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        constant_inputs["constant_input"] = Timeseries(
            np.hstack([self.initial_time, self.times()]),
            np.hstack(([1.0], np.linspace(1.0, 0.0, 21))),
        )
        return constant_inputs

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def set_timeseries(self, timeseries_id, timeseries, ensemble_member, **kwargs):
        # Do nothing
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options

    def constraints(self, ensemble_member):
        return [
            (self.state_at("x", 0.5, ensemble_member=ensemble_member), 1.0, np.inf),
            (self.state_at("x", 0.7, ensemble_member=ensemble_member), -np.inf, 0.8),
            (self.integral("x", 0.1, 1.0, ensemble_member=ensemble_member), -np.inf, 1.0),
        ]

    def path_objective(self, ensemble_member):
        return self.state("u") ** 2


class ModelConstraints(Model):
    pass


class ModelConstraintsVector(ModelConstraints):
    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        g, lbg, ubg = zip(*constraints, strict=True)

        constraint = (ca.vertcat(*g), np.array(lbg), np.array(ubg))
        return [constraint]


class ModelPathConstraintsSimple(Model):
    def constraints(self, ensemble_member):
        return []

    def path_constraints(self, ensemble_member):
        x_constr = (self.state("x"), 0.1, 1.2)
        z_constr = (self.state("z"), 0.0, 2.2)
        return [x_constr, z_constr]


class ModelPathConstraintsSimpleVector(ModelPathConstraintsSimple):
    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        g, lbg, ubg = zip(*constraints, strict=True)

        constraint = (ca.vertcat(*g), np.array(lbg), np.array(ubg))

        return [constraint]


class ModelPathConstraintsTimeseries(Model):
    def constraints(self, ensemble_member):
        return []

    def path_constraints(self, ensemble_member):
        m = np.full_like(self.times(), 0.1)
        M = np.full_like(self.times(), np.inf)

        m[10] = 1.0
        M[14] = 0.8

        x_constr = (self.state("x"), Timeseries(self.times(), m), Timeseries(self.times(), M))
        z_constr = (self.state("z"), 0.0, 2.2)
        return [x_constr, z_constr]


class ModelPathConstraintsTimeseriesVector(ModelPathConstraintsTimeseries):
    def constraints(self, ensemble_member):
        return []

    def path_constraints(self, ensemble_member):
        x_min = np.full_like(self.times(), 0.1)
        x_max = np.full_like(self.times(), np.inf)

        x_min[10] = 1.0
        x_max[14] = 0.8

        z_min = np.full_like(self.times(), 0.0)
        z_max = np.full_like(self.times(), 2.2)

        g = ca.vertcat(self.state("x"), self.state("z"))
        m = np.stack((x_min, z_min), axis=1)
        M = np.stack((x_max, z_max), axis=1)

        lbg = Timeseries(self.times(), m)
        ubg = Timeseries(self.times(), M)

        constraint = (g, lbg, ubg)

        return [constraint]


class ModelAdditionalVariables(Model):
    def pre(self):
        super().pre()

        self._additional_vars = []

        for i in range(len(self.times())):
            sym = ca.MX.sym(f"u2_t{i}")
            self._additional_vars.append(sym)

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member).copy()

        for sym, t in zip(self._additional_vars, self.times(), strict=True):
            x_sym = self.extra_variable(sym.name(), ensemble_member)
            constraints.append((x_sym - self.state_at("u", t) ** 2, 0, np.inf))

        return constraints

    def path_objective(self, ensemble_member):
        return ca.MX(0)

    def objective(self, ensemble_member):
        return ca.sum1(
            ca.vertcat(
                *[self.extra_variable(x.name(), ensemble_member) for x in self._additional_vars]
            )
        )

    @property
    def extra_variables(self):
        return self._additional_vars

    def bounds(self):
        bounds = super().bounds()

        for s in self._additional_vars:
            bounds[s.name()] = (0.0, 4.0)

        return bounds

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)

        for s in self._additional_vars:
            seed[s.name()] = 0.0

        return seed


class ModelAdditionalVariablesVector(Model):
    # Want to test bounds() and seed() for both ways of setting values. Either
    # with floats, or with np.arrays. We therefore make 2 vector variables,
    # and one scalar variable.

    def pre(self):
        super().pre()

        n_times = len(self.times())

        size1 = int(n_times / 2)
        size2 = 1
        size3 = n_times - size1 - size2

        sym1 = ca.MX.sym("u2_1", size1)
        sym2 = ca.MX.sym("u2_2", size2)
        sym3 = ca.MX.sym("u2_3", size3)

        assert sum([sym1.size1(), sym2.size1(), sym3.size1()]) == len(self.times())

        self._additional_vars = [sym1, sym2, sym3]

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member).copy()

        all_u = ca.vertcat(*[self.state_at("u", t) for t in self.times()])

        sym1, sym2, sym3 = self._additional_vars
        size1 = sym1.size1()

        s1 = self.extra_variable(sym1.name(), ensemble_member)
        s2 = self.extra_variable(sym2.name(), ensemble_member)
        s3 = self.extra_variable(sym3.name(), ensemble_member)

        assert s1.size1() == size1

        constraints.append((s1 - all_u[:size1] ** 2, 0, np.inf))
        constraints.append((s2 - all_u[size1] ** 2, 0, np.inf))
        constraints.append((s3 - all_u[size1 + 1 :] ** 2, 0, np.inf))

        return constraints

    def path_objective(self, ensemble_member):
        return ca.MX(0)

    def objective(self, ensemble_member):
        return ca.sum1(
            ca.vertcat(
                *[self.extra_variable(x.name(), ensemble_member) for x in self._additional_vars]
            )
        )

    @property
    def extra_variables(self):
        return self._additional_vars

    def bounds(self):
        bounds = super().bounds()

        # Use scalar value, and rely on broadcasting for the first vector symbol
        for s in self._additional_vars[:2]:
            bounds[s.name()] = (0.0, 4.0)

        # Use numpy array as bounds for the last vector symbol
        s = self._additional_vars[-1]
        bounds[s.name()] = (np.full(s.size1(), 0.0), np.full(s.size1(), 4.0))

        return bounds

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)

        # Use scalar value, and rely on broadcasting for the first vector symbol
        for s in self._additional_vars[:2]:
            seed[s.name()] = 0.0

        # Use numpy array as bounds for the last vector symbol
        s = self._additional_vars[-1]
        seed[s.name()] = np.full(s.size1(), 0.0)

        return seed


class ModelParameters(Model):
    def pre(self):
        super().pre()

        self._param_vars = [f"par_{i}" for i in range(5)]
        self._additional_vars = [ca.MX.sym(f"par2_{i}") for i in range(5)]

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        for t_sym, p_name in zip(self._additional_vars, self._param_vars, strict=True):
            t = self.extra_variable(t_sym.name(), ensemble_member)
            p = self.parameters(ensemble_member)[p_name]

            constraints.append((t - p**2, 0.0, 0.0))

        return constraints

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        for i, p in enumerate(self._param_vars):
            parameters[p] = i

        return parameters

    @property
    def extra_variables(self):
        return self._additional_vars


class ModelParametersVector(Model):
    def pre(self):
        super().pre()
        self._param_var = "par_x5"
        self._additional_var = ca.MX.sym("par2_x5", 5)

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        t = self.extra_variable(self._additional_var.name(), ensemble_member)
        p = self.parameters(ensemble_member)[self._param_var]

        constraints.append((t - p**2, 0.0, 0.0))

        return constraints

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        parameters[self._param_var] = np.arange(5)
        return parameters

    @property
    def extra_variables(self):
        return [self._additional_var]


class ModelAdditionalPathVariables(Model):
    def pre(self):
        super().pre()

        u1 = ca.MX.sym("u**1")
        u2 = ca.MX.sym("u**2")
        u3 = ca.MX.sym("u**3")

        self._additional_path_vars = [u1, u2, u3]

    @property
    def path_variables(self):
        return self._additional_path_vars

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        for x in self._additional_path_vars:
            p = int(x.name()[-1])
            constraints.append((self.state(x.name()) - self.state("u") ** p, 0.0, 0.0))

        return constraints


class ModelAdditionalPathVariablesVector(Model):
    def pre(self):
        super().pre()

        self._additional_path_var = ca.MX.sym("u**1,2,3", 3)

    @property
    def path_variables(self):
        return [self._additional_path_var]

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        x = self._additional_path_var
        constraints.append(
            (
                self.state(x.name())
                - ca.vertcat(self.state("u") ** 1, self.state("u") ** 2, self.state("u") ** 3),
                0.0,
                0.0,
            )
        )

        return constraints


class ModelAdditionalPathVariablesStatetAt(Model):
    def pre(self):
        super().pre()

        u1 = ca.MX.sym("u**1")
        u2 = ca.MX.sym("u**2")
        u3 = ca.MX.sym("u**3")

        self._additional_path_vars = [u1, u2, u3]

    @property
    def path_variables(self):
        return self._additional_path_vars

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        for t in self.times():
            for x in self._additional_path_vars:
                p = int(x.name()[-1])
                constraints.append(
                    (self.state_at(x.name(), t) - self.state_at("u", t) ** p, 0.0, 0.0)
                )

        return constraints


class ModelAdditionalPathVariablesStatetAtVector(Model):
    def pre(self):
        super().pre()

        self._additional_path_var = ca.MX.sym("u**1,2,3", 3)

    @property
    def path_variables(self):
        return [self._additional_path_var]

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        x = self._additional_path_var
        for t in self.times():
            constraints.append(
                (
                    self.state_at(x.name(), t)
                    - ca.vertcat(
                        self.state_at("u", t) ** 1,
                        self.state_at("u", t) ** 2,
                        self.state_at("u", t) ** 3,
                    ),
                    0.0,
                    0.0,
                )
            )

        return constraints


class ModelConstantInputs(Model):
    def pre(self):
        super().pre()

        self._input_vars = [ca.MX.sym(f"ci_{i}") for i in range(3)]
        self._additional_vars = [ca.MX.sym(f"ci2_{i}") for i in range(3)]

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        for t, p in zip(self._additional_vars, self._input_vars, strict=True):
            constraints.append((self.state(t.name()) - self.state(p.name()) ** 2, 0.0, 0.0))

        return constraints

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)

        for i, p in enumerate(self._input_vars):
            constant_inputs[p.name()] = Timeseries(
                self.times(), (i + 1) * np.arange(len(self.times()))
            )

        return constant_inputs

    @property
    def path_variables(self):
        return self._additional_vars


class ModelConstantInputsVector(Model):
    def pre(self):
        super().pre()

        self._input_var = ca.MX.sym("ci", 3)
        self._additional_var = ca.MX.sym("ci2", 3)

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)
        t = self.state(self._additional_var.name())
        p = self.state(self._input_var.name())
        constraints.append((t - p**2, np.zeros(3), np.zeros(3)))
        return constraints

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        n_times = len(self.times())
        constant_inputs[self._input_var.name()] = Timeseries(
            self.times(),
            np.stack(
                (1 * np.arange(n_times), 2 * np.arange(n_times), 3 * np.arange(n_times)), axis=1
            ),
        )
        return constant_inputs

    @property
    def path_variables(self):
        return [self._additional_var]


class TestVectorConstraints(TestCase):
    """
    NOTE: As long as the order of constraints is the same, whether or not they are passed
    as a vector or not should not matter. Therefore we often check to see if two problems
    are _exactly_ equal.
    """

    def test_vector_constraints(self):
        self.problem1 = ModelConstraints()
        self.problem2 = ModelConstraintsVector()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)

    def test_path_vector_constraints_simple(self):
        self.problem1 = ModelPathConstraintsSimple()
        self.problem2 = ModelPathConstraintsSimpleVector()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)

    def test_path_vector_constraints_timeseries(self):
        self.problem1 = ModelPathConstraintsTimeseries()
        self.problem2 = ModelPathConstraintsTimeseriesVector()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)

    def test_additional_variables_sanity(self):
        # Sanity check that the model is as it is intended, i.e. giving the
        # same answer as the original model.
        self.problem1 = ModelConstraints()
        self.problem2 = ModelAdditionalVariables()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertAlmostEqual(self.problem1.objective_value, self.problem2.objective_value, 1e-6)

    def test_additional_variables(self):
        self.problem1 = ModelAdditionalVariables()
        self.problem2 = ModelAdditionalVariablesVector()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)

    def test_vector_parameters(self):
        self.problem1 = ModelParameters()
        self.problem2 = ModelParametersVector()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        v1 = np.stack(tuple(results1[p.name()] for p in self.problem1._additional_vars), axis=1)
        v2 = results2[self.problem2._additional_var.name()]

        self.assertTrue(np.array_equal(v1, v2))

        ref = np.arange(len(self.problem1._additional_vars)) ** 2
        self.assertAlmostEqual(v1[0, :].flatten(), ref, 1e-6)

    def test_additional_path_variables(self):
        self.problem1 = ModelAdditionalPathVariables()
        self.problem2 = ModelAdditionalPathVariablesVector()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        ref = np.stack((results1["u"], results1["u"] ** 2, results1["u"] ** 3), axis=1)

        v1 = np.stack(
            tuple(results1[p.name()] for p in self.problem1._additional_path_vars), axis=1
        )
        v2 = results2[self.problem2._additional_path_var.name()]

        self.assertTrue(np.array_equal(v1, v2))
        self.assertAlmostEqual(v1, ref, 1e-6)

    def test_additional_path_variables_state_at_notimplemented(self):
        self.problem = ModelAdditionalPathVariablesStatetAtVector()
        with self.assertRaises(NotImplementedError):
            self.problem.optimize()

    @unittest.skip("state_at() not implemented yet for vector variables")
    def test_additional_path_variables_state_at(self):
        self.problem1 = ModelAdditionalPathVariables()
        self.problem2 = ModelAdditionalPathVariablesStatetAt()
        self.problem3 = ModelAdditionalPathVariablesStatetAtVector()
        self.problem1.optimize()
        self.problem2.optimize()
        self.problem3.optimize()

        # Order of constraints is different between problem1 and problem2
        self.assertAlmostEqual(self.problem1.objective_value, self.problem2.objective_value, 1e-6)
        self.assertEqual(self.problem2.objective_value, self.problem3.objective_value)

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()
        results3 = self.problem3.extract_results()

        ref = np.stack((results1["u"], results1["u"] ** 2, results1["u"] ** 3), axis=1)

        v1 = np.stack(
            tuple(results1[p.name()] for p in self.problem1._additional_path_vars), axis=1
        )
        v2 = np.stack(
            tuple(results2[p.name()] for p in self.problem2._additional_path_vars), axis=1
        )
        v3 = results3[self.problem3._additional_path_var.name()]

        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(np.array_equal(v1, v3))
        self.assertAlmostEqual(v1, ref, 1e-6)

    def test_vector_constant_inputs(self):
        self.problem1 = ModelConstantInputs()
        self.problem2 = ModelConstantInputsVector()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        refs = []
        for s in self.problem1._input_vars:
            refs.append(self.problem1.constant_inputs(0)[s.name()].values)
        ref = np.stack(refs, axis=1) ** 2

        v1 = np.stack(tuple(results1[p.name()] for p in self.problem1._additional_vars), axis=1)
        v2 = results2[self.problem2._additional_var.name()]

        self.assertTrue(np.array_equal(v1, v2))
        self.assertAlmostEqual(v1, ref, 1e-6)


class AdditionalNominals:
    def pre(self):
        super().pre()

        self._additional_vars_nominals = {}

    def variable_nominal(self, variable):
        if variable in self._additional_vars_nominals:
            return self._additional_vars_nominals[variable]
        else:
            return super().variable_nominal(variable)


class ModelAdditionalVariablesNominals(AdditionalNominals, ModelAdditionalVariables):
    def pre(self):
        super().pre()

        nominals = np.linspace(0.5, 2.0, len(self.times()))

        for i, v in enumerate(self._additional_vars):
            self._additional_vars_nominals[v.name()] = nominals[i]


class ModelAdditionalVariablesVectorNominals(AdditionalNominals, ModelAdditionalVariablesVector):
    def pre(self):
        super().pre()

        nominals = np.linspace(0.5, 2.0, len(self.times()))

        offset = 0
        for v in self._additional_vars:
            self._additional_vars_nominals[v.name()] = nominals[offset : offset + v.size1()]
            offset += v.size1()


class ModelAdditionalPathVariablesNominals(AdditionalNominals, ModelAdditionalPathVariables):
    def pre(self):
        super().pre()

        for i, v in enumerate(self._additional_path_vars):
            self._additional_vars_nominals[v.name()] = float(2 + i)


class ModelAdditionalPathVariablesVectorNominals(
    AdditionalNominals, ModelAdditionalPathVariablesVector
):
    def pre(self):
        super().pre()

        self._additional_vars_nominals = {self._additional_path_var.name(): np.array([2, 3, 4])}


class TestVectorNominals(TestCase):
    def test_additional_variables(self):
        self.problem1 = ModelAdditionalVariablesNominals()
        self.problem2 = ModelAdditionalVariablesVectorNominals()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)
        self.assertTrue(np.array_equal(self.problem1.solver_output, self.problem2.solver_output))

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        v1 = np.hstack([results1[p.name()].ravel() for p in self.problem1._additional_vars])
        v2 = np.hstack([results2[p.name()].ravel() for p in self.problem2._additional_vars])

        self.assertTrue(np.array_equal(v1, v2))

    def test_additional_path_variables(self):
        self.problem1 = ModelAdditionalPathVariablesNominals()
        self.problem2 = ModelAdditionalPathVariablesVectorNominals()
        self.problem1.optimize()
        self.problem2.optimize()

        self.assertEqual(self.problem1.objective_value, self.problem2.objective_value)
        self.assertTrue(np.array_equal(self.problem1.solver_output, self.problem2.solver_output))

        results1 = self.problem1.extract_results()
        results2 = self.problem2.extract_results()

        ref = np.stack((results1["u"], results1["u"] ** 2, results1["u"] ** 3), axis=1)

        v1 = np.stack(
            tuple(results1[p.name()] for p in self.problem1._additional_path_vars), axis=1
        )
        v2 = results2[self.problem2._additional_path_var.name()]

        self.assertTrue(np.array_equal(v1, v2))
        self.assertAlmostEqual(v1, ref, 1e-6)


class InvalidVectorConstraintLbg(ModelAdditionalVariablesVector):
    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        g, lbg, ubg = constraints[-1]

        assert g.size1() > 1
        assert np.isscalar(lbg)

        # Make a wrong lbg, to check that we get an exception
        constraints[-1] = (g, np.full(g.size1() - 1, lbg), ubg)

        return constraints


class InvalidVectorConstraintUbg(ModelAdditionalVariablesVector):
    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        g, lbg, ubg = constraints[-1]

        assert g.size1() > 1
        assert np.isscalar(ubg)

        # Make a wrong ubg, to check that we get an exception
        constraints[-1] = (g, lbg, np.full(g.size1() - 1, ubg))

        return constraints


class ModelScalarConstraints(Model):
    """Returns plain Python float and numpy scalar as constraints."""

    def constraints(self, ensemble_member):
        return [
            (1.0, 1.0, 1.0),
            (np.float64(0.5), 0.0, 1.0),
        ]


class TestScalarConstraints(TestCase):
    def test_scalar_constraints_do_not_raise(self):
        """Scalar float/numpy constraint values must be accepted without raising AttributeError."""
        ModelScalarConstraints().transcribe()


class TestInvalidVectorConstraints(TestCase):
    def test_vector_constraint_lbg(self):
        self.problem = InvalidVectorConstraintLbg()

        with self.assertRaisesRegex(Exception, "Shape mismatch .* lower bound"):
            self.problem.optimize()

    def test_vector_constraint_ubg(self):
        self.problem = InvalidVectorConstraintUbg()

        with self.assertRaisesRegex(Exception, "Shape mismatch .* upper bound"):
            self.problem.optimize()
