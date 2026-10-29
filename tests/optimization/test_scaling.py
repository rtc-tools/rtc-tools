import collections
import unittest

import casadi as ca

from ..test_case import TestCase
from .test_integration import SingleShootingModel
from .test_modelica_mixin import ModelHistory


def repeated_scalings(nlp, nominals):
    """Products X[k] * nominal that the expanded NLP computes more than once"""
    f = ca.Function("nlp", [nlp["x"]], [nlp["f"], nlp["g"]]).expand()
    work = {}
    products = collections.Counter()
    for k in range(f.n_instructions()):
        op = f.instruction_id(k)
        if op == ca.OP_OUTPUT:
            continue
        if op == ca.OP_INPUT:
            value = ("x", f.instruction_input(k)[1])
        elif op == ca.OP_CONST:
            value = ("c", f.instruction_constant(k))
        else:
            if op == ca.OP_MUL:
                inputs = [work[i] for i in f.instruction_input(k)]
                if {inputs[0][0], inputs[1][0]} == {"x", "c"}:
                    c, x = sorted(inputs)
                    if c[1] in nominals:
                        products[x[1], c[1]] += 1
            value = ("other", k)
        for i in f.instruction_output(k):
            work[i] = value
    return [p for p, n in products.items() if n > 1]


class ModelHistoryScaled(ModelHistory):
    # Not 2, because CasADi expands x * 2 to x + x. 30 is the nominal of initial_der(x), 3 / dt.
    nominals = {3.0, 30.0}

    def variable_nominal(self, variable):
        if variable in {"x", "w"}:
            return 3.0
        else:
            return super().variable_nominal(variable)

    def constraints(self, ensemble_member):
        t0 = self.initial_time
        return [
            (self.der_at("x", t0, ensemble_member), -10.0, 10.0),
            (self.state_at("x", t0 + 0.05, ensemble_member), -10.0, 10.0),
            (ca.sum1(self.states_in("x", ensemble_member=ensemble_member)), -100.0, 100.0),
        ]


class SingleShootingModelScaled(SingleShootingModel):
    nominals = {3.0}

    def variable_nominal(self, variable):
        if variable == "x":
            return 3.0
        else:
            return super().variable_nominal(variable)

    def objective(self, ensemble_member):
        return ca.sumsqr(self.states_in("x", ensemble_member=ensemble_member))


class TestScaleSolverInputOnce(TestCase):
    def assert_scaled_once(self, problem):
        problem.pre()
        nlp = problem.transcribe()[-1]
        self.assertEqual(repeated_scalings(nlp, problem.nominals), [])

    def test_collocation(self):
        self.assert_scaled_once(ModelHistoryScaled())

    @unittest.skipIf(
        tuple(int(v) for v in ca.__version__.split(".")[:2]) < (3, 7),
        "CasADi < 3.7 cannot expand the integrator's rootfinder",
    )
    def test_integrate_states(self):
        self.assert_scaled_once(SingleShootingModelScaled())
