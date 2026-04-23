"""
Regression tests for path-variable assembly in transcribe().

--- Empty path variables (CasADi 3.6+) ---

ca.horzcat(*[]) now returns DM(1,0) instead of the old Sparsity(0,0).
This caused a row-count mismatch when assembling accumulation_U for models
with no path variables, crashing transcribe() before the solver ran.

--- NaN placeholder miscounts for delayed feedback ---

When delayed expressions are present, delayed_feedback_function is called
numerically with NaN placeholders for path variables and extra constant inputs.
The placeholder sizes must equal path_variables_size and
extra_constant_inputs_size (sum of size1() across all symbols). Using len()
instead counts symbols, not elements, so any vector-valued path variable or
extra constant input (size1() > 1) causes a RuntimeError before the solver runs.
"""

from pathlib import Path

import casadi as ca
import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from ..test_case import TestCase
from .data_path import data_path
from .test_delay_expressions import ModelCompleteHistory

MODEL_SOURCE = """\
model SimpleIntegrator
    Real x(start=0.0, nominal=1.0);
    input Real u(fixed=false, nominal=1.0);
equation
    der(x) = u;
end SimpleIntegrator;
"""

MODEL_NAME = "SimpleIntegrator"


class _BaseProblem(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name=MODEL_NAME,
            model_folder=data_path(),
        )

    def times(self, variable=None):
        return np.linspace(0.0, 1.0, 11)

    def bounds(self):
        bounds = super().bounds()
        bounds["u"] = (-2.0, 2.0)
        return bounds

    def path_objective(self, ensemble_member):
        return self.state("u") ** 2

    def set_timeseries(self, timeseries_id, timeseries, ensemble_member, **kwargs):
        pass

    def compiler_options(self):
        opts = super().compiler_options()
        opts["cache"] = False
        opts["library_folders"] = []
        return opts


class ModelNoPathVariables(_BaseProblem):
    pass


class TestNoPathVariables(TestCase):
    """transcribe() must not crash when no path variables are present."""

    def setUp(self):
        self.model_path = Path(data_path()) / f"{MODEL_NAME}.mo"
        self.model_path.write_text(MODEL_SOURCE)
        self.problem = ModelNoPathVariables()
        self.problem.optimize()

    def tearDown(self):
        self.model_path.unlink(missing_ok=True)

    def test_transcribe_does_not_crash(self):
        """optimize() completing without error is the regression check."""
        self.assertIsNotNone(self.problem.objective_value)


class ModelVectorPathVariableWithDelay(ModelCompleteHistory):
    """ModelDelay extended with a size-3 path variable.

    The delayed expressions force delayed_feedback_function to be called
    numerically with a NaN placeholder whose size must equal
    path_variables_size, not len(path_variables).
    """

    def pre(self):
        super().pre()
        self._pv = ca.MX.sym("pv", 3)

    @property
    def path_variables(self):
        return [self._pv]

    def bounds(self):
        bounds = super().bounds()
        bounds["pv"] = (np.full(3, -2.0), np.full(3, 2.0))
        return bounds

    def path_constraints(self, ensemble_member):
        u3 = ca.vertcat(self.state("u"), self.state("u"), self.state("u"))
        return [(self.state("pv") - u3, np.zeros(3), np.zeros(3))]


class TestVectorPathVariableWithDelayedFeedback(TestCase):
    """Vector path variable must not corrupt the NaN placeholder for delayed feedback.

    The delayed_feedback_function is called numerically with a placeholder of
    size path_variables_size. Using len(path_variables) instead produces a
    size mismatch and a RuntimeError before the solver runs.
    """

    def test_optimize_does_not_crash(self):
        ModelVectorPathVariableWithDelay().optimize()


class ModelVectorExtraConstantInputWithDelay(ModelCompleteHistory):
    """ModelDelay extended with a size-2 extra constant input.

    The delayed expressions force delayed_feedback_function to be called
    numerically with a NaN placeholder whose size must equal
    extra_constant_inputs_size, not len(self.__extra_constant_inputs).
    """

    def constant_inputs(self, ensemble_member):
        inputs = super().constant_inputs(ensemble_member)
        times = self.times()
        # A 2-element vector timeseries not present in the DAE becomes an
        # extra constant input with size1()=2, while len() counts it as 1.
        inputs["extra_vec"] = Timeseries(times, np.ones((len(times), 2)))
        return inputs


class TestVectorExtraConstantInputWithDelay(TestCase):
    """Vector extra constant inputs must not corrupt the NaN placeholder for delayed feedback.

    delayed_feedback_function is called numerically with a placeholder of size
    extra_constant_inputs_size (sum of size1()). Using len() instead produces a
    size mismatch and a RuntimeError before the solver runs.
    """

    def test_optimize_does_not_crash(self):
        ModelVectorExtraConstantInputWithDelay().optimize()
