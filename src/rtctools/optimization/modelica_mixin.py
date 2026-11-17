import itertools
import logging

import casadi as ca
import numpy as np

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.caching import cached
from rtctools._internal.casadi_helpers import substitute_in_external
from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check
from rtctools._internal.modelica_mixin_base import _ModelicaMixinBase

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class ModelicaMixin(_ModelicaMixinBase, OptimizationProblem):
    """
    Adds a `Modelica <http://www.modelica.org/>`_ model to your optimization problem.

    During preprocessing, the Modelica files located inside the ``model`` subfolder are loaded.

    :cvar modelica_library_folders:
        Folders in which any referenced Modelica libraries are to be found.
        Default is an empty list.
    """

    # Folders in which the referenced Modelica libraries are found
    modelica_library_folders = []

    def _mo_extra_variable_groups(self):
        return {
            "string_parameters": [
                v.name
                for v in (
                    *self._mo_pymoca_model.string_parameters,
                    *self._mo_pymoca_model.string_constants,
                )
            ],
            "control_inputs": [],
        }

    def _mo_input_group(self, variable):
        return "constant_inputs" if variable.fixed else "control_inputs"

    def _mo_init_from_model(self, **kwargs):
        # Initialize nominals and types
        # These are not in @cached dictionary properties for backwards compatibility.
        self.__python_types = AliasDict(self.alias_relation)
        for v in itertools.chain(
            self._mo_pymoca_model.states,
            self._mo_pymoca_model.alg_states,
            self._mo_pymoca_model.inputs,
        ):
            self.__python_types[v.symbol.name()] = v.python_type

    def _mo_init_after_residuals(self, **kwargs):
        # Log variables in debug mode
        self._mo_log_dae_variables("ModelicaMixin")

    @cached
    def compiler_options(self) -> dict[str, str | bool]:
        compiler_options = super().compiler_options()

        # Disallow aliasing to derivative states
        compiler_options["allow_derivative_aliases"] = False

        return compiler_options

    def delayed_feedback(self):
        delayed_feedback = super().delayed_feedback()

        # Create delayed feedback
        for delay_state, delay_argument in zip(
            self._mo_pymoca_model.delay_states, self._mo_pymoca_model.delay_arguments, strict=True
        ):
            delayed_feedback.append((delay_argument.expr, delay_state, delay_argument.duration))
        return delayed_feedback

    @property
    def dae_residual(self):
        return self._mo_dae_residual

    @property
    def dae_variables(self):
        return self._mo_mx

    @property
    @cached
    def output_variables(self):
        output_variables = [ca.MX.sym(variable) for variable in self._mo_pymoca_model.outputs]
        output_variables.extend(self._mo_mx["control_inputs"])
        return output_variables

    @cached
    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super().parameters(ensemble_member)

        # Return parameter values from pymoca model
        parameters.update({v.symbol.name(): v.value for v in self._mo_pymoca_model.parameters})

        # Done
        return parameters

    @cached
    def string_parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super().string_parameters(ensemble_member)

        # Return parameter values from pymoca model
        parameters.update({v.name: v.value for v in self._mo_pymoca_model.string_parameters})
        parameters.update({v.name: v.value for v in self._mo_pymoca_model.string_constants})

        # Done
        return parameters

    @cached
    def history(self, ensemble_member):
        history = super().history(ensemble_member)

        initial_time = np.array([self.initial_time])

        # Parameter values
        parameters = self.parameters(ensemble_member)
        parameter_values = [
            parameters.get(param.name(), param) for param in self._mo_mx["parameters"]
        ]

        # Initial conditions obtained from start attributes.
        for v in self._mo_pymoca_model.states:
            if v.fixed:
                sym_name = v.symbol.name()
                start = v.start

                if isinstance(start, ca.MX):
                    # If start contains symbolics, try substituting parameter values
                    if isinstance(start, ca.MX) and not start.is_constant():
                        [start] = substitute_in_external(
                            [start],
                            self._mo_mx["parameters"],
                            parameter_values,
                            resolve_numerically=True,
                        )
                        if not start.is_constant() or np.isnan(float(start)):
                            raise Exception(
                                f"ModelicaMixin: Could not resolve initial value for {sym_name}"
                            )

                    start = v.python_type(start)

                history[sym_name] = Timeseries(initial_time, start)

                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(f"ModelicaMixin: Initial state variable {sym_name} = {start}")

        return history

    @property
    def initial_residual(self):
        return self._mo_initial_residual

    @cached
    @ensemble_bounds_check
    def bounds(self, ensemble_member: int | None = None):
        bounds = (
            super().bounds(ensemble_member) if self.ensemble_specific_bounds else super().bounds()
        )

        ensemble_member = ensemble_member if self.ensemble_specific_bounds else 0

        # Parameter values
        parameters = self.parameters(ensemble_member)
        parameter_values = [
            parameters.get(param.name(), param) for param in self._mo_mx["parameters"]
        ]

        # Load additional bounds from model
        for v in itertools.chain(
            self._mo_pymoca_model.states,
            self._mo_pymoca_model.alg_states,
            self._mo_pymoca_model.inputs,
        ):
            sym_name = v.symbol.name()

            try:
                (m, M) = bounds[sym_name]
            except KeyError:
                if self.__python_types.get(sym_name, float) is bool:
                    (m, M) = (0, 1)
                else:
                    (m, M) = (-np.inf, np.inf)

            m_ = v.min
            if isinstance(m_, ca.MX) and not m_.is_constant():
                [m_] = substitute_in_external(
                    [m_], self._mo_mx["parameters"], parameter_values, resolve_numerically=True
                )
                if not m_.is_constant() or np.isnan(float(m_)):
                    raise Exception(f"Could not resolve lower bound for variable {sym_name}")
            m_ = float(m_)

            M_ = v.max
            if isinstance(M_, ca.MX) and not M_.is_constant():
                [M_] = substitute_in_external(
                    [M_], self._mo_mx["parameters"], parameter_values, resolve_numerically=True
                )
                if not M_.is_constant() or np.isnan(float(M_)):
                    raise Exception(f"Could not resolve upper bound for variable {sym_name}")
            M_ = float(M_)

            # We take the intersection of all provided bounds
            m = max(m, m_)
            M = min(M, M_)

            bounds[sym_name] = (m, M)

        return bounds

    @cached
    def seed(self, ensemble_member):
        # Call parent class first for default values.
        seed = super().seed(ensemble_member)

        # Parameter values
        parameters = self.parameters(ensemble_member)
        parameter_values = [
            parameters.get(param.name(), param) for param in self._mo_mx["parameters"]
        ]

        # Load seeds
        for var in itertools.chain(self._mo_pymoca_model.states, self._mo_pymoca_model.alg_states):
            if var.fixed:
                # Values will be set from import timeseries
                continue

            start = var.start

            if isinstance(start, ca.MX) or start != 0.0:
                sym_name = var.symbol.name()

                # If start contains symbolics, try substituting parameter values
                if isinstance(start, ca.MX) and not start.is_constant():
                    [start] = substitute_in_external(
                        [start],
                        self._mo_mx["parameters"],
                        parameter_values,
                        resolve_numerically=True,
                    )
                    if not start.is_constant() or np.isnan(float(start)):
                        logger.error(f"ModelicaMixin: Could not resolve seed value for {sym_name}")
                        continue

                times = self.times(sym_name)
                start = var.python_type(start)
                s = Timeseries(times, np.full_like(times, start))
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(f"ModelicaMixin: Seeded variable {sym_name} = {start}")
                seed[sym_name] = s

        return seed

    def variable_is_discrete(self, variable):
        return self.__python_types.get(variable, float) is not float

    @property
    @cached
    def __nominals(self):
        # Make the dict
        nominal_dict = AliasDict(self.alias_relation)

        # Grab parameters and their values
        parameters = self.parameters(0)
        parameter_values = [
            parameters.get(param.name(), param) for param in self._mo_mx["parameters"]
        ]

        # Iterate over nominalizable states
        for v in itertools.chain(
            self._mo_pymoca_model.states,
            self._mo_pymoca_model.alg_states,
            self._mo_pymoca_model.inputs,
        ):
            sym_name = v.symbol.name()
            nominal = v.nominal

            # If nominal contains parameter symbols, substitute them
            if isinstance(nominal, ca.MX) and not nominal.is_constant():
                [nominal] = substitute_in_external(
                    [nominal], self._mo_mx["parameters"], parameter_values, resolve_numerically=True
                )
                if not nominal.is_constant() or np.isnan(float(nominal)):
                    logger.error(f"ModelicaMixin: Could not resolve nominal value for {sym_name}")
                    continue

            nominal = float(nominal)

            if not np.isnan(nominal):
                # Take absolute value (nominal sign is meaningless- a nominal is a magnitude)
                nominal = abs(nominal)

                # If nominal is 0 or 1, we just use the default (1.0)
                if nominal in (0.0, 1.0):
                    continue

                nominal_dict[sym_name] = nominal

                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(
                        f"ModelicaMixin: Set nominal value for variable {sym_name} to {nominal}"
                    )
            else:
                logger.warning(f"ModelicaMixin: Could not set nominal value for {sym_name}")

        return nominal_dict

    def variable_nominal(self, variable):
        try:
            return self.__nominals[variable]
        except KeyError:
            return super().variable_nominal(variable)
