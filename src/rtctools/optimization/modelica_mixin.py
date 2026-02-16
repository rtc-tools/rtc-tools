import importlib.resources
import itertools
import logging
from importlib import metadata as importlib_metadata

import casadi as ca
import numpy as np
import pymoca
import pymoca.backends.casadi.api

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.caching import cached
from rtctools._internal.casadi_helpers import substitute_in_external
from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class ModelicaMixin(OptimizationProblem):
    """
    Adds a `Modelica <http://www.modelica.org/>`_ model to your optimization problem.

    During preprocessing, the Modelica files located inside the ``model`` subfolder are loaded.

    :cvar modelica_library_folders:
        Folders in which any referenced Modelica libraries are to be found.
        Default is an empty list.
    """

    # Folders in which the referenced Modelica libraries are found
    modelica_library_folders = []

    def __init__(self, **kwargs):
        # Check arguments
        assert "model_folder" in kwargs

        # Log pymoca version
        logger.debug(f"Using pymoca {pymoca.__version__}.")

        # Transfer model from the Modelica .mo file to CasADi using pymoca
        if "model_name" in kwargs:
            model_name = kwargs["model_name"]
        else:
            if hasattr(self, "model_name"):
                model_name = self.model_name
            else:
                model_name = self.__class__.__name__

        compiler_options = self.compiler_options()
        logger.info(f"Loading/compiling model {model_name}.")
        try:
            self.__pymoca_model = pymoca.backends.casadi.api.transfer_model(
                kwargs["model_folder"], model_name, compiler_options
            )
        except (RuntimeError, ModuleNotFoundError) as error:
            if not compiler_options.get("cache", False):
                raise error
            compiler_options["cache"] = False
            logger.warning(f"Loading model {model_name} using a cache file failed: {error}.")
            logger.info(f"Compiling model {model_name}.")
            self.__pymoca_model = pymoca.backends.casadi.api.transfer_model(
                kwargs["model_folder"], model_name, compiler_options
            )

        # Extract the CasADi MX variables used in the model
        self.__mx = {}
        self.__mx["time"] = [self.__pymoca_model.time]
        self.__mx["states"] = [v.symbol for v in self.__pymoca_model.states]
        self.__mx["derivatives"] = [v.symbol for v in self.__pymoca_model.der_states]
        self.__mx["algebraics"] = [v.symbol for v in self.__pymoca_model.alg_states]
        self.__mx["parameters"] = [v.symbol for v in self.__pymoca_model.parameters]
        self.__mx["string_parameters"] = [
            v.name
            for v in (*self.__pymoca_model.string_parameters, *self.__pymoca_model.string_constants)
        ]
        self.__mx["control_inputs"] = []
        self.__mx["constant_inputs"] = []
        self.__mx["lookup_tables"] = []

        # Merge with user-specified delayed feedback
        for v in self.__pymoca_model.inputs:
            if v.symbol.name() in self.__pymoca_model.delay_states:
                # Delayed feedback variables are local to each ensemble, and
                # therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                self.__mx["algebraics"].append(v.symbol)
            else:
                if v.symbol.name() in kwargs.get("lookup_tables", []):
                    self.__mx["lookup_tables"].append(v.symbol)
                elif v.fixed:
                    self.__mx["constant_inputs"].append(v.symbol)
                else:
                    self.__mx["control_inputs"].append(v.symbol)

        # Initialize nominals and types
        # These are not in @cached dictionary properties for backwards compatibility.
        self.__python_types = AliasDict(self.alias_relation)
        for v in itertools.chain(
            self.__pymoca_model.states, self.__pymoca_model.alg_states, self.__pymoca_model.inputs
        ):
            self.__python_types[v.symbol.name()] = v.python_type

        # Initialize dae, initial residuals, as well as delay arguments
        # These are not in @cached dictionary properties so that we need to create the list
        # of function arguments only once.
        variable_lists = ["states", "der_states", "alg_states", "inputs", "constants", "parameters"]
        function_arguments = [self.__pymoca_model.time] + [
            ca.veccat(*[v.symbol for v in getattr(self.__pymoca_model, variable_list)])
            for variable_list in variable_lists
        ]

        self.__dae_residual = self.__pymoca_model.dae_residual_function(*function_arguments)
        if self.__dae_residual is None:
            self.__dae_residual = ca.MX()

        self.__initial_residual = self.__pymoca_model.initial_residual_function(*function_arguments)
        if self.__initial_residual is None:
            self.__initial_residual = ca.MX()

        # Log variables in debug mode
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(
                "ModelicaMixin: Found states {}".format(
                    ", ".join([var.name() for var in self.__mx["states"]])
                )
            )
            logger.debug(
                "ModelicaMixin: Found derivatives {}".format(
                    ", ".join([var.name() for var in self.__mx["derivatives"]])
                )
            )
            logger.debug(
                "ModelicaMixin: Found algebraics {}".format(
                    ", ".join([var.name() for var in self.__mx["algebraics"]])
                )
            )
            logger.debug(
                "ModelicaMixin: Found control inputs {}".format(
                    ", ".join([var.name() for var in self.__mx["control_inputs"]])
                )
            )
            logger.debug(
                "ModelicaMixin: Found constant inputs {}".format(
                    ", ".join([var.name() for var in self.__mx["constant_inputs"]])
                )
            )
            logger.debug(
                "ModelicaMixin: Found parameters {}".format(
                    ", ".join([var.name() for var in self.__mx["parameters"]])
                )
            )

        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

    @cached
    def compiler_options(self) -> dict[str, str | bool]:
        """
        Subclasses can configure the `pymoca <http://github.com/pymoca/pymoca>`_ compiler options
        here.

        :returns:
            A dictionary of pymoca compiler options.  See the pymoca documentation for details.
        """

        # Default options
        compiler_options = {}

        # Expand vector states to multiple scalar component states.
        compiler_options["expand_vectors"] = True

        # Where imported model libraries are located.
        library_folders = self.modelica_library_folders.copy()

        for ep in importlib_metadata.entry_points(group="rtctools.libraries.modelica"):
            if ep.name == "library_folder":
                library_folders.append(str(importlib.resources.files(ep.module).joinpath(ep.attr)))

        compiler_options["library_folders"] = library_folders

        # Eliminate equations of the type 'var = const'.
        compiler_options["eliminate_constant_assignments"] = True

        # Eliminate constant symbols from model, replacing them with the values
        # specified in the model.
        compiler_options["replace_constant_values"] = True

        # Replace any constant expressions into the model.
        compiler_options["replace_constant_expressions"] = True

        # Replace any parameter expressions into the model.
        compiler_options["replace_parameter_expressions"] = True

        # Eliminate variables starting with underscores.
        compiler_options["eliminable_variable_expression"] = r"(.*[.]|^)_\w+(\[[\d,]+\])?\Z"

        # Pymoca currently requires `expand_mx` to be set for
        # `eliminable_variable_expression` to work.
        compiler_options["expand_mx"] = True

        # Automatically detect and eliminate alias variables.
        compiler_options["detect_aliases"] = True

        # Disallow aliasing to derivative states
        compiler_options["allow_derivative_aliases"] = False

        # Cache the model on disk
        compiler_options["cache"] = True

        # Done
        return compiler_options

    def delayed_feedback(self):
        delayed_feedback = super().delayed_feedback()

        # Create delayed feedback
        for delay_state, delay_argument in zip(
            self.__pymoca_model.delay_states, self.__pymoca_model.delay_arguments, strict=True
        ):
            delayed_feedback.append((delay_argument.expr, delay_state, delay_argument.duration))
        return delayed_feedback

    @property
    def dae_residual(self):
        return self.__dae_residual

    @property
    def dae_variables(self):
        return self.__mx

    @property
    @cached
    def output_variables(self):
        output_variables = [ca.MX.sym(variable) for variable in self.__pymoca_model.outputs]
        output_variables.extend(self.__mx["control_inputs"])
        return output_variables

    @cached
    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super().parameters(ensemble_member)

        # Return parameter values from pymoca model
        parameters.update({v.symbol.name(): v.value for v in self.__pymoca_model.parameters})

        # Done
        return parameters

    @cached
    def string_parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super().string_parameters(ensemble_member)

        # Return parameter values from pymoca model
        parameters.update({v.name: v.value for v in self.__pymoca_model.string_parameters})
        parameters.update({v.name: v.value for v in self.__pymoca_model.string_constants})

        # Done
        return parameters

    @cached
    def history(self, ensemble_member):
        history = super().history(ensemble_member)

        initial_time = np.array([self.initial_time])

        # Parameter values
        parameters = self.parameters(ensemble_member)
        parameter_values = [
            parameters.get(param.name(), param) for param in self.__mx["parameters"]
        ]

        # Initial conditions obtained from start attributes.
        for v in self.__pymoca_model.states:
            if v.fixed:
                sym_name = v.symbol.name()
                start = v.start

                if isinstance(start, ca.MX):
                    # If start contains symbolics, try substituting parameter values
                    if isinstance(start, ca.MX) and not start.is_constant():
                        [start] = substitute_in_external(
                            [start],
                            self.__mx["parameters"],
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
        return self.__initial_residual

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
            parameters.get(param.name(), param) for param in self.__mx["parameters"]
        ]

        # Load additional bounds from model
        for v in itertools.chain(
            self.__pymoca_model.states, self.__pymoca_model.alg_states, self.__pymoca_model.inputs
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
                    [m_], self.__mx["parameters"], parameter_values, resolve_numerically=True
                )
                if not m_.is_constant() or np.isnan(float(m_)):
                    raise Exception(f"Could not resolve lower bound for variable {sym_name}")
            m_ = float(m_)

            M_ = v.max
            if isinstance(M_, ca.MX) and not M_.is_constant():
                [M_] = substitute_in_external(
                    [M_], self.__mx["parameters"], parameter_values, resolve_numerically=True
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
            parameters.get(param.name(), param) for param in self.__mx["parameters"]
        ]

        # Load seeds
        for var in itertools.chain(self.__pymoca_model.states, self.__pymoca_model.alg_states):
            if var.fixed:
                # Values will be set from import timeseries
                continue

            start = var.start

            if isinstance(start, ca.MX) or start != 0.0:
                sym_name = var.symbol.name()

                # If start contains symbolics, try substituting parameter values
                if isinstance(start, ca.MX) and not start.is_constant():
                    [start] = substitute_in_external(
                        [start], self.__mx["parameters"], parameter_values, resolve_numerically=True
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
    def alias_relation(self):
        return self.__pymoca_model.alias_relation

    @property
    @cached
    def __nominals(self):
        # Make the dict
        nominal_dict = AliasDict(self.alias_relation)

        # Grab parameters and their values
        parameters = self.parameters(0)
        parameter_values = [
            parameters.get(param.name(), param) for param in self.__mx["parameters"]
        ]

        # Iterate over nominalizable states
        for v in itertools.chain(
            self.__pymoca_model.states, self.__pymoca_model.alg_states, self.__pymoca_model.inputs
        ):
            sym_name = v.symbol.name()
            nominal = v.nominal

            # If nominal contains parameter symbols, substitute them
            if isinstance(nominal, ca.MX) and not nominal.is_constant():
                [nominal] = substitute_in_external(
                    [nominal], self.__mx["parameters"], parameter_values, resolve_numerically=True
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
