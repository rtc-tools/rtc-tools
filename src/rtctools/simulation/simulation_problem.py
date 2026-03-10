import copy
import importlib.resources
import itertools
import logging
import math
from collections import OrderedDict
from importlib import metadata as importlib_metadata

import casadi as ca
import numpy as np
import pymoca
import pymoca.backends.casadi.api

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.caching import cached
from rtctools._internal.debug_check_helpers import DebugLevel
from rtctools._internal.modelica_errors import raise_if_missing_msl
from rtctools.data.storage import DataStoreAccessor

logger = logging.getLogger("rtctools")


class Variable:
    """
    Modeled after the Variable class in pymoca.backends.casadi.model, with modifications to make it
    easier for the common case in RTC-Tools to instantiate them.

    That means:
    - pass in name instead of ca.MX symbol
    - only scalars are allowed (shape = (1, 1))
    - no aliases
    - no "python_type"
    - able to specify nominal/min/max in constructor
    """

    def __init__(self, name, /, min=-np.inf, max=np.inf, nominal=1.0):
        self.name = name
        self.min = min
        self.max = max
        self.nominal = nominal
        self._symbol = ca.MX.sym(name)

    @property
    def symbol(self):
        return self._symbol


class SimulationProblem(DataStoreAccessor):
    """
    Implements the `BMI <http://csdms.colorado.edu/wiki/BMI_Description>`_ Interface.

    Base class for all Simulation problems. Loads the Modelica Model.

    :cvar modelica_library_folders: Folders containing any referenced Modelica libraries. Default
        is an empty list.

    """

    _debug_check_level = DebugLevel.MEDIUM
    _debug_check_options = {}

    # Folders in which the referenced Modelica libraries are found
    modelica_library_folders = []

    # Force workaround for delay support by assuming zero delay. This flag
    # will be removed when proper delay support is added.
    _force_zero_delay = False

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

        # Load model from pymoca backend
        compiler_options = self.compiler_options()
        logger.info(f"Loading/compiling model {model_name}.")
        try:
            self.__pymoca_model = pymoca.backends.casadi.api.transfer_model(
                kwargs["model_folder"], model_name, compiler_options
            )
        except RuntimeError as error:
            raise_if_missing_msl(error, kwargs["model_folder"])
            if compiler_options.get("cache", False):
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
        self.__mx["constant_inputs"] = []
        self.__mx["lookup_tables"] = []

        for v in self.__pymoca_model.inputs:
            if v.symbol.name() in self.__pymoca_model.delay_states:
                # Delayed feedback variables are local to each ensemble, and
                # therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                self.__mx["algebraics"].append(v.symbol)
            else:
                if v.symbol.name() in kwargs.get("lookup_tables", []):
                    self.__mx["lookup_tables"].append(v.symbol)
                else:
                    # All inputs are constant inputs
                    self.__mx["constant_inputs"].append(v.symbol)

        # Set timestep size
        self._dt_is_fixed = False
        self.__dt = None
        fixed_dt = kwargs.get("fixed_dt", None)
        if fixed_dt is not None:
            self._dt_is_fixed = True
            self.__dt = fixed_dt

        # Add auxiliary variables for keeping track of delay expressions to the algebraic states
        n_delay_states = len(self.__pymoca_model.delay_states)
        self.__delay_times = []
        if n_delay_states > 0:
            if fixed_dt is None and not self._force_zero_delay:
                raise ValueError("fixed_dt should be set when using delay equations.")
            self.__delay_times = self._get_delay_times()
            delay_expression_states = self._create_delay_expression_states()
            self.__mx["algebraics"] += delay_expression_states

        # Log variables in debug mode
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(
                "SimulationProblem: Found states {}".format(
                    ", ".join([var.name() for var in self.__mx["states"]])
                )
            )
            logger.debug(
                "SimulationProblem: Found derivatives {}".format(
                    ", ".join([var.name() for var in self.__mx["derivatives"]])
                )
            )
            logger.debug(
                "SimulationProblem: Found algebraics {}".format(
                    ", ".join([var.name() for var in self.__mx["algebraics"]])
                )
            )
            logger.debug(
                "SimulationProblem: Found constant inputs {}".format(
                    ", ".join([var.name() for var in self.__mx["constant_inputs"]])
                )
            )
            logger.debug(
                "SimulationProblem: Found parameters {}".format(
                    ", ".join([var.name() for var in self.__mx["parameters"]])
                )
            )

        # Get the extra variables that are user defined
        self.__extra_variables = self.extra_variables()
        self.__extra_variables_symbols = [v.symbol for v in self.__extra_variables]

        # Store the types in an AliasDict
        self.__python_types = AliasDict(self.alias_relation)
        model_variable_types = [
            "states",
            "der_states",
            "alg_states",
            "inputs",
            "constants",
            "parameters",
        ]
        for t in model_variable_types:
            for v in getattr(self.__pymoca_model, t):
                self.__python_types[v.symbol.name()] = v.python_type

        # Store the nominals in an AliasDict
        self.__nominals = AliasDict(self.alias_relation)
        for v in itertools.chain(self.__pymoca_model.states, self.__pymoca_model.alg_states):
            sym_name = v.symbol.name()

            # If the nominal is 0.0 or 1.0 or -1.0, ignore: get_variable_nominal returns a default
            # of 1.0
            # TODO: handle nominal vectors (update() will need to load them)
            if (
                ca.MX(v.nominal).is_zero()
                or ca.MX(v.nominal - 1).is_zero()
                or ca.MX(v.nominal + 1).is_zero()
            ):
                continue
            else:
                if ca.MX(v.nominal).size1() != 1:
                    logger.error(f"Vector Nominals not supported yet. ({sym_name})")
                self.__nominals[sym_name] = ca.fabs(v.nominal)
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(
                        f"SimulationProblem: Setting nominal value for variable {sym_name} "
                        f"to {self.__nominals[sym_name]}"
                    )

        for v in self.__extra_variables:
            self.__nominals[v.name] = v.nominal

        # Initialize DAE and initial residuals
        variable_lists = ["states", "der_states", "alg_states", "inputs", "constants", "parameters"]
        function_arguments = [self.__pymoca_model.time] + [
            ca.veccat(*[v.symbol for v in getattr(self.__pymoca_model, variable_list)])
            for variable_list in variable_lists
        ]

        self.__dae_residual = self.__pymoca_model.dae_residual_function(*function_arguments)

        if self.__dae_residual is None:
            # DAE is empty, that can happen if we add the only (non-aliasing) equations
            # in Python.
            self.__dae_residual = ca.MX()

        self.__initial_residual = self.__pymoca_model.initial_residual_function(*function_arguments)
        if self.__initial_residual is None:
            self.__initial_residual = ca.MX()

        # Construct state vector
        self.__sym_list = (
            self.__mx["states"]
            + self.__mx["algebraics"]
            + self.__mx["derivatives"]
            + self.__extra_variables_symbols
            + self.__mx["time"]
            + self.__mx["constant_inputs"]
            + self.__mx["parameters"]
        )
        n_elements = np.array([var.numel() for var in self.__sym_list])
        i_end = n_elements.cumsum()
        i_start = np.array([0, *(i_end[:-1])])
        self.__state_vector = np.full(n_elements.sum(), np.nan)

        # A very handy index
        self.__n_state_symbols = (
            len(self.__mx["states"])
            + len(self.__mx["algebraics"])
            + len(self.__mx["derivatives"])
            + len(self.__extra_variables)
        )
        self.__n_states = i_end[self.__n_state_symbols - 1]

        # NOTE: Backwards compatibility allowing set_var() for parameters. These
        # variables check that this is only done before calling initialize().
        self.__parameters = AliasDict(self.alias_relation)
        self.__parameters.update({v.name(): v for v in self.__mx["parameters"]})
        self.__parameters_set_var = True

        # Construct a dict to look up symbols by name (or iterate over)
        self.__sym_dict = OrderedDict((sym.name(), sym) for sym in self.__sym_list)

        # Generate a dictionary that we can use to lookup the index in the state vector.
        # To avoid repeated and relatively expensive `canonical_signed` calls, we
        # make a dictionary for all variables and their aliases.
        self.__i_start = {}
        self.__i_end = {}
        for i, k in enumerate(self.__sym_dict.keys()):
            for alias in self.alias_relation.aliases(k):
                if alias.startswith("-"):
                    self.__i_start[alias[1:]] = (i_start[i], -1.0)
                    self.__i_end[alias[1:]] = (i_end[i], -1.0)
                else:
                    self.__i_start[alias] = (i_start[i], 1.0)
                    self.__i_end[alias] = (i_end[i], 1.0)
        self.__indices = self.__i_start

        # Call parent class for default behaviour.
        super().__init__(**kwargs)

    def _get_delay_times(self):
        """
        Get the delay times for each delay equation.
        """
        if self._force_zero_delay:
            return [0] * len(self.__pymoca_model.delay_states)
        parameter_symbols = [v.symbol for v in self.__pymoca_model.parameters]
        parameter_values = [v.value for v in self.__pymoca_model.parameters]
        delay_time_expressions = [
            delay_arg.duration for delay_arg in self.__pymoca_model.delay_arguments
        ]
        delay_time_fun = ca.Function(
            "delay_time_function", parameter_symbols, delay_time_expressions
        )
        delay_time_values = delay_time_fun(*parameter_values)
        if len(delay_time_expressions) == 1:
            return [delay_time_values]
        return list(delay_time_values)

    def _create_delay_expression_states(self):
        """
        Create auxiliary states for delay equations.

        Create states to keep track of the history of delay expressions.
        For example, if we have a delay equation of the form

        .. math::
            x = delay(5 * y, 2 * dt),

        Then we need variables to store :math:`5 * y` at time :math`t` and time :math:`t - dt`
        (For each state, we also store the previous value,
        so if we have a state for :math:`5 * y` at :math:`t - dt`,
        then its previous value is :math:`5 * y` at :math:`t - 2 * dt`).
        """
        delay_expression_states = []
        for delay_state, delay_time in zip(
            self.__pymoca_model.delay_states, self.__delay_times, strict=True
        ):
            if delay_time > 0:
                n_previous_values = int(np.ceil(delay_time / self.get_time_step()))
            else:
                n_previous_values = 1
            expression_state = delay_state + "_expr"
            expression_symbol = ca.MX.sym(expression_state, n_previous_values)
            delay_expression_states.append(expression_symbol)
        return delay_expression_states

    def initialize(self, config_file=None):
        """
        Initialize state vector with default values

        Initial values are first read from the given Modelica files.
        If an initial value equals zero or is not provided by a Modelica file,
        and the variable is not marked as fixed,
        then the initial value is tried to be set with the initial_state method.
        When using CSVMixin, this method by default looks for initial values
        in an initial_state.csv file.
        Furthermore, if a variable is not marked as fixed
        and no initial value is given by the initial_state method,
        the initial value can be overwritten using the seed method.
        When a variable is marked as fixed, the initial value is only read from the Modelica file.

        :param config_file: Path to an initialization file.
        """
        if config_file:
            # TODO read start and stop time from config_file and call:
            # self.setup_experiment(start,stop)
            # for now, assume that setup_experiment was called beforehand
            raise NotImplementedError

        # Short-hand notation for the model
        model = self.__pymoca_model

        # Set values of parameters defined in the model into the state vector
        for var in self.__pymoca_model.parameters:
            # First check to see if parameter is already set (this allows child classes to override
            # model defaults)
            if np.isfinite(self.get_var(var.symbol.name())):
                continue

            # Also test to see if the value is constant
            if isinstance(var.value, ca.MX) and not var.value.is_constant():
                continue

            # Try to extract the value
            try:
                # Extract the value as a python type
                val = var.python_type(var.value)
            except ValueError:
                # var.value is a float NaN being cast to non-float
                continue
            else:
                # If val is finite, we set it
                if np.isfinite(val):
                    logger.debug(
                        f"SimulationProblem: Setting parameter {var.symbol.name()} = {val}"
                    )
                    self.set_var(var.symbol.name(), val)

        # Nominals can be symbolic, written in terms of parameters. After all
        # parameter values are known, we evaluate the numeric values of the
        # nominals.
        nominal_vars = list(self.__nominals.keys())
        symbolic_nominals = ca.vertcat(*[self.get_variable_nominal(v) for v in nominal_vars])
        nominal_evaluator = ca.Function(
            "nominal_evaluator", self.__mx["parameters"], [symbolic_nominals]
        )

        n_parameters = len(self.__mx["parameters"])
        if n_parameters > 0:
            [evaluated_nominals] = nominal_evaluator.call(self.__state_vector[-n_parameters:])
        else:
            [evaluated_nominals] = nominal_evaluator.call([])

        evaluated_nominals = np.array(evaluated_nominals).ravel()

        nominal_dict = dict(zip(nominal_vars, evaluated_nominals, strict=True))

        self.__nominals.update(nominal_dict)

        # The variables that need a mutually consistent initial condition
        X = ca.vertcat(*self.__sym_list[: self.__n_state_symbols])
        X_prev = ca.vertcat(
            *[
                ca.MX.sym(sym.name() + "_prev", sym.shape)
                for sym in self.__sym_list[: self.__n_state_symbols]
            ]
        )

        # Assemble initial residuals and set values from start attributes into the state vector
        minimized_residuals = []
        for var in itertools.chain(self.__pymoca_model.states, self.__pymoca_model.alg_states):
            var_name = var.symbol.name()
            var_nominal = self.get_variable_nominal(var_name)
            start_values = {}

            # Attempt to cast var.start to python type
            mx_start = ca.MX(var.start)
            if mx_start.is_constant():
                # cast var.start to python type
                start_value_pymoca = var.python_type(mx_start.to_DM())
                if start_value_pymoca is not None and start_value_pymoca != 0:
                    start_values["modelica"] = start_value_pymoca
            else:
                start_values["modelica"] = mx_start

            if not var.fixed:
                # To make initialization easier, we allow setting initial states by providing
                # timeseries with names that match a symbol in the model. We only check for this
                # matching if the start and fixed attributes were left as default
                try:
                    start_values["initial_state"] = self.initial_state()[var_name]
                except KeyError:
                    pass
                else:
                    # An initial state was found- add it to the constrained residuals
                    logger.debug(
                        "Initialize: Added {} = {} to initial equations "
                        "(found matching timeseries).".format(
                            var_name, start_values["initial_state"]
                        )
                    )
                    # Set var to be fixed
                    var.fixed = True

            if not var.fixed:
                # To make initialization easier, we allow setting initial guesses by providing
                # timeseries with names that match a symbol in the model. We only check for this
                # matching if the start and fixed attributes were left as default
                try:
                    start_values["seed"] = self.seed()[var_name]
                except KeyError:
                    pass
                else:
                    # An initial state was found- add it to the constrained residuals
                    logger.debug(
                        "Initialize: Added {} = {} as initial guess "
                        "(found matching timeseries).".format(var_name, start_values["seed"])
                    )

            # Set the start value based on the different inputs.
            if "seed" in start_values:
                input_source = "seed"
                source_description = "seed method"
            elif "modelica" in start_values:
                input_source = "modelica"
                source_description = "modelica file"
            elif "initial_state" in start_values:
                input_source = "initial_state"
                source_description = "initial_state method (typically reads initial_state.csv)"
            else:
                start_values["modelica"] = start_value_pymoca
                input_source = "modelica"
                source_description = "modelica file or default value"
            start_val = start_values.get(input_source, None)
            start_is_numeric = start_val is not None and not isinstance(start_val, ca.MX)
            numeric_start_val = start_val if start_is_numeric else 0.0
            if len(start_values) > 1:
                logger.warning(
                    f"Initialize: Multiple initial values for {var_name} are provided: "
                    f"{start_values}. Value from {source_description} will be used to continue."
                )

            # Attempt to set start_val in the state vector. Default to zero if unknown.
            try:
                self.set_var(var_name, numeric_start_val)
            except KeyError:
                logger.warning(
                    f"Initialize: {var_name} not found in state vector. "
                    f"Initial value of {numeric_start_val} not set."
                )

            # Add a residual for the difference between the state and its starting expression
            start_expr = start_val
            min_is_symbolic = isinstance(var.min, ca.MX)
            max_is_symbolic = isinstance(var.max, ca.MX)
            if var.fixed:
                # Set bounds to be equal to each other, such that IPOPT can
                # turn the decision variable into a parameter.
                if min_is_symbolic or max_is_symbolic or var.min != -np.inf or var.max != np.inf:
                    logger.info(
                        f"Initialize: bounds of {var_name} will be overwritten"
                        + f" by the start value given by {source_description}."
                    )
                var.min = start_expr
                var.max = start_expr
            else:
                # minimize residual
                minimized_residuals.append((var.symbol - start_expr) / var_nominal)

            # Check that the start_value is in between the variable bounds.
            if start_is_numeric and not min_is_symbolic and not max_is_symbolic:
                if not (var.min <= start_val and start_val <= var.max):
                    logger.log(
                        (
                            logging.WARNING
                            if source_description != "modelica file or default value"
                            else logging.DEBUG
                        ),
                        f"Initialize: start value {var_name} = {start_val} "
                        f"is not in between bounds {var.min} and {var.max} and will be adjusted.",
                    )

        # Default start var for ders is zero
        for der_var in self.__mx["derivatives"]:
            self.set_var(der_var.name(), 0.0)

        # Residuals for initial values for the delay states / expressions.
        for delay_state, delay_argument in zip(
            model.delay_states, model.delay_arguments, strict=True
        ):
            expression_state = delay_state + "_expr"
            i_delay_state, _ = self.__indices[delay_state]
            i_expr_start, _ = self.__i_start[expression_state]
            i_expr_end, _ = self.__i_end[expression_state]
            minimized_residuals.append(X[i_expr_start:i_expr_end] - delay_argument.expr)
            minimized_residuals.append(X[i_delay_state] - delay_argument.expr)

        # Warn for nans in state vector (verify we didn't miss anything)
        self.__warn_for_nans()

        # Optionally encourage a steady-state initial condition
        if getattr(self, "encourage_steady_state_initial_conditions", False):
            # add penalty for der(var) != 0.0
            for d in self.__mx["derivatives"]:
                logger.debug(f"Added {d.name()} to the minimized residuals.")
                minimized_residuals.append(d)

        # Make minimized_residuals into a single symbolic object
        if minimized_residuals:
            minimized_residual = ca.vertcat(*minimized_residuals)
        else:
            # DAE is empty
            minimized_residual = ca.MX(0)

        # Extra equations
        extra_equations = self.extra_equations()

        # Assemble symbolics needed to make a function describing the initial condition of the model
        # We constrain every entry in this MX to zero
        equality_constraints = ca.vertcat(
            self.__dae_residual, self.__initial_residual, *extra_equations
        )

        # Make a list of unscaled symbols and a list of their scaled equivalent
        unscaled_symbols = []
        scaled_symbols = []
        for sym_name, nominal in self.__nominals.items():
            # Note that sym_name is always a canonical state
            index, _ = self.__indices[sym_name]

            # If the symbol is a state, Add the symbol to the lists
            if index <= self.__n_states:
                unscaled_symbols.append(X[index])
                scaled_symbols.append(X[index] * nominal)

                # Also scale previous states
                unscaled_symbols.append(X_prev[index])
                scaled_symbols.append(X_prev[index] * nominal)

        unscaled_symbols = ca.vertcat(*unscaled_symbols)
        scaled_symbols = ca.vertcat(*scaled_symbols)

        # Substitute unscaled terms for scaled terms
        equality_constraints = ca.substitute(equality_constraints, unscaled_symbols, scaled_symbols)
        minimized_residual = ca.substitute(minimized_residual, unscaled_symbols, scaled_symbols)

        logger.debug("SimulationProblem: Initial Equations are " + str(equality_constraints))
        logger.debug("SimulationProblem: Minimized Residuals are " + str(minimized_residual))

        # State bounds can be symbolic, written in terms of parameters. After all
        # parameter values are known, we evaluate the numeric values of bounds.
        bound_vars = (
            self.__pymoca_model.states
            + self.__pymoca_model.alg_states
            + self.__pymoca_model.der_states
            + self.__extra_variables
        )

        symbolic_bounds = ca.vertcat(*[ca.horzcat(v.min, v.max) for v in bound_vars])
        bound_evaluator = ca.Function("bound_evaluator", self.__mx["parameters"], [symbolic_bounds])

        # Evaluate bounds using values of parameters
        n_parameters = len(self.__mx["parameters"])
        if n_parameters > 0:
            [evaluated_bounds] = bound_evaluator.call(self.__state_vector[-n_parameters:])
        else:
            [evaluated_bounds] = bound_evaluator.call([])

        # Scale the bounds with the nominals
        nominals = []
        for var in bound_vars:
            nominals.append(self.get_variable_nominal(var.symbol.name()))

        evaluated_bounds = np.array(evaluated_bounds) / np.array(nominals)[:, None]

        # Update with the bounds of delayed states / expressions
        if model.delay_states:
            i_start_first_delay_state, _ = self.__indices[model.delay_states[0]]
            i_end_last_delay_expr, _ = self.__i_end[model.delay_states[-1] + "_expr"]
            n_delay = i_end_last_delay_expr - i_start_first_delay_state
            delay_bounds = np.array([-np.inf, np.inf] * n_delay).reshape((n_delay, 2))
            # offset = len(self.__pymoca_model.states) + len(self.__pymoca_model.alg_states)
            offset = i_start_first_delay_state
            evaluated_bounds = np.vstack(
                (evaluated_bounds[:offset, :], delay_bounds, evaluated_bounds[offset:, :])
            )

        # Construct arrays of state bounds (used in the initialize() nlp, but not in __do_step
        # rootfinder)
        self.__lbx = evaluated_bounds[:, 0]
        self.__ubx = evaluated_bounds[:, 1]

        # Constrain model equation residuals to zero
        lbg = np.zeros(equality_constraints.size1())
        ubg = np.zeros(equality_constraints.size1())

        # Construct objective function from the input residual
        objective_function = ca.dot(minimized_residual, minimized_residual)

        # Substitute constants and parameters
        const_and_par = ca.vertcat(
            *self.__mx["time"], *self.__mx["constant_inputs"], *self.__mx["parameters"]
        )
        const_and_par_values = self.__state_vector[self.__n_states :]

        objective_function = ca.substitute(objective_function, const_and_par, const_and_par_values)
        equality_constraints = ca.substitute(
            equality_constraints, const_and_par, const_and_par_values
        )

        # Construct nlp and solver to find initial state using ipopt
        # Note that some operations cannot be expanded, e.g. ca.interpolant. So
        # we _try_ to expand, but fall back on ca.MX evaluation if we cannot.
        try:
            expand_f_g = ca.Function("f", [X], [objective_function, equality_constraints]).expand()
        except RuntimeError as e:
            if "eval_sx" not in str(e):
                raise
            else:
                logger.info(
                    "Cannot expand objective/constraints to SX, falling back to MX evaluation"
                )
                nlp = {"x": X, "f": objective_function, "g": equality_constraints}
        else:
            X_sx = ca.SX.sym("X", X.shape)
            objective_function_sx, equality_constraints_sx = expand_f_g(X_sx)
            nlp = {"x": X_sx, "f": objective_function_sx, "g": equality_constraints_sx}

        solver = ca.nlpsol("solver", "ipopt", nlp, self.solver_options())

        # Construct guess
        guess = ca.vertcat(*np.nan_to_num(self.__state_vector[: self.__n_states]))

        # Find initial state
        initial_state = solver(x0=guess, lbx=self.__lbx, ubx=self.__ubx, lbg=lbg, ubg=ubg)

        # If unsuccessful, stop.
        return_status = solver.stats()["return_status"]
        if return_status not in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}:
            if return_status == "Infeasible_Problem_Detected":
                message = (
                    f"Initialization Failed with return status: {return_status}. "
                    + "This means no initial state could be found "
                    + "that satisfies all equations and constraints."
                )
            else:
                message = f"Initialization Failed with return status: {return_status}. "
            raise Exception(message)

        # Update state vector with initial conditions
        self.__state_vector[: self.__n_states] = initial_state["x"][: self.__n_states].T

        # make a copy of the initialized initial state vector in case we want to run the model again
        self.__initialized_state_vector = copy.deepcopy(self.__state_vector)

        # Warn for nans in state vector after initialization
        self.__warn_for_nans()

        # No longer allow setting parameters with set_var(), as we want to be
        # clear that that does not work
        self.__parameters_set_var = False

        self.__parameter_names_including_aliases = set()
        for p in self.__parameters.keys():
            self.__parameter_names_including_aliases |= self.alias_relation.aliases(p)

        # Construct the rootfinder

        # Assemble some symbolics, including those needed for a backwards Euler derivative
        # approximation
        dt = ca.MX.sym("delta_t")
        parameters = ca.vertcat(*self.__mx["parameters"])
        if n_parameters > 0:
            constants = ca.vertcat(X_prev, *self.__sym_list[self.__n_state_symbols : -n_parameters])
        else:
            constants = ca.vertcat(X_prev, *self.__sym_list[self.__n_state_symbols :])

        # Make a list of derivative approximations using backwards Euler formulation
        derivative_approximation_residuals = []
        for index, derivative_state in enumerate(self.__mx["derivatives"]):
            derivative_approximation_residuals.append(
                derivative_state - (X[index] - X_prev[index]) / dt
            )

        # Delayed feedback
        delay_equations = []
        for delay_state, delay_argument, delay_time in zip(
            model.delay_states,
            model.delay_arguments,
            self.__delay_times,
            strict=True,
        ):
            expression_state = delay_state + "_expr"
            i_delay_state, _ = self.__indices[delay_state]
            i_expr_start, _ = self.__i_start[expression_state]
            i_expr_end, _ = self.__i_end[expression_state]
            delay_equations.append(X[i_expr_start] - delay_argument.expr)
            delay_equations.append(
                X[i_expr_start + 1 : i_expr_end] - X_prev[i_expr_start : i_expr_end - 1]
            )
            n_previous_values = self.__sym_dict[expression_state].numel()
            interpolation_weight = n_previous_values - delay_time / self.__dt
            delay_equations.append(
                X[i_delay_state]
                - interpolation_weight * X[i_expr_end - 1]
                - (1 - interpolation_weight) * X_prev[i_expr_end - 1]
            )

        # Append residuals for derivative approximations
        dae_residual = ca.vertcat(
            self.__dae_residual,
            *derivative_approximation_residuals,
            *delay_equations,
            *extra_equations,
        )

        # TODO: implement lookup_tables

        # Substitute unscaled terms for scaled terms
        dae_residual = ca.substitute(dae_residual, unscaled_symbols, scaled_symbols)

        # Substitute the parameters
        if n_parameters > 0:
            parameters_values = self.__state_vector[-n_parameters:]
            dae_residual = ca.substitute(dae_residual, parameters, parameters_values)

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("SimulationProblem: DAE Residual is " + str(dae_residual))

        if X.size1() != dae_residual.size1():
            logger.error(
                f"Formulation Error: Number of states ({X.size1()}) "
                f"does not equal number of equations ({dae_residual.size1()})"
            )

        # Construct a function res_vals that returns the numerical residuals of a numerical state
        self.__res_vals = ca.Function("res_vals", [X, dt, constants], [dae_residual])
        try:
            self.__res_vals = self.__res_vals.expand()
        except RuntimeError as e:
            if "eval_sx" not in str(e):
                raise
            else:
                pass

        # Use rootfinder() to make a function that takes a step forward in time by trying to zero
        # res_vals()
        options = self.rootfinder_options()
        solver = options["solver"]
        solver_options = options["solver_options"]
        self.__do_step = ca.rootfinder("next_state", solver, self.__res_vals, solver_options)

    def pre(self):
        """
        Any preprocessing takes place here.
        """
        pass

    def post(self):
        """
        Any postprocessing takes place here.
        """
        pass

    def setup_experiment(self, start, stop, dt):
        """
        Method for subclasses (PIMixin, CSVMixin, or user classes) to set timing information for a
        simulation run.

        :param start: Start time for the simulation.
        :param stop:  Final time for the simulation.
        :param dt:    Time step size.
        """

        # Set class vars with start/stop/dt values
        self.__start = start
        self.__stop = stop
        self.set_time_step(dt)

        # Set time in state vector
        self.set_var("time", start)

    def update(self, dt):
        """
        Performs one timestep.

        The methods ``setup_experiment`` and ``initialize`` must have been called before.

        :param dt: Time step size.
        """
        if dt > 0:
            self.set_time_step(dt)
        dt = self.get_time_step()

        logger.debug(f"Taking a step at {self.get_current_time()} with size {dt}")

        # increment time
        self.set_var("time", self.get_current_time() + dt)

        # take a step
        if np.isnan(self.__state_vector).any():
            logger.error("Found a nan in the state vector (before making the step)")
        guess = self.__state_vector[: self.__n_states]
        if len(self.__mx["parameters"]) > 0:
            next_state = self.__do_step(
                guess, dt, self.__state_vector[: -len(self.__mx["parameters"])]
            )
        else:
            next_state = self.__do_step(guess, dt, self.__state_vector)

        try:
            if np.isnan(next_state).any():
                index_to_name = {index[0]: name for name, index in self.__indices.items()}
                named_next_state = {
                    index_to_name[i]: float(next_state[i]) for i in range(0, next_state.shape[0])
                }
                variables_with_nan = [
                    name for name, value in named_next_state.items() if np.isnan(value)
                ]
                if variables_with_nan:
                    logger.error(
                        f"Found nan(s) in the next_state vector for:\n\t {variables_with_nan}"
                    )
        except (KeyError, IndexError, TypeError):
            logger.warning("Something went wrong while checking for nans in the next_state vector")

        # Check convergence of rootfinder
        rootfinder_stats = self.__do_step.stats()

        if not rootfinder_stats["success"]:
            message = (
                "Simulation has failed to converge at time {}. Solver failed with status {}"
            ).format(self.get_current_time(), rootfinder_stats["nlpsol"]["return_status"])
            logger.error(message)
            raise Exception(message)

        if logger.getEffectiveLevel() == logging.DEBUG:
            # compute max residual
            largest_res = ca.norm_inf(
                self.__res_vals(
                    next_state, self.__dt, self.__state_vector[: -len(self.__mx["parameters"])]
                )
            )
            logger.debug(f"Residual maximum magnitude: {float(largest_res):.2E}")

        # Update state vector
        self.__state_vector[: self.__n_states] = next_state.toarray().ravel()

    def simulate(self):
        """
        Run model from start_time to end_time.
        """

        # Do any preprocessing, which may include changing parameter values on
        # the model
        logger.info("Preprocessing")
        self.pre()

        # Initialize model
        logger.info("Initializing")
        self.initialize()

        # Perform all timesteps
        logger.info("Running")
        while self.get_current_time() < self.get_end_time():
            self.update(-1)

        # Do any postprocessing
        logger.info("Postprocessing")
        self.post()

    def reset(self):
        """
        Reset the FMU.
        """
        self.__state_vector = copy.deepcopy(self.__initialized_state_vector)

    def get_start_time(self):
        """
        Return start time of experiment.

        :returns: The start time of the experiment.
        """
        return self.__start

    def get_end_time(self):
        """
        Return end time of experiment.

        :returns: The end time of the experiment.
        """
        return self.__stop

    def get_current_time(self):
        """
        Return current time of simulation.

        :returns: The current simulation time.
        """
        return self.get_var("time")

    def get_time_step(self):
        """
        Return simulation timestep.

        :returns: The simulation timestep.
        """
        return self.__dt

    def get_var(self, name):
        """
        Return a numpy array from FMU.

        :param name: Variable name.

        :returns: The value of the variable.
        """

        # Get the index of the canonical state and sign
        index, sign = self.__indices[name]
        value = self.__state_vector[index]

        # Adjust sign if needed
        if sign < 0:
            value *= sign

        # Adjust for nominal value if not default
        if index <= self.__n_states:
            nominal = self.get_variable_nominal(name)
            value *= nominal

        return value

    def get_var_count(self):
        """
        Return the number of variables in the model.

        :returns: The number of variables in the model.
        """
        return len(self.get_variables())

    def get_var_name(self, i):
        """
        Returns the name of a variable.

        :param i: Index in ordered dictionary returned by method get_variables.

        :returns: The name of the variable.
        """
        return list(self.get_variables())[i]

    def get_var_type(self, name):
        """
        Return type, compatible with numpy.

        :param name: String variable name.

        :returns: The numpy-compatible type of the variable.

        :raises: KeyError
        """
        return self.__python_types[name]

    def get_var_rank(self, name):
        """
        Not implemented
        """
        raise NotImplementedError

    def get_var_shape(self, name):
        """
        Not implemented
        """
        raise NotImplementedError

    def get_variables(self):
        """
        Return all variables (both internal and user defined)

        :returns: An ordered dictionary of all variables supported by the model.
        """
        return AliasDict(self.alias_relation, self.__sym_dict)

    @cached
    def get_state_variables(self):
        return AliasDict(
            self.alias_relation,
            {sym.name(): sym for sym in (self.__mx["states"] + self.__mx["algebraics"])},
        )

    @cached
    def get_parameter_variables(self):
        return AliasDict(self.alias_relation, {sym.name(): sym for sym in self.__mx["parameters"]})

    @cached
    def get_input_variables(self):
        return AliasDict(
            self.alias_relation, {sym.name(): sym for sym in self.__mx["constant_inputs"]}
        )

    @cached
    def get_output_variables(self):
        return self.__pymoca_model.outputs

    def __warn_for_nans(self):
        """
        Test state vector for missing values and warn
        """
        value_is_nan = np.isnan(self.__state_vector)
        if any(value_is_nan):
            for name in self.__sym_dict:
                i_start, _ = self.__i_start[name]
                i_end, _ = self.__i_end[name]
                if any(value_is_nan[i_start:i_end]):
                    logger.warning(f"Variable {name} has no value.")

    def set_time_step(self, dt):
        """
        Set the timestep size.

        :param dt: Timestep size of the simulation.
        """
        if self._dt_is_fixed:
            assert math.isclose(self.__dt, dt), (
                "Timestep size dt is marked as constant and cannot be changed."
            )
        else:
            self.__dt = dt

    def set_var(self, name, value):
        """
        Set the value of the given variable.

        :param name: Name of variable to set.
        :param value:  Value(s).
        """

        # TODO: sanitize input

        # Check if it is a parameter, and if it is allowed to be set
        if not self.__parameters_set_var:
            if name in self.__parameter_names_including_aliases:
                raise Exception("Cannot set parameters after initialize() has been called.")

        # Get the index of the canonical state and sign
        index, sign = self.__indices[name]
        if sign < 0:
            value *= sign

        # Adjust for nominal value if not default
        if index <= self.__n_states:
            nominal = self.get_variable_nominal(name)
            value /= nominal

        # Store value in state vector
        self.__state_vector[index] = value

    def set_var_slice(self, name, start, count, var):
        """
        Not implemented.
        """
        raise NotImplementedError

    def set_var_index(self, name, index, var):
        """
        Not implemented.
        """
        raise NotImplementedError

    def inq_compound(self, name):
        """
        Not implemented.
        """
        raise NotImplementedError

    def inq_compound_field(self, name, index):
        """
        Not implemented.
        """
        raise NotImplementedError

    def solver_options(self):
        """
        Returns a dictionary of CasADi nlpsol() solver options.

        :returns: A dictionary of CasADi :class:`nlpsol` options. See the CasADi
            documentation for details.
        """
        return {
            "ipopt.fixed_variable_treatment": "make_parameter",
            "ipopt.print_level": 0,
            "print_time": False,
            "error_on_fail": False,
        }

    def rootfinder_options(self):
        """
        Returns a dictionary of CasADi rootfinder() options.

        The dictionary has the following items:
        * solver: the solver used for rootfinding, e.g. nlpsol, fast_newton, etc.
        * solver_options: options for the solver
        See the CasADi documentation for details on solvers and solver options.

        :returns: A dictionary of CasADi :class:`rootfinder` options.
        """
        return {
            "solver": "nlpsol",
            "solver_options": {
                "nlpsol": "ipopt",
                "nlpsol_options": self.solver_options(),
                "error_on_fail": False,
            },
        }

    def get_variable_nominal(self, variable) -> float | ca.MX:
        """
        Get the value of the nominal attribute of a variable

        NOTE: Due to backwards compatibility for allowing parameters to be set
        with set_var() instead of overriding parameters(), this method can
        return a symbolic value for nominals defined in the Modelica file. It
        can only do so until the initializion() method in this class is
        called/completed, after which it will return numeric values only.
        """
        return self.__nominals.get(variable, 1.0)

    def timeseries_at(self, variable, t):
        """
        Get value of timeseries variable at time t: should be overridden by pi or csv mixin
        """
        raise NotImplementedError

    @cached
    def initial_state(self) -> AliasDict[str, float]:
        """
        The initial state.

        :returns: A dictionary of variable names and initial state (t0) values.
        """
        t0 = self.get_start_time()
        initial_state_dict = AliasDict(self.alias_relation)

        for variable in list(self.get_state_variables()) + list(self.get_input_variables()):
            try:
                initial_state_dict[variable] = self.timeseries_at(variable, t0)
            except KeyError:
                pass
            except NotImplementedError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(f"Read intial state for {variable}")

        return initial_state_dict

    @cached
    def seed(self) -> AliasDict[str, float]:
        """
        Seed values providing an initial guess for the t0 states.

        :returns: A dictionary of variable names and seed (t0) values.
        """
        return AliasDict(self.alias_relation)

    @cached
    def parameters(self):
        """
        Return a dictionary of parameter values extracted from the Modelica model
        """
        # Create AliasDict
        parameters = AliasDict(self.alias_relation)

        # Update with model parameters
        parameters.update({p.symbol.name(): p.value for p in self.__pymoca_model.parameters})

        return parameters

    def extra_variables(self) -> list[Variable]:
        return []

    def extra_equations(self) -> list[ca.MX]:
        return []

    @property
    @cached
    def alias_relation(self):
        return self.__pymoca_model.alias_relation

    @cached
    def compiler_options(self):
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

        # Cache the model on disk
        compiler_options["cache"] = True

        # Done
        return compiler_options
