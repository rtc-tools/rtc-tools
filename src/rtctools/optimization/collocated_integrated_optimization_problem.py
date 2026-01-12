import itertools
import logging
import warnings
from abc import ABCMeta, abstractmethod

import casadi as ca
import numpy as np
from numpy.typing import NDArray

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.casadi_helpers import (
    interpolate,
    is_affine,
    nullvertcat,
    reduce_matvec,
    substitute_in_external,
)
from rtctools._internal.debug_check_helpers import DebugLevel, debug_check

from .optimization_problem import BT, OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class CollocatedIntegratedOptimizationProblem(OptimizationProblem, metaclass=ABCMeta):
    """
    Discretizes your model using a mixed collocation/integration scheme.

    Collocation means that the discretized model equations are included as constraints
    between state variables in the optimization problem.

    Integration means that the model equations are solved from one time step to the next
    in a sequential fashion, using a rootfinding algorithm at each and every step.  The
    results of the integration procedure feature as inputs to the objective functions
    as well as to any constraints that do not originate from the DAE model.

    .. note::

        To ensure that your optimization problem only has globally optimal solutions,
        any model equations that are collocated must be linear.  By default, all
        model equations are collocated, and linearity of the model equations is
        verified.  Working with non-linear models is possible, but discouraged.

    :cvar check_collocation_linearity:
        If ``True``, check whether collocation constraints are linear. Default is ``True``.
    :cvar inline_delay_expressions:
        If ``True``, delay expressions are inlined into constraints and objectives instead of
        being added as separate equality constraints. Default is ``False``.
    """

    #: Check whether the collocation constraints are linear
    check_collocation_linearity = True

    #: Inline delay expressions instead of adding them as separate equality constraints
    inline_delay_expressions = False

    #: Whether or not the collocation constraints are linear (affine)
    linear_collocation = None

    def __init__(self, **kwargs):
        # Variables that will be optimized
        self.dae_variables["free_variables"] = (
            self.dae_variables["states"]
            + self.dae_variables["algebraics"]
            + self.dae_variables["control_inputs"]
        )

        # Cache names of states
        self.__differentiated_states = [
            variable.name() for variable in self.dae_variables["states"]
        ]
        self.__differentiated_states_map = {
            v: i for i, v in enumerate(self.__differentiated_states)
        }

        self.__algebraic_states = [variable.name() for variable in self.dae_variables["algebraics"]]
        self.__algebraic_states_map = {v: i for i, v in enumerate(self.__algebraic_states)}

        self.__controls = [variable.name() for variable in self.dae_variables["control_inputs"]]
        self.__controls_map = {v: i for i, v in enumerate(self.__controls)}

        self.__derivative_names = [
            variable.name() for variable in self.dae_variables["derivatives"]
        ]

        self.__initial_derivative_names = [
            "initial_" + variable for variable in self.__derivative_names
        ]

        self.__initial_derivative_nominals = {}

        # DAE cache
        self.__integrator_step_function = None
        self.__dae_residual_function_collocated = None
        self.__initial_residual_with_params_fun_map = None

        # Create dictionary of variables so that we have O(1) state lookup available
        self.__variables = AliasDict(self.alias_relation)
        for var in itertools.chain(
            self.dae_variables["states"],
            self.dae_variables["algebraics"],
            self.dae_variables["control_inputs"],
            self.dae_variables["constant_inputs"],
            self.dae_variables["parameters"],
            self.dae_variables["time"],
        ):
            self.__variables[var.name()] = var

        # Call super
        super().__init__(**kwargs)

    @abstractmethod
    def times(self, variable=None):
        """
        List of time stamps for variable (to optimize for).

        :param variable: Variable name.

        :returns: A list of time stamps for the given variable.
        """
        pass

    def interpolation_method(self, variable=None):
        """
        Interpolation method for variable.

        :param variable: Variable name.

        :returns: Interpolation method for the given variable.
        """
        return self.INTERPOLATION_LINEAR

    @property
    def integrate_states(self):
        """
        TRUE if all states are to be integrated rather than collocated.
        """
        return False

    @property
    def theta(self):
        r"""
        RTC-Tools discretizes differential equations of the form

        .. math::

            \dot{x} = f(x, u)

        using the :math:`\theta`-method

        .. math::

            x_{i+1} = x_i + \Delta t \left[\theta f(x_{i+1}, u_{i+1})
            + (1 - \theta) f(x_i, u_i)\right]

        The default is :math:`\theta = 1`, resulting in the implicit or backward Euler method. Note
        that in this case, the control input at the initial time step is not used.

        Set :math:`\theta = 0` to use the explicit or forward Euler method.  Note that in this
        case, the control input at the final time step is not used.

        .. warning:: This is an experimental feature for :math:`0 < \theta < 1`.

        .. deprecated:: 2.4
           Support for semi-explicit collocation (theta < 1) will be removed in a future release.
        """

        # Default to implicit Euler collocation, which is cheaper to evaluate
        # than the trapezoidal method, while being A-stable.
        #
        # N.B.  Setting theta to 0 will cause problems with algebraic equations,
        #       unless a consistent initialization is supplied for the algebraics.
        # N.B.  Setting theta to any value strictly between 0 and 1 will cause
        #       algebraic equations to be solved in an average sense.  This may
        #       induce unexpected oscillations.
        # TODO Fix these issue by performing index reduction and splitting DAE into ODE and
        #      algebraic parts. Theta then only applies to the ODE part.
        return 1.0

    def map_options(self) -> dict[str, str | int]:
        """
        Returns a dictionary of CasADi ``map()`` options.

        +---------------+-----------+---------------+
        | Option        | Type      | Default value |
        +===============+===========+===============+
        | ``mode``      | ``str`    | ``openmp``    |
        +---------------+-----------+---------------+
        | ``n_threads`` | ``int``   | ``None``      |
        +---------------+-----------+---------------+

        The ``mode`` option controls the mode of the ``map()`` call.  Valid values include
        ``openmp``, ``thread``, and ``unroll``.  See the CasADi and documentation for detailed
        documentation on these modes.

        The ``n_threads`` option controls the number of threads used when in ``thread`` mode.

        .. note::

            Not every CasADi build has support for OpenMP enabled.  For such builds, the `thread`
            mode offers an alternative parallelization mode.

        .. note::

            The use of ``expand=True`` in ``solver_options()`` may negate the parallelization
            benefits obtained using ``map()``.

        :returns: A dictionary of options for the `map()` call used to evaluate constraints on
            every time stamp.
        """
        return {"mode": "openmp"}

    def transcribe(self):
        # DAE residual
        dae_residual = self.dae_residual

        # Initial residual
        initial_residual = self.initial_residual

        logger.info(
            f"Transcribing problem with a DAE of {dae_residual.size1()} equations, "
            f"{len(self.times())} collocation points, "
            f"and {len(self.dae_variables['free_variables'])} free variables"
        )

        # Reset dictionary of variables
        for var in itertools.chain(self.path_variables, self.extra_variables):
            self.__variables[var.name()] = var

        # Split the constant inputs into those used in the DAE, and additional
        # ones used for just the objective and/or constraints
        dae_constant_inputs_names = [x.name() for x in self.dae_variables["constant_inputs"]]
        extra_constant_inputs_name_and_size = []
        for ensemble_member in range(self.ensemble_size):
            extra_constant_inputs_name_and_size.extend(
                [
                    (x, v.values.shape[1] if v.values.ndim > 1 else 1)
                    for x, v in self.constant_inputs(ensemble_member).items()
                    if x not in dae_constant_inputs_names
                ]
            )

        self.__extra_constant_inputs = []
        for var_name, size in extra_constant_inputs_name_and_size:
            var = ca.MX.sym(var_name, size)
            self.__variables[var_name] = var
            self.__extra_constant_inputs.append(var)

        # Cache extra and path variable names, and variable sizes
        self.__path_variable_names = [variable.name() for variable in self.path_variables]
        self.__extra_variable_names = [variable.name() for variable in self.extra_variables]

        # Cache the variable sizes, as a repeated call to .name() and .size1()
        # is expensive due to SWIG call overhead.
        self.__variable_sizes = {}

        for variable in itertools.chain(
            self.differentiated_states,
            self.algebraic_states,
            self.controls,
            self.__initial_derivative_names,
        ):
            self.__variable_sizes[variable] = 1

        for mx_symbol, variable in zip(
            self.path_variables, self.__path_variable_names, strict=True
        ):
            self.__variable_sizes[variable] = mx_symbol.size1()

        for mx_symbol, variable in zip(
            self.extra_variables, self.__extra_variable_names, strict=True
        ):
            self.__variable_sizes[variable] = mx_symbol.size1()

        # Calculate nominals for the initial derivatives. We assume that the
        # history has (roughly) identical time steps for the entire ensemble.
        self.__initial_derivative_nominals = {}
        history_0 = self.history(0)
        for variable, initial_der_name in zip(
            self.__differentiated_states, self.__initial_derivative_names, strict=True
        ):
            times = self.times(variable)
            default_time_step_size = 0
            if len(times) > 1:
                default_time_step_size = times[1] - times[0]
            try:
                h = history_0[variable]
                if h.times[0] == times[0] or len(h.values) == 1:
                    dt = default_time_step_size
                else:
                    assert h.times[-1] == times[0]
                    dt = h.times[-1] - h.times[-2]
            except KeyError:
                dt = default_time_step_size

            if dt > 0:
                self.__initial_derivative_nominals[initial_der_name] = (
                    self.variable_nominal(variable) / dt
                )
            else:
                self.__initial_derivative_nominals[initial_der_name] = self.variable_nominal(
                    variable
                )

        # Check that the removed (because broken) integrated_states option is not used
        try:
            _ = self.integrated_states
        except AttributeError:
            # We expect there to be an error as users should use self.integrate_states
            pass
        else:
            raise Exception(
                "The integrated_states property is no longer supported. "
                "Use integrate_states instead."
            )

        # Variables that are integrated states are not yet allowed to have size > 1
        if self.integrate_states:
            self.__integrated_states = [*self.differentiated_states, *self.algebraic_states]

            for variable in self.__integrated_states:
                if self.__variable_sizes.get(variable, 1) > 1:
                    raise NotImplementedError(
                        f"Vector symbol not supported for integrated state '{variable}'"
                    )
        else:
            self.__integrated_states = []

        # The same holds for controls
        for variable in self.controls:
            if self.__variable_sizes.get(variable, 1) > 1:
                raise NotImplementedError(
                    f"Vector symbol not supported for control state '{variable}'"
                )

        # Collocation times
        collocation_times = self.times()
        n_collocation_times = len(collocation_times)

        # Dynamic parameters
        dynamic_parameters = self.dynamic_parameters()
        dynamic_parameter_names = set()

        # Parameter symbols
        symbolic_parameters = ca.vertcat(*self.dae_variables["parameters"])

        def _interpolate_constant_inputs(variables, raw_constant_inputs):
            constant_inputs_interpolated = {}
            for variable in variables:
                variable = variable.name()
                try:
                    constant_input = raw_constant_inputs[variable]
                except KeyError:
                    raise Exception(f"No values found for constant input {variable}")
                else:
                    values = constant_input.values
                    interpolation_method = self.interpolation_method(variable)
                    constant_inputs_interpolated[variable] = self.interpolate(
                        collocation_times,
                        constant_input.times,
                        values,
                        0.0,
                        0.0,
                        interpolation_method,
                    )

            return constant_inputs_interpolated

        # Create a store of all ensemble-member-specific data for all ensemble members
        # N.B. Don't use n * [{}], as it creates n refs to the same dict.
        ensemble_store = [{} for i in range(self.ensemble_size)]
        for ensemble_member in range(self.ensemble_size):
            ensemble_data = ensemble_store[ensemble_member]

            # Store parameters
            parameters = self.parameters(ensemble_member)
            parameter_values = [None] * len(self.dae_variables["parameters"])
            for i, symbol in enumerate(self.dae_variables["parameters"]):
                variable = symbol.name()
                try:
                    parameter_values[i] = parameters[variable]
                except KeyError:
                    raise Exception(f"No value specified for parameter {variable}")

            if len(dynamic_parameters) > 0:
                jac_1 = ca.jacobian(symbolic_parameters, ca.vertcat(*dynamic_parameters))
                jac_2 = ca.jacobian(ca.vertcat(*parameter_values), ca.vertcat(*dynamic_parameters))
                for i, symbol in enumerate(self.dae_variables["parameters"]):
                    if jac_1[i, :].nnz() > 0 or jac_2[i, :].nnz() > 0:
                        dynamic_parameter_names.add(symbol.name())

            if np.any(
                [isinstance(value, ca.MX) and not value.is_constant() for value in parameter_values]
            ):
                parameter_values = nullvertcat(*parameter_values)
                [parameter_values] = substitute_in_external(
                    [parameter_values],
                    self.dae_variables["parameters"],
                    ca.vertsplit(parameter_values),
                )
            else:
                parameter_values = nullvertcat(*parameter_values)

            if ensemble_member == 0:
                # Store parameter values of member 0, as variable bounds may depend on these.
                self.__parameter_values_ensemble_member_0 = parameter_values
            ensemble_data["parameters"] = parameter_values

            # Store constant inputs
            raw_constant_inputs = self.constant_inputs(ensemble_member)

            ensemble_data["constant_inputs"] = _interpolate_constant_inputs(
                self.dae_variables["constant_inputs"], raw_constant_inputs
            )
            ensemble_data["extra_constant_inputs"] = _interpolate_constant_inputs(
                self.__extra_constant_inputs, raw_constant_inputs
            )

            # Handle all extra constant input data uniformly as 2D arrays
            for k, v in ensemble_data["extra_constant_inputs"].items():
                if v.ndim == 1:
                    ensemble_data["extra_constant_inputs"][k] = v[:, None]

        if self.ensemble_specific_bounds:
            bounds = [self.bounds(ensemble_member=i) for i in range(self.ensemble_size)]
        else:
            bounds = self.bounds()

        # Initialize control discretization
        (
            control_size,
            discrete_control,
            lbx_control,
            ubx_control,
            x0_control,
            indices_control,
        ) = self.discretize_controls(bounds)

        # Initialize state discretization
        (
            state_size,
            discrete_state,
            lbx_state,
            ubx_state,
            x0_state,
            indices_state,
        ) = self.discretize_states(bounds)

        # Merge state vector offset dictionary
        self.__indices = indices_control
        for ensemble_member in range(self.ensemble_size):
            for key, value in indices_state[ensemble_member].items():
                if isinstance(value, slice):
                    value = slice(value.start + control_size, value.stop + control_size)
                else:
                    value += control_size
                self.__indices[ensemble_member][key] = value

        # Initialize vector of optimization symbols
        X = ca.MX.sym("X", control_size + state_size)
        self.__solver_input = X

        # Later on, we will be slicing MX/SX objects a few times for vectorized operations (to
        # reduce the overhead induced for each CasADi call). When slicing MX/SX objects, we want
        # to do that with a list of Python ints. Slicing with something else (e.g. a list of
        # np.int32, or a numpy array) is significantly slower.
        x_inds = list(range(X.size1()))
        self.__indices_as_lists = [{} for ensemble_member in range(self.ensemble_size)]

        for ensemble_member in range(self.ensemble_size):
            for k, v in self.__indices[ensemble_member].items():
                if isinstance(v, slice):
                    self.__indices_as_lists[ensemble_member][k] = x_inds[v]
                elif isinstance(v, int):
                    self.__indices_as_lists[ensemble_member][k] = [v]
                else:
                    self.__indices_as_lists[ensemble_member][k] = [int(i) for i in v]

        # Initialize bound and seed vectors
        discrete = np.zeros(X.size1(), dtype=bool)

        lbx = -np.inf * np.ones(X.size1())
        ubx = np.inf * np.ones(X.size1())

        x0 = np.zeros(X.size1())

        discrete[: len(discrete_control)] = discrete_control
        discrete[len(discrete_control) :] = discrete_state
        lbx[: len(lbx_control)] = lbx_control
        lbx[len(lbx_control) :] = lbx_state
        ubx[: len(ubx_control)] = ubx_control
        ubx[len(lbx_control) :] = ubx_state
        x0[: len(x0_control)] = x0_control
        x0[len(x0_control) :] = x0_state

        # Provide a state for self.state_at() and self.der() to work with.
        self.__control_size = control_size
        self.__state_size = state_size
        self.__symbol_cache = {}

        # Free variables for the collocated optimization problem
        if self.integrate_states:
            integrated_variables = self.dae_variables["states"] + self.dae_variables["algebraics"]
            collocated_variables = []
        else:
            integrated_variables = []
            collocated_variables = self.dae_variables["states"] + self.dae_variables["algebraics"]
        collocated_variables += self.dae_variables["control_inputs"]

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug(f"Integrating variables {repr(integrated_variables)}")
            logger.debug(f"Collocating variables {repr(collocated_variables)}")

        integrated_variable_names = [v.name() for v in integrated_variables]
        integrated_variable_nominals = np.array(
            [self.variable_nominal(v) for v in integrated_variable_names]
        )

        collocated_variable_names = [v.name() for v in collocated_variables]
        collocated_variable_nominals = np.array(
            [self.variable_nominal(v) for v in collocated_variable_names]
        )

        # Split derivatives into "integrated" and "collocated" lists.

        if self.integrate_states:
            integrated_derivatives = self.dae_variables["derivatives"][:]
            collocated_derivatives = []
        else:
            integrated_derivatives = []
            collocated_derivatives = self.dae_variables["derivatives"][:]
        self.__algebraic_and_control_derivatives = []
        for var in self.dae_variables["algebraics"]:
            sym = ca.MX.sym(f"der({var.name()})")
            self.__algebraic_and_control_derivatives.append(sym)
            if self.integrate_states:
                integrated_derivatives.append(sym)
            else:
                collocated_derivatives.append(sym)
        for var in self.dae_variables["control_inputs"]:
            sym = ca.MX.sym(f"der({var.name()})")
            self.__algebraic_and_control_derivatives.append(sym)
            collocated_derivatives.append(sym)

        # Path objective
        path_objective = self.path_objective(0)

        # Path constraints
        path_constraints = self.path_constraints(0)
        path_constraint_expressions = ca.vertcat(
            *[f_constraint for (f_constraint, lb, ub) in path_constraints]
        )

        # Delayed feedback
        delayed_feedback_expressions, delayed_feedback_states, delayed_feedback_durations = (
            [],
            [],
            [],
        )
        delayed_feedback = self.delayed_feedback()
        if delayed_feedback:
            delayed_feedback_expressions, delayed_feedback_states, delayed_feedback_durations = zip(
                *delayed_feedback, strict=True
            )
        # Make sure the original data cannot be used anymore, because it will
        # become incorrect/stale with the inlining of constant parameters.
        del delayed_feedback

        # Initial time
        t0 = self.initial_time

        # Establish integrator theta
        theta = self.theta
        if theta < 1:
            warnings.warn(
                (
                    "Explicit collocation/integration is deprecated "
                    "and will be removed in a future version."
                ),
                FutureWarning,
                stacklevel=1,
            )

        # Set CasADi function options
        options = self.solver_options()
        function_options = {"max_num_dir": options["optimized_num_dir"]}

        # Update the store of all ensemble-member-specific data for all ensemble members
        # with initial states, derivatives, and path variables.
        # Use vectorized approach to avoid SWIG call overhead for each CasADi call.
        n = len(integrated_variables) + len(collocated_variables)
        for ensemble_member in range(self.ensemble_size):
            ensemble_data = ensemble_store[ensemble_member]

            initial_state_indices = [None] * n

            # Derivatives take a bit more effort to vectorize, as we can have
            # both constant values and elements in the state vector
            initial_derivatives = ca.MX.zeros((n, 1))
            init_der_variable = []
            init_der_variable_indices = []
            init_der_variable_nominals = []
            init_der_constant = []
            init_der_constant_values = []

            history = self.history(ensemble_member)

            for j, variable in enumerate(integrated_variable_names + collocated_variable_names):
                initial_state_indices[j] = self.__indices_as_lists[ensemble_member][variable][0]

                try:
                    i = self.__differentiated_states_map[variable]

                    initial_der_name = self.__initial_derivative_names[i]
                    init_der_variable_nominals.append(self.variable_nominal(initial_der_name))
                    init_der_variable_indices.append(
                        self.__indices[ensemble_member][initial_der_name]
                    )
                    init_der_variable.append(j)

                except KeyError:
                    # We do interpolation here instead of relying on der_at. This is faster because:
                    # 1. We can reuse the history variable.
                    # 2. We know that "variable" is a canonical state
                    # 3. We know that we are only dealing with history (numeric values, not
                    #    symbolics)
                    try:
                        h = history[variable]
                        if h.times[0] == t0 or len(h.values) == 1:
                            init_der = 0.0
                        else:
                            assert h.times[-1] == t0
                            init_der = (h.values[-1] - h.values[-2]) / (h.times[-1] - h.times[-2])
                    except KeyError:
                        init_der = 0.0

                    init_der_constant_values.append(init_der)
                    init_der_constant.append(j)

            initial_derivatives[init_der_variable] = X[init_der_variable_indices] * np.array(
                init_der_variable_nominals
            )
            if len(init_der_constant_values) > 0:
                initial_derivatives[init_der_constant] = init_der_constant_values

            ensemble_data["initial_state"] = X[initial_state_indices] * np.concatenate(
                (integrated_variable_nominals, collocated_variable_nominals)
            )
            ensemble_data["initial_derivatives"] = initial_derivatives

            # Store initial path variables
            initial_path_variable_inds = []

            path_variables_size = sum(self.__variable_sizes[v] for v in self.__path_variable_names)
            path_variables_nominals = np.ones(path_variables_size)

            offset = 0
            for variable in self.__path_variable_names:
                step = len(self.times(variable))
                initial_path_variable_inds.extend(
                    self.__indices_as_lists[ensemble_member][variable][0::step]
                )

                variable_size = self.__variable_sizes[variable]
                path_variables_nominals[offset : offset + variable_size] = self.variable_nominal(
                    variable
                )
                offset += variable_size

            ensemble_data["initial_path_variables"] = (
                X[initial_path_variable_inds] * path_variables_nominals
            )

        # Replace parameters which are constant across the entire ensemble
        constant_parameters = []
        constant_parameter_values = []

        ensemble_parameters = []
        ensemble_parameter_values = [[] for i in range(self.ensemble_size)]

        for i, parameter in enumerate(self.dae_variables["parameters"]):
            values = [
                ensemble_store[ensemble_member]["parameters"][i]
                for ensemble_member in range(self.ensemble_size)
            ]
            if (
                len(values) == 1 or all(v == values[0] for v in values)
            ) and parameter.name() not in dynamic_parameter_names:
                constant_parameters.append(parameter)
                constant_parameter_values.append(values[0])
            else:
                ensemble_parameters.append(parameter)
                for ensemble_member in range(self.ensemble_size):
                    ensemble_parameter_values[ensemble_member].append(values[ensemble_member])

        symbolic_parameters = ca.vertcat(*ensemble_parameters)

        # Inline constant parameter values
        if constant_parameters:
            delayed_feedback_expressions = ca.substitute(
                delayed_feedback_expressions, constant_parameters, constant_parameter_values
            )

            delayed_feedback_durations = ca.substitute(
                delayed_feedback_durations, constant_parameters, constant_parameter_values
            )

            path_objective, path_constraint_expressions = ca.substitute(
                [path_objective, path_constraint_expressions],
                constant_parameters,
                constant_parameter_values,
            )

        # Collect extra variable symbols
        symbolic_extra_variables = ca.vertcat(*self.extra_variables)

        # Aggregate ensemble data
        ensemble_aggregate = {}
        ensemble_aggregate["parameters"] = ca.horzcat(
            *[nullvertcat(*p) for p in ensemble_parameter_values]
        )
        ensemble_aggregate["initial_constant_inputs"] = ca.horzcat(
            *[
                nullvertcat(
                    *[
                        float(d["constant_inputs"][variable.name()][0])
                        for variable in self.dae_variables["constant_inputs"]
                    ]
                )
                for d in ensemble_store
            ]
        )
        ensemble_aggregate["initial_extra_constant_inputs"] = ca.horzcat(
            *[
                nullvertcat(
                    *[
                        d["extra_constant_inputs"][variable.name()][0, :]
                        for variable in self.__extra_constant_inputs
                    ]
                )
                for d in ensemble_store
            ]
        )
        ensemble_aggregate["initial_state"] = ca.horzcat(
            *[d["initial_state"] for d in ensemble_store]
        )
        ensemble_aggregate["initial_state"] = reduce_matvec(
            ensemble_aggregate["initial_state"], self.solver_input
        )
        ensemble_aggregate["initial_derivatives"] = ca.horzcat(
            *[d["initial_derivatives"] for d in ensemble_store]
        )
        ensemble_aggregate["initial_derivatives"] = reduce_matvec(
            ensemble_aggregate["initial_derivatives"], self.solver_input
        )
        ensemble_aggregate["initial_path_variables"] = ca.horzcat(
            *[d["initial_path_variables"] for d in ensemble_store]
        )
        ensemble_aggregate["initial_path_variables"] = reduce_matvec(
            ensemble_aggregate["initial_path_variables"], self.solver_input
        )

        if (self.__dae_residual_function_collocated is None) and (
            self.__integrator_step_function is None
        ):
            # Insert lookup tables.  No support yet for different lookup tables per ensemble member.
            lookup_tables = self.lookup_tables(0)

            for sym in self.dae_variables["lookup_tables"]:
                sym_name = sym.name()

                try:
                    lookup_table = lookup_tables[sym_name]
                except KeyError:
                    raise Exception(f"Unable to find lookup table function for {sym_name}")
                else:
                    input_syms = [
                        self.variable(input_sym.name()) for input_sym in lookup_table.inputs
                    ]

                    value = lookup_table.function(*input_syms)
                    [dae_residual] = ca.substitute([dae_residual], [sym], [value])

            if len(self.dae_variables["lookup_tables"]) > 0 and self.ensemble_size > 1:
                logger.warning("Using lookup tables of ensemble member #0 for all members.")

            # Insert constant parameter values
            dae_residual, initial_residual = ca.substitute(
                [dae_residual, initial_residual], constant_parameters, constant_parameter_values
            )

            # Allocate DAE to an integrated or to a collocated part
            if self.integrate_states:
                dae_residual_integrated = dae_residual
                dae_residual_collocated = ca.MX()
            else:
                dae_residual_integrated = ca.MX()
                dae_residual_collocated = dae_residual

            # Check linearity of collocated part
            if self.check_collocation_linearity and dae_residual_collocated.size1() > 0:
                # Check linearity of collocation constraints, which is a necessary condition for the
                # optimization problem to be convex
                self.linear_collocation = True

                # Aside from decision variables, the DAE expression also contains parameters
                # and constant inputs. We need to inline them before we do the affinity check.
                # Note that this not an exhaustive check, as other values for the
                # parameters/constant inputs may result in a non-affine DAE (or vice-versa).
                np.random.seed(42)
                fixed_vars = ca.vertcat(
                    *self.dae_variables["time"],
                    *self.dae_variables["constant_inputs"],
                    ca.MX(symbolic_parameters),
                )
                fixed_var_values = np.random.rand(fixed_vars.size1())

                if not is_affine(
                    ca.substitute(dae_residual_collocated, fixed_vars, fixed_var_values),
                    ca.vertcat(
                        *collocated_variables
                        + integrated_variables
                        + collocated_derivatives
                        + integrated_derivatives
                    ),
                ):
                    self.linear_collocation = False

                    logger.warning(
                        "The DAE residual contains equations that are not affine. "
                        "There is therefore no guarantee that the optimization problem is convex. "
                        "This will, in general, result in the existence of multiple local optima "
                        "and trouble finding a feasible initial solution."
                    )

            # Transcribe DAE using theta method collocation
            if self.integrate_states:
                I = ca.MX.sym("I", len(integrated_variables))  # noqa: E741
                I0 = ca.MX.sym("I0", len(integrated_variables))
                C0 = [ca.MX.sym(f"C0[{i}]") for i in range(len(collocated_variables))]
                CI0 = [
                    ca.MX.sym(f"CI0[{i}]")
                    for i in range(len(self.dae_variables["constant_inputs"]))
                ]
                dt_sym = ca.MX.sym("dt")

                integrated_finite_differences = (I - I0) / dt_sym

                [dae_residual_integrated_0] = ca.substitute(
                    [dae_residual_integrated],
                    (
                        integrated_variables
                        + collocated_variables
                        + integrated_derivatives
                        + self.dae_variables["constant_inputs"]
                        + self.dae_variables["time"]
                    ),
                    (
                        [I0[i] for i in range(len(integrated_variables))]
                        + [C0[i] for i in range(len(collocated_variables))]
                        + [
                            integrated_finite_differences[i]
                            for i in range(len(integrated_derivatives))
                        ]
                        + [CI0[i] for i in range(len(self.dae_variables["constant_inputs"]))]
                        + [self.dae_variables["time"][0] - dt_sym]
                    ),
                )
                [dae_residual_integrated_1] = ca.substitute(
                    [dae_residual_integrated],
                    (integrated_variables + integrated_derivatives),
                    (
                        [I[i] for i in range(len(integrated_variables))]
                        + [
                            integrated_finite_differences[i]
                            for i in range(len(integrated_derivatives))
                        ]
                    ),
                )

                if theta == 0:
                    dae_residual_integrated = dae_residual_integrated_0
                elif theta == 1:
                    dae_residual_integrated = dae_residual_integrated_1
                else:
                    dae_residual_integrated = (
                        1 - theta
                    ) * dae_residual_integrated_0 + theta * dae_residual_integrated_1

                dae_residual_function_integrated = ca.Function(
                    "dae_residual_function_integrated",
                    [
                        I,
                        I0,
                        symbolic_parameters,
                        ca.vertcat(
                            *(
                                [C0[i] for i in range(len(collocated_variables))]
                                + [
                                    CI0[i]
                                    for i in range(len(self.dae_variables["constant_inputs"]))
                                ]
                                + [dt_sym]
                                + collocated_variables
                                + collocated_derivatives
                                + self.dae_variables["constant_inputs"]
                                + self.dae_variables["time"]
                            )
                        ),
                    ],
                    [dae_residual_integrated],
                    function_options,
                )

                # Expand the residual function if possible.
                try:
                    dae_residual_function_integrated = dae_residual_function_integrated.expand()
                except RuntimeError as e:
                    if "'eval_sx' not defined for" in str(e):
                        pass
                    else:
                        raise

                options = self.integrator_options()
                self.__integrator_step_function = ca.rootfinder(
                    "integrator_step_function",
                    "fast_newton",
                    dae_residual_function_integrated,
                    options,
                )

            # Initialize a Function for the DAE residual (collocated part)
            elif len(collocated_variables) > 0:
                self.__dae_residual_function_collocated = ca.Function(
                    "dae_residual_function_collocated",
                    [
                        symbolic_parameters,
                        ca.vertcat(
                            *(
                                collocated_variables
                                + collocated_derivatives
                                + self.dae_variables["constant_inputs"]
                                + self.dae_variables["time"]
                            )
                        ),
                    ],
                    [dae_residual_collocated],
                    function_options,
                )
                # Expand the residual function if possible.
                try:
                    self.__dae_residual_function_collocated = (
                        self.__dae_residual_function_collocated.expand()
                    )
                except RuntimeError as e:
                    if "'eval_sx' not defined for" in str(e):
                        pass
                    else:
                        raise

        if self.integrate_states:
            integrator_step_function = self.__integrator_step_function
            dae_residual_collocated_size = 0
        elif len(collocated_variables) > 0:
            dae_residual_function_collocated = self.__dae_residual_function_collocated
            dae_residual_collocated_size = dae_residual_function_collocated.mx_out(0).size1()
        else:
            dae_residual_collocated_size = 0

        # Note that this list is stored, such that it can be reused in the
        # map_path_expression() method.
        self.__func_orig_inputs = [
            symbolic_parameters,
            ca.vertcat(
                *integrated_variables,
                *collocated_variables,
                *integrated_derivatives,
                *collocated_derivatives,
                *self.dae_variables["constant_inputs"],
                *self.dae_variables["time"],
                *self.path_variables,
                *self.__extra_constant_inputs,
            ),
            symbolic_extra_variables,
        ]

        # Initialize a Function for the path objective
        # Note that we assume that the path objective expression is the same for all ensemble
        # members
        path_objective_function = ca.Function(
            "path_objective", self.__func_orig_inputs, [path_objective], function_options
        )
        path_objective_function = path_objective_function.expand()

        # Initialize a Function for the path constraints
        # Note that we assume that the path constraint expression is the same for all ensemble
        # members
        path_constraints_function = ca.Function(
            "path_constraints",
            self.__func_orig_inputs,
            [path_constraint_expressions],
            function_options,
        )
        path_constraints_function = path_constraints_function.expand()

        # Initialize a Function for the delayed feedback
        delayed_feedback_function = ca.Function(
            "delayed_feedback",
            self.__func_orig_inputs,
            delayed_feedback_expressions,
            function_options,
        )
        delayed_feedback_function = delayed_feedback_function.expand()

        # Set up accumulation over time (integration, and generation of
        # collocation constraints)
        if self.integrate_states:
            accumulated_X = ca.MX.sym("accumulated_X", len(integrated_variables))
        else:
            accumulated_X = ca.MX.sym("accumulated_X", 0)

        path_variables_size = sum(x.size1() for x in self.path_variables)
        extra_constant_inputs_size = sum(x.size1() for x in self.__extra_constant_inputs)

        accumulated_U = ca.MX.sym(
            "accumulated_U",
            (
                2 * (len(collocated_variables) + len(self.dae_variables["constant_inputs"]) + 1)
                + path_variables_size
                + extra_constant_inputs_size
            ),
        )

        integrated_states_0 = accumulated_X[0 : len(integrated_variables)]
        integrated_states_1 = ca.MX.sym("integrated_states_1", len(integrated_variables))
        collocated_states_0 = accumulated_U[0 : len(collocated_variables)]
        collocated_states_1 = accumulated_U[
            len(collocated_variables) : 2 * len(collocated_variables)
        ]
        constant_inputs_0 = accumulated_U[
            2 * len(collocated_variables) : 2 * len(collocated_variables)
            + len(self.dae_variables["constant_inputs"])
        ]
        constant_inputs_1 = accumulated_U[
            2 * len(collocated_variables) + len(self.dae_variables["constant_inputs"]) : 2
            * len(collocated_variables)
            + 2 * len(self.dae_variables["constant_inputs"])
        ]

        offset = 2 * (len(collocated_variables) + len(self.dae_variables["constant_inputs"]))
        collocation_time_0 = accumulated_U[offset + 0]
        collocation_time_1 = accumulated_U[offset + 1]
        path_variables_1 = accumulated_U[offset + 2 : offset + 2 + len(self.path_variables)]
        extra_constant_inputs_1 = accumulated_U[offset + 2 + len(self.path_variables) :]

        # Approximate derivatives using backwards finite differences
        dt = collocation_time_1 - collocation_time_0
        integrated_finite_differences = ca.MX()  # Overwritten later if integrate_states is True
        collocated_finite_differences = (collocated_states_1 - collocated_states_0) / dt

        # We use ca.vertcat to compose the list into an MX.  This is, in
        # CasADi 2.4, faster.
        accumulated_Y = []

        # Integrate integrated states
        if self.integrate_states:
            # Perform step by computing implicit function
            # CasADi shares subexpressions that are bundled into the same Function.
            # The first argument is the guess for the new value of
            # integrated_states.
            [integrated_states_1] = integrator_step_function.call(
                [
                    integrated_states_0,
                    integrated_states_0,
                    symbolic_parameters,
                    ca.vertcat(
                        collocated_states_0,
                        constant_inputs_0,
                        dt,
                        collocated_states_1,
                        collocated_finite_differences,
                        constant_inputs_1,
                        collocation_time_1 - t0,
                    ),
                ],
                False,
                True,
            )
            accumulated_Y.append(integrated_states_1)

            # Recompute finite differences with computed new state.
            # We don't use substititute() for this, as it becomes expensive over long
            # integration horizons.
            integrated_finite_differences = (integrated_states_1 - integrated_states_0) / dt

        # Call DAE residual at collocation point
        # Time stamp following paragraph 3.6.7 of the Modelica
        # specifications, version 3.3.
        elif len(collocated_variables) > 0:
            if theta < 1:
                # Obtain state vector
                [dae_residual_0] = dae_residual_function_collocated.call(
                    [
                        symbolic_parameters,
                        ca.vertcat(
                            collocated_states_0,
                            collocated_finite_differences,
                            constant_inputs_0,
                            collocation_time_0 - t0,
                        ),
                    ],
                    False,
                    True,
                )
            if theta > 0:
                # Obtain state vector
                [dae_residual_1] = dae_residual_function_collocated.call(
                    [
                        symbolic_parameters,
                        ca.vertcat(
                            collocated_states_1,
                            collocated_finite_differences,
                            constant_inputs_1,
                            collocation_time_1 - t0,
                        ),
                    ],
                    False,
                    True,
                )
            if theta == 0:
                accumulated_Y.append(dae_residual_0)
            elif theta == 1:
                accumulated_Y.append(dae_residual_1)
            else:
                accumulated_Y.append((1 - theta) * dae_residual_0 + theta * dae_residual_1)

        self.__func_inputs_implicit = [
            symbolic_parameters,
            ca.vertcat(
                integrated_states_1,
                collocated_states_1,
                integrated_finite_differences,
                collocated_finite_differences,
                constant_inputs_1,
                collocation_time_1 - t0,
                path_variables_1,
                extra_constant_inputs_1,
            ),
            symbolic_extra_variables,
        ]

        accumulated_Y.extend(path_objective_function.call(self.__func_inputs_implicit, False, True))

        accumulated_Y.extend(
            path_constraints_function.call(self.__func_inputs_implicit, False, True)
        )

        accumulated_Y.extend(
            delayed_feedback_function.call(self.__func_inputs_implicit, False, True)
        )

        # Save the accumulated inputs such that can be used later in map_path_expression()
        self.__func_accumulated_inputs = (
            accumulated_X,
            accumulated_U,
            ca.veccat(symbolic_parameters, symbolic_extra_variables),
        )

        # Use map/mapaccum to capture integration and collocation constraint generation over the
        # entire time horizon with one symbolic operation. This saves a lot of memory.
        if n_collocation_times > 1:
            if self.integrate_states:
                accumulated = ca.Function(
                    "accumulated",
                    self.__func_accumulated_inputs,
                    [accumulated_Y[0], ca.vertcat(*accumulated_Y[1:])],
                    function_options,
                )
                accumulation = accumulated.mapaccum("accumulation", n_collocation_times - 1)
            else:
                # Fully collocated problem. Use map(), so that we can use
                # parallelization along the time axis.
                accumulated = ca.Function(
                    "accumulated",
                    self.__func_accumulated_inputs,
                    [ca.vertcat(*accumulated_Y)],
                    function_options,
                )
                options = self.map_options()
                if options["mode"] == "thread":
                    accumulation = accumulated.map(
                        n_collocation_times - 1, options["mode"], options["n_threads"]
                    )
                else:
                    accumulation = accumulated.map(n_collocation_times - 1, options["mode"])
        else:
            accumulation = None

        # Start collecting constraints
        f = []
        g = []
        lbg = []
        ubg = []

        # Add constraints for initial conditions
        if self.__initial_residual_with_params_fun_map is None:
            initial_residual_with_params_fun = ca.Function(
                "initial_residual_total",
                [
                    symbolic_parameters,
                    ca.vertcat(
                        *(
                            self.dae_variables["states"]
                            + self.dae_variables["algebraics"]
                            + self.dae_variables["control_inputs"]
                            + integrated_derivatives
                            + collocated_derivatives
                            + self.dae_variables["constant_inputs"]
                            + self.dae_variables["time"]
                        )
                    ),
                ],
                [ca.veccat(dae_residual, initial_residual)],
                function_options,
            )
            self.__initial_residual_with_params_fun_map = initial_residual_with_params_fun.map(
                self.ensemble_size
            )
        initial_residual_with_params_fun_map = self.__initial_residual_with_params_fun_map
        [res] = initial_residual_with_params_fun_map.call(
            [
                ensemble_aggregate["parameters"],
                ca.vertcat(
                    *[
                        ensemble_aggregate["initial_state"],
                        ensemble_aggregate["initial_derivatives"],
                        ensemble_aggregate["initial_constant_inputs"],
                        ca.repmat([0.0], 1, self.ensemble_size),
                    ]
                ),
            ],
            False,
            True,
        )

        res = ca.vec(res)
        g.append(res)
        zeros = [0.0] * res.size1()
        lbg.extend(zeros)
        ubg.extend(zeros)

        # The initial values and the interpolated mapped arguments are saved
        # such that can be reused in map_path_expression().
        self.__func_initial_inputs = []
        self.__func_map_args = []

        # Integrators are saved for result extraction later on
        self.__integrators = []

        # If inlining delay expressions, prepare for the single call to ca.substitute()
        # at the end
        if self.inline_delay_expressions:
            delayed_feedback_variable_replacement = ca.MX.zeros(X.numel())
            delayed_feedback_variable_replacement[:] = X

        # Process the objectives and constraints for each ensemble member separately.
        # Note that we don't use map here for the moment, so as to allow each ensemble member to
        # define its own constraints and objectives. Path constraints are applied for all ensemble
        # members simultaneously at the moment. We can get rid of map again, and allow every
        # ensemble member to specify its own path constraints as well, once CasADi has some kind
        # of loop detection.
        for ensemble_member in range(self.ensemble_size):
            logger.info(f"Transcribing ensemble member {ensemble_member + 1}/{self.ensemble_size}")

            initial_state = ensemble_aggregate["initial_state"][:, ensemble_member]
            initial_derivatives = ensemble_aggregate["initial_derivatives"][:, ensemble_member]
            initial_path_variables = ensemble_aggregate["initial_path_variables"][
                :, ensemble_member
            ]
            initial_constant_inputs = ensemble_aggregate["initial_constant_inputs"][
                :, ensemble_member
            ]
            initial_extra_constant_inputs = ensemble_aggregate["initial_extra_constant_inputs"][
                :, ensemble_member
            ]
            parameters = ensemble_aggregate["parameters"][:, ensemble_member]
            extra_variables = ca.vertcat(
                *[self.extra_variable(var.name(), ensemble_member) for var in self.extra_variables]
            )

            constant_inputs = ensemble_store[ensemble_member]["constant_inputs"]
            extra_constant_inputs = ensemble_store[ensemble_member]["extra_constant_inputs"]

            # Initial conditions specified in history timeseries
            history = self.history(ensemble_member)
            for variable in itertools.chain(
                self.differentiated_states, self.algebraic_states, self.controls
            ):
                try:
                    history_timeseries = history[variable]
                except KeyError:
                    pass
                else:
                    interpolation_method = self.interpolation_method(variable)
                    val = self.interpolate(
                        t0,
                        history_timeseries.times,
                        history_timeseries.values,
                        np.nan,
                        np.nan,
                        interpolation_method,
                    )
                    val /= self.variable_nominal(variable)

                    if not np.isnan(val):
                        idx = self.__indices_as_lists[ensemble_member][variable][0]

                        if val < lbx[idx] or val > ubx[idx]:
                            logger.warning(
                                f"Initial value {val} for variable '{variable}' outside bounds."
                            )

                        lbx[idx] = ubx[idx] = val

            initial_derivative_constraints = []

            for i, variable in enumerate(self.differentiated_states):
                try:
                    history_timeseries = history[variable]
                except KeyError:
                    pass
                else:
                    if len(history_timeseries.times) <= 1 or np.isnan(
                        history_timeseries.values[-2]
                    ):
                        continue

                    assert history_timeseries.times[-1] == t0

                    if np.isnan(history_timeseries.values[-1]):
                        t0_val = self.state_vector(variable, ensemble_member=ensemble_member)[0]
                        t0_val *= self.variable_nominal(variable)

                        val = (t0_val - history_timeseries.values[-2]) / (
                            t0 - history_timeseries.times[-2]
                        )
                        sym = initial_derivatives[i]
                        initial_derivative_constraints.append(sym - val)
                    else:
                        interpolation_method = self.interpolation_method(variable)

                        t0_val = self.interpolate(
                            t0,
                            history_timeseries.times,
                            history_timeseries.values,
                            np.nan,
                            np.nan,
                            interpolation_method,
                        )
                        initial_der_name = self.__initial_derivative_names[i]

                        val = (t0_val - history_timeseries.values[-2]) / (
                            t0 - history_timeseries.times[-2]
                        )
                        val /= self.variable_nominal(initial_der_name)

                        idx = self.__indices[ensemble_member][initial_der_name]
                        lbx[idx] = ubx[idx] = val

            if len(initial_derivative_constraints) > 0:
                g.append(ca.vertcat(*initial_derivative_constraints))
                lbg.append(np.zeros(len(initial_derivative_constraints)))
                ubg.append(np.zeros(len(initial_derivative_constraints)))

            # Initial conditions for integrator
            accumulation_X0 = []
            if self.integrate_states:
                for variable in integrated_variable_names:
                    value = self.state_vector(variable, ensemble_member=ensemble_member)[0]
                    nominal = self.variable_nominal(variable)
                    if nominal != 1:
                        value *= nominal
                    accumulation_X0.append(value)
            accumulation_X0 = ca.vertcat(*accumulation_X0)

            # Input for map
            logger.info("Interpolating states")

            accumulation_U = [None] * (
                1
                + 2 * len(self.dae_variables["constant_inputs"])
                + 3
                + len(self.__extra_constant_inputs)
            )

            # Most variables have collocation times equal to the global
            # collocation times. Use a vectorized approach to process them.
            interpolated_states_explicit = []
            interpolated_states_implicit = []

            place_holder = [-1] * n_collocation_times
            for variable in collocated_variable_names:
                var_inds = self.__indices_as_lists[ensemble_member][variable]

                # If the variable times != collocation times, what we do here is just a placeholder
                if len(var_inds) != n_collocation_times:
                    var_inds = var_inds.copy()
                    var_inds.extend(place_holder)
                    var_inds = var_inds[:n_collocation_times]

                interpolated_states_explicit.extend(var_inds[:-1])
                interpolated_states_implicit.extend(var_inds[1:])

            repeated_nominals = np.tile(
                np.repeat(collocated_variable_nominals, n_collocation_times - 1), 2
            )
            interpolated_states = (
                ca.vertcat(X[interpolated_states_explicit], X[interpolated_states_implicit])
                * repeated_nominals
            )
            interpolated_states = interpolated_states.reshape(
                (n_collocation_times - 1, len(collocated_variables) * 2)
            )

            # Handle variables that have different collocation times.
            for j, variable in enumerate(collocated_variable_names):
                times = self.times(variable)
                if n_collocation_times == len(times):
                    # Already handled
                    continue

                interpolation_method = self.interpolation_method(variable)
                values = self.state_vector(variable, ensemble_member=ensemble_member)
                interpolated = interpolate(
                    times, values, collocation_times, False, interpolation_method
                )

                nominal = self.variable_nominal(variable)
                if nominal != 1:
                    interpolated *= nominal

                interpolated_states[:, j] = interpolated[:-1]
                interpolated_states[:, len(collocated_variables) + j] = interpolated[1:]

            # We do not cache the Jacobians, as the structure may change from ensemble member to
            # member, and from goal programming/homotopy run to run.
            # We could, of course, pick the states apart into controls and states, and generate
            # Jacobians for each set separately and for each ensemble member separately, but in
            # this case the increased complexity may well offset the performance gained by
            # caching.
            interpolated_states = reduce_matvec(interpolated_states, self.solver_input)

            accumulation_U[0] = interpolated_states

            for j, variable in enumerate(self.dae_variables["constant_inputs"]):
                variable = variable.name()
                constant_input = constant_inputs[variable]
                accumulation_U[1 + j] = ca.MX(constant_input[0 : n_collocation_times - 1])
                accumulation_U[1 + len(self.dae_variables["constant_inputs"]) + j] = ca.MX(
                    constant_input[1:n_collocation_times]
                )

            accumulation_U[1 + 2 * len(self.dae_variables["constant_inputs"])] = ca.MX(
                collocation_times[0 : n_collocation_times - 1]
            )
            accumulation_U[1 + 2 * len(self.dae_variables["constant_inputs"]) + 1] = ca.MX(
                collocation_times[1:n_collocation_times]
            )

            path_variables = [None] * len(self.path_variables)
            for j, variable in enumerate(self.__path_variable_names):
                variable_size = self.__variable_sizes[variable]
                values = self.state_vector(variable, ensemble_member=ensemble_member)

                nominal = self.variable_nominal(variable)
                if isinstance(nominal, np.ndarray):
                    nominal = (
                        np.broadcast_to(nominal, (n_collocation_times, variable_size))
                        .transpose()
                        .ravel()
                    )
                    values *= nominal
                elif nominal != 1:
                    values *= nominal

                path_variables[j] = values.reshape((n_collocation_times, variable_size))[1:, :]

            path_variables = reduce_matvec(ca.horzcat(*path_variables), self.solver_input)

            accumulation_U[1 + 2 * len(self.dae_variables["constant_inputs"]) + 2] = path_variables

            for j, variable in enumerate(self.__extra_constant_inputs):
                variable = variable.name()
                constant_input = extra_constant_inputs[variable]
                accumulation_U[1 + 2 * len(self.dae_variables["constant_inputs"]) + 3 + j] = ca.MX(
                    constant_input[1:n_collocation_times, :]
                )

            # Construct matrix using O(states) CasADi operations
            # This is faster than using blockcat, presumably because of the
            # row-wise scaling operations.
            logger.info("Aggregating and de-scaling variables")

            accumulation_U = [var for var in accumulation_U if var.numel() > 0]
            accumulation_U = ca.transpose(ca.horzcat(*accumulation_U))

            # Map to all time steps
            logger.info("Mapping")

            # Save these inputs such that can be used later in map_path_expression()
            self.__func_initial_inputs.append(
                [
                    parameters,
                    ca.vertcat(
                        initial_state,
                        initial_derivatives,
                        initial_constant_inputs,
                        0.0,
                        initial_path_variables,
                        initial_extra_constant_inputs,
                    ),
                    extra_variables,
                ]
            )

            if accumulation is not None:
                integrators_and_collocation_and_path_constraints = accumulation(
                    accumulation_X0,
                    accumulation_U,
                    ca.repmat(ca.vertcat(parameters, extra_variables), 1, n_collocation_times - 1),
                )
            else:
                integrators_and_collocation_and_path_constraints = None

            if accumulation is not None and self.integrate_states:
                integrators = integrators_and_collocation_and_path_constraints[0]
                integrators_and_collocation_and_path_constraints = (
                    integrators_and_collocation_and_path_constraints[1]
                )
            if (
                accumulation is not None
                and integrators_and_collocation_and_path_constraints.numel() > 0
            ):
                collocation_constraints = ca.vec(
                    integrators_and_collocation_and_path_constraints[
                        :dae_residual_collocated_size, 0 : n_collocation_times - 1
                    ]
                )
                discretized_path_objective = ca.vec(
                    integrators_and_collocation_and_path_constraints[
                        dae_residual_collocated_size : dae_residual_collocated_size
                        + path_objective.size1(),
                        0 : n_collocation_times - 1,
                    ]
                )
                discretized_path_constraints = ca.vec(
                    integrators_and_collocation_and_path_constraints[
                        dae_residual_collocated_size
                        + path_objective.size1() : dae_residual_collocated_size
                        + path_objective.size1()
                        + path_constraint_expressions.size1(),
                        0 : n_collocation_times - 1,
                    ]
                )
                discretized_delayed_feedback = integrators_and_collocation_and_path_constraints[
                    dae_residual_collocated_size
                    + path_objective.size1()
                    + path_constraint_expressions.size1() :,
                    0 : n_collocation_times - 1,
                ]
            else:
                collocation_constraints = ca.MX()
                discretized_path_objective = ca.MX()
                discretized_path_constraints = ca.MX()
                discretized_delayed_feedback = ca.MX()

            logger.info("Composing NLP segment")

            # Store integrators for result extraction
            if self.integrate_states:
                # Store integrators for result extraction
                self.__integrators.append(
                    {
                        variable: integrators[i, :]
                        for i, variable in enumerate(integrated_variable_names)
                    }
                )
            else:
                # Add collocation constraints
                g.append(collocation_constraints)
                zeros = np.zeros(collocation_constraints.size1())
                lbg.extend(zeros)
                ubg.extend(zeros)

            # Prepare arguments for map_path_expression() calls to ca.map()
            if len(integrated_variables) + len(collocated_variables) > 0:
                if self.integrate_states:
                    # Inputs
                    states_and_algebraics_and_controls = ca.vertcat(
                        *[
                            self.variable_nominal(variable)
                            * self.__integrators[ensemble_member][variable]
                            for variable in integrated_variable_names
                        ],
                        interpolated_states[
                            :,
                            len(collocated_variables) :,
                        ].T,
                    )
                    states_and_algebraics_and_controls_derivatives = (
                        (
                            states_and_algebraics_and_controls
                            - ca.horzcat(
                                ensemble_store[ensemble_member]["initial_state"],
                                states_and_algebraics_and_controls[:, :-1],
                            )
                        ).T
                        / (collocation_times[1:] - collocation_times[:-1])
                    ).T
                else:
                    states_and_algebraics_and_controls = interpolated_states[
                        :, len(collocated_variables) :
                    ].T
                    states_and_algebraics_and_controls_derivatives = (
                        (
                            interpolated_states[:, len(collocated_variables) :]
                            - interpolated_states[:, : len(collocated_variables)]
                        )
                        / (collocation_times[1:] - collocation_times[:-1])
                    ).T
            else:
                states_and_algebraics_and_controls = ca.MX()
                states_and_algebraics_and_controls_derivatives = ca.MX()

            self.__func_map_args.append(
                [
                    ca.repmat(
                        ca.vertcat(*ensemble_parameter_values[ensemble_member]),
                        1,
                        n_collocation_times - 1,
                    ),
                    ca.vertcat(
                        states_and_algebraics_and_controls,
                        states_and_algebraics_and_controls_derivatives,
                        *[
                            ca.MX(constant_inputs[variable][1:]).T
                            for variable in dae_constant_inputs_names
                        ],
                        ca.MX(collocation_times[1:]).T,
                        path_variables.T if path_variables.numel() > 0 else ca.MX(),
                        *[
                            ca.MX(extra_constant_inputs[variable][1:]).T
                            for (variable, _) in extra_constant_inputs_name_and_size
                        ],
                    ),
                    ca.repmat(extra_variables, 1, n_collocation_times - 1),
                ]
            )

            # Delayed feedback
            # Make an array of all unique times in history series
            history_times = np.unique(
                np.hstack(
                    (np.array([]), *[history_series.times for history_series in history.values()])
                )
            )
            # By convention, the last timestep in history series is the initial time. We drop this
            # index
            history_times = history_times[:-1]

            # Find the historical values of states, extrapolating backward if necessary
            history_values = np.empty(
                (history_times.shape[0], len(integrated_variables) + len(collocated_variables))
            )
            if history_times.shape[0] > 0:
                for j, var in enumerate(integrated_variables + collocated_variables):
                    var_name = var.name()
                    try:
                        history_series = history[var_name]
                    except KeyError:
                        history_values[:, j] = np.nan
                    else:
                        interpolation_method = self.interpolation_method(var_name)
                        history_values[:, j] = self.interpolate(
                            history_times,
                            history_series.times,
                            history_series.values,
                            np.nan,
                            np.nan,
                            interpolation_method,
                        )

            # Calculate the historical derivatives of historical values
            history_derivatives = ca.repmat(np.nan, 1, history_values.shape[1])
            if history_times.shape[0] > 1:
                history_derivatives = ca.vertcat(
                    history_derivatives,
                    np.diff(history_values, axis=0) / np.diff(history_times)[:, None],
                )

            # Find the historical values of constant inputs, extrapolating backward if necessary
            constant_input_values = np.empty(
                (history_times.shape[0], len(self.dae_variables["constant_inputs"]))
            )
            if history_times.shape[0] > 0:
                for j, var in enumerate(self.dae_variables["constant_inputs"]):
                    var_name = var.name()
                    try:
                        constant_input_series = raw_constant_inputs[var_name]
                    except KeyError:
                        constant_input_values[:, j] = np.nan
                    else:
                        interpolation_method = self.interpolation_method(var_name)
                        constant_input_values[:, j] = self.interpolate(
                            history_times,
                            constant_input_series.times,
                            constant_input_series.values,
                            np.nan,
                            np.nan,
                            interpolation_method,
                        )

            if len(delayed_feedback_expressions) > 0:
                delayed_feedback_history = np.zeros(
                    (history_times.shape[0], len(delayed_feedback_expressions))
                )
                for i, time in enumerate(history_times):
                    history_delayed_feedback_res = delayed_feedback_function.call(
                        [
                            parameters,
                            ca.veccat(
                                ca.transpose(history_values[i, :]),
                                ca.transpose(history_derivatives[i, :]),
                                ca.transpose(constant_input_values[i, :]),
                                time,
                                ca.repmat(np.nan, len(self.path_variables)),
                                ca.repmat(np.nan, len(self.__extra_constant_inputs)),
                            ),
                            ca.repmat(np.nan, len(self.extra_variables)),
                        ]
                    )
                    delayed_feedback_history[i, :] = [
                        float(val) for val in history_delayed_feedback_res
                    ]

                initial_delayed_feedback = delayed_feedback_function.call(
                    self.__func_initial_inputs[ensemble_member], False, True
                )

                path_variables_nominal = np.ones(path_variables_size)
                offset = 0
                for variable in self.__path_variable_names:
                    variable_size = self.__variable_sizes[variable]
                    path_variables_nominal[offset : offset + variable_size] = self.variable_nominal(
                        variable
                    )
                    offset += variable_size

                nominal_delayed_feedback = delayed_feedback_function.call(
                    [
                        parameters,
                        ca.vertcat(
                            [
                                self.variable_nominal(var.name())
                                for var in integrated_variables + collocated_variables
                            ],
                            np.zeros((initial_derivatives.size1(), 1)),
                            initial_constant_inputs,
                            0.0,
                            path_variables_nominal,
                            initial_extra_constant_inputs,
                        ),
                        extra_variables,
                    ]
                )

            if delayed_feedback_expressions:
                # Resolve delay values
                # First, substitute parameters for values all at once. Make
                # sure substitute() gets called with the right signature. This
                # means we need at least one element that is of type MX.
                delayed_feedback_durations = list(delayed_feedback_durations)
                delayed_feedback_durations[0] = ca.MX(delayed_feedback_durations[0])

                substituted_delay_durations = ca.substitute(
                    delayed_feedback_durations,
                    [ca.vertcat(symbolic_parameters)],
                    [ca.vertcat(parameters)],
                )

                # Use mapped function to evaluate delay in terms of constant inputs
                mapped_delay_function = ca.Function(
                    "delay_values",
                    self.dae_variables["time"] + self.dae_variables["constant_inputs"],
                    substituted_delay_durations,
                ).map(len(collocation_times))

                # Call mapped delay function with inputs as arrays
                evaluated_delay_durations = mapped_delay_function.call(
                    [collocation_times]
                    + [constant_inputs[v.name()] for v in self.dae_variables["constant_inputs"]]
                )

                for i in range(len(delayed_feedback_expressions)):
                    in_variable_name = delayed_feedback_states[i]
                    expression = delayed_feedback_expressions[i]
                    delay = evaluated_delay_durations[i]

                    # Resolve aliases
                    in_canonical, in_sign = self.alias_relation.canonical_signed(in_variable_name)
                    in_times = self.times(in_canonical)
                    in_nominal = self.variable_nominal(in_canonical)
                    in_values = in_nominal * self.state_vector(
                        in_canonical, ensemble_member=ensemble_member
                    )
                    if in_sign < 0:
                        in_values *= in_sign

                    # Cast delay from DM to np.array
                    delay = delay.toarray().flatten()

                    assert np.all(np.isfinite(delay)), (
                        "Delay duration must be resolvable to real values at transcribe()"
                    )

                    out_times = np.concatenate([history_times, collocation_times])
                    out_values = ca.veccat(
                        delayed_feedback_history[:, i],
                        initial_delayed_feedback[i],
                        ca.transpose(discretized_delayed_feedback[i, :]),
                    )

                    # Check whether enough history has been specified, and that no
                    # needed history values are missing
                    hist_earliest = np.min(collocation_times - delay)
                    hist_start_ind = np.searchsorted(out_times, hist_earliest)
                    if out_times[hist_start_ind] != hist_earliest:
                        # We need an earlier value to interpolate with
                        hist_start_ind -= 1

                    if hist_start_ind < 0 or np.any(
                        np.isnan(delayed_feedback_history[hist_start_ind:, i])
                    ):
                        logger.warning(
                            f"Incomplete history for delayed expression {expression}. "
                            "Extrapolating t0 value backwards in time."
                        )
                        out_times = out_times[len(history_times) :]
                        out_values = out_values[len(history_times) :]

                    # Set up delay constraints
                    if len(collocation_times) != len(in_times):
                        interpolation_method = self.interpolation_method(in_canonical)
                        x_in = interpolate(
                            in_times, in_values, collocation_times, False, interpolation_method
                        )
                    else:
                        x_in = in_values
                    interpolation_method = self.interpolation_method(in_canonical)
                    x_out_delayed = interpolate(
                        out_times,
                        out_values,
                        collocation_times - delay,
                        False,
                        interpolation_method,
                    )

                    nominal = nominal_delayed_feedback[i]

                    if self.inline_delay_expressions:
                        # Get the indices for the delayed feedback variable in the optimization
                        # vector
                        indices = self.__indices[ensemble_member][in_canonical]

                        # Insert replacement values into appropriate slot
                        delayed_feedback_variable_replacement[indices] = (
                            x_out_delayed / in_nominal / in_sign
                        )

                        # Equate delayed feedback variable to zero so that the numerical solver can
                        # remove it from the problem.
                        #
                        # Note: The alternative approach would be to shrink self.solver_input=X,
                        # self.__indices, and any other variables that directly or indirectly
                        # depend on the size and order of 'X'.  This would be a complex operation,
                        # which would however be redundant for solvers that (like Ipopt) already
                        # automatically detect and remove variables where lbx==ubx.
                        lbx[indices] = 0.0
                        ubx[indices] = 0.0
                    else:
                        # Default behavior: add delay expressions as equality constraints
                        g.append((x_in - x_out_delayed) / nominal)
                        zeros = np.zeros(n_collocation_times)
                        lbg.extend(zeros)
                        ubg.extend(zeros)

            # Objective
            f_member = self.objective(ensemble_member)
            if f_member.size1() == 0:
                f_member = 0
            if path_objective.size1() > 0:
                initial_path_objective = path_objective_function.call(
                    self.__func_initial_inputs[ensemble_member], False, True
                )
                f_member += initial_path_objective[0] + ca.sum1(discretized_path_objective)
            f.append(self.ensemble_member_probability(ensemble_member) * f_member)

            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.debug(f"Adding objective {f_member}")

            # Constraints
            constraints = self.constraints(ensemble_member)
            if constraints is None:
                raise Exception(
                    "The `constraints` method returned None, but should always return a list."
                )

            if logger.getEffectiveLevel() == logging.DEBUG:
                for constraint in constraints:
                    logger.debug("Adding constraint {}, {}, {}".format(*constraint))

            if constraints:
                g_constraint, lbg_constraint, ubg_constraint = list(zip(*constraints, strict=True))

                lbg_constraint = list(lbg_constraint)
                ubg_constraint = list(ubg_constraint)

                # Broadcast lbg/ubg if it's a vector constraint
                for i, (g_i, lbg_i, ubg_i) in enumerate(
                    zip(g_constraint, lbg_constraint, ubg_constraint, strict=True)
                ):
                    s = g_i.size1()
                    if s > 1:
                        if not isinstance(lbg_i, np.ndarray) or lbg_i.shape[0] == 1:
                            lbg_constraint[i] = np.full(s, lbg_i)
                        elif lbg_i.shape[0] != g_i.shape[0]:
                            raise Exception(
                                "Shape mismatch between constraint "
                                f"#{i} ({g_i.shape[0]},) and its lower bound ({lbg_i.shape[0]},)"
                            )

                        if not isinstance(ubg_i, np.ndarray) or ubg_i.shape[0] == 1:
                            ubg_constraint[i] = np.full(s, ubg_i)
                        elif ubg_i.shape[0] != g_i.shape[0]:
                            raise Exception(
                                "Shape mismatch between constraint "
                                f"#{i} ({g_i.shape[0]},) and its upper bound ({ubg_i.shape[0]},)"
                            )

                g.extend(g_constraint)
                lbg.extend(lbg_constraint)
                ubg.extend(ubg_constraint)

            # Path constraints
            # We need to call self.path_constraints() again here,
            # as the bounds may change from ensemble member to member.
            if ensemble_member > 0:
                path_constraints = self.path_constraints(ensemble_member)

            if len(path_constraints) > 0:
                # We need to evaluate the path constraints at t0, as the initial time is not
                # included in the accumulation.
                [initial_path_constraints] = path_constraints_function.call(
                    self.__func_initial_inputs[ensemble_member], False, True
                )
                g.append(initial_path_constraints)
                g.append(discretized_path_constraints)

                lbg_path_constraints = np.empty(
                    (path_constraint_expressions.size1(), n_collocation_times)
                )
                ubg_path_constraints = np.empty(
                    (path_constraint_expressions.size1(), n_collocation_times)
                )

                j = 0
                for path_constraint in path_constraints:
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        logger.debug("Adding path constraint {}, {}, {}".format(*path_constraint))

                    s = path_constraint[0].size1()

                    lb = path_constraint[1]
                    if isinstance(lb, ca.MX) and not lb.is_constant():
                        [lb] = ca.substitute(
                            [lb], symbolic_parameters, self.__parameter_values_ensemble_member_0
                        )
                    elif isinstance(lb, Timeseries):
                        lb = self.interpolate(
                            collocation_times, lb.times, lb.values, -np.inf, -np.inf
                        ).transpose()
                    elif isinstance(lb, np.ndarray):
                        lb = np.broadcast_to(lb, (n_collocation_times, s)).transpose()

                    ub = path_constraint[2]
                    if isinstance(ub, ca.MX) and not ub.is_constant():
                        [ub] = ca.substitute(
                            [ub], symbolic_parameters, self.__parameter_values_ensemble_member_0
                        )
                    elif isinstance(ub, Timeseries):
                        ub = self.interpolate(
                            collocation_times, ub.times, ub.values, np.inf, np.inf
                        ).transpose()
                    elif isinstance(ub, np.ndarray):
                        ub = np.broadcast_to(ub, (n_collocation_times, s)).transpose()

                    lbg_path_constraints[j : j + s, :] = lb
                    ubg_path_constraints[j : j + s, :] = ub

                    j += s

                lbg.extend(lbg_path_constraints.transpose().ravel())
                ubg.extend(ubg_path_constraints.transpose().ravel())

        # If inlining delay expressions, carry out the single call to ca.substitute()
        # A single call is more efficient as it constructs a single ca.Function internally,
        # rather than one for each element of the list. Also note that we do this _outside_
        # the ensemble member loop for speed reasons, as each substitute() roughly takes the
        # same amount of time regardless how many expressions we are replacing.
        if self.inline_delay_expressions:
            g = [ca.substitute(ca.vertcat(*g), X, delayed_feedback_variable_replacement)]

        # NLP function
        logger.info("Creating NLP dictionary")

        nlp = {"x": X, "f": ca.sum1(ca.vertcat(*f)), "g": ca.vertcat(*g)}

        # Done
        logger.info("Done transcribing problem")

        # Debug check coefficients
        self.__debug_check_transcribe_linear_coefficients(discrete, lbx, ubx, lbg, ubg, x0, nlp)

        return discrete, lbx, ubx, lbg, ubg, x0, nlp

    def clear_transcription_cache(self):
        """
        Clears the DAE ``Function``s that were cached by ``transcribe``.
        """
        self.__dae_residual_function_collocated = None
        self.__integrator_step_function = None
        self.__initial_residual_with_params_fun_map = None

    def extract_results(self, ensemble_member=0):
        logger.info("Extracting results")

        # Gather results in a dictionary
        control_results = self.extract_controls(ensemble_member)
        state_results = self.extract_states(ensemble_member)

        # Merge dictionaries
        results = AliasDict(self.alias_relation)
        results.update(control_results)
        results.update(state_results)

        logger.info("Done extracting results")

        # Return results dictionary
        return results

    @property
    def solver_input(self):
        return self.__solver_input

    def solver_options(self):
        options = super().solver_options()

        solver = options["solver"]
        assert solver in ["bonmin", "ipopt"]

        # Set the option in both cases, to avoid one inadvertently remaining in the cache.
        options[solver]["jac_c_constant"] = "yes" if self.linear_collocation else "no"
        return options

    def integrator_options(self):
        """
        Configures the implicit function used for time step integration.

        :returns: A dictionary of CasADi :class:`rootfinder` options.  See the CasADi documentation
            for details.
        """
        return {}

    @property
    def controls(self):
        return self.__controls

    def _collint_get_lbx_ubx(
        self,
        bounds: dict[str, BT] | list[dict[str, BT]],
        count: int,
        indices: list[dict[str, slice | int]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lbx = np.full(count, -np.inf, dtype=np.float64)
        ubx = np.full(count, np.inf, dtype=np.float64)

        # Variables that are not collocated, and only have a single entry in the state vector
        scalar_variables_set = set(self.__extra_variable_names) | set(self.__integrated_states)

        variable_sizes = self.__variable_sizes

        # Bounds, defaulting to +/- inf, if not set
        for ensemble_member in range(self.ensemble_size):
            if self.ensemble_specific_bounds:
                bounds_member = bounds[ensemble_member]
            else:
                bounds_member = bounds

            for variable, inds in indices[ensemble_member].items():
                variable_size = variable_sizes[variable]

                if variable in scalar_variables_set:
                    times = self.initial_time
                    n_times = 1
                else:
                    times = self.times(variable)
                    n_times = len(times)

                try:
                    bound = bounds_member[variable]
                except KeyError:
                    pass
                else:
                    nominal = self.variable_nominal(variable)
                    interpolation_method = self.interpolation_method(variable)
                    if isinstance(nominal, np.ndarray):
                        nominal = (
                            np.broadcast_to(nominal, (n_times, variable_size)).transpose().ravel()
                        )

                    if bound[0] is not None:
                        if isinstance(bound[0], Timeseries):
                            lower_bound = self.interpolate(
                                times,
                                bound[0].times,
                                bound[0].values,
                                -np.inf,
                                -np.inf,
                                interpolation_method,
                            ).ravel()
                        elif isinstance(bound[0], np.ndarray):
                            lower_bound = (
                                np.broadcast_to(bound[0], (n_times, variable_size))
                                .transpose()
                                .ravel()
                            )
                        else:
                            lower_bound = bound[0]
                        lbx[inds] = np.maximum(lbx[inds], lower_bound / nominal)

                    if bound[1] is not None:
                        if isinstance(bound[1], Timeseries):
                            upper_bound = self.interpolate(
                                times,
                                bound[1].times,
                                bound[1].values,
                                +np.inf,
                                +np.inf,
                                interpolation_method,
                            ).ravel()
                        elif isinstance(bound[1], np.ndarray):
                            upper_bound = (
                                np.broadcast_to(bound[1], (n_times, variable_size))
                                .transpose()
                                .ravel()
                            )
                        else:
                            upper_bound = bound[1]
                        ubx[inds] = np.minimum(ubx[inds], upper_bound / nominal)

                # Warn for NaNs
                if np.any(np.isnan(lbx[inds])):
                    logger.error(f"Lower bound on variable {variable} contains NaN")
                if np.any(np.isnan(ubx[inds])):
                    logger.error(f"Upper bound on variable {variable} contains NaN")

                # Check that the lower bounds are not higher than the upper
                # bounds. To avoid spam, we just log the first offending one per
                # variable, not _all_ time steps.
                if np.any(lbx[inds] > ubx[inds]):
                    error_inds = np.where(lbx[inds] > ubx[inds])[0].tolist()
                    logger.error(
                        f"Lower bound {lbx[inds][error_inds[0]] * nominal} is higher than "
                        f"upper bound {ubx[inds][error_inds[0]] * nominal} for variable {variable}"
                    )

        return lbx, ubx

    def _collint_get_x0(self, count, indices):
        x0 = np.zeros(count, dtype=np.float64)

        # Variables that are not collocated, and only have a single entry in the state vector
        scalar_variables_set = set(self.__extra_variable_names) | set(self.__integrated_states)

        variable_sizes = self.__variable_sizes

        for ensemble_member in range(self.ensemble_size):
            seed = self.seed(ensemble_member)
            for variable, inds in indices[ensemble_member].items():
                variable_size = variable_sizes[variable]

                if variable in scalar_variables_set:
                    times = self.initial_time
                    n_times = 1
                else:
                    times = self.times(variable)
                    n_times = len(times)

                try:
                    seed_k = seed[variable]
                    nominal = self.variable_nominal(variable)
                    interpolation_method = self.interpolation_method(variable)
                    if isinstance(nominal, np.ndarray):
                        nominal = (
                            np.broadcast_to(nominal, (n_times, variable_size)).transpose().ravel()
                        )

                    if isinstance(seed_k, Timeseries):
                        seed_k = (
                            self.interpolate(
                                times, seed_k.times, seed_k.values, 0, 0, interpolation_method
                            )
                            .transpose()
                            .ravel()
                        )
                    if isinstance(inds, int) and isinstance(seed_k, np.ndarray):
                        seed_k = seed_k.item()
                    x0[inds] = seed_k

                    x0[inds] /= nominal
                except KeyError:
                    pass
        return x0

    def _collint_get_discrete(self, count, indices):
        discrete = np.zeros(count, dtype=bool)

        for ensemble_member in range(self.ensemble_size):
            for variable, inds in indices[ensemble_member].items():
                discrete[inds] = self.variable_is_discrete(variable)

        return discrete

    def discretize_control(self, variable, ensemble_member, times, offset):
        # Default implementation: One single set of control inputs for all
        # ensembles
        try:
            return self.__discretize_control_cache[variable]
        except KeyError:
            control_indices = slice(offset, offset + len(times))
            self.__discretize_control_cache[variable] = control_indices
            return control_indices

    def discretize_controls(self, bounds):
        self.__discretize_control_cache = {}

        indices = [{} for ensemble_member in range(self.ensemble_size)]

        count = 0
        for variable in self.controls:
            times = self.times(variable)

            for ensemble_member in range(self.ensemble_size):
                control_indices = self.discretize_control(variable, ensemble_member, times, count)
                indices[ensemble_member][variable] = control_indices
                control_indices_stop = (
                    control_indices.stop
                    if isinstance(control_indices, slice)
                    else (int(np.max(control_indices)) + 1)
                )  # indices need not be ordered
                count = max(count, control_indices_stop)

        discrete = self._collint_get_discrete(count, indices)
        lbx, ubx = self._collint_get_lbx_ubx(bounds, count, indices)
        x0 = self._collint_get_x0(count, indices)

        # Return number of control variables
        return count, discrete, lbx, ubx, x0, indices

    def extract_controls(self, ensemble_member=0):
        X = self.solver_output.copy()

        indices = self.__indices[ensemble_member]

        results = {}
        for variable in self.controls:
            inds = indices[variable]
            results[variable] = self.variable_nominal(variable) * X[inds]

        return results

    def control_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        canonical, sign = self.alias_relation.canonical_signed(variable)

        if canonical not in self.__controls_map:
            raise KeyError(variable)

        return self.state_at(variable, t, ensemble_member, scaled, extrapolate)

    @property
    def differentiated_states(self):
        return self.__differentiated_states

    @property
    def algebraic_states(self):
        return self.__algebraic_states

    def discretize_states(self, bounds):
        # Default implementation: States for all ensemble members
        variable_sizes = self.__variable_sizes

        # Space for collocated states
        ensemble_member_size = 0
        if self.integrate_states:
            n_model_states = len(self.differentiated_states) + len(self.algebraic_states)
            if len(self.__integrated_states) != n_model_states:
                error_msg = (
                    "CollocatedIntegratedOptimizationProblem: "
                    "integrated_states should specify all model states, or none at all"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

            # Count initial states only
            ensemble_member_size += n_model_states
        else:
            # Count discretised states over optimization horizon
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states):
                ensemble_member_size += variable_sizes[variable] * len(self.times(variable))
        # Count any additional path variables (which cannot be integrated)
        for variable in self.__path_variable_names:
            ensemble_member_size += variable_sizes[variable] * len(self.times(variable))

        # Space for extra variables
        for variable in self.__extra_variable_names:
            ensemble_member_size += variable_sizes[variable]

        # Space for initial states and derivatives
        ensemble_member_size += len(self.dae_variables["derivatives"])

        # Total space requirement
        count = self.ensemble_size * ensemble_member_size

        # Allocate arrays
        indices = [{} for ensemble_member in range(self.ensemble_size)]

        for ensemble_member in range(self.ensemble_size):
            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states):
                variable_size = variable_sizes[variable]

                if self.integrate_states:
                    assert variable_size == 1
                    indices[ensemble_member][variable] = offset

                    offset += 1
                else:
                    times = self.times(variable)
                    n_times = len(times)

                    indices[ensemble_member][variable] = slice(
                        offset, offset + n_times * variable_size
                    )

                    offset += n_times * variable_size

            for variable in self.__path_variable_names:
                variable_size = variable_sizes[variable]

                times = self.times(variable)
                n_times = len(times)

                indices[ensemble_member][variable] = slice(offset, offset + n_times * variable_size)

                offset += n_times * variable_size

            for extra_variable in self.__extra_variable_names:
                variable_size = variable_sizes[extra_variable]

                indices[ensemble_member][extra_variable] = slice(offset, offset + variable_size)

                offset += variable_size

            for initial_der_name in self.__initial_derivative_names:
                indices[ensemble_member][initial_der_name] = offset

                offset += 1

        discrete = self._collint_get_discrete(count, indices)
        lbx, ubx = self._collint_get_lbx_ubx(bounds, count, indices)
        x0 = self._collint_get_x0(count, indices)

        # Return number of state variables
        return count, discrete, lbx, ubx, x0, indices

    def extract_states(self, ensemble_member=0):
        # Solver output
        X = self.solver_output.copy()

        indices = self.__indices[ensemble_member]

        # Extract control inputs
        results = {}

        # Perform integration, in order to extract integrated variables
        # We bundle all integrations into a single Function, so that subexpressions
        # are evaluated only once.
        if self.integrate_states:
            # Use integrators to facilitate common subexpression
            # elimination.
            f = ca.Function(
                "f",
                [self.solver_input],
                [
                    ca.vertcat(
                        *[
                            self.__integrators[ensemble_member][variable]
                            for variable in self.__integrated_states
                        ]
                    )
                ],
            )
            integrators_output = f(X)
            j = 0
            for variable in self.__integrated_states:
                inds = indices[variable]
                initial_value = X[inds]
                n = self.__integrators[ensemble_member][variable].size1()
                results[variable] = self.variable_nominal(variable) * np.concatenate(
                    [[initial_value], np.array(integrators_output[j : j + n, :]).ravel()]
                )
                j += n

        # Extract initial derivatives
        for initial_der_name in self.__initial_derivative_names:
            inds = indices[initial_der_name]

            try:
                nominal = self.variable_nominal(initial_der_name)
                results[initial_der_name] = nominal * X[inds].ravel()
            except KeyError:
                pass

        # Extract all other variables
        variable_sizes = self.__variable_sizes

        for variable in itertools.chain(
            self.differentiated_states,
            self.algebraic_states,
            self.__path_variable_names,
            self.__extra_variable_names,
        ):
            if variable in results:
                continue

            inds = indices[variable]
            variable_size = variable_sizes[variable]

            if variable_size > 1:
                results[variable] = X[inds].reshape((variable_size, -1)).transpose()
            else:
                results[variable] = X[inds]

            results[variable] = self.variable_nominal(variable) * results[variable]

        # Extract constant input aliases
        constant_inputs = self.constant_inputs(ensemble_member)
        for variable in self.dae_variables["constant_inputs"]:
            variable = variable.name()
            try:
                constant_input = constant_inputs[variable]
            except KeyError:
                pass
            else:
                results[variable] = np.interp(
                    self.times(variable), constant_input.times, constant_input.values
                )

        return results

    def state_vector(self, variable, ensemble_member=0):
        indices = self.__indices[ensemble_member][variable]
        return self.solver_input[indices]

    def state_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        if isinstance(variable, ca.MX):
            variable = variable.name()

        if self.__variable_sizes.get(variable, 1) > 1:
            raise NotImplementedError("state_at() not supported for vector states")

        name = "{}[{},{}]{}".format(
            variable, ensemble_member, t - self.initial_time, "S" if scaled else ""
        )
        if extrapolate:
            name += "E"
        try:
            return self.__symbol_cache[name]
        except KeyError:
            # Look up transcribe_problem() state.
            t0 = self.initial_time
            X = self.solver_input

            # Fetch appropriate symbol, or value.
            canonical, sign = self.alias_relation.canonical_signed(variable)
            found = False

            # Check if it is in the state vector
            try:
                inds = self.__indices[ensemble_member][canonical]
            except KeyError:
                pass
            else:
                times = self.times(canonical)

                if self.integrate_states:
                    nominal = 1
                    if t == self.initial_time:
                        sym = sign * X[inds]
                        found = True
                    else:
                        variable_values = ca.horzcat(
                            sign * X[inds], self.__integrators[ensemble_member][canonical]
                        ).T
                else:
                    nominal = self.variable_nominal(canonical)
                    variable_values = X[inds]

                if not found:
                    f_left, f_right = np.nan, np.nan
                    if t < t0:
                        history = self.history(ensemble_member)
                        try:
                            history_timeseries = history[canonical]
                        except KeyError:
                            if extrapolate:
                                sym = variable_values[0]
                            else:
                                sym = np.nan
                        else:
                            if extrapolate:
                                f_left = history_timeseries.values[0]
                                f_right = history_timeseries.values[-1]
                            interpolation_method = self.interpolation_method(canonical)
                            sym = self.interpolate(
                                t,
                                history_timeseries.times,
                                history_timeseries.values,
                                f_left,
                                f_right,
                                interpolation_method,
                            )
                        if scaled and nominal != 1:
                            sym /= nominal
                    else:
                        if not extrapolate and (t < times[0] or t > times[-1]):
                            raise Exception(
                                f"Cannot interpolate for {canonical}: "
                                f"Point {t} outside of range [{times[0]}, {times[-1]}]"
                            )

                        interpolation_method = self.interpolation_method(canonical)
                        sym = interpolate(times, variable_values, [t], False, interpolation_method)
                        if not scaled and nominal != 1:
                            sym *= nominal
                    if sign < 0:
                        sym *= -1
                    found = True

            if not found:
                constant_inputs = self.constant_inputs(ensemble_member)
                try:
                    constant_input = constant_inputs[variable]
                    found = True
                except KeyError:
                    pass
                else:
                    times = self.times(variable)
                    f_left, f_right = np.nan, np.nan
                    if extrapolate:
                        f_left = constant_input.values[0]
                        f_right = constant_input.values[-1]
                    interpolation_method = self.interpolation_method(variable)
                    sym = self.interpolate(
                        t,
                        constant_input.times,
                        constant_input.values,
                        f_left,
                        f_right,
                        interpolation_method,
                    )
            if not found:
                parameters = self.parameters(ensemble_member)
                try:
                    sym = parameters[variable]
                    found = True
                except KeyError:
                    pass
            if not found:
                raise KeyError(variable)

            # Cache symbol.
            self.__symbol_cache[name] = sym

            return sym

    def variable(self, variable):
        return self.__variables[variable]

    def variable_nominal(self, variable):
        try:
            return self.__initial_derivative_nominals[variable]
        except KeyError:
            return super().variable_nominal(variable)

    def extra_variable(self, extra_variable, ensemble_member=0):
        indices = self.__indices[ensemble_member][extra_variable]
        return self.solver_input[indices] * self.variable_nominal(extra_variable)

    def __states_times_in(self, variable, t0=None, tf=None, ensemble_member=0):
        # Time stamps for this variable
        times = self.times(variable)

        # Set default values
        if t0 is None:
            t0 = times[0]
        if tf is None:
            tf = times[-1]

        # Find canonical variable
        canonical, sign = self.alias_relation.canonical_signed(variable)
        nominal = self.variable_nominal(canonical)
        state = self.state_vector(canonical, ensemble_member)
        if self.integrate_states and canonical in self.__integrators[ensemble_member]:
            state = ca.vertcat(state, ca.transpose(self.__integrators[ensemble_member][canonical]))
        state *= nominal
        if sign < 0:
            state *= -1

        # Compute combined points
        if t0 < times[0]:
            history = self.history(ensemble_member)
            try:
                history_timeseries = history[canonical]
            except KeyError:
                raise Exception(
                    f"No history found for variable {variable}, "
                    "but a historical value was requested"
                )
            else:
                history_times = history_timeseries.times[:-1]
                history = history_timeseries.values[:-1]
                if sign < 0:
                    history *= -1
        else:
            history_times = np.empty(0)
            history = np.empty(0)

        # Collect time stamps and states, "knots".
        (indices,) = np.where(np.logical_and(times >= t0, times <= tf))
        (history_indices,) = np.where(np.logical_and(history_times >= t0, history_times <= tf))
        if (t0 not in times[indices]) and (t0 not in history_times[history_indices]):
            x0 = self.state_at(variable, t0, ensemble_member)
        else:
            t0 = x0 = ca.MX()
        if (tf not in times[indices]) and (tf not in history_times[history_indices]):
            xf = self.state_at(variable, tf, ensemble_member)
        else:
            tf = xf = ca.MX()
        t = ca.vertcat(t0, history_times[history_indices], times[indices], tf)
        x = ca.vertcat(x0, history[history_indices], state[indices], xf)

        return x, t

    def states_in(
        self,
        variable: str,
        t0: float | None = None,
        tf: float | None = None,
        ensemble_member: int = 0,
        *,
        return_times: bool = False,
    ) -> ca.MX | tuple[ca.DM, ca.MX]:
        x, t = self.__states_times_in(variable, t0, tf, ensemble_member)

        if return_times:
            return x, t
        else:
            return x

    def integral(self, variable, t0=None, tf=None, ensemble_member=0):
        x, t = self.__states_times_in(variable, t0, tf, ensemble_member)

        if x.size1() > 1:
            # Integrate knots using trapezoid rule
            x_avg = 0.5 * (x[: x.size1() - 1] + x[1:])
            dt = t[1:] - t[: x.size1() - 1]
            return ca.sum1(x_avg * dt)
        else:
            return ca.MX(0)

    def der(self, variable):
        # Look up the derivative variable for the given non-derivative variable
        canonical, sign = self.alias_relation.canonical_signed(variable)
        try:
            i = self.__differentiated_states_map[canonical]
            return sign * self.dae_variables["derivatives"][i]
        except KeyError:
            try:
                i = self.__algebraic_states_map[canonical]
            except KeyError:
                i = len(self.algebraic_states) + self.__controls_map[canonical]
            return sign * self.__algebraic_and_control_derivatives[i]

    def der_at(self, variable, t, ensemble_member=0):
        # Special case t being t0 for differentiated states
        if t == self.initial_time:
            # We have a special symbol for t0 derivatives
            X = self.solver_input

            canonical, sign = self.alias_relation.canonical_signed(variable)
            try:
                i = self.__differentiated_states_map[canonical]
            except KeyError:
                # Fall through, in case 'variable' is not a differentiated state.
                pass
            else:
                initial_der_name = self.__initial_derivative_names[i]
                nominal = self.variable_nominal(initial_der_name)
                idx = self.__indices[ensemble_member][initial_der_name]

                return nominal * sign * X[idx]

        # Time stamps for this variable
        times = self.times(variable)

        if t <= self.initial_time:
            # Derivative requested for t0 or earlier.  We need the history.
            history = self.history(ensemble_member)
            try:
                htimes = history[variable].times[:-1]
                history_and_times = np.hstack((htimes, times))
            except KeyError:
                history_and_times = times
        else:
            history_and_times = times

        # Special case t being the initial available point.  In this case, we have
        # no derivative information available.
        if t == history_and_times[0]:
            return 0.0

        # Handle t being an interior point, or t0 for a non-differentiated
        # state
        for i in range(len(history_and_times)):
            # Use finite differences when between collocation points, and
            # backward finite differences when on one.
            if t > history_and_times[i] and t <= history_and_times[i + 1]:
                dx = self.state_at(
                    variable, history_and_times[i + 1], ensemble_member=ensemble_member
                ) - self.state_at(variable, history_and_times[i], ensemble_member=ensemble_member)
                dt = history_and_times[i + 1] - history_and_times[i]
                return dx / dt

        # t does not belong to any collocation point interval
        raise IndexError

    def map_path_expression(self, expr, ensemble_member):
        f = ca.Function("f", self.__func_orig_inputs, [expr]).expand()
        initial_values = f(*self.__func_initial_inputs[ensemble_member])

        # Map
        number_of_timeslots = len(self.times())
        if number_of_timeslots > 1:
            fmap = f.map(number_of_timeslots - 1)
            values = fmap(*self.__func_map_args[ensemble_member])

            all_values = ca.horzcat(initial_values, values)
        else:
            all_values = initial_values

        return ca.transpose(all_values)

    def solver_success(self, *args, **kwargs):
        self.__debug_check_state_output_scaling()

        return super().solver_success(*args, **kwargs)

    def _debug_get_named_nlp(self, nlp):
        x = nlp["x"]
        f = nlp["f"]
        g = nlp["g"]

        expand_f_g = ca.Function("f_g", [x], [f, g]).expand()
        x_sx = ca.SX.sym("X", *x.shape)
        f_sx, g_sx = expand_f_g(x_sx)

        x, f, g = x_sx, f_sx, g_sx

        # Build a vector of symbols with the descriptive names for useful
        # logging of constraints. Some decision variables may be shared
        # between ensemble members, so first we build a complete mapping of
        # state_index -> (canonical name, ensemble members, time step index)
        state_index_map = {}
        for ensemble_member in range(self.ensemble_size):
            indices = self.__indices_as_lists[ensemble_member]
            for k, v in indices.items():
                for t_i, i in enumerate(v):
                    if i in state_index_map:
                        # Shared state vector entry between ensemble members
                        assert k == state_index_map[i][0]
                        assert t_i == state_index_map[i][2]

                        state_index_map[i][1].append(ensemble_member)
                    else:
                        state_index_map[i] = [k, [ensemble_member], t_i]

        assert len(state_index_map) == x.size1()

        # Build descriptive decision variables for each state vector entry
        var_names = []
        for i in range(len(state_index_map)):
            var_name, ensemble_members, t_i = state_index_map[i]

            if len(ensemble_members) == 1:
                ensemble_members = ensemble_members[0]
            else:
                ensemble_members = "[{}]".format(",".join(str(x) for x in sorted(ensemble_members)))

            var_names.append(f"{var_name}__e{ensemble_members}__t{t_i}")

        # Create named versions of the constraints
        named_x = ca.vertcat(*(ca.SX.sym(v) for v in var_names))
        named_g = ca.vertsplit(ca.Function("tmp", [x], [g])(named_x))
        named_f = ca.vertsplit(ca.Function("tmp", [x], [f])(named_x))[0]

        return var_names, named_x, named_f, named_g

    @debug_check(DebugLevel.VERYHIGH)
    def __debug_check_transcribe_linear_coefficients(
        self,
        discrete,
        lbx,
        ubx,
        lbg,
        ubg,
        x0,
        nlp,
        tol_rhs=1e6,
        tol_zero=1e-12,
        tol_up=1e2,
        tol_down=1e-2,
        tol_range=1e3,
        evaluate_at_x0=False,
    ):
        nlp = nlp.copy()

        expand_f_g = ca.Function("f_g", [nlp["x"]], [nlp["f"], nlp["g"]]).expand()
        X_sx = ca.SX.sym("X", *nlp["x"].shape)
        f_sx, g_sx = expand_f_g(X_sx)

        nlp["x"] = X_sx
        nlp["f"] = f_sx
        nlp["g"] = g_sx

        lbg = np.array(ca.veccat(*lbg)).ravel()
        ubg = np.array(ca.veccat(*ubg)).ravel()

        var_names, named_x, named_f, named_g = self._debug_get_named_nlp(nlp)

        def constr_to_str(i):
            c_str = str(named_g[i])

            lb, ub = lbg[i], ubg[i]

            if np.isfinite(lb) and np.isfinite(ub) and lb == ub:
                c_str = f"{c_str} = {lb}"
            elif np.isfinite(lb) and np.isfinite(ub):
                c_str = f"{lb} <= {c_str} <= {ub}"
            elif np.isfinite(lb):
                c_str = f"{c_str} >= {lb}"
            elif np.isfinite(ub):
                c_str = f"{c_str} <= {ub}"

            return c_str

        # Checking for right hand side of constraints
        logger.info(f"Sanity check of lbg and ubg, checking for small values (<{tol_zero})")

        lbg_abs_no_zero = np.abs(lbg.copy())
        lbg_abs_no_zero[lbg_abs_no_zero == 0.0] = +np.inf
        ind = np.argmin(lbg_abs_no_zero)
        if np.any(np.isfinite(lbg_abs_no_zero)):
            logger.info(f"Smallest (absolute) lbg coefficient {lbg_abs_no_zero[ind]}")
            logger.info(f"E.g., {constr_to_str(ind)}")
        lbg_inds = lbg_abs_no_zero < tol_zero
        if np.any(lbg_inds):
            logger.info(f"Too small of a (absolute) lbg found: {min(lbg[lbg_inds])}")

        ubg_abs_no_zero = np.abs(ubg.copy())
        ubg_abs_no_zero[ubg_abs_no_zero == 0.0] = +np.inf
        ind = np.argmin(ubg_abs_no_zero)
        if np.any(np.isfinite(ubg_abs_no_zero)):
            logger.info(f"Smallest (absolute) ubg coefficient {ubg_abs_no_zero[ind]}")
            logger.info(f"E.g., {constr_to_str(ind)}")
        ubg_inds = ubg_abs_no_zero < tol_zero
        if np.any(ubg_inds):
            logger.info(f"Too small of a (absolute) ubg found: {min(ubg[ubg_inds])}")

        logger.info(f"Sanity check of lbg and ubg, checking for large values (>{tol_rhs})")

        lbg_abs_no_inf = np.abs(lbg.copy())
        lbg_abs_no_inf[~np.isfinite(lbg_abs_no_inf)] = -np.inf
        ind = np.argmax(lbg_abs_no_inf)
        if np.any(np.isfinite(lbg_abs_no_inf)):
            logger.info(f"Largest (absolute) lbg coefficient {lbg_abs_no_inf[ind]}")
            logger.info(f"E.g., {constr_to_str(ind)}")

        lbg_inds = lbg_abs_no_inf > tol_rhs
        if np.any(lbg_inds):
            raise Exception(f"Too large of a (absolute) lbg found: {max(lbg[lbg_inds])}")

        ubg_abs_no_inf = np.abs(ubg.copy())
        ubg_abs_no_inf[~np.isfinite(ubg)] = -np.inf
        ind = np.argmax(ubg_abs_no_inf)
        if np.any(np.isfinite(ubg_abs_no_inf)):
            logger.info(f"Largest (absolute) ubg coefficient {ubg_abs_no_inf[ind]}")
            logger.info(f"E.g., {constr_to_str(ind)}")

        ubg_inds = ubg_abs_no_inf > tol_rhs
        if np.any(ubg_inds):
            raise Exception(f"Too large of a (absolute) ubg found: {max(ubg[ubg_inds])}")

        eval_point = x0 if evaluate_at_x0 else 1.0
        eval_point_str = "x0" if evaluate_at_x0 else "1.0"

        # Check coefficient matrix
        logger.info(
            "Sanity check on objective and constraints Jacobian matrix/constant coefficients values"
        )

        in_var = nlp["x"]
        out = []
        for o in [nlp["f"], nlp["g"]]:
            Af = ca.Function("Af", [in_var], [ca.jacobian(o, in_var)])
            bf = ca.Function("bf", [in_var], [o])

            A = Af(eval_point)
            A = ca.sparsify(A)

            b = bf(0)
            b = ca.sparsify(b)

            out.append((A.tocsc().tocoo(), b.tocsc().tocoo()))

        # Objective
        A_obj, b_obj = out[0]
        logger.info(
            f"Statistics of objective: max & min of abs(jac(f, {eval_point_str}))) "
            f"f({eval_point_str}), constants"
        )
        max_obj_A = max(np.abs(A_obj.data), default=None)
        min_obj_A = min(np.abs(A_obj.data[A_obj.data != 0.0]), default=None)
        obj_x0 = np.array(ca.Function("tmp", [in_var], [nlp["f"]])(eval_point)).ravel()[0]
        obj_b = b_obj.data[0] if len(b_obj.data) > 0 else 0.0

        logger.info(f"{max_obj_A} & {min_obj_A}, {obj_x0}, {obj_b}")

        if abs(obj_b) > tol_up:
            logger.info(f"Constant '{obj_b}' in objective exceeds upper tolerance of '{tol_up}'")
        if abs(obj_b) > tol_up:
            logger.info(f"Objective value at x0 '{obj_x0}' exceeds upper tolerance of '{tol_up}'")

        # Constraints
        A_constr, b_constr = out[1]
        logger.info(
            "Statistics of constraints: max & min of abs(jac(g, x0))), max & min of abs(g(x0))"
        )
        max_constr_A = max(np.abs(A_constr.data), default=None)
        min_constr_A = min(np.abs(A_constr.data[A_constr.data != 0.0]), default=None)
        max_constr_b = max(np.abs(b_constr.data), default=None)
        min_constr_b = min(np.abs(b_constr.data[b_constr.data != 0.0]), default=None)
        logger.info(f"{max_constr_A} & {min_constr_A}, {max_constr_b} & {min_constr_b}")

        # Filter out exactly zero, as those entries do not show up in the
        # matrix. Shut up SonarCloud warning about this exact-to-zero
        # comparison.
        maxs = [
            x
            for x in [max_constr_A, max_constr_b, max_obj_A, obj_b]
            if x is not None and x != 0.0  # NOSONAR
        ]
        mins = [
            x
            for x in [min_constr_A, min_constr_b, min_obj_A, obj_b]
            if x is not None and x != 0.0  # NOSONAR
        ]
        if (maxs and max(maxs) > tol_up) or (mins and min(mins) < tol_down):
            logger.info("Jacobian matrix /constants coefficients values outside typical range!")

        # Check on individual constraints. (Only check values of constraint's Jacobian.)
        A_constr_csr = A_constr.tocsr()

        exceedences = []

        for i in range(A_constr_csr.shape[0]):
            r = A_constr_csr.getrow(i)
            data = r.data

            try:
                max_r = max(np.abs(data))
                min_r = min(np.abs(data))
            except ValueError:
                # Emtpy constraint?
                continue

            assert min_r != 0.0

            if max_r > tol_up or min_r < tol_down or max_r / min_r > tol_range:
                c_str = constr_to_str(i)
                exceedences.append((i, max_r, min_r, c_str))

        if exceedences:
            logger.info(
                "Exceedence in jacobian of constraints evaluated at x0"
                f" (max > {tol_up:g}, min < {tol_down:g}, or max / min > {tol_range:g}):"
            )

            exceedences = sorted(exceedences, key=lambda x: x[1] / x[2], reverse=True)

            for i, (r, max_r, min_r, c) in enumerate(exceedences):
                logger.info(f"row {r} (max: {max_r}, min: {min_r}, range: {max_r / min_r}):  {c}")

                if i >= 9:
                    logger.info(
                        f"Too many warnings of same type ({len(exceedences) - 10} others remain)."
                    )
                    break

        # Columns
        A_constr_csc = A_constr.tocsc()

        coeffs = []

        max_range_found = 1.0

        logger.info(
            "Checking for range exceedence for each variable (i.e., check Jacobian matrix columns)"
        )
        exceedences = []

        for c in range(A_constr_csc.shape[1]):
            cur_col = A_constr_csc.getcol(c)
            cur_coeffs = cur_col.data

            if len(cur_coeffs) == 0:
                coeffs.append(None)
                continue

            abs_coeffs = np.abs(cur_coeffs)

            max_r_i = np.argmax(abs_coeffs)
            min_r_i = np.argmin(abs_coeffs)

            max_r = abs_coeffs[max_r_i]
            min_r = abs_coeffs[min_r_i]

            assert min_r != 0.0

            max_range_found = max(max_r / min_r, max_range_found)

            if max_r / min_r > tol_range:
                inds = cur_col.indices

                c_min = inds[min_r_i]
                c_max = inds[max_r_i]

                r = A_constr_csr.getrow(c_min)
                c_min_str = constr_to_str(c_min)
                r = A_constr_csr.getrow(c_max)
                c_max_str = constr_to_str(c_max)

                exceedences.append((c, max_r / min_r, min_r, max_r, c_min_str, c_max_str))

            coeffs.append((min_r, max_r))

        exceedences = sorted(exceedences, key=lambda x: x[1], reverse=True)

        logger.info(f"Max range found: {max_range_found}")
        if exceedences:
            logger.info(f"Exceedence in range per column (max / min > {tol_range:g}):")

            for i, (c, exc, min_, max_, c_min_str, c_max_str) in enumerate(exceedences):
                logger.info(f"col {c} ({var_names[c]}):  range {exc}, min {min_}, max {max_}")
                logger.info(c_min_str)
                logger.info(c_max_str)

                if i >= 9:
                    logger.info(
                        f"Too many warnings of same type ({len(exceedences) - 10} others remain)."
                    )
                    break

        logger.info("Checking for range exceedence for variables in the objective function")
        max_range_found = 1.0

        exceedences = []
        for c, d in zip(A_obj.col, A_obj.data, strict=True):
            cofc = coeffs[c]

            if cofc is None:
                # Variable does not appear in constraints
                continue

            min_r, max_r = cofc

            obj_coeff = abs(d)

            max_range = max(obj_coeff / min_r, max_r / obj_coeff)

            max_range_found = max(max_range, max_range_found)

            if max_range > tol_range:
                exceedences.append((c, max_range, obj_coeff, min_r, max_r))

        logger.info(f"Max range found: {max_range_found}")
        if exceedences:
            logger.info(f"Exceedence in range of objective variable (range > {tol_range:g}):")

            for i, (c, max_range, obj_coeff, min_r, max_r) in enumerate(exceedences):
                logger.info(
                    f"col {c} ({var_names[c]}): range: {max_range}, obj: {obj_coeff}, "
                    f"min constr: {min_r}, max constr {max_r}"
                )

                if i >= 9:
                    logger.info(
                        f"Too many warnings of same type ({len(exceedences) - 10} others remain)."
                    )
                    break

    @debug_check(DebugLevel.VERYHIGH)
    def __debug_check_state_output_scaling(self, tol_up=1e4, tol_down=1e-2, ignore_all_zero=True):
        """
        Check the scaling using the resulting/optimized solver output.

        Exceedences of the (absolute) state vector of `tol_up` are rather
        unambiguously bad. If a certain variable has _any_ violation, we
        report it.

        Exceedences on `tol_down` are more difficult as maybe the scaling is
        correct, but the answer just happened to be (almost) zero. We only
        report if _all_ values are in violation (and even then can't really be
        certain).
        """

        abs_output = np.abs(self.solver_output)

        inds_up = np.flatnonzero(abs_output >= tol_up)
        inds_down = np.flatnonzero(abs_output <= tol_down)

        indices = self.__indices_as_lists

        variable_to_all_indices = {k: set(v) for k, v in indices[0].items()}
        for ensemble_indices in indices[1:]:
            for k, v in ensemble_indices.items():
                variable_to_all_indices[k] |= set(v)

        if len(inds_up) > 0:
            exceedences = []

            for k, v in variable_to_all_indices.items():
                inds = v.intersection(inds_up)
                if inds:
                    exceedences.append((k, max(abs_output[list(inds)])))

            exceedences = sorted(exceedences, key=lambda x: x[1], reverse=True)

            if exceedences:
                logger.info(
                    "Variables with at least one (absolute) state vector entry/entries "
                    f"larger than {tol_up}"
                )

            for k, v in exceedences:
                logger.info(f"{k}: abs max = {v}")

        if len(inds_down) > 0:
            exceedences = []

            for k, v in variable_to_all_indices.items():
                if v.issubset(inds_down):
                    exceedences.append((k, max(abs_output[list(v)])))

            exceedences = sorted(exceedences, key=lambda x: x[1], reverse=True)

            if next((v for k, v in exceedences if not ignore_all_zero or v > 0.0), None):
                ignore_all_zero_string = " (but not all zero)" if ignore_all_zero else ""
                logger.info(
                    "Variables with all (absolute) state vector entry/entries "
                    f"smaller than {tol_down}{ignore_all_zero_string}"
                )

            for k, v in exceedences:
                if ignore_all_zero and v == 0.0:
                    continue

                logger.info(f"{k}: abs max = {v}")
