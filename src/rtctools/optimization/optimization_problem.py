import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Literal, overload

import casadi as ca
import numpy as np

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.debug_check_helpers import DebugLevel, debug_check
from rtctools._internal.ensemble_bounds_decorator import ensemble_bounds_check
from rtctools.data.storage import DataStoreAccessor

from rtctools._internal.casadi_to_LP import (
    build_objective,
    build_constraints,
    build_bounds,
    sanitize_var_names,
    write_lp_file
)

from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


# Typical type for a bound on a variable
BT = float | np.ndarray | Timeseries


class LookupTable:
    """
    Base class for LookupTables.
    """

    @property
    def inputs(self) -> list[ca.MX]:
        """
        List of lookup table input variables.
        """
        raise NotImplementedError

    @property
    def function(self) -> ca.Function:
        """
        Lookup table CasADi :class:`Function`.
        """
        raise NotImplementedError


class OptimizationProblem(DataStoreAccessor, metaclass=ABCMeta):
    """
    Base class for all optimization problems.
    """

    _debug_check_level = DebugLevel.MEDIUM
    _debug_check_options = {}

    #: Enable ensemble-specific bounds functionality
    ensemble_specific_bounds = False

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        self.__mixed_integer = False

    def optimize(
        self,
        preprocessing: bool = True,
        postprocessing: bool = True,
        log_solver_failure_as_error: bool = True,
    ) -> bool:
        """
        Perform one initialize-transcribe-solve-finalize cycle.

        :param preprocessing:  True to enable a call to ``pre`` preceding the optimization.
        :param postprocessing: True to enable a call to ``post`` following the optimization.

        :returns: True on success.
        """

        # Deprecations / removals
        if hasattr(self, "initial_state"):
            raise RuntimeError(
                "Support for `initial_state()` has been removed. Please use `history()` instead."
            )

        logger.info("Entering optimize()")

        # Do any preprocessing, which may include changing parameter values on
        # the model
        if preprocessing:
            self.pre()

            # Check if control inputs are bounded
            self.__check_bounds_control_input()
        else:
            logger.debug("Skipping Preprocessing in OptimizationProblem.optimize()")

        # Transcribe problem
        discrete, lbx, ubx, lbg, ubg, x0, nlp = self.transcribe()

        # Create an NLP solver
        logger.debug("Collecting solver options")

        self.__mixed_integer = np.any(discrete)
        options = {}
        options.update(self.solver_options())  # Create a copy

        logger.debug("Creating solver")

        expand = options.pop("expand", False)
        export_model = options.pop("export_model", False)
        if expand or export_model:
            # NOTE: CasADi only supports the "expand" option for nlpsol. To
            # also be able to expand with e.g. qpsol, we do the expansion
            # ourselves here.
            logger.debug("Expanding objective and constraints to SX")

            expand_f_g = ca.Function("f_g", [nlp["x"]], [nlp["f"], nlp["g"]]).expand()
            X_sx = ca.SX.sym("X", *nlp["x"].shape)
            f_sx, g_sx = expand_f_g(X_sx)

            nlp["f"] = f_sx
            nlp["g"] = g_sx
            nlp["x"] = X_sx

            if export_model:
                # Export LF if collocation constraints are linear
                if self.linear_collocation:
                    # Get variable names - note the [0] to get the dictionary from the list
                    indices_dict = self._CollocatedIntegratedOptimizationProblem__indices[0]
                    var_names = sanitize_var_names(
                        indices_dict,  # Pass the dictionary, not the list
                        expand_f_g.nnz_in()
                    )
                    
                    # Build LP components
                    obj_str = build_objective(expand_f_g, var_names)
                    constr_str = build_constraints(expand_f_g, lbg, ubg, var_names)
                    bounds_str = build_bounds(var_names, lbx, ubx)
                    
                    # Write LP file
                    import time
                    filename = "myproblem_{}.lp".format(int(time.time()))
                    write_lp_file(
                        filename,
                        obj_str,
                        constr_str,
                        bounds_str,
                        var_names,
                        discrete
                    )
                else:
                    logger.warning("Skipping LP export: collocation constraints are not linear")
                    return
                

        # Debug check for non-linearity in constraints
        self.__debug_check_linearity_constraints(nlp)

        # Debug check for linear independence of the constraints
        self.__debug_check_linear_independence(lbx, ubx, lbg, ubg, nlp)

        # Solver option
        my_solver = options["solver"]
        del options["solver"]

        # Already consumed
        del options["optimized_num_dir"]

        # Iteration callback
        iteration_callback = options.pop("iteration_callback", None)

        # CasADi solver to use
        casadi_solver = options.pop("casadi_solver")
        if isinstance(casadi_solver, str):
            casadi_solver = getattr(ca, casadi_solver)

        nlpsol_options = {**options}

        if self.__mixed_integer:
            nlpsol_options["discrete"] = discrete
        if iteration_callback:
            nlpsol_options["iteration_callback"] = iteration_callback

        # Remove ipopt and bonmin defaults if they are not used
        if my_solver != "ipopt":
            nlpsol_options.pop("ipopt", None)
        if my_solver != "bonmin":
            nlpsol_options.pop("bonmin", None)

        solver = casadi_solver("nlp", my_solver, nlp, nlpsol_options)

        # Solve NLP
        logger.info("Calling solver")

        results = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=ca.veccat(*lbg), ubg=ca.veccat(*ubg))

        # Extract relevant stats
        self.__objective_value = float(results["f"])
        self.__solver_output = np.array(results["x"]).ravel()
        self.__transcribed_problem = {
            "lbx": lbx,
            "ubx": ubx,
            "lbg": lbg,
            "ubg": ubg,
            "x0": x0,
            "nlp": nlp,
        }
        self.__lam_g = results.get("lam_g")
        self.__lam_x = results.get("lam_x")

        self.__solver_stats = solver.stats()

        success, log_level = self.solver_success(self.__solver_stats, log_solver_failure_as_error)

        return_status = self.__solver_stats["return_status"]
        if "secondary_return_status" in self.__solver_stats:
            return_status = "{}: {}".format(
                return_status, self.__solver_stats["secondary_return_status"]
            )
        wall_clock_time = "elapsed time not read"
        if "t_wall_total" in self.__solver_stats:
            wall_clock_time = "{} seconds".format(self.__solver_stats["t_wall_total"])
        elif "t_wall_solver" in self.__solver_stats:
            wall_clock_time = "{} seconds".format(self.__solver_stats["t_wall_solver"])

        if success:
            logger.log(
                log_level,
                f"Solver succeeded with status {return_status} ({wall_clock_time}).",
            )
        else:
            try:
                ii = [y[0] for y in self.loop_over_error].index(self.priority)
                loop_error_indicator = self.loop_over_error[ii][1]
                try:
                    loop_error = self.loop_over_error[ii][2]
                    if loop_error_indicator and loop_error in return_status:
                        log_level = logging.INFO
                except IndexError:
                    if loop_error_indicator:
                        log_level = logging.INFO
                logger.log(
                    log_level,
                    f"Solver failed with status {return_status} ({wall_clock_time}).",
                )
            except (AttributeError, ValueError):
                logger.log(
                    log_level,
                    f"Solver failed with status {return_status} ({wall_clock_time}).",
                )

        # Do any postprocessing
        if postprocessing:
            self.post()
        else:
            logger.debug("Skipping Postprocessing in OptimizationProblem.optimize()")

        # Done
        logger.info("Done with optimize()")

        return success

    def __check_bounds_control_input(self) -> None:
        # Checks if at the control inputs have bounds, log warning when a control input is not
        # bounded.
        if self.ensemble_specific_bounds:
            bounds_list = [
                self.bounds(ensemble_member) for ensemble_member in range(self.ensemble_size)
            ]
        else:
            bounds_list = [self.bounds()]

        for ensemble_member, bounds in enumerate(bounds_list):
            for variable in self.dae_variables["control_inputs"]:
                variable = variable.name()
                if variable not in bounds:
                    logger.warning(
                        "OptimizationProblem: control input {} has no bounds{}.".format(
                            variable,
                            f" ({ensemble_member=})" if self.ensemble_specific_bounds else "",
                        )
                    )

    @abstractmethod
    def transcribe(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, ca.MX]
    ]:
        """
        Transcribe the continuous optimization problem to a discretized, solver-ready
        optimization problem.
        """
        pass

    def solver_options(self) -> dict[str, str | int | float | bool | str]:
        """
        Returns a dictionary of CasADi optimization problem solver options.

        The default solver for continuous problems is `Ipopt
        <https://projects.coin-or.org/Ipopt/>`_.

        The default solver for mixed integer problems is `Bonmin
        <http://projects.coin-or.org/Bonmin/>`_.

        :returns: A dictionary of solver options. See the CasADi and
                  respective solver documentation for details.
        """
        options = {"error_on_fail": False, "optimized_num_dir": 3, "casadi_solver": ca.nlpsol, 
                   "export_model": False}

        if self.__mixed_integer:
            options["solver"] = "bonmin"

            bonmin_options = options["bonmin"] = {}
            bonmin_options["algorithm"] = "B-BB"
            bonmin_options["nlp_solver"] = "Ipopt"
            bonmin_options["nlp_log_level"] = 2
            bonmin_options["linear_solver"] = "mumps"
        else:
            options["solver"] = "ipopt"

            ipopt_options = options["ipopt"] = {}
            ipopt_options["linear_solver"] = "mumps"
        return options

    def solver_success(
        self, solver_stats: dict[str, str | bool], log_solver_failure_as_error: bool
    ) -> tuple[bool, int]:
        """
        Translates the returned solver statistics into a boolean and log level
        to indicate whether the solve was succesful, and how to log it.

        :param solver_stats: Dictionary containing information about the
                             solver status. See explanation below.
        :param log_solver_failure_as_error: Indicates whether a solve failure
            Should be logged as an error or info message.

        ``solver_stats`` typically consist of three fields:

        * return_status: ``str``
        * secondary_return_status: ``str``
        * success: ``bool``

        By default we rely on CasADi's interpretation of the return_status
        (and secondary status) to the success variable, with an exception for
        IPOPT (see below).

        The logging level is typically ``logging.INFO`` for success, and
        ``logging.ERROR`` for failure. Only for IPOPT an exception is made for
        `Not_Enough_Degrees_Of_Freedom`, which returns ``logging.WARNING`` instead.
        For example, this can happen when too many goals are specified, and
        lower priority goals cannot improve further on the current result.

        :returns: A tuple indicating whether or not the solver has succeeded, and what level to log
            it with.
        """
        success = solver_stats["success"]
        log_level = logging.INFO if success else logging.ERROR

        if self.solver_options()["solver"].lower() in ["bonmin", "ipopt"] and solver_stats[
            "return_status"
        ] in ["Not_Enough_Degrees_Of_Freedom"]:
            log_level = logging.WARNING

        if log_level == logging.ERROR and not log_solver_failure_as_error:
            log_level = logging.INFO

        if self.solver_options()["solver"].lower() == "knitro":
            list_feas_flags = [
                "KN_RC_OPTIMAL_OR_SATISFACTORY",
                "KN_RC_ITER_LIMIT_FEAS",
                "KN_RC_NEAR_OPT",
                "KN_RC_FEAS_XTOL",
                "KN_RC_FEAS_NO_IMPROVE",
                "KN_RC_FEAS_FTOL",
                "KN_RC_TIME_LIMIT_FEAS",
                "KN_RC_FEVAL_LIMIT_FEAS",
                "KN_RC_MIP_EXH_FEAS",
                "KN_RC_MIP_TERM_FEAS",
                "KN_RC_MIP_SOLVE_LIMIT_FEAS",
                "KN_RC_MIP_NODE_LIMIT_FEAS",
            ]
            if solver_stats["return_status"] in list_feas_flags:
                success = True

        return success, log_level

    @abstractproperty
    def solver_input(self) -> ca.MX:
        """
        The symbolic input to the NLP solver.
        """
        pass

    @abstractmethod
    def extract_results(self, ensemble_member: int = 0) -> dict[str, np.ndarray]:
        """
        Extracts state and control input time series from optimizer results.

        :returns: A dictionary of result time series.
        """
        pass

    @property
    def objective_value(self) -> float:
        """
        The last obtained objective function value.
        """
        return self.__objective_value

    @property
    def solver_output(self) -> np.ndarray:
        """
        The raw output from the last NLP solver run.
        """
        return self.__solver_output

    @property
    def solver_stats(self) -> dict[str, Any]:
        """
        The stats from the last NLP solver run.
        """
        return self.__solver_stats

    @property
    def lagrange_multipliers(self) -> tuple[Any, Any]:
        """
        The lagrange multipliers at the solution.
        """
        return self.__lam_g, self.__lam_x

    @property
    def transcribed_problem(self) -> dict[str, Any]:
        """
        The transcribed problem.
        """
        return self.__transcribed_problem

    def pre(self) -> None:
        """
        Preprocessing logic is performed here.
        """
        pass

    @abstractproperty
    def dae_residual(self) -> ca.MX:
        """
        Symbolic DAE residual of the model.
        """
        pass

    @abstractproperty
    def dae_variables(self) -> dict[str, list[ca.MX]]:
        """
        Dictionary of symbolic variables for the DAE residual.
        """
        pass

    @property
    def path_variables(self) -> list[ca.MX]:
        """
        List of additional, time-dependent optimization variables, not covered by the DAE model.
        """
        return []

    @abstractmethod
    def variable(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the given variable.

        :param variable: Variable name.

        :returns: The associated CasADi :class:`MX` symbol.
        """
        raise NotImplementedError

    @property
    def extra_variables(self) -> list[ca.MX]:
        """
        List of additional, time-independent optimization variables, not covered by the DAE model.
        """
        return []

    @property
    def output_variables(self) -> list[ca.MX]:
        """
        List of variables that the user requests to be included in the output files.
        """
        return []

    def delayed_feedback(self) -> list[tuple[str, str, float]]:
        """
        Returns the delayed feedback mappings.  These are given as a list of triples
        :math:`(x, y, \\tau)`, to indicate that :math:`y = x(t - \\tau)`.

        :returns: A list of triples.

        Example::

            def delayed_feedback(self):
                fb1 = ['x', 'y', 0.1]
                fb2 = ['x', 'z', 0.2]
                return [fb1, fb2]

        """
        return []

    @property
    def ensemble_size(self) -> int:
        """
        The number of ensemble members.
        """
        return 1

    def ensemble_member_probability(self, ensemble_member: int) -> float:
        """
        The probability of an ensemble member occurring.

        :param ensemble_member: The ensemble member index.

        :returns: The probability of an ensemble member occurring.

        :raises: IndexError
        """
        return 1.0

    def parameters(self, ensemble_member: int) -> AliasDict[str, bool | int | float | ca.MX]:
        """
        Returns a dictionary of parameters.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of parameter names and values.
        """
        return AliasDict(self.alias_relation)

    def string_parameters(self, ensemble_member: int) -> dict[str, str]:
        """
        Returns a dictionary of string parameters.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of string parameter names and values.
        """
        return {}

    def constant_inputs(self, ensemble_member: int) -> AliasDict[str, Timeseries]:
        """
        Returns a dictionary of constant inputs.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of constant input names and time series.
        """
        return AliasDict(self.alias_relation)

    def lookup_tables(self, ensemble_member: int) -> AliasDict[str, LookupTable]:
        """
        Returns a dictionary of lookup tables.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and lookup tables.
        """
        return AliasDict(self.alias_relation)

    @staticmethod
    def merge_bounds(a: tuple[BT, BT], b: tuple[BT, BT]) -> tuple[BT, BT]:
        """
        Returns a pair of bounds which is the intersection of the two pairs of
        bounds given as input.

        :param a: First pair ``(upper, lower)`` bounds
        :param b: Second pair ``(upper, lower)`` bounds

        :returns: A pair of ``(upper, lower)`` bounds which is the
                  intersection of the two input bounds.
        """
        a, A = a
        b, B = b

        # Make sure we are dealing with the correct types
        if __debug__:
            for v in (a, A, b, B):
                if isinstance(v, np.ndarray):
                    assert v.ndim == 1
                    assert np.issubdtype(v.dtype, np.number)
                else:
                    assert isinstance(v, (float, int, Timeseries))

        all_bounds = [a, A, b, B]

        # First make sure that we treat single element vectors as scalars
        for i, v in enumerate(all_bounds):
            if isinstance(v, np.ndarray) and np.prod(v.shape) == 1:
                all_bounds[i] = v.item()

        # Upcast lower bounds to be of equal type, and upper bounds as well.
        for i, j in [(0, 2), (2, 0), (1, 3), (3, 1)]:
            v1 = all_bounds[i]
            v2 = all_bounds[j]

            # We only check for v1 being of a "smaller" type than v2, as we
            # know we will encounter the reverse as well.
            if isinstance(v1, type(v2)):
                # Same type, nothing to do.
                continue
            elif isinstance(v1, (int, float)) and isinstance(v2, Timeseries):
                all_bounds[i] = Timeseries(v2.times, np.full_like(v2.values, v1))
            elif isinstance(v1, np.ndarray) and isinstance(v2, Timeseries):
                if v2.values.ndim != 2 or len(v1) != v2.values.shape[1]:
                    raise Exception(
                        f"Mismatching vector size when upcasting to Timeseries, {v1} vs. {v2}."
                    )
                all_bounds[i] = Timeseries(v2.times, np.broadcast_to(v1, v2.values.shape))
            elif isinstance(v1, (int, float)) and isinstance(v2, np.ndarray):
                all_bounds[i] = np.full_like(v2, v1)

        a, A, b, B = all_bounds

        assert isinstance(a, type(b))
        assert isinstance(A, type(B))

        # Merge the bounds
        m, M = None, None

        if isinstance(a, np.ndarray):
            if not a.shape == b.shape:
                raise Exception("Cannot merge vector minimum bounds of non-equal size")
            m = np.maximum(a, b)
        elif isinstance(a, Timeseries):
            if len(a.times) != len(b.times):
                raise Exception("Cannot merge Timeseries minimum bounds with different lengths")
            elif not np.all(a.times == b.times):
                raise Exception("Cannot merge Timeseries minimum bounds with non-equal times")
            elif not a.values.shape == b.values.shape:
                raise Exception("Cannot merge vector Timeseries minimum bounds of non-equal size")
            m = Timeseries(a.times, np.maximum(a.values, b.values))
        else:
            m = max(a, b)

        if isinstance(A, np.ndarray):
            if not A.shape == B.shape:
                raise Exception("Cannot merge vector maximum bounds of non-equal size")
            M = np.minimum(A, B)
        elif isinstance(A, Timeseries):
            if len(A.times) != len(B.times):
                raise Exception("Cannot merge Timeseries maximum bounds with different lengths")
            elif not np.all(A.times == B.times):
                raise Exception("Cannot merge Timeseries maximum bounds with non-equal times")
            elif not A.values.shape == B.values.shape:
                raise Exception("Cannot merge vector Timeseries maximum bounds of non-equal size")
            M = Timeseries(A.times, np.minimum(A.values, B.values))
        else:
            M = min(A, B)

        return m, M

    @ensemble_bounds_check
    def bounds(self, ensemble_member: int | None = None) -> AliasDict[str, tuple[BT, BT]]:
        """
        Returns variable bounds as a dictionary mapping variable names to a pair of bounds.
        A bound may be a constant, or a time series.

        :param ensemble_member: The ensemble member index. Must be None (or not provided) when
                               ensemble_specific_bounds is False. Must be an integer when
                               ensemble_specific_bounds is True.

        :returns: A dictionary of variable names and ``(upper, lower)`` bound pairs.
                  The bounds may be numbers or :class:`.Timeseries` objects.

        Example::
            # ensemble_specific_bounds is False
            def bounds(self):
                return {'x': (1.0, 2.0), 'y': (2.0, 3.0)}

            # ensemble_specific_bounds is True
            def bounds(self, ensemble_member: int):
                if ensemble_member == 0:
                    return {'x': (1.0, 2.0), 'y': (2.0, 3.0)}
                else:
                    return {'x': (0.0, 3.0), 'y': (1.0, 4.0)}
        """
        return AliasDict(self.alias_relation)

    def history(self, ensemble_member: int) -> AliasDict[str, Timeseries]:
        """
        Returns the state history.

        :param ensemble_member: The ensemble member index.

        :returns:
            A dictionary of variable names and historical time series (up to and including t0).
        """
        return AliasDict(self.alias_relation)

    def variable_is_discrete(self, variable: str) -> bool:
        """
        Returns ``True`` if the provided variable is discrete.

        :param variable: Variable name.

        :returns: ``True`` if variable is discrete (integer).
        """
        return False

    def variable_nominal(self, variable: str) -> float | np.ndarray:
        """
        Returns the nominal value of the variable.  Variables are scaled by replacing them with
        their nominal value multiplied by the new variable.

        :param variable: Variable name.

        :returns: The nominal value of the variable.
        """
        return 1

    @property
    def initial_time(self) -> float:
        """
        The initial time in seconds.
        """
        return self.times()[0]

    @property
    def initial_residual(self) -> ca.MX:
        """
        The initial equation residual.

        Initial equations are used to find consistent initial conditions.

        :returns: An :class:`MX` object representing F in the initial equation F = 0.
        """
        return ca.MX(0)

    def seed(self, ensemble_member: int) -> AliasDict[str, float | Timeseries]:
        """
        Seeding data.  The optimization algorithm is seeded with the data returned by this method.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and seed time series.
        """
        return AliasDict(self.alias_relation)

    def objective(self, ensemble_member: int) -> ca.MX:
        """
        The objective function for the given ensemble member.

        Call :func:`OptimizationProblem.state_at` to return a symbol representing a model variable
        at a given time.

        :param ensemble_member: The ensemble member index.

        :returns: An :class:`MX` object representing the objective function.

        Example::

            def objective(self, ensemble_member):
                # Return value of state 'x' at final time:
                times = self.times()
                return self.state_at('x', times[-1], ensemble_member)

        """
        return ca.MX(0)

    def path_objective(self, ensemble_member: int) -> ca.MX:
        """
        Returns a path objective the given ensemble member.

        Path objectives apply to all times and ensemble members simultaneously.
        Call :func:`OptimizationProblem.state` to return a time- and ensemble-member-independent
        symbol representing a model variable.

        :param ensemble_member: The ensemble member index. This index is currently unused,
                                and here for future use only.

        :returns: A :class:`MX` object representing the path objective.

        Example::

            def path_objective(self, ensemble_member):
                # Minimize x(t) for all t
                return self.state('x')

        """
        return ca.MX(0)

    def constraints(
        self, ensemble_member: int
    ) -> list[tuple[ca.MX, float | np.ndarray, float | np.ndarray]]:
        """
        Returns a list of constraints for the given ensemble member.

        Call :func:`OptimizationProblem.state_at` to return a symbol representing a model variable
        at a given time.

        :param ensemble_member: The ensemble member index.

        :returns: A list of triples ``(f, m, M)``, with an :class:`MX` object representing
                  the constraint function ``f``, lower bound ``m``, and upper bound ``M``.
                  The bounds must be numbers.

        Example::

            def constraints(self, ensemble_member):
                t = 1.0
                constraint1 = (
                    2 * self.state_at('x', t, ensemble_member),
                    2.0, 4.0)
                constraint2 = (
                    self.state_at('x', t, ensemble_member) + self.state_at('y', t, ensemble_member),
                    2.0, 3.0)
                return [constraint1, constraint2]

        """
        return []

    def path_constraints(
        self, ensemble_member: int
    ) -> list[tuple[ca.MX, float | np.ndarray, float | np.ndarray]]:
        """
        Returns a list of path constraints.

        Path constraints apply to all times and ensemble members simultaneously.
        Call :func:`OptimizationProblem.state` to return a time- and ensemble-member-independent
        symbol representing a model variable.

        :param ensemble_member: The ensemble member index. This index may only
                                be used to supply member-dependent bounds.

        :returns: A list of triples ``(f, m, M)``, with an :class:`MX` object representing
                  the path constraint function ``f``, lower bound ``m``, and upper bound ``M``.
                  The bounds may be numbers or :class:`.Timeseries` objects.

        Example::

            def path_constraints(self, ensemble_member):
                # 2 * x must lie between 2 and 4 for every time instance.
                path_constraint1 = (2 * self.state('x'), 2.0, 4.0)
                # x + y must lie between 2 and 3 for every time instance
                path_constraint2 = (self.state('x') + self.state('y'), 2.0, 3.0)
                return [path_constraint1, path_constraint2]

        """
        return []

    def post(self) -> None:
        """
        Postprocessing logic is performed here.
        """
        pass

    @property
    def equidistant(self) -> bool:
        """
        ``True`` if all time series are equidistant.
        """
        return False

    INTERPOLATION_LINEAR = 0
    INTERPOLATION_PIECEWISE_CONSTANT_FORWARD = 1
    INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD = 2

    def interpolate(
        self,
        t: float | np.ndarray,
        ts: np.ndarray,
        fs: np.ndarray,
        f_left: float = np.nan,
        f_right: float = np.nan,
        mode: int = INTERPOLATION_LINEAR,
    ) -> float | np.ndarray:
        """
        Linear interpolation over time.

        :param t:       Time at which to evaluate the interpolant.
        :type t:        float or vector of floats
        :param ts:      Time stamps.
        :type ts:       numpy array
        :param fs:      Function values at time stamps ts.
        :param f_left:  Function value left of leftmost time stamp.
        :param f_right: Function value right of rightmost time stamp.
        :param mode:    Interpolation mode.

        :returns: The interpolated value.
        """

        if isinstance(fs, np.ndarray) and fs.ndim == 2:
            # 2-D array of values. Interpolate each column separately.
            if len(t) == len(ts) and np.all(t == ts):
                # Early termination; nothing to interpolate
                return fs.copy()

            fs_int = [
                self.interpolate(t, ts, fs[:, i], f_left, f_right, mode) for i in range(fs.shape[1])
            ]
            return np.stack(fs_int, axis=1)
        elif hasattr(t, "__iter__"):
            if len(t) == len(ts) and np.all(t == ts):
                # Early termination; nothing to interpolate
                return fs.copy()

            return self.__interpolate(t, ts, fs, f_left, f_right, mode)
        else:
            if ts[0] == t:
                # Early termination; nothing to interpolate
                return fs[0]

            return self.__interpolate(t, ts, fs, f_left, f_right, mode)

    def __interpolate(self, t, ts, fs, f_left=np.nan, f_right=np.nan, mode=INTERPOLATION_LINEAR):
        """
        Linear interpolation over time.

        :param t:       Time at which to evaluate the interpolant.
        :type t:        float or vector of floats
        :param ts:      Time stamps.
        :type ts:       numpy array
        :param fs:      Function values at time stamps ts.
        :param f_left:  Function value left of leftmost time stamp.
        :param f_right: Function value right of rightmost time stamp.
        :param mode:    Interpolation mode.

        Note that it is assumed that `ts` is sorted. No such assumption is made for `t`.

        :returns: The interpolated value.
        """

        if f_left is None:
            if (min(t) if hasattr(t, "__iter__") else t) < ts[0]:
                raise Exception(f"Interpolation: Point {t} left of range")

        if f_right is None:
            if (max(t) if hasattr(t, "__iter__") else t) > ts[-1]:
                raise Exception(f"Interpolation: Point {t} right of range")

        if mode == self.INTERPOLATION_LINEAR:
            # No need to handle f_left / f_right; NumPy already does this for us
            return np.interp(t, ts, fs, f_left, f_right)
        elif mode == self.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD:
            v = fs[np.maximum(np.searchsorted(ts, t, side="right") - 1, 0)]
        elif mode == self.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD:
            v = fs[np.minimum(np.searchsorted(ts, t, side="left"), len(ts) - 1)]
        else:
            raise NotImplementedError

        # Handle f_left / f_right:
        if hasattr(t, "__iter__"):
            v[t < ts[0]] = f_left
            v[t > ts[-1]] = f_right
        else:
            if t < ts[0]:
                v = f_left
            elif t > ts[-1]:
                v = f_right

        return v

    @abstractproperty
    def controls(self) -> list[str]:
        """
        List of names of the control variables (excluding aliases).
        """
        pass

    @abstractmethod
    def discretize_controls(
        self, resolved_bounds: AliasDict
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the discretization of the control inputs, filling lower and upper
        bound vectors for the resulting optimization variables, as well as an initial guess.

        :param resolved_bounds: :class:`AliasDict` of numerical bound values. This is the
                                same dictionary as returned by :func:`bounds`, but with all
                                parameter symbols replaced with their numerical values.

        :returns: The number of control variables in the optimization problem, a lower
                  bound vector, an upper bound vector, a seed vector, and a dictionary
                  of offset values.
        """
        pass

    def dynamic_parameters(self) -> list[ca.MX]:
        """
        Returns a list of parameter symbols that may vary from run to run.  The values
        of these parameters are not cached.

        :returns: A list of parameter symbols.
        """
        return []

    @abstractmethod
    def extract_controls(self, ensemble_member: int = 0) -> dict[str, np.ndarray]:
        """
        Extracts state time series from optimizer results.

        Must return a dictionary of result time series.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of control input time series.
        """
        pass

    def control_vector(self, variable: str, ensemble_member: int = 0) -> ca.MX | list[ca.MX]:
        """
        Return the optimization variables for the entire time horizon of the given state.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: A vector of control input symbols for the entire time horizon.

        :raises: KeyError
        """
        return self.state_vector(variable, ensemble_member)

    def control(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the given control input, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given control input.

        :raises: KeyError
        """
        return self.variable(variable)

    @abstractmethod
    def control_at(
        self, variable: str, t: float, ensemble_member: int = 0, scaled: bool = False
    ) -> ca.MX:
        """
        Returns an :class:`MX` symbol representing the given control input at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.
        :param scaled:          True to return the scaled variable.

        :returns: :class:`MX` symbol representing the control input at the given time.

        :raises: KeyError
        """
        pass

    @abstractproperty
    def differentiated_states(self) -> list[str]:
        """
        List of names of the differentiated state variables (excluding aliases).
        """
        pass

    @abstractproperty
    def algebraic_states(self) -> list[str]:
        """
        List of names of the algebraic state variables (excluding aliases).
        """
        pass

    @abstractmethod
    def discretize_states(
        self, resolved_bounds: AliasDict
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the discretization of the states.

        Fills lower and upper bound vectors for the resulting optimization
        variables, as well as an initial guess.

        :param resolved_bounds: :class:`AliasDict` of numerical bound values.
            This is the same dictionary as returned by :func:`bounds`, but
            with all parameter symbols replaced with their numerical values.

        :returns: The number of control variables in the optimization problem,
                  a lower bound vector, an upper bound vector, a seed vector,
                  and a dictionary of vector offset values.
        """
        pass

    @abstractmethod
    def extract_states(self, ensemble_member: int = 0) -> dict[str, np.ndarray]:
        """
        Extracts state time series from optimizer results.

        Must return a dictionary of result time series.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of state time series.
        """
        pass

    @abstractmethod
    def state_vector(self, variable: str, ensemble_member: int = 0) -> ca.MX | list[ca.MX]:
        """
        Return the optimization variables for the entire time horizon of the given state.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: A vector of state symbols for the entire time horizon.

        :raises: KeyError
        """
        pass

    def state(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the given state, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given state.

        :raises: KeyError
        """
        return self.variable(variable)

    @abstractmethod
    def state_at(
        self, variable: str, t: float, ensemble_member: int = 0, scaled: bool = False
    ) -> ca.MX:
        """
        Returns an :class:`MX` symbol representing the given variable at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.
        :param scaled:          True to return the scaled variable.

        :returns: :class:`MX` symbol representing the state at the given time.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def extra_variable(self, variable: str, ensemble_member: int = 0) -> ca.MX:
        """
        Returns an :class:`MX` symbol representing the extra variable inside the state vector.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` symbol representing the extra variable.

        :raises: KeyError
        """
        pass

    @overload
    def states_in(
        self,
        variable: str,
        t0: float | None = ...,
        tf: float | None = ...,
        ensemble_member: int = ...,
        *,
        return_times: Literal[False] = ...,
    ) -> ca.MX: ...

    @overload
    def states_in(
        self,
        variable: str,
        t0: float | None = ...,
        tf: float | None = ...,
        ensemble_member: int = ...,
        *,
        return_times: Literal[True],
    ) -> tuple[ca.DM, ca.MX]: ...

    @abstractmethod
    def states_in(
        self,
        variable: str,
        t0: float | None = None,
        tf: float | None = None,
        ensemble_member: int = 0,
        *,
        return_times: bool = False,
    ) -> ca.MX | tuple[ca.DM, ca.MX]:
        """
        Iterates over symbols for states in the interval [t0, tf].

        :param variable:        Variable name.
        :param t0:              Left bound of interval.  If equal to None, the initial time is used.
        :param tf:              Right bound of interval.  If equal to None, the final time is used.
        :param ensemble_member: The ensemble member index.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def integral(
        self,
        variable: str,
        t0: float | None = None,
        tf: float | None = None,
        ensemble_member: int = 0,
    ) -> ca.MX:
        """
        Returns an expression for the integral over the interval [t0, tf].

        :param variable:        Variable name.
        :param t0:              Left bound of interval.  If equal to None, the initial time is used.
        :param tf:              Right bound of interval.  If equal to None, the final time is used.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` object representing the integral.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def der(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the time derivative given state, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given state.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def der_at(self, variable: str, t: float, ensemble_member: int = 0) -> ca.MX:
        """
        Returns an expression for the time derivative of the specified variable at time t.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` object representing the derivative.

        :raises: KeyError
        """
        pass

    def get_timeseries(self, variable: str, ensemble_member: int = 0) -> Timeseries:
        """
        Looks up a timeseries from the internal data store.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: The requested time series.
        :rtype: :class:`.Timeseries`

        :raises: KeyError
        """
        raise NotImplementedError

    def set_timeseries(
        self,
        variable: str,
        timeseries: Timeseries,
        ensemble_member: int = 0,
        output: bool = True,
        check_consistency: bool = True,
    ) -> None:
        """
        Sets a timeseries in the internal data store.

        :param variable:          Variable name.
        :param timeseries:        Time series data.
        :type timeseries:         iterable of floats, or :class:`.Timeseries`
        :param ensemble_member:   The ensemble member index.
        :param output:            Whether to include this time series in output data files.
        :param check_consistency: Whether to check consistency between the time stamps on
                                  the new timeseries object and any existing time stamps.
        """
        raise NotImplementedError

    def timeseries_at(self, variable: str, t: float, ensemble_member: int = 0) -> float:
        """
        Return the value of a time series at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.

        :returns: The interpolated value of the time series.

        :raises: KeyError
        """
        raise NotImplementedError

    def map_path_expression(self, expr: ca.MX, ensemble_member: int) -> ca.MX:
        """
        Maps the path expression `expr` over the entire time horizon of the optimization problem.

        :param expr: An :class:`MX` path expression.

        :returns: An :class:`MX` expression evaluating `expr` over the entire time horizon.
        """
        raise NotImplementedError

    @debug_check(DebugLevel.HIGH)
    def __debug_check_linearity_constraints(self, nlp):
        x = nlp["x"]
        f = nlp["f"]
        g = nlp["g"]

        expand_f_g = ca.Function("f_g", [x], [f, g]).expand()
        X_sx = ca.SX.sym("X", *x.shape)
        f_sx, g_sx = expand_f_g(X_sx)

        jac = ca.Function("j", [X_sx], [ca.jacobian(g_sx, X_sx)]).expand()
        if jac(np.nan).is_regular():
            logger.info("The constraints are linear")
        else:
            hes = ca.Function("j", [X_sx], [ca.jacobian(ca.jacobian(g_sx, X_sx), X_sx)]).expand()
            if hes(np.nan).is_regular():
                logger.info("The constraints are quadratic")
            else:
                logger.info("The constraints are nonlinear")

    @debug_check(DebugLevel.VERYHIGH)
    def __debug_check_linear_independence(self, lbx, ubx, lbg, ubg, nlp):
        x = nlp["x"]
        f = nlp["f"]
        g = nlp["g"]

        expand_f_g = ca.Function("f_g", [x], [f, g]).expand()
        x_sx = ca.SX.sym("X", *x.shape)
        f_sx, g_sx = expand_f_g(x_sx)

        x, f, g = x_sx, f_sx, g_sx

        lbg = np.array(ca.vertsplit(ca.veccat(*lbg))).ravel()
        ubg = np.array(ca.vertsplit(ca.veccat(*ubg))).ravel()

        # Find the linear constraints
        g_sjac = ca.Function("Af", [x], [ca.jtimes(g, x, x.ones(*x.shape))])

        res = g_sjac(np.nan)
        res = np.array(res).ravel()
        g_is_linear = ~np.isnan(res)

        # Find the rows in the jacobian with only a single entry
        g_jac_csr = ca.DM(ca.Function("tmp", [x], [g]).sparsity_jac(0, 0)).tocsc().tocsr()
        g_single_variable = np.diff(g_jac_csr.indptr) == 1

        # Find the rows which are equality constraints
        g_eq_constraint = lbg == ubg

        # The intersection of all selections are constraints like we want
        g_constant_assignment = g_is_linear & g_single_variable & g_eq_constraint

        # Map of variable (index) to constraints/row numbers
        var_index_assignment = {}
        for i in range(g.size1()):
            if g_constant_assignment[i]:
                var_ind = g_jac_csr.getrow(i).indices[0]
                var_index_assignment.setdefault(var_ind, []).append(i)

        var_names, named_x, named_f, named_g = self._debug_get_named_nlp(nlp)

        for vi, g_inds in var_index_assignment.items():
            if len(g_inds) > 1:
                logger.info(
                    f"Variable '{var_names[vi]}' has duplicate constraints setting its value:"
                )
                for g_i in g_inds:
                    logger.info(f"row {g_i}: {named_g[g_i]} = {lbg[g_i]}")

        # Find variables for which the bounds are equal, but also an equality
        # constraint is set. This would result in a constraint `1 = 1` with
        # the default IPOPT option `fixed_variable_treatment = make_parameter`
        x_inds = np.flatnonzero(lbx == ubx)

        for vi in x_inds:
            if vi in var_index_assignment:
                logger.info(
                    f"Variable '{var_names[vi]}' has equal bounds (value = {lbx[vi]}), "
                    "but also the following equality constraints:"
                )
                for g_i in var_index_assignment[vi]:
                    logger.info(f"row {g_i}: {named_g[g_i]} = {lbg[g_i]}")
