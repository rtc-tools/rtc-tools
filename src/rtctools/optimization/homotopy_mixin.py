import logging
from typing import Any

import numpy as np

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")

# Half the microsecond resolution of datetimes: timestamps closer than this are equivalent.
_TIME_TOL = 5e-7


class HomotopyMixin(OptimizationProblem):
    """
    Adds homotopy to your optimization problem.  A homotopy is a continuous transformation between
    two optimization problems, parametrized by a single parameter :math:`\\theta \\in [0, 1]`.

    Homotopy may be used to solve non-convex optimization problems, by starting with a convex
    approximation at :math:`\\theta = 0.0` and ending with the non-convex problem at
    :math:`\\theta = 1.0`.

    .. note::

        It is advised to look for convex reformulations of your problem, before resorting to a use
        of the (potentially expensive) homotopy process.

    """

    def __init__(self, **kwargs):
        self.__theta: float | np.ndarray | None = None
        self.__theta_val: float | None = None
        self.__results: list | None = None
        super().__init__(**kwargs)
        self._gp_prune_subproblems = False

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)
        options = self.homotopy_options()

        # Overwrite the seed only when the results of the latest run are
        # stored within this class. That is, when the GoalProgrammingMixin
        # class is not used or at the first run of the goal programming loop.
        theta_val = self.__theta_val
        overwrite_seed = (
            self.__results is not None
            and theta_val is not None
            and theta_val > float(options["theta_start"])
        )

        if overwrite_seed and getattr(self, "_gp_first_run", True):
            for key, result in self.__results[ensemble_member].items():
                times = self.times(key)
                if (result.ndim == 1 and len(result) == len(times)) or (
                    result.ndim == 2 and result.shape[0] == len(times)
                ):
                    # Only include seed timeseries which are consistent
                    # with the specified time stamps.
                    seed[key] = Timeseries(times, result)
                elif (result.ndim == 1 and len(result) == 1) or (
                    result.ndim == 2 and result.shape[0] == 1
                ):
                    seed[key] = result
        return seed

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)
        options = self.homotopy_options()
        theta = self.homotopy_theta
        if theta is not None:
            # Modelica parameters remain scalar. Time-varying theta must be a fixed input.
            parameters[options["homotopy_parameter"]] = float(np.asarray(theta).flat[0])
        return parameters

    def __sync_theta(self, ensemble_member):
        """Map forecast theta onto IO timestamps, including history, for one member."""
        theta = self.__theta
        if theta is None or not hasattr(self, "io"):
            return
        # DataStore.datetimes raises AttributeError before any timeseries have been read.
        datetimes = getattr(self.io, "datetimes", None)
        if datetimes is None:
            return
        if np.ndim(theta) == 0:
            values = np.full(len(datetimes), float(theta))
        else:
            # Previous-value mapping preserves the cutoff on finer IO grids. History uses
            # the first forecast value; the threshold counts forecast steps only. The
            # tolerance snaps numerically equivalent timestamps (e.g. 0.6 vs 0.6000000000000001).
            forecast_times = np.asarray(self.times(), dtype=float)
            io_times = np.asarray(self.io.times_sec, dtype=float)
            indices = np.searchsorted(forecast_times, io_times + _TIME_TOL, side="right") - 1
            values = theta[np.clip(indices, 0, len(theta) - 1)]
        self.io.set_timeseries(
            self.homotopy_options()["homotopy_parameter"], datetimes, values, ensemble_member
        )

    def __theta_history_times(self, existing, forecast_start, ensemble_member):
        """Collect past timestamps from the parent input, IO and history() of this member."""
        arrays = []
        if isinstance(existing, Timeseries):
            arrays.append(np.asarray(existing.times, dtype=float))
        io = getattr(self, "io", None)
        if getattr(io, "datetimes", None) is not None:
            arrays.append(np.asarray(io.times_sec, dtype=float))
        # History supplied without IO timestamps must also see theta in the past.
        arrays.extend(
            np.asarray(series.times, dtype=float)
            for series in self.history(ensemble_member).values()
        )
        if not arrays:
            return np.empty(0)
        # Concatenate once; many history series share the same grid.
        times = np.concatenate(arrays)
        return np.unique(times[times < forecast_start - _TIME_TOL])

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)

        options = self.homotopy_options()
        param_name = options["homotopy_parameter"]

        theta = self.homotopy_theta
        if theta is None:
            return constant_inputs

        dae_constant_inputs = [
            v.name() for v in getattr(self, "dae_variables", {}).get("constant_inputs", [])
        ]
        canonical_name, _ = self.alias_relation.canonical_signed(param_name)
        if canonical_name in dae_constant_inputs or param_name in constant_inputs:
            # Use the forecast grid directly; an IO grid may have different timestamps.
            times = np.asarray(self.times(), dtype=float)
            values = (
                np.full(len(times), float(theta))
                if np.ndim(theta) == 0
                else np.asarray(theta, dtype=float)
            )
            if len(times) > 0:
                history_times = self.__theta_history_times(
                    constant_inputs.get(param_name), times[0], ensemble_member
                )
                times = np.hstack([history_times, times])
                values = np.hstack([np.full(len(history_times), values[0]), values])
            # Write through the configured alias so AliasDict applies its sign.
            constant_inputs[param_name] = Timeseries(times, values)

        return constant_inputs

    def __is_theta_input(self, variable) -> bool:
        """True if ``variable`` (or an alias) is the homotopy variable used as a fixed input."""
        canonical_name, _ = self.alias_relation.canonical_signed(variable)
        theta_name, _ = self.alias_relation.canonical_signed(
            self.homotopy_options()["homotopy_parameter"]
        )
        dae_constant_inputs = {
            v.name() for v in getattr(self, "dae_variables", {}).get("constant_inputs", [])
        }
        return canonical_name == theta_name and canonical_name in dae_constant_inputs

    @staticmethod
    def __snap_to_grid(t, grid):
        """Replace times within ``_TIME_TOL`` of a grid time by that grid time."""
        t_arr = np.asarray(t, dtype=float)
        idx = np.searchsorted(grid, t_arr)
        lower = grid[np.clip(idx - 1, 0, len(grid) - 1)]
        upper = grid[np.clip(idx, 0, len(grid) - 1)]
        nearest = np.where(np.abs(lower - t_arr) <= np.abs(upper - t_arr), lower, upper)
        snapped = np.where(np.abs(nearest - t_arr) <= _TIME_TOL, nearest, t_arr)
        return snapped if np.ndim(t) else float(snapped)

    def timeseries_at(self, variable, t, ensemble_member=0):
        # IO stores theta on its own grid, which may not contain the forecast cutoff.
        # Interpolate the forecast-grid fixed input that the equations use instead.
        # Unlike IOMixin.timeseries_at(), which is always linear, use the configured
        # interpolation method so that theta matches the transcription between time steps.
        if self.__theta is not None and self.__is_theta_input(variable):
            series = self.constant_inputs(ensemble_member)[variable]
            canonical_name, _ = self.alias_relation.canonical_signed(variable)
            interpolation_method = getattr(self, "interpolation_method", None)
            mode = (
                interpolation_method(canonical_name)
                if interpolation_method is not None
                else self.INTERPOLATION_LINEAR
            )
            times = np.asarray(series.times, dtype=float)
            return self.interpolate(
                self.__snap_to_grid(t, times), times, series.values, np.nan, np.nan, mode
            )
        return super().timeseries_at(variable, t, ensemble_member)

    def homotopy_options(self) -> dict[str, Any]:
        """
        Returns a dictionary of options controlling the homotopy process.

        +------------------------------------+------------+---------------+
        | Option                             | Type       | Default value |
        +====================================+============+===============+
        | ``theta_start``                    | ``float``  | ``0.0``       |
        +------------------------------------+------------+---------------+
        | ``delta_theta_0``                  | ``float``  | ``1.0``       |
        +------------------------------------+------------+---------------+
        | ``delta_theta_min``                | ``float``  | ``0.01``      |
        +------------------------------------+------------+---------------+
        | ``homotopy_parameter``             | ``string`` | ``theta``     |
        +------------------------------------+------------+---------------+
        | ``non_linear_thresh_time_idx``     | ``int``    | ``None``      |
        +------------------------------------+------------+---------------+
        | ``prune_intermediate_subproblems`` | ``bool``   | ``False``     |
        +------------------------------------+------------+---------------+

        The homotopy process is controlled by the homotopy parameter in the model, specified by the
        option ``homotopy_parameter``.  The homotopy parameter is initialized to ``theta_start``,
        and increases to a value of ``1.0`` with a dynamically changing step size.  This step size
        is initialized with the value of the option ``delta_theta_0``.  If this step size is too
        large, i.e., if the problem with the increased homotopy parameter fails to converge, the
        step size is halved.  The process of halving terminates when the step size falls below the
        minimum value specified by the option ``delta_theta_min``.

        If ``non_linear_thresh_time_idx`` is set (either via this option or via the ``parameters``
        dictionary), ``theta`` is treated as a timeseries where time steps at or beyond this index
        are held at 0.0. The index counts steps in ``times()`` (excluding IO history), and must
        be a non-negative integer. An explicit parameter takes precedence over the option;
        ``None`` or an index at or beyond the horizon length retains scalar homotopy.
        Time-varying homotopy requires the model to declare theta as a fixed input, e.g.
        ``input Real theta(fixed=true)``, rather than a Modelica parameter. Historical IO
        timestamps use the first forecast theta. Equations must be formulated so that
        theta zero gives the intended linear approximation. An index of ``0`` keeps the whole
        horizon linear; the problem is then solved only once.

        .. note::

            The linearity check of the collocation equations substitutes random values for
            fixed inputs. When theta is a fixed input, the :math:`\\theta = 0` step is therefore
            not recognized as linear: it is solved correctly, but without the constant-Jacobian
            solver setting, and a warning about non-affine equations is logged.

        If ``prune_intermediate_subproblems`` is set to ``True`` and goal programming is used,
        all but the first available priority are omitted during intermediate homotopy
        iterations (:math:`\\theta < 1.0`), and only solved at the final iteration
        (:math:`\\theta = 1.0`). This requires ``HomotopyMixin`` to precede
        ``GoalProgrammingMixin`` in the inheritance order; otherwise a warning is logged and
        the option is ignored.

        :returns: A dictionary of homotopy options.
        """

        return {
            "theta_start": 0.0,
            "delta_theta_0": 1.0,
            "delta_theta_min": 0.01,
            "homotopy_parameter": "theta",
            "non_linear_thresh_time_idx": None,
            "prune_intermediate_subproblems": False,
        }

    @property
    def homotopy_theta(self) -> float | np.ndarray | None:
        """
        Current value of the homotopy parameter :math:`\\theta`.
        """
        return self.__theta

    @property
    def _homotopy_is_intermediate(self) -> bool:
        """
        True if currently in an intermediate homotopy step (theta < 1.0).
        """
        return self.__theta_val is not None and self.__theta_val < 1.0

    def dynamic_parameters(self):
        dynamic_parameters = super().dynamic_parameters()

        options = self.homotopy_options()
        param_name = options["homotopy_parameter"]
        canonical_name, _ = self.alias_relation.canonical_signed(param_name)
        theta = self.homotopy_theta
        # For theta = 0, we don't mark the homotopy parameter as being dynamic,
        # so that the correct sparsity structure is obtained for the linear model.
        if theta is not None and np.any(np.asarray(theta) > 0):
            # CasADi needs a symbol, not the negated expression of a negative alias.
            # The name is validated in optimize(), so a lookup error is not swallowed.
            dynamic_parameters.append(self.variable(canonical_name))

        return dynamic_parameters

    def _validate_homotopy_settings(self, options, parameters):
        delta_theta = float(options["delta_theta_0"])
        delta_theta_min = float(options["delta_theta_min"])
        theta_start = float(options["theta_start"])
        if not np.isfinite(theta_start) or not 0.0 <= theta_start <= 1.0:
            raise ValueError("theta_start must be finite and between 0 and 1.")
        if not np.isfinite(delta_theta) or delta_theta <= 0.0:
            raise ValueError("delta_theta_0 must be finite and positive.")
        if not np.isfinite(delta_theta_min) or delta_theta_min <= 0.0:
            raise ValueError("delta_theta_min must be finite and positive.")

        thresh_idx = parameters.get(
            "non_linear_thresh_time_idx", options.get("non_linear_thresh_time_idx")
        )

        n_times = len(self.times())
        if n_times == 0:
            raise ValueError("Homotopy requires a non-empty optimization time grid.")
        is_fixed_input = self._validate_homotopy_variable(options["homotopy_parameter"])
        if thresh_idx is not None:
            if (
                not isinstance(thresh_idx, (int, float, np.integer, np.floating))
                or isinstance(thresh_idx, (bool, np.bool_))
                or not np.isfinite(thresh_idx)
                or thresh_idx < 0
                or (isinstance(thresh_idx, (float, np.floating)) and not thresh_idx.is_integer())
            ):
                raise ValueError("non_linear_thresh_time_idx must be a non-negative integer.")
            thresh_idx = int(thresh_idx)
        is_vector_theta = thresh_idx is not None and thresh_idx < n_times
        if is_vector_theta and not is_fixed_input:
            raise ValueError(
                "Time-varying homotopy requires the homotopy variable to be a fixed "
                "input (e.g. input Real theta(fixed=true)), not a Modelica parameter."
            )
        return (
            theta_start,
            delta_theta,
            delta_theta_min,
            thresh_idx,
            n_times,
            is_vector_theta,
            is_fixed_input,
        )

    def _validate_homotopy_variable(self, name) -> bool:
        """Raise if the (alias-resolved) name is unknown; return whether it is a fixed input."""
        canonical_name, _ = self.alias_relation.canonical_signed(name)
        input_names = {v.name() for v in self.dae_variables.get("constant_inputs", [])}
        parameter_names = {v.name() for v in self.dae_variables.get("parameters", [])}
        if canonical_name not in input_names | parameter_names:
            raise ValueError(
                f"Homotopy parameter '{name}' is not a parameter or fixed input of the model."
            )
        return canonical_name in input_names

    def __pruning_supported(self) -> bool:
        """True if a goal programming mixin that honours pruning follows this mixin in the MRO."""
        mro = type(self).__mro__
        later = mro[mro.index(HomotopyMixin) + 1 :]
        return any(vars(cls).get("_supports_homotopy_pruning", False) for cls in later)

    def __prune_enabled(self, options) -> bool:
        if not options.get("prune_intermediate_subproblems", False):
            return False
        if not self.__pruning_supported():
            logger.warning(
                "prune_intermediate_subproblems is ignored: it requires GoalProgrammingMixin "
                "to follow HomotopyMixin in the inheritance order."
            )
            return False
        return True

    def __initial_theta(self, theta_start, thresh_idx, is_vector_theta) -> float:
        if not is_vector_theta:
            return theta_start
        if thresh_idx == 0:
            # All time steps stay linear, so every continuation step would be identical.
            logger.info("non_linear_thresh_time_idx = 0: the horizon is linear; solving once.")
            return 1.0
        logger.info(
            "Time-dependent homotopy uses theta as a fixed input; a warning about "
            "non-affine equations at theta = 0 is expected."
        )
        return theta_start

    def __log_continuation_failure(self, theta_val, delta_theta_min, as_error):
        failure_message = (
            f"Solver failed with homotopy parameter theta = {theta_val}. Theta cannot "
            f"be decreased further, as that would violate the minimum delta "
            f"theta of {delta_theta_min}."
        )
        if as_error:
            logger.error(failure_message)
        else:
            # In this case we expect some higher level process to deal
            # with the solver failure, so we only log it as info here.
            logger.info(failure_message)

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do not expose a previous run's theta or results during preprocessing or configuration.
        self.__theta = None
        self.__theta_val = None
        self.__results = None
        # Pre-processing
        if preprocessing:
            self.pre()

        options = self.homotopy_options()
        parameters = self.parameters(0)
        (
            theta_start,
            delta_theta,
            delta_theta_min,
            thresh_idx,
            n_times,
            is_vector_theta,
            is_fixed_input,
        ) = self._validate_homotopy_settings(options, parameters)

        def _set_theta(theta_val: float):
            self.__theta_val = theta_val
            if is_vector_theta:
                vec = np.zeros(n_times, dtype=float)
                vec[:thresh_idx] = theta_val
                self.__theta = vec
            else:
                self.__theta = theta_val
            # Keep IO consistent once per step; only fixed inputs are time series.
            if is_fixed_input:
                for ensemble_member in range(self.ensemble_size):
                    self.__sync_theta(ensemble_member)

        theta_val = self.__initial_theta(theta_start, thresh_idx, is_vector_theta)
        last_successful_theta = None
        success = False
        prune_subproblems = self.__prune_enabled(options)

        while theta_val <= 1.0:
            _set_theta(theta_val)
            if is_vector_theta:
                logger.info(
                    f"Solving with homotopy parameter theta = {theta_val} "
                    f"(threshold index {thresh_idx}/{n_times})."
                )
            else:
                logger.info(f"Solving with homotopy parameter theta = {theta_val}.")

            self._gp_prune_subproblems = prune_subproblems and self._homotopy_is_intermediate

            try:
                success = super().optimize(
                    preprocessing=False, postprocessing=False, log_solver_failure_as_error=False
                )
            finally:
                self._gp_prune_subproblems = False

            if success:
                self.__results = [
                    self.extract_results(ensemble_member)
                    for ensemble_member in range(self.ensemble_size)
                ]

                if theta_val <= 0.0:
                    self.check_collocation_linearity = False
                    self.linear_collocation = False

                    # Recompute the sparsity structure for the nonlinear model family.
                    self.clear_transcription_cache()

                last_successful_theta = theta_val
                if theta_val >= 1.0:
                    break

                theta_val = min(1.0, theta_val + delta_theta)
                if 1.0 - theta_val < 1e-9:
                    theta_val = 1.0

            else:
                if last_successful_theta is None:
                    break

                delta_theta = (theta_val - last_successful_theta) / 2.0

                if delta_theta < delta_theta_min:
                    self.__log_continuation_failure(
                        theta_val, options["delta_theta_min"], log_solver_failure_as_error
                    )
                    break

                theta_val = last_successful_theta + delta_theta

        # Post-processing
        if postprocessing:
            self.post()

        return success
