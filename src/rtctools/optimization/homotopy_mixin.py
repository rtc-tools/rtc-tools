import logging
from typing import Any

import numpy as np

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


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
        super().__init__(**kwargs)
        self._gp_prune_subproblems = False

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)
        options = self.homotopy_options()

        # Overwrite the seed only when the results of the latest run are
        # stored within this class. That is, when the GoalProgrammingMixin
        # class is not used or at the first run of the goal programming loop.
        theta_val = getattr(self, "_HomotopyMixin__theta_val", None)
        overwrite_seed = theta_val is not None and theta_val > float(options["theta_start"])

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
            self.__sync_theta(ensemble_member)
        return parameters

    def __sync_theta(self, ensemble_member):
        """Map forecast theta onto IO timestamps, including history, for one member."""
        theta = self.homotopy_theta
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
            # the first forecast value; the threshold counts forecast steps only.
            indices = np.searchsorted(self.times(), self.io.times_sec, side="right") - 1
            values = theta[np.clip(indices, 0, len(theta) - 1)]
        self.io.set_timeseries(
            self.homotopy_options()["homotopy_parameter"], datetimes, values, ensemble_member
        )

    def constant_inputs(self, ensemble_member):
        self.__sync_theta(ensemble_member)
        constant_inputs = super().constant_inputs(ensemble_member)

        options = self.homotopy_options()
        param_name = options["homotopy_parameter"]

        theta = self.homotopy_theta
        if theta is None:
            return constant_inputs

        dae_constant_inputs = [
            v.name() for v in getattr(self, "dae_variables", {}).get("constant_inputs", [])
        ]
        if param_name in dae_constant_inputs or param_name in constant_inputs:
            # Use the forecast grid directly; an IO grid may have different timestamps.
            times = self.times()
            values = np.full(len(times), float(theta)) if np.ndim(theta) == 0 else theta
            constant_inputs[param_name] = Timeseries(times, values)

        return constant_inputs

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
        theta zero gives the intended linear approximation.

        If ``prune_intermediate_subproblems`` is set to ``True`` and goal programming is used,
        all but the first available priority are omitted during intermediate homotopy
        iterations (:math:`\\theta < 1.0`), and only solved at the final iteration
        (:math:`\\theta = 1.0`).

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
        try:
            return self.__theta
        except AttributeError:
            return None

    @property
    def _homotopy_is_intermediate(self) -> bool:
        """
        True if currently in an intermediate homotopy step (theta < 1.0).
        """
        theta_val = getattr(self, "_HomotopyMixin__theta_val", None)
        return theta_val is not None and theta_val < 1.0

    def dynamic_parameters(self):
        dynamic_parameters = super().dynamic_parameters()

        options = self.homotopy_options()
        param_name = options["homotopy_parameter"]

        try:
            var = self.variable(param_name)
        except KeyError:
            var = None

        theta = self.homotopy_theta
        if var is not None and theta is not None:
            is_active = bool(np.any(np.asarray(theta) > 0))
            if is_active:
                # For theta = 0, we don't mark the homotopy parameter as being dynamic,
                # so that the correct sparsity structure is obtained for the linear model.
                dynamic_parameters.append(var)

        return dynamic_parameters

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do not expose a previous run's theta during preprocessing or configuration lookup.
        self.__theta = None
        self.__theta_val = None
        # Pre-processing
        if preprocessing:
            self.pre()

        options = self.homotopy_options()
        parameters = self.parameters(0)
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
        if thresh_idx is not None:
            if (
                not isinstance(thresh_idx, (int, float, np.integer, np.floating))
                or isinstance(thresh_idx, (bool, np.bool_))
                or not np.isfinite(thresh_idx)
                or thresh_idx < 0
                or int(thresh_idx) != thresh_idx
            ):
                raise ValueError("non_linear_thresh_time_idx must be a non-negative integer.")
            thresh_idx = int(thresh_idx)
        is_vector_theta = thresh_idx is not None and thresh_idx < n_times
        if is_vector_theta:
            input_names = {v.name() for v in self.dae_variables.get("constant_inputs", [])}
            if options["homotopy_parameter"] not in input_names:
                raise ValueError(
                    "Time-varying homotopy requires the homotopy variable to be a fixed "
                    "input (e.g. input Real theta(fixed=true)), not a Modelica parameter."
                )

        def _set_theta(theta_val: float):
            self.__theta_val = theta_val
            if is_vector_theta:
                vec = np.zeros(n_times, dtype=float)
                vec[:thresh_idx] = theta_val
                self.__theta = vec
            else:
                self.__theta = theta_val

        theta_val = theta_start
        last_successful_theta = None
        success = False
        prune_subproblems = bool(options.get("prune_intermediate_subproblems", False))

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

                if theta_val == 0.0:
                    self.check_collocation_linearity = False
                    self.linear_collocation = False

                    # Recompute the sparsity structure for the nonlinear model family.
                    self.clear_transcription_cache()

                last_successful_theta = theta_val
                if theta_val == 1.0:
                    break

                theta_val = min(1.0, theta_val + delta_theta)
                if 1.0 - theta_val < 1e-9:
                    theta_val = 1.0

            else:
                if last_successful_theta is None:
                    break

                delta_theta = (theta_val - last_successful_theta) / 2.0

                if delta_theta < delta_theta_min:
                    failure_message = (
                        f"Solver failed with homotopy parameter theta = {theta_val}. Theta cannot "
                        f"be decreased further, as that would violate the minimum delta "
                        f"theta of {options['delta_theta_min']}."
                    )
                    if log_solver_failure_as_error:
                        logger.error(failure_message)
                    else:
                        # In this case we expect some higher level process to deal
                        # with the solver failure, so we only log it as info here.
                        logger.info(failure_message)
                    break

                theta_val = last_successful_theta + delta_theta

        # Post-processing
        if postprocessing:
            self.post()

        return success
