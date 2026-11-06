import logging
import math

import numpy as np

from .goal_programming_mixin import GoalProgrammingMixin
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

    # The fixed input that holds a time-dependent theta, or None.
    __theta_input = None

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)
        options = self.homotopy_options()

        # Overwrite the seed only when the results of the latest run are
        # stored within this class. That is, when the GoalProgrammingMixin
        # class is not used or at the first run of the goal programming loop.
        if self.__theta > options["theta_start"] and getattr(self, "_gp_first_run", True):
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
        try:
            # Only set the theta if we are in the optimization loop. We want
            # to avoid accidental usage of the parameter value in e.g. pre().
            # Note that we use a try-except here instead of hasattr, to avoid
            # explicit name mangling.
            parameters[options["homotopy_parameter"]] = self.__theta
        except AttributeError:
            pass

        return parameters

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        if self.__theta_input is None:
            return constant_inputs

        times = self.times()
        values = np.full(len(times), self.__theta)
        values[self.__thresh_idx :] = 0.0

        # Delayed expressions also read theta at the history times.
        history_start = min(
            (series.times[0] for series in self.history(ensemble_member).values()),
            default=times[0],
        )
        if history_start < times[0]:
            times = np.hstack([history_start, times])
            values = np.hstack([values[0], values])

        constant_inputs[self.__theta_input] = Timeseries(times, values)
        return constant_inputs

    def homotopy_options(self) -> dict[str, str | float | bool | int | None]:
        """
        Returns a dictionary of options controlling the homotopy process.

        +----------------------------------+------------+---------------+
        | Option                           | Type       | Default value |
        +==================================+============+===============+
        | ``theta_start``                  | ``float``  | ``0.0``       |
        +----------------------------------+------------+---------------+
        | ``delta_theta_0``                | ``float``  | ``1.0``       |
        +----------------------------------+------------+---------------+
        | ``delta_theta_min``              | ``float``  | ``0.01``      |
        +----------------------------------+------------+---------------+
        | ``homotopy_parameter``           | ``string`` | ``theta``     |
        +----------------------------------+------------+---------------+
        | ``skip_intermediate_priorities`` | ``bool``   | ``False``     |
        +----------------------------------+------------+---------------+
        | ``non_linear_thresh_time_idx``   | ``int``    | ``None``      |
        +----------------------------------+------------+---------------+

        The homotopy process is controlled by the homotopy parameter in the model, specified by the
        option ``homotopy_parameter``.  The homotopy parameter is initialized to ``theta_start``,
        and increases to a value of ``1.0`` with a dynamically changing step size.  This step size
        is initialized with the value of the option ``delta_theta_0``.  If this step size is too
        large, i.e., if the problem with the increased homotopy parameter fails to converge, the
        step size is halved.  The process of halving terminates when the step size falls below the
        minimum value specified by the option ``delta_theta_min``. The last step is shortened if
        needed, so that the problem is always solved at :math:`\\theta = 1.0`.

        ``theta_start`` must not exceed 1, both step sizes must be positive, and
        ``homotopy_parameter`` must name a parameter or a fixed input of the model. Otherwise,
        :py:meth:`optimize` raises a ``ValueError``.

        If ``skip_intermediate_priorities`` is ``True``, goal programming solves only its first
        priority at the intermediate steps (:math:`\\theta < 1.0`), and all priorities at
        :math:`\\theta = 1.0`. This requires ``GoalProgrammingMixin`` to follow
        ``HomotopyMixin`` in the inheritance order.

        If ``non_linear_thresh_time_idx`` is set, :math:`\\theta` follows the homotopy process
        only at the time steps of :py:meth:`times` before this index, and is ``0.0`` after it. The
        history uses the :math:`\\theta` of the first time step. :math:`\\theta` must then be a
        fixed input of the model, e.g. ``input Real theta(fixed=true)``. The step at
        :math:`\\theta = 0` logs a warning about non-affine equations, because the linearity
        check substitutes random values for fixed inputs.

        :returns: A dictionary of homotopy options.
        """

        return {
            "theta_start": 0.0,
            "delta_theta_0": 1.0,
            "delta_theta_min": 0.01,
            "homotopy_parameter": "theta",
            "skip_intermediate_priorities": False,
            "non_linear_thresh_time_idx": None,
        }

    def dynamic_parameters(self):
        dynamic_parameters = super().dynamic_parameters()

        if self.__theta > 0 and self.__theta_input is None:
            # For theta = 0, we don't mark the homotopy parameter as being dynamic,
            # so that the correct sparsity structure is obtained for the linear model.
            options = self.homotopy_options()
            dynamic_parameters.append(self.variable(options["homotopy_parameter"]))

        return dynamic_parameters

    def __check_options(self, options):
        if not options["theta_start"] <= 1.0:
            raise ValueError("theta_start must not exceed 1.")
        if not (options["delta_theta_0"] > 0.0 and options["delta_theta_min"] > 0.0):
            raise ValueError("delta_theta_0 and delta_theta_min must be positive.")
        name = options["homotopy_parameter"]
        canonical_name, _ = self.alias_relation.canonical_signed(name)
        input_names = {v.name() for v in self.dae_variables["constant_inputs"]}
        parameter_names = {v.name() for v in self.dae_variables["parameters"]}
        if canonical_name not in parameter_names | input_names:
            raise ValueError(
                f"Homotopy parameter '{name}' is not a parameter or fixed input of the model."
            )

        thresh_idx = options["non_linear_thresh_time_idx"]
        if thresh_idx is not None and thresh_idx < 0:
            raise ValueError("non_linear_thresh_time_idx must not be negative.")
        if thresh_idx is not None and canonical_name not in input_names:
            raise ValueError(
                "non_linear_thresh_time_idx requires theta to be a fixed input of the model, "
                "e.g. input Real theta(fixed=true)."
            )

        mro = type(self).__mro__
        if options["skip_intermediate_priorities"] and (
            GoalProgrammingMixin not in mro[mro.index(HomotopyMixin) :]
        ):
            raise ValueError(
                "skip_intermediate_priorities requires GoalProgrammingMixin "
                "to follow HomotopyMixin in the inheritance order."
            )

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Pre-processing
        if preprocessing:
            self.pre()

        # Overrides of homotopy_options() need not return the optional keys.
        options = {**HomotopyMixin.homotopy_options(self), **self.homotopy_options()}
        self.__check_options(options)
        self.__thresh_idx = options["non_linear_thresh_time_idx"]
        self.__theta_input = (
            options["homotopy_parameter"] if self.__thresh_idx is not None else None
        )
        delta_theta = options["delta_theta_0"]

        # Homotopy loop
        self.__theta = options["theta_start"]

        while self.__theta <= 1.0:
            logger.info(f"Solving with homotopy parameter theta = {self.__theta}.")

            self._gp_skip_later_priorities = (
                options["skip_intermediate_priorities"] and self.__theta < 1.0
            )
            success = super().optimize(
                preprocessing=False, postprocessing=False, log_solver_failure_as_error=False
            )
            if success:
                self.__results = [
                    self.extract_results(ensemble_member)
                    for ensemble_member in range(self.ensemble_size)
                ]

                if self.__theta == 0.0:
                    self.check_collocation_linearity = False
                    self.linear_collocation = False

                    # Recompute the sparsity structure for the nonlinear model family.
                    self.clear_transcription_cache()

                if self.__theta == 1.0:  # NOSONAR
                    break
                theta_converged = self.__theta

            else:
                if self.__theta == options["theta_start"]:
                    break

                delta_theta = (self.__theta - theta_converged) / 2

                if delta_theta < options["delta_theta_min"]:
                    self.__theta = theta_converged
                    failure_message = (
                        "Solver failed with homotopy parameter theta = {}. Theta cannot "
                        "be decreased further, as that would violate the minimum delta "
                        "theta of {}.".format(self.__theta, options["delta_theta_min"])
                    )
                    if log_solver_failure_as_error:
                        logger.error(failure_message)
                    else:
                        # In this case we expect some higher level process to deal
                        # with the solver failure, so we only log it as info here.
                        logger.info(failure_message)
                    break

            self.__theta = min(theta_converged + delta_theta, 1.0)
            # Rounding can leave the sum of the steps just below 1.0.
            if success and math.isclose(self.__theta, 1.0):
                self.__theta = 1.0

        # Post-processing
        if postprocessing:
            self.post()

        return success
