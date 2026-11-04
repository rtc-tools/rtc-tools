import logging
import math

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

    def homotopy_options(self) -> dict[str, str | float]:
        """
        Returns a dictionary of options controlling the homotopy process.

        +------------------------+------------+---------------+
        | Option                 | Type       | Default value |
        +========================+============+===============+
        | ``theta_start``        | ``float``  | ``0.0``       |
        +------------------------+------------+---------------+
        | ``delta_theta_0``      | ``float``  | ``1.0``       |
        +------------------------+------------+---------------+
        | ``delta_theta_min``    | ``float``  | ``0.01``      |
        +------------------------+------------+---------------+
        | ``homotopy_parameter`` | ``string`` | ``theta``     |
        +------------------------+------------+---------------+

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

        :returns: A dictionary of homotopy options.
        """

        return {
            "theta_start": 0.0,
            "delta_theta_0": 1.0,
            "delta_theta_min": 0.01,
            "homotopy_parameter": "theta",
        }

    def dynamic_parameters(self):
        dynamic_parameters = super().dynamic_parameters()

        if self.__theta > 0:
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
        variables = self.dae_variables["parameters"] + self.dae_variables["constant_inputs"]
        if canonical_name not in {v.name() for v in variables}:
            raise ValueError(
                f"Homotopy parameter '{name}' is not a parameter or fixed input of the model."
            )

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Pre-processing
        if preprocessing:
            self.pre()

        options = self.homotopy_options()
        self.__check_options(options)
        delta_theta = options["delta_theta_0"]

        # Homotopy loop
        self.__theta = options["theta_start"]

        while self.__theta <= 1.0:
            logger.info(f"Solving with homotopy parameter theta = {self.__theta}.")

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

                if self.__theta == 1.0:
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
