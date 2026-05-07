import cProfile
import logging
import os
import pstats
import re
import sys

import casadi

from . import __version__
from ._internal.alias_tools import OrderedSet
from .data import pi
from .optimization.csv_mixin import CSVMixin as OptimizationCSVMixin
from .optimization.io_mixin import IOMixin as OptimizationIOMixin
from .optimization.pi_mixin import PIMixin as OptimizationPIMixin
from .simulation.csv_mixin import CSVMixin as SimulationCSVMixin
from .simulation.io_mixin import IOMixin as SimulationIOMixin
from .simulation.pi_mixin import PIMixin as SimulationPIMixin


def _resolve_folder(kwargs, base_folder, subfolder_kw, default):
    subfolder = kwargs.pop(subfolder_kw, default)
    if os.path.isabs(subfolder):
        return subfolder
    else:
        return os.path.join(base_folder, subfolder)


def _configure_extra_casadi_path(logger):
    """Prepend RTCTOOLS_EXTRA_CASADIPATH to CasADi's plugin search path."""
    if extra_casadi_path := os.getenv("RTCTOOLS_EXTRA_CASADIPATH", "").strip():
        current_path = casadi.GlobalOptions.getCasadiPath()
        current_parts = current_path.split(os.pathsep) if current_path else []
        seen = {os.path.normcase(os.path.normpath(p)) for p in current_parts}
        new_parts = []
        for p in extra_casadi_path.split(os.pathsep):
            norm = os.path.normcase(os.path.normpath(p))
            if norm not in seen:
                seen.add(norm)
                new_parts.append(p)
                if not os.path.isdir(norm):
                    logger.warning(
                        f"RTCTOOLS_EXTRA_CASADIPATH entry {p!r} does not exist"
                        " or is not a directory."
                    )
        if new_parts:
            combined = os.pathsep.join(new_parts + current_parts)
            logger.debug(f"Setting CasADi plugin search path to '{combined}'.")
            casadi.GlobalOptions.setCasadiPath(combined)


def run_optimization_problem(
    optimization_problem_class, base_folder="..", log_level=logging.INFO, profile=False, **kwargs
):
    """
    Sets up and solves an optimization problem.

    This function makes the following assumptions:

    1. That the ``base_folder`` contains subfolders ``input``, ``output``, and ``model``,
       containing input data, output data, and the model, respectively.
    2. When using :class:`.CSVLookupTableMixin`, that the base folder contains
       a subfolder ``lookup_tables``.
    3. When using :class:`.ModelicaMixin`, that the base folder contains
       a subfolder ``model``.
    4. When using :class:`.ModelicaMixin`, that the toplevel Modelica model name
       equals the class name.

    :param optimization_problem_class: Optimization problem class to solve.
    :param base_folder:                Base folder.
    :param log_level:                  The log level to use.
    :param profile:                    Whether or not to enable profiling.

    :returns: :class:`.OptimizationProblem` instance.
    """

    # Do some checks on the problem class
    if issubclass(optimization_problem_class, SimulationCSVMixin):
        raise ValueError(
            "Optimization problem class cannot be derived from simulation.csv_mixin.CSVMixin."
        )
    if issubclass(optimization_problem_class, SimulationIOMixin):
        raise ValueError(
            "Optimization problem class cannot be derived from simulation.csv_mixin.IOMixin."
        )
    if issubclass(optimization_problem_class, SimulationPIMixin):
        raise ValueError(
            "Optimization problem class cannot be derived from simulation.csv_mixin.PIMixin."
        )

    # Set input/output folders
    if not os.path.isabs(base_folder):
        # Resolve base folder relative to script folder
        base_folder = os.path.join(sys.path[0], base_folder)

    model_folder = _resolve_folder(kwargs, base_folder, "model_folder", "model")
    input_folder = _resolve_folder(kwargs, base_folder, "input_folder", "input")
    output_folder = _resolve_folder(kwargs, base_folder, "output_folder", "output")

    # Set up logging
    logger = logging.getLogger("rtctools")

    # Add stream handler if it does not already exist.
    if not logger.hasHandlers() and not any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Add pi.DiagHandler, if using PIMixin. Only add it if it does not already exist.
    if issubclass(optimization_problem_class, OptimizationPIMixin) and not any(
        isinstance(h, pi.DiagHandler) for h in logger.handlers
    ):
        handler = pi.DiagHandler(output_folder)
        logger.addHandler(handler)

    # Set log level
    logger.setLevel(log_level)

    # Log version info
    logger.info(f"Using RTC-Tools {__version__}.")
    logger.debug(f"Using CasADi {casadi.__version__}.")

    _configure_extra_casadi_path(logger)

    # Check for some common mistakes in inheritance order
    suggested_order = OrderedSet(
        [
            "HomotopyMixin",
            "MinAbsGoalProgrammingMixin",
            "LinearizedOrderGoalProgrammingMixin",
            "SinglePassGoalProgrammingMixin",
            "GoalProgrammingMixin",
            "PIMixin",
            "CSVMixin",
            "IOMixin",
            "ModelicaMixin",
            "PlanningMixin",
            "ControlTreeMixin",
            "CollocatedIntegratedOptimizationProblem",
            "OptimizationProblem",
        ]
    )
    base_names = OrderedSet([b.__name__ for b in optimization_problem_class.__bases__])
    if suggested_order & base_names != base_names & suggested_order:
        msg = (
            "Please inherit from base classes in the following order: "
            f"{list(base_names & suggested_order)}"
        )
        logger.error(msg)
        raise Exception(msg)

    # Run
    try:
        prob = optimization_problem_class(
            model_folder=model_folder,
            input_folder=input_folder,
            output_folder=output_folder,
            **kwargs,
        )
        if profile:
            filename = os.path.join(base_folder, "profile.prof")

            cProfile.runctx("prob.optimize()", globals(), locals(), filename)

            s = pstats.Stats(filename)
            s.strip_dirs().sort_stats("time").print_stats()
        else:
            prob.optimize()
        return prob
    except Exception as e:
        logger.error(str(e))
        if isinstance(e, TypeError):
            exc_info = sys.exc_info()
            value = exc_info[1]
            try:
                failed_class = re.search(
                    "Can't instantiate (.*) with abstract methods", str(value)
                ).group(1)
                abstract_method = re.search(" with abstract methods (.*)", str(value)).group(1)
                logger.error(
                    f"The {failed_class} is missing a mixin. Please add a mixin that instantiates "
                    f"abstract method {abstract_method}, so that the optimizer can run."
                )
            except Exception:
                pass
        for handler in logger.handlers:
            handler.flush()
        raise


def run_simulation_problem(
    simulation_problem_class, base_folder="..", log_level=logging.INFO, **kwargs
):
    """
    Sets up and runs a simulation problem.

    :param simulation_problem_class: Simulation problem class to solve.
    :param base_folder:              Folder within which subfolders "input", "output", and "model"
        exist, containing input and output data, and the model, respectively.
    :param log_level:                The log level to use.

    :returns: :class:`SimulationProblem` instance.
    """

    # Do some checks on the problem class
    if issubclass(simulation_problem_class, OptimizationCSVMixin):
        raise ValueError(
            "Simulation problem class cannot be derived from optimization.csv_mixin.CSVMixin."
        )
    if issubclass(simulation_problem_class, OptimizationIOMixin):
        raise ValueError(
            "Simulation problem class cannot be derived from optimization.csv_mixin.IOMixin."
        )
    if issubclass(simulation_problem_class, OptimizationPIMixin):
        raise ValueError(
            "Simulation problem class cannot be derived from optimization.csv_mixin.PIMixin."
        )

    # Set input/output folders
    if base_folder is None:
        # Check command line arguments
        if len(sys.argv) != 2:
            raise Exception(f"Usage: {sys.argv[0]} BASE_FOLDER")

        base_folder = sys.argv[1]
    else:
        if not os.path.isabs(base_folder):
            # Resolve base folder relative to script folder
            base_folder = os.path.join(sys.path[0], base_folder)

    model_folder = _resolve_folder(kwargs, base_folder, "model_folder", "model")
    input_folder = _resolve_folder(kwargs, base_folder, "input_folder", "input")
    output_folder = _resolve_folder(kwargs, base_folder, "output_folder", "output")

    # Set up logging
    logger = logging.getLogger("rtctools")
    if not logger.hasHandlers() and not any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Add pi.DiagHandler, if using PIMixin. Only add it if it does not already exist.
    if issubclass(simulation_problem_class, SimulationPIMixin) and not any(
        isinstance(h, pi.DiagHandler) for h in logger.handlers
    ):
        handler = pi.DiagHandler(output_folder)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    logger.info(f"Using RTC-Tools {__version__}")
    logger.debug(f"Using CasADi {casadi.__version__}.")

    _configure_extra_casadi_path(logger)

    # Run
    prob = simulation_problem_class(
        model_folder=model_folder, input_folder=input_folder, output_folder=output_folder, **kwargs
    )
    prob.simulate()
    return prob
