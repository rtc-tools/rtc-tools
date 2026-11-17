import importlib.resources
import logging
from importlib import metadata as importlib_metadata

import casadi as ca
import pymoca
import pymoca.backends.casadi.api

from .caching import cached

logger = logging.getLogger("rtctools")


class _ModelicaMixinBase:
    """
    The Modelica front end shared by ``ModelicaMixin`` and ``SimulationProblem``: it compiles
    the model with `pymoca <http://github.com/pymoca/pymoca>`_ and groups its symbols.

    ``__init__`` loads the model and builds the variable groups and residuals. What differs per
    class goes through the hooks :py:meth:`_mo_extra_variable_groups`, :py:meth:`_mo_input_group`,
    :py:meth:`_mo_init_from_model` and :py:meth:`_mo_init_after_residuals`. ``super().__init__()``
    comes last, so that the rest of the chain finds the model data in place. Each class defines
    its own ``modelica_library_folders``.
    """

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

        model = self._mo_pymoca_model = self.pymoca_model(
            kwargs["model_folder"], model_name, self.compiler_options()
        )

        # Extract the CasADi MX variables used in the model
        mx = self._mo_mx = {}
        mx["time"] = [model.time]
        mx["states"] = [v.symbol for v in model.states]
        mx["derivatives"] = [v.symbol for v in model.der_states]
        mx["algebraics"] = [v.symbol for v in model.alg_states]
        mx["parameters"] = [v.symbol for v in model.parameters]
        mx.update(self._mo_extra_variable_groups())
        mx["constant_inputs"] = []
        mx["lookup_tables"] = []

        for v in model.inputs:
            if v.symbol.name() in model.delay_states:
                # Delayed feedback variables are local to each ensemble, and
                # therefore belong to the collection of algebraic variables,
                # rather than to the control inputs.
                mx["algebraics"].append(v.symbol)
            elif v.symbol.name() in kwargs.get("lookup_tables", []):
                mx["lookup_tables"].append(v.symbol)
            else:
                mx[self._mo_input_group(v)].append(v.symbol)

        # Set up what the class builds on the model before the residuals
        self._mo_init_from_model(**kwargs)

        # Initialize DAE and initial residuals
        variable_lists = ["states", "der_states", "alg_states", "inputs", "constants", "parameters"]
        function_arguments = [model.time] + [
            ca.veccat(*[v.symbol for v in getattr(model, variable_list)])
            for variable_list in variable_lists
        ]

        # The DAE can be empty, e.g. if the only (non-aliasing) equations are added in Python.
        self._mo_dae_residual = model.dae_residual_function(*function_arguments)
        if self._mo_dae_residual is None:
            self._mo_dae_residual = ca.MX()

        self._mo_initial_residual = model.initial_residual_function(*function_arguments)
        if self._mo_initial_residual is None:
            self._mo_initial_residual = ca.MX()

        # Set up what the class builds after the residuals, still before the rest of the chain
        self._mo_init_after_residuals(**kwargs)

        super().__init__(**kwargs)

    def _mo_extra_variable_groups(self) -> dict[str, list]:
        """
        Hook for variable groups of the subclass, placed after ``parameters`` in
        ``self._mo_mx``. Default: none.
        """
        return {}

    def _mo_input_group(self, variable: pymoca.backends.casadi.model.Variable) -> str:
        """
        Hook for the group of an input that is neither delayed feedback nor a lookup table.
        Default: ``"constant_inputs"``.
        """
        return "constant_inputs"

    def _mo_init_from_model(self, **kwargs) -> None:
        """
        Hook for the setup of the subclass that needs the model, called after the variable
        groups and before the residuals are built. Default: nothing.
        """
        pass

    def _mo_init_after_residuals(self, **kwargs) -> None:
        """
        Hook for the setup of the subclass after the residuals are built, called before
        ``super().__init__()``. Default: nothing.
        """
        pass

    def _mo_log_dae_variables(self, prefix: str) -> None:
        """
        Log the variable groups in debug mode.
        """
        if logger.getEffectiveLevel() == logging.DEBUG:
            for group in [
                "states",
                "derivatives",
                "algebraics",
                "control_inputs",
                "constant_inputs",
                "parameters",
            ]:
                if group in self._mo_mx:
                    logger.debug(
                        "{}: Found {} {}".format(
                            prefix,
                            group.replace("_", " "),
                            ", ".join([var.name() for var in self._mo_mx[group]]),
                        )
                    )

    def pymoca_model(
        self, model_folder: str, model_name: str, compiler_options: dict[str, str | bool]
    ) -> pymoca.backends.casadi.model.Model:
        """
        Load or compile the model named ``model_name`` from the Modelica files in
        ``model_folder``.

        Override this method to supply a model from elsewhere, e.g. one composed in memory.
        The model is requested once, from ``__init__``.

        :param model_folder: Folder containing the Modelica files.
        :param model_name: Name of the model class to compile.
        :param compiler_options: The options returned by :py:meth:`compiler_options`.

        :returns: A pymoca CasADi model.
        """
        logger.info(f"Loading/compiling model {model_name}.")
        try:
            return pymoca.backends.casadi.api.transfer_model(
                model_folder, model_name, compiler_options
            )
        except (RuntimeError, ModuleNotFoundError) as error:
            if not compiler_options.get("cache", False):
                raise error
            compiler_options["cache"] = False
            logger.warning(f"Loading model {model_name} using a cache file failed: {error}.")
            logger.info(f"Compiling model {model_name}.")
            return pymoca.backends.casadi.api.transfer_model(
                model_folder, model_name, compiler_options
            )

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

        # Cache the model on disk
        compiler_options["cache"] = True

        # Done
        return compiler_options

    @property
    @cached
    def alias_relation(self):
        return self._mo_pymoca_model.alias_relation
