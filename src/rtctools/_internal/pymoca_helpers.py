import logging
from collections.abc import Mapping
from typing import Any

import pymoca.backends.casadi.api

from rtctools._internal.modelica_errors import raise_if_missing_msl

_logger = logging.getLogger("rtctools")


def load_pymoca_model(
    model_folder: str,
    model_name: str,
    compiler_options: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> Any:
    """Load a pymoca model with cache-retry logic and actionable MSL error messages.

    compiler_options is never mutated; a copy with cache=False is used for the retry.
    """
    if logger is None:
        logger = _logger
    try:
        return pymoca.backends.casadi.api.transfer_model(model_folder, model_name, compiler_options)
    except (RuntimeError, ModuleNotFoundError) as error:
        raise_if_missing_msl(error, model_folder)
        if not compiler_options.get("cache", False):
            raise
        compiler_options = {**compiler_options, "cache": False}
        logger.warning(f"Loading model {model_name} using a cache file failed: {error}.")
        logger.info(f"Compiling model {model_name}.")
        try:
            return pymoca.backends.casadi.api.transfer_model(
                model_folder, model_name, compiler_options
            )
        except (ModuleNotFoundError, RuntimeError) as retry_error:
            raise_if_missing_msl(retry_error, model_folder)
            raise
