import re


def raise_if_missing_msl(error: Exception, model_folder: str) -> None:
    """
    If the pymoca error indicates a missing Modelica Standard Library class,
    re-raise with a clear, actionable message.
    """
    match = re.search(r"[Cc]lass (Modelica\.[^\s'\"]+) not found", str(error))
    if match:
        missing = match.group(1)
        pkg = ".".join(missing.split(".")[:2])
        raise RuntimeError(
            f"Failed to load Modelica model from '{model_folder}'.\n"
            f"  Missing class: '{missing}'\n\n"
            f"  '{pkg}' is not included in 'rtc-tools-standard-library' "
            f"(currently only Modelica.Units is supported).\n\n"
            f"  To resolve this:\n"
            f"  1. Run: rtc-tools-migrate-model <path/to/model/folder>\n"
            f"     to check your model for known compatibility issues.\n"
            f"  2. Contribute the missing package to:\n"
            f"     https://github.com/rtc-tools/rtc-tools-standard-library\n"
            f"  3. Or add the .mo files manually via 'modelica_library_folders' "
            f"in your problem class.\n\n"
            f"  Original error: {error}"
        ) from error
