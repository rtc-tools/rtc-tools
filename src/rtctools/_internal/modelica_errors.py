import re

# Matches pymoca ClassNotFoundError messages in both known phrasings:
# - "Could not find class 'X'"  (pymoca >= 0.10)
# - "Class 'X' not found"       (alternative phrasing)
_MODELICA_PATTERN = re.compile(
    r"(?:[Cc]ould not find class|[Cc]lass)\s+'?(Modelica\.[^\s'\"]+)'?(?:\s+not found)?"
)
_SI_ALIAS_PATTERN = re.compile(
    r"(?:[Cc]ould not find class|[Cc]lass)\s+'?(SI\.[^\s'\"]+)'?(?:\s+not found)?"
)


def _extract_missing_class(msg: str, pattern: re.Pattern) -> str | None:
    match = pattern.search(msg)
    return match.group(1) if match else None


def _msl_error(model_folder: str, missing: str, detail: str, original: Exception) -> RuntimeError:
    return RuntimeError(
        f"Failed to load Modelica model from '{model_folder}'.\n"
        f"  Missing class: '{missing}'\n\n"
        f"{detail}\n\n"
        f"  Original error: {original}"
    )


def raise_if_missing_msl(error: Exception, model_folder: str) -> None:
    """Re-raise with an actionable message if the error matches a known MSL failure pattern."""
    msg = str(error)

    # Bug 1: Modelica.* class not in rtc-tools-standard-library
    missing = _extract_missing_class(msg, _MODELICA_PATTERN)
    if missing:
        pkg = ".".join(missing.split(".")[:2])
        if pkg == "Modelica.Units":
            raise _msl_error(
                model_folder,
                missing,
                f"  'Modelica.Units' is included in 'rtc-tools-standard-library',"
                f" but '{missing}' was not found.\n"
                f"  The class may not be covered by the installed version"
                f" of 'rtc-tools-standard-library'.\n"
                f"  You can also add the missing .mo files manually via"
                f" 'modelica_library_folders' in your problem class.",
                error,
            ) from error
        raise _msl_error(
            model_folder,
            missing,
            f"  '{pkg}' is not included in 'rtc-tools-standard-library'"
            f" (currently only Modelica.Units is supported).\n"
            f"  To resolve this, contribute the missing package to:\n"
            f"    https://github.com/rtc-tools/rtc-tools-standard-library\n"
            f"  Or add the .mo files manually via 'modelica_library_folders'"
            f" in your problem class.",
            error,
        ) from error

    # Bug 2: unresolved SI.* alias — assumes SI always means Modelica.Units.SI.
    missing = _extract_missing_class(msg, _SI_ALIAS_PATTERN)
    if missing:
        fq = missing.replace("SI.", "Modelica.Units.SI.", 1)
        raise _msl_error(
            model_folder,
            missing,
            f"  Known pymoca >= 0.11 regression: 'import SI = Modelica.Units.SI'"
            f" aliases are silently dropped.\n"
            f"  Replace '{missing}' with '{fq}' and remove the alias"
            f" in the library that defines it,\n"
            f"  or downgrade to rtc-tools 2.7.x (pymoca 0.9.2).",
            error,
        ) from error
