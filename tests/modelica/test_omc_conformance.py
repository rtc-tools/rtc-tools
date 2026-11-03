"""
Check the example models with the OpenModelica compiler (omc).

pymoca, which RTC-Tools uses to compile Modelica, accepts some models that are
not valid Modelica (issue #1873). This test runs omc on every example model:

- checkModel: is the model valid Modelica?
- translateModel: can omc turn the model into a system of equations?
- messages: which errors and warnings does omc report?

EXAMPLE_MODELS lists every example model and its known failures, as regular
expressions of the messages omc reports. The test fails when omc reports an
error or warning that is not listed, and, as a strict expected failure (xfail),
while a listed message is still reported. Once a failure is fixed, its message
disappears, pytest reports XPASS(strict) as a failure, and you remove the
failure from EXAMPLE_MODELS. So every fix shows up as a change to EXAMPLE_MODELS
below.

The known failures and the message texts depend on the omc version. CI declares
that version in .github/workflows/ci.yml and passes it on in the OMC_VERSION
environment variable. The test only runs when omc of that version is on the
PATH, and is skipped otherwise. It also requires the Modelica Standard Library.
To install the library, run:

    omc tests/modelica/install_msl.mos
"""

import functools
import importlib.metadata
import importlib.resources
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest


class Model:
    """An example model and its known failures.

    path: "<folder in examples/>/<name of the model class>"
    fails_check: the errors checkModel reports
    fails_translate: the errors translateModel reports
    warns: the warnings omc reports

    Each takes a regular expression or a tuple of them.
    """

    def __init__(self, path, *, fails_check=(), fails_translate=(), warns=()):
        self.path = path
        self.example, self.name = path.split("/")
        self.fails_check = as_tuple(fails_check)
        self.fails_translate = as_tuple(fails_translate)
        self.warns = as_tuple(warns)

    @property
    def known_messages(self):
        return self.fails_check + self.fails_translate + self.warns


def as_tuple(value):
    return (value,) if isinstance(value, str) else tuple(value)


# Known failures shared by several models, named after the argument they go in.

# Deltares.ChannelFlow calls max() with two arrays (rtc-tools-channel-flow).
CHECK_DELTARES_MAX = r"No matching function found for max\(\S+\.H_b\["
# The model has fewer equations than unknowns, for example because top-level
# inputs are bound in their declaration.
TRANSLATE_UNDER_DETERMINED = r"Too few equations, under-determined system"
WARNING_NO_EQUATION_LEFT = r"Variable \S+ does not have any remaining equation"

# Every example model and its known failures.
EXAMPLE_MODELS = [
    Model("basic/Example"),
    Model("cascading_channels/Example", fails_check=CHECK_DELTARES_MAX),
    Model("channel_pulse/Example", fails_check=CHECK_DELTARES_MAX),
    Model("channel_wave_damping/ExampleOptimization", fails_check=CHECK_DELTARES_MAX),
    Model("channel_wave_damping/ExampleLocalControl", fails_check=CHECK_DELTARES_MAX),
    Model("ensemble/Example"),
    Model(
        "goal_programming/Example",
        fails_translate=TRANSLATE_UNDER_DETERMINED,
        warns=WARNING_NO_EQUATION_LEFT,
    ),
    Model("integrator_delay/Example"),
    Model("lookup_table/Example"),
    Model(
        "mixed_integer/Example",
        fails_translate=TRANSLATE_UNDER_DETERMINED,
        warns=WARNING_NO_EQUATION_LEFT,
    ),
    # Under-determined by design: Turbine_is_on is a decision variable, constrained in Python.
    Model(
        "pumped_hydropower_system/PumpedStoragePlant",
        fails_translate=TRANSLATE_UNDER_DETERMINED,
        warns=WARNING_NO_EQUATION_LEFT,
    ),
    Model("simulation/Example"),
    # Under-determined by design: the equation for y is implemented in Python.
    Model(
        "simulation_with_custom_equations/SimpleModel",
        fails_translate=TRANSLATE_UNDER_DETERMINED,
        warns=WARNING_NO_EQUATION_LEFT,
    ),
    Model("single_reservoir/SingleReservoir"),
    Model("step_response/Example", fails_check=CHECK_DELTARES_MAX),
]

# Warnings that are accepted for all models.
ALLOWED_WARNINGS = [
    # RTC-Tools sets the initial states from timeseries data, not in the model.
    r"The initial conditions are not fully specified",
    # Raised by the Deltares library.
    r"The model contains alias variables with redundant start and/or conflicting nominal values",
    # Parameters such as theta and step_size are set by RTC-Tools at run time.
    r"Parameter \S+ has no value, and is fixed during initialization",
]

EXAMPLES_PATH = Path(__file__).resolve().parents[2] / "examples"

# The omc version the known failures are recorded for, declared in ci.yml.
OMC_VERSION = os.environ.get("OMC_VERSION")

OMC = shutil.which("omc")


def installed_omc_version():
    """Return the version of omc on the PATH, or None if there is none."""
    if OMC is None:
        return None
    version = subprocess.run([OMC, "--version"], capture_output=True, text=True).stdout
    return version.strip().removeprefix("OpenModelica ")


def skip_reason():
    if OMC_VERSION is None:
        return "OMC_VERSION is not set, see .github/workflows/ci.yml"
    if installed_omc_version() != OMC_VERSION:
        return f"requires omc {OMC_VERSION}, found {installed_omc_version()}"
    return None


pytestmark = pytest.mark.skipif(skip_reason() is not None, reason=str(skip_reason()))


@dataclass
class OmcResult:
    check_ok: bool
    translate_ok: bool
    messages: list[str]  # first line of each error and warning, without duplicates
    log: str  # all output of omc, for failure messages


@functools.cache
def run_omc(model):
    """Run checkModel and translateModel on a model, once per test session."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "run.mos").write_text(omc_script(model))
        subprocess.run([OMC, "run.mos"], cwd=tmp, check=True)
        check = Path(tmp, "check.txt").read_text()
        translate = Path(tmp, "translate.txt").read_text()

    messages = re.findall(r"(?:Error|Warning): (.*)", check + translate)
    return OmcResult(
        check_ok=check.startswith(f"Check of {model.name} completed successfully"),
        translate_ok=translate.startswith("true"),
        messages=list(dict.fromkeys(messages)),
        log=f"checkModel:\n{check}\ntranslateModel:\n{translate}",
    )


def omc_script(model):
    """Return an omc script that writes the result and messages of each step to a file."""
    library_path = os.pathsep.join(modelica_library_folders())
    model_files = sorted((EXAMPLES_PATH / model.example / "model").glob("*.mo"))
    # Paths are written with forward slashes: backslashes escape in Modelica strings.
    load_files = "\n".join(f'loadFile("{f.as_posix()}");' for f in model_files)

    # The Modelica version must match install_msl.mos.
    # getErrorString() returns the messages since its last call. Assign it to a
    # variable first: using it in a larger expression does not work after
    # translateModel fails.
    return f"""
        echo(false);
        setModelicaPath("{library_path}{os.pathsep}" + getModelicaPath());
        loadModel(Modelica, {{"4.0.0"}});
        {load_files}

        result := checkModel({model.name});
        messages := getErrorString();
        writeFile("check.txt", result + "\\n" + messages);

        result := String(translateModel({model.name}));
        messages := getErrorString();
        writeFile("translate.txt", result + "\\n" + messages);
    """


def modelica_library_folders():
    """Return the Modelica library folders of installed packages, like RTC-Tools does."""
    return [
        Path(str(importlib.resources.files(ep.module).joinpath(ep.attr))).as_posix()
        for ep in importlib.metadata.entry_points(group="rtctools.libraries.modelica")
        if ep.name == "library_folder"
    ]


def each_model(known_failures=lambda model: ()):
    """Parametrize a test over all models, as an expected failure where one is known."""
    params = []
    for model in EXAMPLE_MODELS:
        reasons = known_failures(model)
        marks = [pytest.mark.xfail(reason="; ".join(reasons), strict=True)] if reasons else []
        params.append(pytest.param(model, id=model.path, marks=marks))
    return pytest.mark.parametrize("model", params)


@each_model(known_failures=lambda model: model.fails_check)
def test_check_model(model):
    result = run_omc(model)
    assert result.check_ok, result.log


@each_model(known_failures=lambda model: model.fails_check + model.fails_translate)
def test_translate_model(model):
    result = run_omc(model)
    assert result.translate_ok, result.log


@each_model()
def test_no_unknown_messages(model):
    """Fail on any error or warning that is not allowed for all models or known for this model."""
    result = run_omc(model)
    known = ALLOWED_WARNINGS + list(model.known_messages)
    unknown = [m for m in result.messages if not any(re.search(k, m) for k in known)]
    assert not unknown, "Unknown messages:\n" + "\n".join(unknown)


@pytest.mark.parametrize(
    ("model", "pattern"),
    [
        pytest.param(
            model,
            pattern,
            id=f"{model.path}: {pattern}",
            marks=pytest.mark.xfail(reason=f"omc reports: {pattern}", strict=True),
        )
        for model in EXAMPLE_MODELS
        for pattern in model.known_messages
    ],
)
def test_known_message(model, pattern):
    """Expected to fail while omc still reports the known error or warning."""
    result = run_omc(model)
    reported = [m for m in result.messages if re.search(pattern, m)]
    assert not reported, "omc reports:\n" + "\n".join(reported)
