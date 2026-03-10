import importlib.resources
import logging
import os
import shutil
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import rtctools

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rtctools")
logger.setLevel(logging.INFO)


def copy_libraries(*args):
    if not args:
        args = sys.argv[1:]

    if not args:
        path = input("Folder to put the Modelica libraries: [.] ") or "."
    else:
        path = args[0]

    if not os.path.exists(path):
        sys.exit(f"Folder '{path}' does not exist")

    def _copytree(src, dst, symlinks=False, ignore=None):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                _copytree(s, d, symlinks, ignore)
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d)
                elif Path(s).name.lower() == "package.mo":
                    # Pick the largest one, assuming that all plugin packages
                    # to not provide a meaningful package.mo
                    if os.stat(s).st_size > os.stat(d).st_size:
                        logger.warning(f"Overwriting '{d}' with '{s}' as the latter is larger.")
                        os.remove(d)
                        shutil.copy2(s, d)
                    else:
                        logger.warning(f"Not copying '{s}' to '{d}' as the latter is larger.")
                else:
                    raise OSError("Could not combine two folders")

    dst = Path(path)

    library_folders = []

    for ep in importlib_metadata.entry_points(group="rtctools.libraries.modelica"):
        if ep.name == "library_folder":
            library_folders.append(Path(importlib.resources.files(ep.module).joinpath(ep.attr)))

    tlds = {}
    for lf in library_folders:
        for x in lf.iterdir():
            if x.is_dir():
                tlds.setdefault(x.name, []).append(x)

    for tld, paths in tlds.items():
        if Path(tld).exists():
            sys.exit(f"Library with name '{tld}'' already exists")

        try:
            for p in paths:
                _copytree(p, dst / p.name)
        except OSError:
            sys.exit(f"Failed merging the libraries in package '{tld}'")

    sys.exit(f"Succesfully copied all library folders to '{dst.resolve()}'")


def download_examples(*args):
    if not args:
        args = sys.argv[1:]

    if not args:
        path = input("Folder to download the examples to: [.] ") or "."
    else:
        path = args[0]

    if not os.path.exists(path):
        sys.exit(f"Folder '{path}' does not exist")

    path = Path(path)

    import urllib.request
    from urllib.error import HTTPError
    from zipfile import ZipFile

    version = rtctools.__version__
    try:
        url = f"https://github.com/rtc-tools/rtc-tools/zipball/{version}"

        opener = urllib.request.build_opener()
        urllib.request.install_opener(opener)
        # The security warning can be dismissed as the url variable is hardcoded to a remote.
        local_filename, _ = urllib.request.urlretrieve(url)  # nosec
    except HTTPError:
        sys.exit(f"Could not found examples for RTC-Tools version {version}.")

    with ZipFile(local_filename, "r") as z:
        target = path / "rtc-tools-examples"
        zip_folder_name = next(x for x in z.namelist() if x.startswith("Deltares-rtc-tools-"))
        prefix = "{}/examples/".format(zip_folder_name.rstrip("/"))
        members = [x for x in z.namelist() if x.startswith(prefix)]
        z.extractall(members=members)
        shutil.move(prefix, target)
        shutil.rmtree(zip_folder_name)

        sys.exit(f"Succesfully downloaded the RTC-Tools examples to '{target.resolve()}'")

    try:
        os.remove(local_filename)
    except OSError:
        pass


def migrate_model(*args):
    """
    Scan Modelica model files for known compatibility issues with pymoca >= 0.11
    and report (or fix) them.

    Usage:
        rtc-tools-migrate-model <model_folder> [--fix]

    Checks performed:
      - Use of deprecated 'SI.*' shorthand (replaced by 'Modelica.Units.SI.*' in pymoca >= 0.10)
      - References to Modelica Standard Library packages not yet in rtc-tools-standard-library

    With --fix, safe automatic rewrites (SI shorthand) are applied in-place.
    """
    import re

    if not args:
        args = sys.argv[1:]

    fix_mode = "--fix" in args
    paths = [a for a in args if not a.startswith("--")]

    if not paths:
        print("Usage: rtc-tools-migrate-model <model_folder> [--fix]")
        sys.exit(1)

    model_folder = Path(paths[0])
    if not model_folder.exists():
        sys.exit(f"Folder '{model_folder}' does not exist.")

    mo_files = list(model_folder.rglob("*.mo"))
    if not mo_files:
        sys.exit(f"No .mo files found in '{model_folder}'.")

    # MSL packages currently covered by rtc-tools-standard-library
    _SUPPORTED = {"Modelica.Units"}

    # Patterns
    _si_re = re.compile(r"\bSI\.([A-Za-z]\w*)")
    _msl_re = re.compile(r"\b(Modelica\.[A-Za-z][A-Za-z0-9_.]*)")

    print(f"Scanning {len(mo_files)} file(s) in '{model_folder}'...\n")
    any_issues = False

    for mo_file in sorted(mo_files):
        content = mo_file.read_text(encoding="utf-8")
        issues = []
        new_content = content

        # Check 1: deprecated SI.* shorthand
        if _si_re.search(content):
            example = _si_re.search(content).group(0)
            issues.append(
                f"  - Uses deprecated '{example}' shorthand. "
                f"Replace with 'Modelica.Units.SI.*' (pymoca >= 0.10 no longer treats 'SI' as a builtin)."
            )
            if fix_mode:
                new_content = _si_re.sub(r"Modelica.Units.SI.\1", new_content)

        # Check 2: MSL packages not covered by rtc-tools-standard-library
        used_pkgs = {".".join(m.split(".")[:2]) for m in _msl_re.findall(content)}
        missing_pkgs = sorted(used_pkgs - _SUPPORTED)
        for pkg in missing_pkgs:
            issues.append(
                f"  - References '{pkg}', which is not yet in rtc-tools-standard-library."
            )

        if issues:
            any_issues = True
            rel = mo_file.relative_to(model_folder)
            print(f"{rel}:")
            for issue in issues:
                print(issue)
            print()

        if fix_mode and new_content != content:
            mo_file.write_text(new_content, encoding="utf-8")
            print(f"  [fixed] {mo_file.name}\n")

    if not any_issues:
        print("No compatibility issues found.")
    elif not fix_mode:
        print(
            "Run with '--fix' to automatically apply safe rewrites (SI shorthand substitution).\n"
            "For missing MSL packages, contribute to: "
            "https://github.com/rtc-tools/rtc-tools-standard-library"
        )
