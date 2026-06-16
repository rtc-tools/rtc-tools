import importlib.resources
import logging
import os
import re
import shutil
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import rtctools


def _zipball_version(version: str) -> str:
    """Return the nearest GitHub zipball version for an RTC-Tools version string."""

    version = version.split("+", 1)[0]

    match = re.match(
        r"^(?P<base>\d+(?:\.\d+)*)(?:(?P<pre>a|b|rc)(?P<pre_num>\d+))?(?:\.post\d+)?(?:\.dev\d+)?$",
        version,
    )
    if not match:
        return version

    release_version = match.group("base")
    pre = match.group("pre")
    if pre is not None:
        release_version += f"{pre}{match.group('pre_num')}"

    return release_version


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

    sys.exit(f"Successfully copied all library folders to '{dst.resolve()}'")


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
    release_version = _zipball_version(version)
    if release_version != version:
        logger.info(
            f"Using RTC-Tools version {release_version} "
            f"(resolved from {version}) to download examples."
        )
    else:
        logger.info(f"Using RTC-Tools version {release_version} to download examples.")

    local_filename = None
    try:
        url = f"https://github.com/rtc-tools/rtc-tools/zipball/{release_version}"

        opener = urllib.request.build_opener()
        urllib.request.install_opener(opener)
        # The security warning can be dismissed as the url variable is hardcoded to a remote.
        local_filename, _ = urllib.request.urlretrieve(url)  # nosec
    except HTTPError:
        sys.exit(f"Could not find examples for RTC-Tools version {release_version}.")

    try:
        with ZipFile(local_filename, "r") as z:
            import tempfile

            target = path / "rtc-tools-examples"
            zip_folder_name = next(
                (name.split("/", 1)[0] for name in z.namelist() if name and "/" in name),
                None,
            )
            if zip_folder_name is None:
                sys.exit("Downloaded archive is malformed or empty.")

            prefix = f"{zip_folder_name}/examples/"
            members = [x for x in z.namelist() if x.startswith(prefix)]

            with tempfile.TemporaryDirectory(dir=path) as extract_dir:
                extract_dir = Path(extract_dir)
                z.extractall(path=extract_dir, members=members)
                shutil.move(str(extract_dir / zip_folder_name / "examples"), target)

                sys.exit(f"Successfully downloaded the RTC-Tools examples to '{target.resolve()}'")
    finally:
        if local_filename is not None:
            try:
                os.remove(local_filename)
            except OSError:
                pass
