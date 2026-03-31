import logging
import os
from unittest.mock import patch

from rtctools.util import _configure_extra_casadi_path

from ..test_case import TestCase

logger = logging.getLogger("rtctools")


class TestExtraCasadiPath(TestCase):
    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": "/custom/path"})
    @patch("rtctools.util.casadi")
    def test_prepends_to_existing_path(self, mock_casadi):
        mock_casadi.GlobalOptions.getCasadiPath.return_value = "/casadi/default"

        with self.assertLogs("rtctools", level=logging.DEBUG) as cm:
            _configure_extra_casadi_path(logger)

        call_args = mock_casadi.GlobalOptions.setCasadiPath.call_args.args[0]
        self.assertEqual(call_args.split(os.pathsep), ["/custom/path", "/casadi/default"])
        # The DEBUG log must mention the resulting combined path.
        # assertLogs output format is "LEVEL:logger:message".
        self.assertTrue(
            any(call_args in r and r.startswith("DEBUG:") for r in cm.output),
            f"Expected combined path in DEBUG log output, got: {cm.output}",
        )

    @patch("rtctools.util.casadi")
    def test_absent_env_var_does_nothing(self, mock_casadi):
        os.environ.pop("RTCTOOLS_EXTRA_CASADIPATH", None)
        with self.assertNoLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)
        mock_casadi.GlobalOptions.getCasadiPath.assert_not_called()
        mock_casadi.GlobalOptions.setCasadiPath.assert_not_called()

    @patch("rtctools.util.casadi")
    def test_empty_or_whitespace_env_var_does_nothing(self, mock_casadi):
        """Empty and whitespace-only values must both be treated as absent."""
        for value in ("", " "):
            mock_casadi.reset_mock()
            with patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": value}):
                with self.assertNoLogs("rtctools", level=logging.DEBUG):
                    _configure_extra_casadi_path(logger)
            mock_casadi.GlobalOptions.getCasadiPath.assert_not_called()
            mock_casadi.GlobalOptions.setCasadiPath.assert_not_called()

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": "/custom/path"})
    @patch("rtctools.util.casadi")
    def test_no_separator_artefacts_when_existing_path_absent(self, mock_casadi):
        """getCasadiPath() returning None or "" must not introduce separator artefacts."""
        for return_value in (None, ""):
            mock_casadi.reset_mock()
            mock_casadi.GlobalOptions.getCasadiPath.return_value = return_value

            with self.assertLogs("rtctools", level=logging.DEBUG):
                _configure_extra_casadi_path(logger)

            call_args = mock_casadi.GlobalOptions.setCasadiPath.call_args.args[0]
            self.assertEqual(
                call_args, "/custom/path", f"Failed for getCasadiPath()={return_value!r}"
            )

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": "/custom/path"})
    @patch("rtctools.util.casadi")
    def test_already_in_existing_path_is_not_prepended(self, mock_casadi):
        """A path already present in the CasADi path must not be prepended again."""
        mock_casadi.GlobalOptions.getCasadiPath.return_value = "/custom/path"

        # No new paths to add, so setCasadiPath must not be called and no log emitted.
        with self.assertNoLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)

        mock_casadi.GlobalOptions.setCasadiPath.assert_not_called()

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": f"/path/one{os.pathsep}/path/two"})
    @patch("rtctools.util.casadi")
    def test_multi_path_env_var_prepends_all(self, mock_casadi):
        """Multiple paths in the env var are all prepended in order."""
        mock_casadi.GlobalOptions.getCasadiPath.return_value = "/casadi/default"

        with self.assertLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)

        call_args = mock_casadi.GlobalOptions.setCasadiPath.call_args.args[0]
        self.assertEqual(call_args.split(os.pathsep), ["/path/one", "/path/two", "/casadi/default"])

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": f"/path/one{os.pathsep}/path/two"})
    @patch("rtctools.util.casadi")
    def test_multi_path_env_var_is_idempotent(self, mock_casadi):
        """Calling twice with the same paths must not produce duplicates.

        On the second call all entries are already in the CasADi path, so
        setCasadiPath must not be called again and no log must be emitted.
        """
        mock_casadi.GlobalOptions.getCasadiPath.return_value = "/casadi/default"

        with self.assertLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)

        first_call_args = mock_casadi.GlobalOptions.setCasadiPath.call_args.args[0]
        mock_casadi.GlobalOptions.getCasadiPath.return_value = first_call_args

        with self.assertNoLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)

        mock_casadi.GlobalOptions.setCasadiPath.assert_called_once()

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": f"/path/one{os.pathsep}/path/one/"})
    @patch("rtctools.util.casadi")
    def test_trailing_slash_deduplication(self, mock_casadi):
        """Paths that differ only by a trailing slash must be treated as duplicates.

        The first occurrence is kept; the second (with trailing slash) is dropped.
        """
        mock_casadi.GlobalOptions.getCasadiPath.return_value = ""

        with self.assertLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)

        call_args = mock_casadi.GlobalOptions.setCasadiPath.call_args.args[0]
        parts = call_args.split(os.pathsep)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], "/path/one")
