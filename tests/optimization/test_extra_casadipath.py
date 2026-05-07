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
        # assertLogs output format is "LEVEL:logger:message".
        self.assertTrue(
            any(call_args in r and r.startswith("DEBUG:") for r in cm.output),
            f"Expected combined path in DEBUG log output, got: {cm.output}",
        )

    @patch("rtctools.util.casadi")
    def test_empty_or_whitespace_env_var_does_nothing(self, mock_casadi):
        """Empty and whitespace-only values must be treated as absent and result in no action."""
        for value in ("", " "):
            with self.subTest(value=repr(value)):
                with patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": value}):
                    with self.assertNoLogs("rtctools", level=logging.INFO):
                        _configure_extra_casadi_path(logger)
                    mock_casadi.GlobalOptions.getCasadiPath.assert_not_called()
                    mock_casadi.GlobalOptions.setCasadiPath.assert_not_called()

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": "/custom/path"})
    @patch("rtctools.util.casadi")
    def test_no_separator_artefacts_when_existing_path_is_none(self, mock_casadi):
        """getCasadiPath() returning None must not introduce separator artefacts."""
        mock_casadi.GlobalOptions.getCasadiPath.return_value = None
        with self.assertLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)
        call_args = mock_casadi.GlobalOptions.setCasadiPath.call_args.args[0]
        self.assertEqual(call_args, "/custom/path")

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": "/custom/path"})
    @patch("rtctools.util.casadi")
    def test_no_separator_artefacts_when_existing_path_is_empty(self, mock_casadi):
        """getCasadiPath() returning "" must not introduce separator artefacts."""
        mock_casadi.GlobalOptions.getCasadiPath.return_value = ""
        with self.assertLogs("rtctools", level=logging.DEBUG):
            _configure_extra_casadi_path(logger)
        call_args = mock_casadi.GlobalOptions.setCasadiPath.call_args.args[0]
        self.assertEqual(call_args, "/custom/path")

    @patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": "/custom/path"})
    @patch("rtctools.util.casadi")
    def test_already_in_existing_path_is_not_prepended(self, mock_casadi):
        """A path already present in the CasADi path must not be prepended again."""
        mock_casadi.GlobalOptions.getCasadiPath.return_value = "/custom/path"

        with self.assertNoLogs("rtctools", level=logging.INFO):
            _configure_extra_casadi_path(logger)

        mock_casadi.GlobalOptions.setCasadiPath.assert_not_called()

    @patch("rtctools.util.casadi")
    def test_nonexistent_path_emits_warning(self, mock_casadi):
        """A path that does not exist on disk must trigger a warning."""
        mock_casadi.GlobalOptions.getCasadiPath.return_value = ""
        nonexistent = "/does/not/exist/on/any/machine"
        with patch.dict(os.environ, {"RTCTOOLS_EXTRA_CASADIPATH": nonexistent}):
            with self.assertLogs("rtctools", level=logging.WARNING) as cm:
                _configure_extra_casadi_path(logger)
        self.assertTrue(
            any(nonexistent in r and r.startswith("WARNING:") for r in cm.output),
            f"Expected WARNING about nonexistent path, got: {cm.output}",
        )
