import unittest

from rtctools._internal.modelica_errors import raise_if_missing_msl


class TestRaiseIfMissingMsl(unittest.TestCase):
    def _call(self, msg):
        raise_if_missing_msl(RuntimeError(msg), "/some/model/folder")

    # Both pymoca phrasings should reach the same branch and carry the same payload.
    def _assert_non_units_package(self, msg):
        with self.assertRaises(RuntimeError) as ctx:
            self._call(msg)
        out = str(ctx.exception)
        self.assertIn("Modelica.SIunits.Velocity", out)
        self.assertIn("is not included in", out)
        self.assertIn("rtc-tools-standard-library", out)
        self.assertIn("https://github.com/rtc-tools/rtc-tools-standard-library", out)

    def test_missing_non_units_package_could_not_find_phrasing(self):
        self._assert_non_units_package("Could not find class 'Modelica.SIunits.Velocity'")

    def test_missing_non_units_package_class_not_found_phrasing(self):
        self._assert_non_units_package("Class 'Modelica.SIunits.Velocity' not found")

    def test_missing_modelica_units_class_raises_version_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._call("Could not find class 'Modelica.Units.SI.UnknownType'")
        out = str(ctx.exception)
        self.assertIn("Modelica.Units.SI.UnknownType", out)
        self.assertIn("Modelica.Units", out)
        self.assertNotIn("is not included in", out)

    def test_si_alias_regression_raises_with_expanded_name(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._call("Could not find class 'SI.VolumeFlowRate'")
        out = str(ctx.exception)
        self.assertIn("SI.VolumeFlowRate", out)
        self.assertIn("Modelica.Units.SI.VolumeFlowRate", out)
        self.assertIn("regression", out)

    def test_non_matching_message_does_not_raise(self):
        self._call("Simulation failed due to numerical instability")
