import unittest

from rtctools._internal.casadi_to_lp import _sanitize_var_names
from rtctools._internal.lp_names import _lp_safe_name, _lp_unique_safe_names


class TestLPNames(unittest.TestCase):
    def test_lp_safe_name(self):
        changed = [
            "inflow",
            "Inf_x",
            "info",
            "Bounds",
            "st",
            "st.",
            "max",
            "End",
            "e",
            "E",
            "ee",
            "e5__t0",
            "E10",
            "ee3",
            "eex",
            "eEx",
            "eels",
            "e5x",
            "1x",
            ".x",
        ]
        unchanged = [
            "storage",
            "max_level",
            "max__t0",
            "bounds_x",
            "e__t0",
            "elevation",
            "example",
            "ex5",
            "e_5",
            "x1",
            "minimise",
            "sos1",
            "_inflow",
        ]
        for name in changed:
            with self.subTest(name=name):
                self.assertEqual(_lp_safe_name(name), "_" + name)
        for name in unchanged:
            with self.subTest(name=name):
                self.assertEqual(_lp_safe_name(name), name)

    def test_unique_safe_names(self):
        cases = [
            ("no_change", ["a", "b"], ["a", "b"]),
            ("many_duplicates", ["a"] * 3 + ["a__d2"], ["a", "a__d1", "a__d3", "a__d2"]),
            ("duplicates", ["a", "a", "a"], ["a", "a__d1", "a__d2"]),
            ("renamed_yields_to_existing", ["inflow", "_inflow"], ["_inflow__d1", "_inflow"]),
            ("suffix_taken", ["a", "a", "a__d1"], ["a", "a__d2", "a__d1"]),
        ]
        for case_name, names, expected in cases:
            with self.subTest(case=case_name):
                self.assertEqual(_lp_unique_safe_names(names), expected)

    def test_sanitize_var_names_renames_keywords_not_indexed_names(self):
        names = _sanitize_var_names([{"max": [0], "x": [1, 2], "max_x": [3, 4]}], 5)
        self.assertEqual(names, ["_max", "x__t0", "x__t1", "max_x__t0", "max_x__t1"])
        names = _sanitize_var_names([{"max": [0, 1]}], 2)
        self.assertEqual(names, ["max__t0", "max__t1"])

    def test_sanitize_var_names_are_unique(self):
        # Examples of #1860
        cases = [
            ([{"u": [0, 1], "u__t0": [2]}], 3),
            ([{"a[1]": [0], "a_I1I_": [1]}], 2),
            ([{"u": [0, 1], "u__t0": [3]}, {"u": [0, 2], "u__t0": [3]}], 4),
            ([{"inflow": [0], "_inflow": [1]}], 2),
        ]
        for indices, n in cases:
            with self.subTest(indices=indices):
                names = _sanitize_var_names(indices, n)
                self.assertEqual(len(set(names)), n, msg=names)

    def test_sanitize_var_names_avoids_zero_helper_variable(self):
        names = _sanitize_var_names([{"__zero": [0]}], 1)
        self.assertNotIn("__zero", names)
