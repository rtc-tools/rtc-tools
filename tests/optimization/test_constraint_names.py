import numpy as np

from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.single_pass_goal_programming_mixin import (
    SinglePassGoalProgrammingMixin,
    SinglePassMethod,
)

from ..test_case import TestCase
from .test_vector_constraints_variables import Model


class UnnamedModel(Model):
    def path_constraints(self, ensemble_member):
        return [(self.state("u"), -2.0, 2.0), (self.state("x"), -10.0, 10.0)]


class NamedModel(UnnamedModel):
    named_constraints = True

    def constraints(self, ensemble_member):
        first, *rest = super().constraints(ensemble_member)
        return [(*first, "x_at_half"), *rest]

    def path_constraints(self, ensemble_member):
        first, *rest = super().path_constraints(ensemble_member)
        return [(*first, "u_range"), *rest]


class NameWithoutFlagModel(NamedModel):
    named_constraints = False


class NonStringNameModel(NamedModel):
    def constraints(self, ensemble_member):
        return [(*c[:3], 1) for c in super().constraints(ensemble_member)]


class TestConstraintTuples(TestCase):
    def test_names_do_not_change_the_problem(self):
        unnamed = UnnamedModel()
        self.assertTrue(unnamed.optimize())
        named = NamedModel()
        self.assertTrue(named.optimize())
        self.assertEqual(named.objective_value, unnamed.objective_value)

    def test_name_requires_flag(self):
        with self.assertRaisesRegex(TypeError, "named_constraints is True"):
            NameWithoutFlagModel().optimize()

    def test_name_must_be_a_string(self):
        with self.assertRaisesRegex(TypeError, "must be a str or None, not int"):
            NonStringNameModel().optimize()


class LabelledModel(NamedModel):
    track_constraint_names = True

    def constraints(self, ensemble_member):
        vector = (self.states_in("x", 0.8, 0.9, ensemble_member=ensemble_member), -5.0, 5.0)
        return [*super().constraints(ensemble_member), vector]


class LabelledEnsembleModel(LabelledModel):
    @property
    def ensemble_size(self):
        return 2


class _XGoal(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 1.0, ensemble_member=ensemble_member)

    function_range = (-10.0, 10.0)


class _XMaxGoal(_XGoal):
    priority = 1
    target_max = 0.5


class _XMinGoal(_XGoal):
    priority = 2
    target_min = 1.5


class LabelledSinglePassModel(SinglePassGoalProgrammingMixin, LabelledModel):
    def goals(self):
        return [_XMaxGoal(), _XMinGoal()]


class TestConstraintLabels(TestCase):
    def labels_and_bounds(self, model_class):
        problem = model_class()
        problem.optimize()
        transcribed = problem.transcribed_problem
        labels = transcribed["g_names"]
        self.assertEqual(len(labels), transcribed["nlp"]["g"].size1())
        lbg = np.concatenate(
            [np.atleast_1d(np.asarray(b, dtype=float)) for b in transcribed["lbg"]]
        )
        return labels, dict(zip(labels, lbg, strict=True))

    def test_labels_name_every_row(self):
        labels, lbg = self.labels_and_bounds(LabelledModel)
        self.assertEqual(len(set(labels)), len(labels))
        self.assertEqual(lbg["x_at_half"], 1.0)
        self.assertEqual(lbg["constraint_2"], -np.inf)
        self.assertEqual(lbg["constraint_3__0"], -5.0)
        self.assertEqual(lbg["u_range__t0"], -2.0)
        self.assertEqual(lbg["path_constraint_1__t20"], -10.0)
        self.assertIn("initial_residual_0", labels)

    def test_ensemble_members_get_a_suffix(self):
        labels, _ = self.labels_and_bounds(LabelledEnsembleModel)
        self.assertEqual(len(set(labels)), len(labels))
        self.assertIn("x_at_half__m1", labels)
        self.assertIn("initial_residual_0__m1", labels)
        self.assertIn("u_range__t3__m0", labels)

    def test_tracking_is_off_by_default(self):
        problem = UnnamedModel()
        problem.optimize()
        self.assertIsNone(problem.transcribed_problem["g_names"])

    def test_mismatched_labels_are_dropped(self):
        class Model(LabelledModel):
            def transcribe(self):
                discrete, lbx, ubx, lbg, ubg, x0, nlp = super().transcribe()
                nlp["g"] = nlp["g"][:-1]
                return discrete, lbx, ubx, lbg[:-1], ubg[:-1], x0, nlp

        problem = Model()
        with self.assertLogs("rtctools", "WARNING"):
            problem.optimize()
        self.assertIsNone(problem.transcribed_problem["g_names"])

    def test_single_pass_labels_objective_rows(self):
        for method, expected in [
            (SinglePassMethod.APPEND_CONSTRAINTS_OBJECTIVE, ["objective_priority_1"]),
            (
                SinglePassMethod.UPDATE_OBJECTIVE_CONSTRAINT_BOUNDS,
                ["objective_priority_1", "objective_priority_2"],
            ),
        ]:
            with self.subTest(method=method):
                model_class = type(
                    "Model", (LabelledSinglePassModel,), {"single_pass_method": method}
                )
                labels, _ = self.labels_and_bounds(model_class)
                self.assertEqual(labels[-len(expected) :], expected)
