import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin, StateGoal
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem


class TargetGoal(StateGoal):
    order = 4

    def __init__(
        self,
        optimization_problem,
        state,
        target_min,
        target_max,
        function_range,
        priority,
        function_nominal=1,
    ):
        self.state = state
        self.target_min = target_min
        self.target_max = target_max
        self.priority = priority
        super().__init__(optimization_problem)
        self.function_range = function_range
        self.function_nominal = function_nominal


class MaxRevenueGoal(StateGoal):
    order = 1
    state = "SystemGeneratingRevenue"
    priority = 30

    def function(self, optimization_problem, ensemble_member):
        return -optimization_problem.state(self.state)


class MinCostGoal(StateGoal):
    order = 1
    state = "PumpCost"
    priority = 25


class PumpStorage(
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    model_name = "PumpedStoragePlant"

    def pre(self):
        super().pre()
        self.power_nominal = np.mean(self.get_timeseries("Target_Power").values)

    def path_constraints(self, ensemble_member):
        """
        These constraints are formulated using the Big-M notation and represent the reversible
        pump-turbine unit used to move water between the upper and lower reservoirs.

        Boolean
        -------
        Turbine_is_on is a boolean which is 1 when the reversible unit is working as a turbine,
        and 0 otherwsie. This is imposed by the following constraints

        Constraints
        -----------
        0 <= PumpFlow + Turbine_is_on * M <= inf
        0 <= TurbineFlow + (1 - Turbine_is_on) * M <= inf
        -inf <= PumpFlow - (1 - Turbine_is_on) * M <= 0
        -inf <= TurbineFlow - Turbine_is_on * M <= 0
        """
        constraints = super().path_constraints(ensemble_member)

        M = 200.0
        constraints.append((self.state("PumpFlow") + self.state("Turbine_is_on") * M, 0.0, np.inf))
        constraints.append(
            (self.state("TurbineFlow") + (1 - self.state("Turbine_is_on")) * M, 0.0, np.inf)
        )
        constraints.append(
            (self.state("PumpFlow") - (1 - self.state("Turbine_is_on")) * M, -np.inf, 0.0)
        )
        constraints.append(
            (self.state("TurbineFlow") - self.state("Turbine_is_on") * M, -np.inf, 0.0)
        )

        return constraints

    def variable_nominal(self, variable=None):
        nom = super().variable_nominal(variable)
        if variable == "TotalGeneratingPower":
            return self.power_nominal
        elif variable == "TurbinePower":
            return self.power_nominal
        else:
            return nom

    def path_goals(self):
        goals = super().path_goals()
        # 020 goal to set the target spill flow as zero
        goals.append(
            TargetGoal(
                self,
                state="ReservoirSpillFlow",
                target_min=np.nan,
                target_max=0.0,
                function_range=(0.0, 100.0),
                priority=20,
            )
        )
        # 030 goal to ensure the power generation meets the target
        target = self.get_timeseries("Target_Power")
        goals.append(
            TargetGoal(
                self,
                state="TotalSystemPower",
                target_min=target,
                target_max=target,
                function_range=(0.0, 4e6),
                function_nominal=self.power_nominal,
                priority=10,
            )
        )
        # 040 goal to minimise the cost of the pump
        goals.append(MinCostGoal(self))
        # 050 goal to maximise the revenue from the turbines
        goals.append(MaxRevenueGoal(self))

        return goals

    def solver_options(self):
        options = super().solver_options()
        options["casadi_solver"] = "qpsol"
        options["solver"] = "highs"
        options["export_model"] = True
        return options

    def post(self):
        super().post()
        results = self.extract_results()
        results["TotalSystemRevenueSum"] = np.sum(results["TotalSystemRevenue"])
        print("Total Revenue is " + str(results["TotalSystemRevenueSum"]))


# Run
run_optimization_problem(PumpStorage)
