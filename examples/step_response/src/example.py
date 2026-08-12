from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem


class Example(HomotopyMixin, CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def parameters(self, ensemble_member):
        p = super().parameters(ensemble_member)
        times = self.times()
        if self.use_semi_implicit:
            p["step_size"] = times[1] - times[0]
        else:
            p["step_size"] = 0.0
        p["Channel.use_convective_acceleration"] = self.use_convective_acceleration
        p["Channel.use_upwind"] = self.use_upwind
        return p

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        times = self.times()

        # Extract the number of nodes in the channel
        parameters = self.parameters(ensemble_member)
        n_level_nodes = int(parameters["Channel.n_level_nodes"])

        # To Mimic HEC-RAS behaviour, enforce steady state both at t0 and at t1.
        for i in range(n_level_nodes):
            state = f"Channel.H[{i + 1}]"
            constraints.append(
                (self.state_at(state, times[0]) - self.state_at(state, times[1]), 0, 0)
            )
        return constraints

    def post(self):
        super().post()

    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "ipopt"
        return options


class ExampleSaintVenant(Example):
    """Saint Venant equation. Convective acceleration discretized with central differences"""

    model_name = "Example"

    use_semi_implicit = False
    use_convective_acceleration = True
    use_upwind = False

    timeseries_export_basename = "timeseries_export_saint_venant"


def run():
    run_optimization_problem(ExampleSaintVenant, base_folder="..")


if __name__ == "__main__":
    run()
