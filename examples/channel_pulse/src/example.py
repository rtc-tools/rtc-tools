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
        # The last level node follows the downstream level boundary at every time, so
        # steady state there is set through the downstream flow instead.
        for i in range(n_level_nodes - 1):
            state = f"Channel.H[{i + 1}]"
            constraints.append(
                (self.state_at(state, times[0]) - self.state_at(state, times[1]), 0, 0)
            )
        q_down = f"Channel.Q[{n_level_nodes + 1}]"
        constraints.append(
            (self.state_at(q_down, times[0]) - self.state_at("Inflow_Q", times[0]), 0, 0)
        )
        return constraints


class ExampleInertialWave(Example):
    """Inertial wave equation (no convective acceleration)"""

    model_name = "Example"

    use_semi_implicit = False
    use_convective_acceleration = False
    use_upwind = False

    timeseries_export_basename = "timeseries_export_inertial_wave"


class ExampleInertialWaveSemiImplicit(Example):
    """Inertial wave equation (no convective acceleration)"""

    model_name = "Example"

    use_semi_implicit = True
    use_convective_acceleration = False
    use_upwind = False

    timeseries_export_basename = "timeseries_export_inertial_wave_semi_implicit"


class ExampleSaintVenant(Example):
    """Saint Venant equation. Convective acceleration discretized with central differences"""

    model_name = "Example"

    use_semi_implicit = False
    use_convective_acceleration = True
    use_upwind = False

    timeseries_export_basename = "timeseries_export_saint_venant"


class ExampleSaintVenantUpwind(Example):
    """Saint Venant equation. Convective acceleration discretized with upwind scheme"""

    model_name = "Example"

    use_semi_implicit = False
    use_convective_acceleration = True
    use_upwind = True

    timeseries_export_basename = "timeseries_export_saint_venant_upwind"


run_optimization_problem(ExampleInertialWave)
run_optimization_problem(ExampleInertialWaveSemiImplicit)
run_optimization_problem(ExampleSaintVenant)
run_optimization_problem(ExampleSaintVenantUpwind)
