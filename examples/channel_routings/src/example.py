from rtctools_channel_flow.channel_flow_parameter_setting import (
    ChannelFlowParameterSettingOpimizationMixin,
)

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem


class ExampleSV(
    ChannelFlowParameterSettingOpimizationMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    linearised_sv = True
    linearised_sv_branches = ["Channel"]
    model_name = "Example"

    def parameters(self, ensemble_member):
        p = super().parameters(ensemble_member)
        p["Channel.H_nominal"] = initial_water_level - p["Channel.H_b_up"]
        p["Channel.H_nominal_down"] = (
            self.get_timeseries("Level_H").values[0] - p["Channel.H_b_down"]
        )

        return p

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        times = self.times()

        # Extract the number of nodes in the channel
        parameters = self.parameters(ensemble_member)
        n_level_nodes = int(parameters["Channel.n_level_nodes"])

        initial_level = 0.0
        for i in range(n_level_nodes - 1):
            state = f"Channel.Y_relative[{i + 1}]"
            constraints.append((self.state_at(state, times[0]), initial_level, initial_level))

        initial_level = 0.0
        for i in range(1, n_level_nodes + 1):
            state = f"Channel.Q_relative[{i + 1}]"
            constraints.append((self.state_at(state, times[0]), initial_level, initial_level))
        return constraints

    def post(self):
        super().post()

        """
        results = self.extract_results()
        for i in range(1,13):
            print( results["Channel.Q["+ str(i) + ']'][0])
        print("Q1")
        for i in range(1,13):
            print( results["Channel.Q["+ str(i) + ']'][1])
        print("Initial der")
        for i in range(1,13):
            print( results["initial_der(Channel.Q_relative["+ str(i) + '])'])
        print("Q_relative")
        for i in range(1,13):
            print( results["Channel.Q_relative["+ str(i) + ']'][0])
        print("Y_relative")
        for i in range(1,12):
            print( results["Channel.Y_relative["+ str(i) + ']'][0])

        for i in range(1,12):
            print( results["Channel.H["+ str(i) + ']'][0])
        """
        """
        n_subplots = 2
        fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))
        axarr[0].set_title("Water Levels and Flow Rates")

        # Upper subplot
        axarr[0].set_ylabel("Flow Rate [m³/s]")
        for i in range(1,13):
            axarr[0].plot(
                results["Channel.Q["+ str(i) + ']'],
                label="Upstream",
                color="xkcd:dark sky blue",
            )
        # Upper subplot
        axarr[1].set_ylabel("Height [m³/s]")
        for i in range(1,12):
            axarr[1].plot(
                results["Channel.H["+ str(i) + ']'],
                label="Upstream",
                color="xkcd:dark sky blue",
            )
        axarr[1].plot(
            results["Channel.H[1]"],
            label="Upstream",
            color="red",
        )
        """

    timeseries_export_basename = "timeseries_export_SV"


class ExampleIDZ(
    ChannelFlowParameterSettingOpimizationMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    idz = True
    idz_branches = ["Channel"]
    model_name = "ExampleIDZ"

    def parameters(self, ensemble_member):
        p = super().parameters(ensemble_member)
        # p["step_size"] = 0.0
        p["Channel.H_nominal"] = initial_water_level
        super().parameters(ensemble_member)
        return p

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        times = self.times()

        # Extract the number of nodes in the channel
        parameters = self.parameters(ensemble_member)
        n_level_nodes = int(parameters["Channel.n_level_nodes"])

        initial_discharge = -self.get_timeseries("Inflow_Q").values[0]
        state = "Channel.HQDown.Q"
        constraints.append((self.state_at(state, times[0]), initial_discharge, initial_discharge))

        # initial_level = 0.0116
        initial_level = initial_water_level
        for _i in range(n_level_nodes - 1):
            state = "Channel.HQUp.H"
            constraints.append((self.state_at(state, times[0]), initial_level, initial_level))

        return constraints

    def post(self):
        super().post()

    """
        results = self.extract_results()
        for i in range(1,13):
            print( results["Channel.Q["+ str(i) + ']'][1])
        print("Initial der")
        for i in range(1,13):
            print( results["initial_der(Channel.Q_relative["+ str(i) + '])'])
        print("Q_relative")
        for i in range(1,13):
            print( results["Channel.Q_relative["+ str(i) + ']'][0])
        print("Y_relative")
        for i in range(1,12):
            print( results["Channel.Y_relative["+ str(i) + ']'][0])

        print("H")

        for i in range(1,12):
            print( results["Channel.H["+ str(i) + ']'][0])

        n_subplots = 2
        fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))
        axarr[0].set_title("Water Levels and Flow Rates")

        # Upper subplot
        axarr[0].set_ylabel("Flow Rate [m³/s]")
        for i in range(1,13):
            axarr[0].plot(
                results["Channel.Q["+ str(i) + ']'],
                label="Upstream",
                color="xkcd:dark sky blue",
            )
        # Upper subplot
        axarr[1].set_ylabel("Height [m³/s]")
        for i in range(1,12):
            axarr[1].plot(
                results["Channel.H["+ str(i) + ']'],
                label="Upstream",
                color="xkcd:dark sky blue",
            )
        axarr[1].plot(
            results["Channel.H[1]"],
            label="Upstream",
            color="red",
        )
        plt.show()

    """

    timeseries_export_basename = "timeseries_export_IDZ"


class ExampleLinear(CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    """Inertial wave equation (no convective acceleration)"""

    use_semi_implicit = False
    use_convective_acceleration = True
    use_upwind = True
    model_name = "ExampleLinear"

    def parameters(self, ensemble_member):
        p = super().parameters(ensemble_member)
        times = self.times()
        if self.use_semi_implicit:
            p["step_size"] = times[1] - times[0]
        else:
            p["step_size"] = 0.0
        p["Channel.use_convective_acceleration"] = self.use_convective_acceleration
        p["Channel.use_upwind"] = self.use_upwind

        p["Channel.uniform_nominal_depth"] = initial_water_level - p["Channel.H_b_up"]

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

        """
        initial_level = 1.212
        for i in range(n_level_nodes-1):
            state = f"Channel.H[{i + 1}]"
            constraints.append(
                (self.state_at(state, times[0]) , initial_level, initial_level)
            )
        """

        return constraints

    def post(self):
        super().post()

        """
        results = self.extract_results()
        n_subplots = 2
        fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))
        axarr[0].set_title("Water Levels and Flow Rates")

        # Upper subplot
        axarr[0].set_ylabel("Flow Rate [m³/s]")
        for i in range(1,13):
            axarr[0].plot(
                results["Channel.Q["+ str(i) + ']'],
                label="Upstream",
                color="xkcd:dark sky blue",
            )
        # Upper subplot
        axarr[1].set_ylabel("Height [m³/s]")
        for i in range(1,12):
            axarr[1].plot(
                results["Channel.H["+ str(i) + ']'],
                label="Upstream",
                color="xkcd:dark sky blue",
            )
        axarr[1].plot(
            results["Channel.H[1]"],
            label="Upstream",
            color="red",
        )
        plt.show()
        """

    timeseries_export_basename = "timeseries_export_linear"


class ExampleID(CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    """Inertial wave equation (no convective acceleration)"""

    model_name = "ExampleID"

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        times = self.times()

        # Extract the number of nodes in the channel
        n_level_nodes = 2

        initial_level = -self.get_timeseries("Inflow_Q").values[0]
        for _i in range(n_level_nodes - 1):
            state = "Channel.HQDown.Q"
            constraints.append((self.state_at(state, times[0]), initial_level, initial_level))

        initial_level = initial_water_level
        for _i in range(n_level_nodes - 1):
            state = "Channel.HQUp.H"
            constraints.append((self.state_at(state, times[0]), initial_level, initial_level))

        return constraints

    timeseries_export_basename = "timeseries_export_id"


class ExampleFullSV(
    HomotopyMixin, CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem
):
    model_name = "ExampleFullSV"
    use_semi_implicit = False
    use_convective_acceleration = True
    use_upwind = True

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
        results = self.extract_results()
        # print(results["Channel.H[1]"][1])
        self.initial_water_level = results["Channel.H[1]"][71]
        # print(results.keys())

    timeseries_export_basename = "timeseries_export_saint_venant_upwind"


res = run_optimization_problem(ExampleFullSV)
initial_water_level = res.initial_water_level
run_optimization_problem(ExampleSV)
run_optimization_problem(ExampleIDZ)
run_optimization_problem(ExampleLinear)
run_optimization_problem(ExampleID)
