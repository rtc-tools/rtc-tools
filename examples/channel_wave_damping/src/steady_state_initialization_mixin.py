import casadi as ca

from rtctools.optimization.optimization_problem import OptimizationProblem


class SteadyStateInitializationMixin(OptimizationProblem):
    def constraints(self, ensemble_member):
        c = super().constraints(ensemble_member)
        times = self.times()
        parameters = self.parameters(ensemble_member)
        # Force steady-state initialization at t0 and at t1.
        for reach in ["upstream", "middle", "downstream"]:
            for i in range(int(parameters[f"{reach}.n_level_nodes"]) + 1):
                state = f"{reach}.Q[{i + 1}]"
                der = self.der_at(state, times[0])
                # A flow that follows from input data has nothing left to constrain
                if isinstance(der, ca.MX):
                    c.append((der, 0, 0))
        return c
