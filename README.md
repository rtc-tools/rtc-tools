# Deltares RTC-Tools

[![Pipeline](https://github.com/rtc-tools/rtc-tools/actions/workflows/ci.yml/badge.svg)](
    https://github.com/rtc-tools/rtc-tools/actions/workflows/ci.yml
)
[![Coverage](
    https://sonarcloud.io/api/project_badges/measure?project=Deltares_rtc-tools&metric=coverage
)](https://sonarcloud.io/summary/new_code?id=Deltares_rtc-tools)

> **NOTE** The rtc-tools repository has been migrated from gitlab to here;
see [migration from gitlab](#migration-from-gitlab).

RTC-Tools is an open-source Python package for simulation and optimization of cyber-physical systems.  It can be used for model-predictive control and operational decision making of water and energy systems.  It is developed and maintained by Deltares, in collaboration with partners.

RTC-Tools offers the following functionalities:

- **Model building using extensible libraries**: Build complex system models using extensible libraries of model components. Modelers can also implement their custom modeling components, for instance, using the [Modelica language](https://modelica.org/language/) or Python script.

    RTC-Tools can be used with the following libraries and packages for specific applications:
    - [rtc-tools-channel-flow](https://gitlab.com/deltares/rtc-tools-channel-flow): water system models
    - [rtc-tools-hydraulic-structures](https://gitlab.com/deltares/rtc-tools-channel-flow): hydraulic assets, such as weirs and pumps
    - [rtc-tools-heat-network](https://github.com/Nieuwe-Warmte-Nu/rtc-tools-heat-network): heat networks

Please note that this list is not exhaustive. Users can also create libraries for other types of applications.

- **Running simulations**: Simulate a given model.

- **Specifying and solving optimization problems**: Define optimization goals, constraints and decision variables to specify the optimization problem for a given model. RTC-Tools supports both open-source solvers (CBC, HiGHS, Ipopt) and commercial solvers (Gurobi, CPLEX) for solving several types of optimization problems:

    - **Linear, non-linear**:  RTC-Tools supports both linear and non-linear optimization problems.

    - **Continuous and discrete**:  RTC-Tools can handle both continuous and discrete decision variables. This makes it suitable for optimizing systems with a mix of continuous controls (such as pump speeds or gate positions) and discrete decisions (such as on/off states of equipment).

    - **Goal programming**: When multiple, and perhaps conflicting, objectives are to be considered (e.g., minimize operational costs while minimizing deviations of water levels from a given range), RTC-Tools offers two approaches to multi-objective optimization: The **Weighting Method**, which assigns weights to each objective and optimizes them simultaneously, and the **Lexicographic Goal Programming method**, which optimizes different objectives sequentially. 

    - **Optimization under uncertainty**: RTC-Tools supports the use of ensemble forecasts for optimization under uncertainty. To reduce the size of the optimization problem, ensemble members can be automatically aggregated using a scenario tree reduction method.

To streamline the integration with user interfaces and data management systems (such as Delft-FEWS), RTC-Tools supports CSV and XML file formats for reading/writing timeseries and other model parameters. Support for other formats can be implemented using mixins.

RTC-Tools uses [CasADi](https://web.casadi.org/) as a symbolic framework for algorithmic differentiation, as well as for interfacing with numerical optimization solvers.


## Install

```bash
pip install rtc-tools
```

## Documentation

Documentation and examples can be found on [readthedocs](https://rtc-tools.readthedocs.io).

## Contributing

We welcome contributions to RTC-Tools! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started, coding standards, and our development process.

The project is governed according to our [Technical Charter](CHARTER.md) and [Governance Model](GOVERNANCE.md).


## Migration from GitLab

The rtc-tools repository has been migrated from gitlab (https://gitlab.com/rtc-tools/rtc-tools)
to here.
To change the git remote url, run

`git remote set-url origin https://github.com/rtc-tools/rtc-tools.git`

and

`git remote set-url --push origin https://github.com/rtc-tools/rtc-tools.git`.


## License
RTC-Tools is licensed under the **[GNU Lesser General Public License v3.0](COPYING)**,
and can be used free of charge. Deltares offers support packages for users who require assistance.


## Acknowledgment
If you use RTC-Tools in your work, please acknowledge it in any resulting publications.
You can do this by citing RTC-Tools and providing a link to our
[website](https://oss.deltares.nl/web/rtc-tools/home) or
[GitHub repository](https://github.com/rtc-tools/rtc-tools).
