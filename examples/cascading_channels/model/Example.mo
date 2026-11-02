model Example
  // Model Elements
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge Inflow annotation(Placement(visible = true, transformation(origin = {-86, 34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level DrinkingWaterPlant(H = 10.) annotation(Placement(visible = true, transformation(origin = {38, 44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level Level annotation(Placement(visible = true, transformation(origin = {66, -32}, extent = {{-10, 10}, {10, -10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicLinear LowerChannel(
    H(each max = 1.0),
    H_b_down = -2.0,
    H_b_up = -1.5,
    friction_coefficient = 10.,
    length = 2000.,
    theta = theta,
    uniform_nominal_depth = 1.75,
    width_down = 10.,
    width_up = 10.,
    semi_implicit_step_size = step_size,
    Q_nominal = 1.0
  ) annotation(Placement(visible = true, transformation(origin = {42, -24}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicLinear MiddleChannel(
    H(each max = 1.5),
    H_b_down = -1.5,
    H_b_up = -1.0,
    friction_coefficient = 10.,
    length = 2000.,
    theta = theta,
    uniform_nominal_depth = 1.75,
    width_down = 10.,
    width_up = 10.,
    semi_implicit_step_size = step_size,
    Q_nominal = 1.0
  ) annotation(Placement(visible = true, transformation(origin = {-10, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicLinear UpperChannel(
    H(each max = 2.0),
    H_b_down = -1.0,
    H_b_up = -0.5,
    friction_coefficient = 10.,
    length = 2000.,
    theta = theta,
    uniform_nominal_depth = 1.75,
    width_down = 10.,
    width_up = 10.,
    semi_implicit_step_size = step_size,
    Q_nominal = 1.0
  ) annotation(Placement(visible = true, transformation(origin = {-62, 26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.Pump DrinkingWaterExtractionPump annotation(Placement(visible = true, transformation(origin = {12, 26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.Pump LowerControlStructure annotation(Placement(visible = true, transformation(origin = {20, 2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.Pump UpperControlStructure annotation(Placement(visible = true, transformation(origin = {-32, 26}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  // Parameters
  parameter Real theta(start = 1.0); // Value driven by HomotopyMixin
  parameter Real step_size(start = 0.0); // Value set in parameters()
  // Inputs
  input Real Inflow_Q(fixed = true);
  input Real UpperControlStructure_Q(fixed = false, min = 0., max = 4.);
  input Real LowerControlStructure_Q(fixed = false, min = 0., max = 4.);
  input Real DrinkingWaterExtractionPump_Q(fixed = false, min = 0., max = 2.);
initial equation
  // Steady state initialization
  der(LowerChannel.Q) = fill(0.0, LowerChannel.n_level_nodes + 1);
  der(MiddleChannel.Q) = fill(0.0, MiddleChannel.n_level_nodes + 1);
  der(UpperChannel.Q) = fill(0.0, UpperChannel.n_level_nodes + 1);
equation
  Inflow_Q = Inflow.Q;
  UpperControlStructure_Q = UpperControlStructure.Q;
  LowerControlStructure_Q = LowerControlStructure.Q;
  DrinkingWaterExtractionPump_Q = DrinkingWaterExtractionPump.Q;
  connect(DrinkingWaterExtractionPump.HQDown, DrinkingWaterPlant.HQ) annotation(Line(points = {{20, 26}, {38, 26}, {38, 36}, {38, 36}}, color = {0, 0, 255}));
  connect(Inflow.HQ, UpperChannel.HQUp) annotation(Line(points = {{-88, 26}, {-70, 26}, {-70, 26}, {-70, 26}}, color = {0, 0, 255}));
  connect(LowerChannel.HQDown, Level.HQ) annotation(Line(points = {{50, -24}, {66, -24}}, color = {0, 0, 255}));
  connect(LowerControlStructure.HQDown, LowerChannel.HQUp) annotation(Line(points = {{26, 2}, {32, 2}, {32, -24}, {32, -24}}, color = {0, 0, 255}));
  connect(MiddleChannel.HQDown, DrinkingWaterExtractionPump.HQUp) annotation(Line(points = {{-2, 2}, {4, 2}, {4, 26}, {4, 26}}, color = {0, 0, 255}));
  connect(MiddleChannel.HQDown, LowerControlStructure.HQUp) annotation(Line(points = {{-4, 2}, {10, 2}, {10, 2}, {10, 2}}, color = {0, 0, 255}));
  connect(UpperChannel.HQDown, UpperControlStructure.HQUp) annotation(Line(points = {{-56, 26}, {-42, 26}, {-42, 26}, {-42, 26}}, color = {0, 0, 255}));
  connect(UpperControlStructure.HQDown, MiddleChannel.HQUp) annotation(Line(points = {{-26, 26}, {-20, 26}, {-20, 2}, {-20, 2}}, color = {0, 0, 255}));
end Example;
