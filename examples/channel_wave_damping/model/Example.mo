model Example
  // Structures
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge Discharge annotation(Placement(visible = true, transformation(origin = {-90, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level Level annotation(Placement(visible = true, transformation(origin = {90, 0}, extent = {{-10, 10}, {10, -10}}, rotation = 90)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicTrapezoidal upstream(
    theta = theta,
    semi_implicit_step_size = step_size,
    H_b_up = 15,
    H_b_down = 15,
    bottom_width_up = 50,
    bottom_width_down = 50,
    length = 20000,
    uniform_nominal_depth = 5,
    friction_coefficient = 35,
    n_level_nodes = 4,
    Q_nominal = 100.0
  ) annotation(Placement(visible = true, transformation(origin = {-60, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicTrapezoidal middle(
    theta = theta,
    semi_implicit_step_size = step_size,
    H_b_up = 10,
    H_b_down = 10,
    bottom_width_up = 50,
    bottom_width_down = 50,
    length = 20000,
    uniform_nominal_depth = 5,
    friction_coefficient = 35,
    n_level_nodes = 4,
    Q_nominal = 100.0
  ) annotation(Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicTrapezoidal downstream(
    theta = theta,
    semi_implicit_step_size = step_size,
    H_b_up = 5,
    H_b_down = 5,
    bottom_width_up = 50,
    bottom_width_down = 50,
    length = 20000,
    uniform_nominal_depth = 5,
    friction_coefficient = 35,
    n_level_nodes = 4,
    Q_nominal = 100.0
  ) annotation(Placement(visible = true, transformation(origin = {58, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.DischargeControlledStructure dam_middle annotation(Placement(visible = true, transformation(origin = {30, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.DischargeControlledStructure dam_upstream annotation(Placement(visible = true, transformation(origin = {-30, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));

  // Inputs
  input Modelica.Units.SI.Position Level_H(fixed = true);
  input Modelica.Units.SI.VolumeFlowRate Inflow_Q(fixed = true);

  // Outputs
  output Modelica.Units.SI.Position H_middle = middle.H[middle.n_level_nodes];
  output Modelica.Units.SI.Position H_upstream = upstream.H[upstream.n_level_nodes];
  output Modelica.Units.SI.VolumeFlowRate Q_in = Inflow_Q;

  // Parameters
  parameter Modelica.Units.SI.Duration step_size(start = 0.0); // Value set in parameters()
  parameter Real theta(start = 1.0); // Value driven by HomotopyMixin
equation
  Level_H = Level.H;
  Inflow_Q = Discharge.Q;
  connect(dam_middle.HQDown, downstream.HQUp) annotation(Line(points = {{38, 0}, {50, 0}}, color = {0, 0, 255}));
  connect(dam_middle.HQUp, middle.HQDown) annotation(Line(points = {{8, 0}, {22, 0}, {22, 0}, {22, 0}}, color = {0, 0, 255}));
  connect(dam_upstream.HQDown, middle.HQUp) annotation(Line(points = {{-22, 0}, {-8, 0}}, color = {0, 0, 255}));
  connect(dam_upstream.HQUp, upstream.HQDown) annotation(Line(points = {{-52, 0}, {-38, 0}}, color = {0, 0, 255}));
  connect(Discharge.HQ, upstream.HQUp) annotation(Line(points = {{-82, 0}, {-68, 0}}, color = {0, 0, 255}));
  connect(Level.HQ, downstream.HQDown) annotation(Line(points = {{66, 0}, {82, 0}, {82, 0}, {82, 0}}, color = {0, 0, 255}));
initial equation
  downstream.Q[2:downstream.n_level_nodes + 1] = fill(Inflow_Q, downstream.n_level_nodes);
  middle.Q[2:middle.n_level_nodes + 1] = fill(Inflow_Q, middle.n_level_nodes);
  upstream.Q[2:upstream.n_level_nodes + 1] = fill(Inflow_Q, upstream.n_level_nodes);
end Example;
