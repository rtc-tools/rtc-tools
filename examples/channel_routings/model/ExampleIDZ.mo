model ExampleIDZ
  // Elements
  Deltares.ChannelFlow.Hydraulic.Branches.IDZ Channel(
  width=500,
  n_QForcing=0,
  length=20000,
  H_b_up=-5.0,
  H_b_down=-5.0,
  side_slope=0.0,
  H_nominal=0.0116,
  Q_nominal=100.0,
  friction_coefficient=0.045
  )annotation(Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level Level annotation(Placement(visible = true, transformation(origin = {60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge Discharge annotation(Placement(visible = true, transformation(origin = {-60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  // Inputs
  input Real Inflow_Q(fixed=true);
  input Real Level_H(fixed=true);
  parameter Real theta(start = 1.0);
  parameter Real step_size(start = 0.0);
  // Output Channel states
  output Real Channel_Q_up = Discharge.Q;
  output Real Channel_Q_dn = Level.HQ.Q;
  output Real Channel_H_up = Discharge.HQ.H;
  output Real Channel_H_dn = Level.H;
equation
  Inflow_Q = Discharge.Q;
  Level_H = Level.H;
  connect(Channel.HQDown, Level.HQ) annotation(Line(points = {{8, 0}, {60, 0}, {60, 0}, {60, 0}}, color = {0, 0, 255}));
  connect(Discharge.HQ, Channel.HQUp) annotation(Line(points = {{-60, 0}, {-8, 0}, {-8, 0}, {-8, 0}}, color = {0, 0, 255}));
//initial equation
//  Channel.Q = fill(Inflow_Q, Channel.n_level_nodes + 1);
end ExampleIDZ;
