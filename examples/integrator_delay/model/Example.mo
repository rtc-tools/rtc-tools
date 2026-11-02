model Example
  parameter Integer delay_hours = 24;
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow control_inflow annotation(
    Placement(visible = true, transformation(origin = {-80, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow external_inflow annotation(
    Placement(visible = true, transformation(origin = {0, 40}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Branches.Delay delay_component(duration = delay_hours * 3600) annotation(
    Placement(visible = true, transformation(origin = {-38, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Branches.Integrator integrator(n_QLateral = 1) annotation(
    Placement(visible = true, transformation(origin = {2, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal outflow annotation(
    Placement(visible = true, transformation(origin = {42, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  input Modelica.Units.SI.VolumeFlowRate Q_in(fixed = true);
  input Modelica.Units.SI.VolumeFlowRate Q_out(fixed = true);
  input Modelica.Units.SI.VolumeFlowRate Q_control(fixed = false);
  output Modelica.Units.SI.Volume V_integrator;
equation
  // connect(control_inflow.QOut, integrator.QIn);
  connect(control_inflow.QOut, delay_component.QIn) annotation(
    Line(points = {{-72, 0}, {-46, 0}}));
  connect(delay_component.QOut, integrator.QIn) annotation(
    Line(points = {{-30, 0}, {-6, 0}}));
  connect(external_inflow.QOut, integrator.QLateral[1]) annotation(
    Line(points = {{8, 40}, {6, 40}, {6, 8}}));
  connect(integrator.QOut, outflow.QIn) annotation(
    Line(points = {{10, 0}, {34, 0}}));
  Q_control = control_inflow.Q;
  Q_in = external_inflow.Q;
  Q_out = outflow.Q;
  V_integrator = integrator.V;
end Example;
