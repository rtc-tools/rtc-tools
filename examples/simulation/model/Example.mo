model Example
  // Elements
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow inflow(Q = Q_in) annotation(Placement(visible = true, transformation(origin = {-55, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Storage.Storage storage(Q_release = P_control, V(start=storage_V_init, fixed=true, nominal=4e5)) annotation(Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal outfall annotation(Placement(visible = true, transformation(origin = {55, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  // Initial States
  parameter Modelica.Units.SI.Volume storage_V_init(start = 0.0); // Value set in parameters.csv
  // Inputs
  input Modelica.Units.SI.VolumeFlowRate P_control(fixed = true);
  input Modelica.Units.SI.VolumeFlowRate Q_in(fixed = true);
  input Modelica.Units.SI.VolumeFlowRate storage_V_target(fixed = true);
  // Outputs
  output Modelica.Units.SI.Volume storage_V = storage.V;
  output Modelica.Units.SI.VolumeFlowRate Q_release = P_control;
equation
  connect(inflow.QOut, storage.QIn) annotation(Line(points = {{-47, 0}, {-10, 0}}));
  connect(storage.QOut, outfall.QIn) annotation(Line(points = {{8, 0}, {47, 0}}));
  annotation(Diagram(coordinateSystem(extent = {{-148.5, -105}, {148.5, 105}}, initialScale = 0.1, grid = {5, 5})));
end Example;