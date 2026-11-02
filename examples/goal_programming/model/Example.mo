model Example
  // Declare Model Elements
  Deltares.ChannelFlow.Hydraulic.Storage.Linear storage(A=1.0e6, H_b=0.0, HQ.H(min=0.0, max=0.5)) annotation(Placement(visible = true, transformation(origin = {32.022, -0}, extent = {{-10, -10}, {10, 10}}, rotation = -270)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge discharge annotation(Placement(visible = true, transformation(origin = {60, -0}, extent = {{10, -10}, {-10, 10}}, rotation = 270)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level level annotation(Placement(visible = true, transformation(origin = {-52.7, 0}, extent = {{-10, -10}, {10, 10}}, rotation = -270)));
  Deltares.ChannelFlow.Hydraulic.Structures.Pump pump annotation(Placement(visible = true, transformation(origin = {0, -20}, extent = {{10, -10}, {-10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.Structures.Pump orifice annotation(Placement(visible = true, transformation(origin = {0, 20}, extent = {{10, -10}, {-10, 10}}, rotation = 0)));

  // Define Input/Output Variables and set them equal to model variables
  input Modelica.Units.SI.VolumeFlowRate Q_pump(fixed=false, min=0.0, max=7.0);
  input Boolean is_downhill;
  input Modelica.Units.SI.VolumeFlowRate Q_in(fixed=true);
  input Modelica.Units.SI.Position H_sea(fixed=true);
  input Modelica.Units.SI.VolumeFlowRate Q_orifice(fixed=false, min=0.0, max=10.0);
  output Modelica.Units.SI.Position storage_level = storage.HQ.H;
  output Modelica.Units.SI.Position sea_level = level.H;
equation
  Q_pump = pump.Q;
  Q_in = discharge.Q;
  H_sea = level.H;
  Q_orifice = orifice.Q;
  // Connect Model Elements
  connect(orifice.HQDown, level.HQ) annotation(Line(visible = true, origin = {-33.025, 10}, points = {{25.025, 10}, {-6.675, 10}, {-6.675, -10}, {-11.675, -10}}, color = {0, 0, 255}));
  connect(storage.HQ, orifice.HQUp) annotation(Line(visible = true, origin = {43.737, 48.702}, points = {{-3.715, -48.702}, {-3.737, -28.702}, {-35.737, -28.702}}, color = {0, 0, 255}));
  connect(storage.HQ, pump.HQUp) annotation(Line(visible = true, origin = {4.669, -31.115}, points = {{35.353, 31.115}, {35.331, 11.115}, {3.331, 11.115}}, color = {0, 0, 255}));
  connect(discharge.HQ, storage.HQ) annotation(Line(visible = true, origin = {46.011, -0}, points = {{5.989, 0}, {-5.989, -0}}, color = {0, 0, 255}));
  connect(pump.HQDown, level.HQ) annotation(Line(visible = true, origin = {-33.025, -10}, points = {{25.025, -10}, {-6.675, -10}, {-6.675, 10}, {-11.675, 10}}, color = {0, 0, 255}));
  annotation(Diagram(coordinateSystem(extent = {{-148.5, -105}, {148.5, 105}}, preserveAspectRatio = true, initialScale = 0.1, grid = {5, 5})));
end Example;
