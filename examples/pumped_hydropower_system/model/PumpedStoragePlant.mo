model PumpedStoragePlant
  import SI = Modelica.Units.SI;
  // Declare Model Elements
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Inflow Inflow annotation(
    Placement(visible = true, transformation(origin = {-68, -4}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.BoundaryConditions.Terminal Outflow annotation(
    Placement(visible = true, transformation(origin = {80, -2}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Storage.Storage UpperBasin annotation(
    Placement(visible = true, transformation(origin = {12, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Reservoir.Reservoir LowerBasin(n_QLateral = 2) annotation(
    Placement(visible = trued, transformation(origin = {-8, 6}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.SimpleRouting.Structures.DischargeControlledStructure Pump annotation(
    Placement(visible = true, transformation(origin = {-24, 56}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Deltares.ChannelFlow.SimpleRouting.Structures.DischargeControlledStructure Turbine annotation(
    Placement(visible = true, transformation(origin = {14, 46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));

  // Define Input/Output Variables and set them equal to model variables
  input SI.VolumeFlowRate Inflow_Q(fixed = true);
  input SI.VolumeFlowRate PumpFlow(min = 0.0, max = 10.0, fixed = false);
  input SI.VolumeFlowRate TurbineFlow(min = 0.0, max = 10.0, fixed = false);
  input SI.VolumeFlowRate ReservoirTurbineFlow(min = 0.0, max = 100.0, fixed = false);
  input SI.VolumeFlowRate ReservoirSpillFlow(min = 0.0, max = 100.0, fixed = false);
  input Real cost_perP(fixed=true);
  output Real PumpPower;
  output Real TurbinePower;
  output Real V_LowerBasin(min = 0.0, max = 1e7);
  output Real V_UpperBasin(min = 0.0, max = 1e7);
  output Real ReservoirPower;
  output Real TotalSystemPower;
  output Real TotalGeneratingPower(min = 0.0);
  output Real SystemGeneratingRevenue;
  output Real TotalSystemRevenue;
  output Real PumpCost;

  // Define Boolean to ensure pumped storage turbine and pump cannot be used simultaneously
  Boolean Turbine_is_on;

  // Define constants for simple power calculations
  parameter Real efficiency_reservoir = 0.88;
  parameter Real efficiency_pump = 0.7;
  parameter Real efficiency_turbine = 0.9;
  parameter Real gravity = 9.81;
  parameter Real rho = 1000.0;
  parameter Real fix_dH_reservoir = 2.5;
  parameter Real fix_dH_pump = 10.0;
  parameter Real fix_dH_turbine = 10.0;

equation
  Inflow.QOut.Q = Inflow_Q;
  Pump.Q = PumpFlow;
  Turbine.Q = TurbineFlow;
  LowerBasin.Q_turbine = ReservoirTurbineFlow;
  LowerBasin.Q_spill = ReservoirSpillFlow;
  V_LowerBasin = LowerBasin.V;
  V_UpperBasin = UpperBasin.V;
  PumpPower = efficiency_pump * PumpFlow * fix_dH_pump * gravity * rho;
  TurbinePower = efficiency_turbine * TurbineFlow * fix_dH_turbine * gravity * rho;
  ReservoirPower = efficiency_reservoir * gravity * rho * fix_dH_reservoir * ReservoirTurbineFlow;
  TotalSystemPower = ReservoirPower + TurbinePower - PumpPower;
  TotalGeneratingPower = ReservoirPower + TurbinePower;
  PumpCost = PumpPower * cost_perP;
  SystemGeneratingRevenue = TotalGeneratingPower*cost_perP;
  TotalSystemRevenue = SystemGeneratingRevenue-PumpCost;

  // Connect elements
  connect(Turbine.QOut, LowerBasin.QLateral[2]) annotation(
    Line(points = {{14, 38}, {-4, 38}, {-4, 14}}));
  connect(Inflow.QOut, LowerBasin.QIn) annotation(
    Line(points = {{-60, -4}, {-37, -4}, {-37, 6}, {-16, 6}}));
  connect(LowerBasin.QOut, Outflow.QIn) annotation(
    Line(points = {{0, 6}, {37, 6}, {37, -2}, {72, -2}}));
  connect(LowerBasin.QLateral[1], Pump.QIn) annotation(
    Line(points = {{-4, 14}, {-24, 14}, {-24, 48}}, thickness = 0.5));
  connect(UpperBasin.QOut, Turbine.QIn) annotation(
    Line(points = {{20, 82}, {14, 82}, {14, 54}}));
  connect(Pump.QOut, UpperBasin.QIn) annotation(
    Line(points = {{-24, 64}, {4, 64}, {4, 82}}));
end PumpedStoragePlant;
