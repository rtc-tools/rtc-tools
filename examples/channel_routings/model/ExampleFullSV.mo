model ExampleFullSV
  // Elements
  Real width=500;
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicLinear Channel(
    Q_nominal = 100.0,
    H_b_down = -5.0,
    H_b_up = -5.0,
    friction_coefficient = 0.045,
    use_manning = true,
    length=20000,
    theta = theta,
    use_inertia = true,
    use_convective_acceleration = false,
    use_upwind = false,
    n_level_nodes = 11,
    uniform_nominal_depth = 5.0,
    //bottom_width_down = 30,
    width_up = width,
    width_down = width,
    semi_implicit_step_size = step_size
  )  annotation(Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Level Level annotation(Placement(visible = true, transformation(origin = {60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge Discharge annotation(Placement(visible = true, transformation(origin = {-60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  // Inputs
  input Real Inflow_Q(fixed=true) = Discharge.Q;
  input Real Level_H(fixed=true) = Level.H;
  parameter Real theta;
  parameter Real step_size;
  // Output Channel states
  output Real Channel_Q_up = Discharge.Q;
  output Real Channel_Q_dn = Level.HQ.Q;
  output Real Channel_H_up = Discharge.HQ.H;
  output Real Channel_H_dn = Level.H;
equation
  connect(Channel.HQDown, Level.HQ) annotation(Line(points = {{8, 0}, {60, 0}, {60, 0}, {60, 0}}, color = {0, 0, 255}));
  connect(Discharge.HQ, Channel.HQUp) annotation(Line(points = {{-60, 0}, {-8, 0}, {-8, 0}, {-8, 0}}, color = {0, 0, 255}));
initial equation
  Channel.Q = fill(Inflow_Q, Channel.n_level_nodes + 1);
end ExampleFullSV;
