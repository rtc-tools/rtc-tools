model Example
  // Elements
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge Outflow(upwind = false) annotation(
    Placement(visible = true, transformation(origin = {60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Deltares.ChannelFlow.Hydraulic.BoundaryConditions.Discharge Inflow annotation(
    Placement(visible = true, transformation(origin = {-60, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  // Inputs
  input Real Inflow_Q(fixed = true) = Inflow.Q;
  input Real Outflow_Q(fixed = true) = Outflow.Q;
  input SI.Stress reach_1_stress_u(fixed = true);
  input SI.Stress reach_1_stress_v(fixed = true);
  parameter Real theta;
  parameter Real step_size;
  // Output Channel states
  output Real QChannel_Q_up = Inflow.Q;
  output Real QChannel_Q_dn = Outflow.Q;
  //output Real QChannel_Q_md2 = Channel.Q[2];
  output Real HChannel_H_up = Inflow.HQ.H;
  output Real HChannel_H_dn = Outflow.HQ.H;
  //output Real HChannel_H_md2 = Channel.H[2];
 
  /*
  output Real Q_relative_1 = Channel.Q[1];
  output Real Q_relative_2 = Channel.Q[2];
  output Real Q_relative_3 = Channel.Q[3];
  output Real Q_relative_4 = Channel.Q[4];
  output Real Q_relative_5 = Channel.Q[5];
  output Real Q_relative_6 = Channel.Q[6];
  output Real Q_relative_7 = Channel.Q[7];
  output Real Q_relative_8 = Channel.Q[8];
  output Real Q_relative_9 = Channel.Q[9];
  output Real Q_relative_10 = Channel.Q[10];
  output Real Q_relative_11 = Channel.Q[11];
  
  output Real Y_relative_1 = Channel.H[1];
  output Real Y_relative_2 = Channel.H[2];
  output Real Y_relative_3 = Channel.H[3];
  output Real Y_relative_4 = Channel.H[4];
  output Real Y_relative_5 = Channel.H[5];
  output Real Y_relative_6 = Channel.H[6];
  output Real Y_relative_7 = Channel.H[7];
  output Real Y_relative_8 = Channel.H[8];
  output Real Y_relative_9 = Channel.H[9];
  output Real Y_relative_10 = Channel.H[10];
  */
  
  Deltares.ChannelFlow.Hydraulic.Branches.HomotopicLinear Channel(H_b_down = 0, H_b_up = 0.2, Q_nominal = 100, friction_coefficient = 0.02, length = 10000, n_level_nodes = 2, theta = theta, uniform_nominal_depth = 3.154, use_inertia = true, use_manning = true, use_upwind = false, width_down = 100, width_up = 100, rotation_deg = 0.0, wind_stress_u = reach_1_stress_u, wind_stress_v = reach_1_stress_v)  annotation(
    Placement(visible = true, transformation(origin = {0, 0}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(Channel.HQDown, Outflow.HQ) annotation(
    Line(points = {{8, 0}, {60, 0}}, color = {0, 0, 255}));
  connect(Inflow.HQ, Channel.HQUp) annotation(
    Line(points = {{-60, 0}, {-8, 0}}, color = {0, 0, 255}));
initial equation
  Channel.Q = fill(Inflow_Q, Channel.n_level_nodes + 1);
end Example;
