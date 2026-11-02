model ExampleOptimization
  extends Example;
  input Modelica.Units.SI.VolumeFlowRate Q_dam_middle(fixed = false, min = 0.0, max = 1000.0, nominal = 100.0) = dam_middle.Q;
  input Modelica.Units.SI.VolumeFlowRate Q_dam_upstream(fixed = false, min = 0.0, max = 1000.0, nominal = 100.0) = dam_upstream.Q;
end ExampleOptimization;
