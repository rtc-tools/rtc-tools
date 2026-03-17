model ModelSIUnits
	Modelica.Units.SI.VolumeFlowRate q;
	input Real u(fixed=false);
equation
	q + u = 1.0;
end ModelSIUnits;
