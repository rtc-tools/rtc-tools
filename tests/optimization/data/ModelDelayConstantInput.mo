model ModelDelayConstantInput
	input Real u(fixed=true);
	output Real y;
equation
	y = 2 * delay(u, 0.1);
end ModelDelayConstantInput;
