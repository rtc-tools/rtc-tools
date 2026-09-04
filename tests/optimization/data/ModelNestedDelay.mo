model ModelNestedDelay
	input Real u(fixed=true);
	output Real y;
equation
	// The factor 2 keeps y from aliasing the delay variable, which is pinned to zero
	y = 2 * delay(delay(u, 0.1), 0.1);
end ModelNestedDelay;
