model ModelDelayLoop
	input Real u(fixed=true);
	Real a;
	Real b;
	output Real y;
equation
	a = delay(b, 1.0);
	b = delay(a, 1.0);
	y = 2 * a + u;
end ModelDelayLoop;
