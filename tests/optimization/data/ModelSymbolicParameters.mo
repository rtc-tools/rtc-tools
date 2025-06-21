model ModelSymbolicParameters
	parameter Real x_initial;
	parameter Real w_seed;

	Real x(start=x_initial, fixed=true);
	Real w(start=w_seed);

	parameter Real k = 1.0;

	parameter Real a = -1.0;
	parameter Real b = 2.0;
	parameter Real u_min = a * b;
	parameter Real u_max;
	input Real u(fixed=false, min = u_min, max = u_max);
equation
	der(x) = k * x + u;
	der(w) = x;
end ModelSymbolicParameters;