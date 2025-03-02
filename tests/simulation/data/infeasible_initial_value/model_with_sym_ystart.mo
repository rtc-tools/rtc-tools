model ModelWithSymbolicStart
	parameter Real x0 = 30;
	Real x(start=x0-10);
equation
	der(x) = -x;
end ModelWithSymbolicStart;
