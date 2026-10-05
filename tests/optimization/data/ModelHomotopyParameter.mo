model ModelHomotopyParameter
    Real x;
    parameter Real u_max;
    parameter Real theta;
    input Real u(fixed=false, min=-2, max=u_max);
    input Real constant_input(fixed=true);
    output Real nonlinear_output;
initial equation
    x = 1.1;
equation
    der(x) = u + constant_input;
    nonlinear_output = x + theta * x^2;
end ModelHomotopyParameter;

