model ModelHomotopyParameter
    Real x;
    parameter Real theta;
    input Real u(fixed=false, min=-2, max=2);
    output Real nonlinear_output;
initial equation
    x = 1.1;
equation
    der(x) = u;
    nonlinear_output = x + theta * x^2;
end ModelHomotopyParameter;
