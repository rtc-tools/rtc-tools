model ModelHomotopyInput
    Real x;
    parameter Real u_max;
    input Real u(fixed=false, min=-2, max=u_max);
    input Real constant_input(fixed=true);
    input Real theta(fixed=true);
    output Real nonlinear_output;
    output Real delayed_nonlinear_output;
initial equation
    x = 1.1;
equation
    der(x) = u + constant_input;
    nonlinear_output = x + theta * x^2;
    delayed_nonlinear_output = delay(x + theta * x^2, 0.4);
end ModelHomotopyInput;

