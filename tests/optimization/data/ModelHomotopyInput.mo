model ModelHomotopyInput
    Real x;
    input Real theta(fixed=true);
    input Real u(fixed=false, min=-2, max=2);
    output Real nonlinear_output;
    output Real delayed_nonlinear_output;
initial equation
    x = 1.1;
equation
    der(x) = u;
    nonlinear_output = x + theta * x^2;
    delayed_nonlinear_output = delay(x + theta * x^2, 0.4);
end ModelHomotopyInput;
