model ModelHomotopyInputAlias
    Real x;
    input Real theta(fixed=true);
    Real negative_theta;
    input Real u(fixed=false, min=-2, max=2);
    output Real nonlinear_output;
initial equation
    x = 1.1;
equation
    der(x) = u;
    negative_theta = -theta;
    nonlinear_output = x + theta * x^2;
end ModelHomotopyInputAlias;
