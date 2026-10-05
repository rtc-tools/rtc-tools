model ModelHomotopyInputAlias
    Real x;
    parameter Real u_max;
    input Real u(fixed=false, min=-2, max=u_max);
    input Real constant_input(fixed=true);
    input Real theta(fixed=true);
    output Real nonlinear_output;
    Real theta_alias;
    Real negative_theta_alias;
initial equation
    x = 1.1;
equation
    der(x) = u + constant_input;
    nonlinear_output = x + theta * x^2;
    theta_alias = theta;
    negative_theta_alias = -theta;
end ModelHomotopyInputAlias;


