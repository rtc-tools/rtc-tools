import logging

import casadi as ca

logger = logging.getLogger("rtctools")

# Limit the number of times we try to substitute in external functions, e.g. in
# case of infinite recursion. Generally unlikely that we will hit this limit for
# any reasonable use case.
MAX_SUBSTITUTE_DEPTH = 10


def is_affine(expr, symbols):
    try:
        Af = ca.Function("f", [symbols], [ca.jacobian(expr, symbols)]).expand()
    except RuntimeError as error:
        if "'eval_sx' not defined for" in str(error):
            Af = ca.Function("f", [symbols], [ca.jacobian(expr, symbols)])
        else:
            raise
    return Af.sparsity_jac(0, 0).nnz() == 0


def nullvertcat(*L):
    """
    Like vertcat, but creates an MX with consistent dimensions even if L is empty.
    """
    if len(L) == 0:
        return ca.DM(0, 1)
    else:
        return ca.vertcat(*L)


def reduce_matvec(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product.

    This reduces the number of nodes required to represent the linear operations.
    """
    Af = ca.Function("Af", [ca.MX()], [ca.jacobian(e, v)])
    A = Af(ca.DM())
    return ca.reshape(ca.mtimes(A, v), e.shape)


def substitute_in_external(expr, symbols, values):
    if len(symbols) == 0 or all(isinstance(x, ca.DM) for x in expr):
        return expr
    elif not expr:
        return []
    else:
        for _ in range(MAX_SUBSTITUTE_DEPTH):
            f = ca.Function("f", symbols, expr).expand()
            expr = f.call(values, True, False)
            if expr[0].is_constant():
                break

        return expr


def interpolate(ts, xs, t, equidistant, mode=0):
    if mode == 0:
        mode_str = "linear"
    elif mode == 1:
        mode_str = "floor"
    else:
        mode_str = "ceil"

    # CasADi fails if there is just a single point. Just "extrapolate" based on
    # that point, just as CasADi would do for entries in 't' outside the range
    # of 'ts'.
    if len(ts) == 1:
        assert xs.size1() == 1
        return ca.vertcat(*[xs] * len(t))

    return ca.interp1d(ts, xs, t, mode_str, equidistant)
