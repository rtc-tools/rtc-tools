import os
import textwrap

import casadi as ca
import numpy as np

# The LP file format limits line length to 255 characters.
LP_MAX_LINE_WIDTH = 255

# Threshold for treating coefficients and constants as zero (absorbs floating-point noise)
LP_COEFF_EPSILON = 1e-10


def build_objective(expand_f_g: ca.Function, var_names: list[str]) -> str:
    """
    Build the LP objective string from the symbolic function and variable names.

    The objective is wrapped to respect the LP format 255-character line limit,
    but only at whitespace boundaries to preserve variable names and coefficients.

    Args:
        expand_f_g (ca.Function): The expanded CasADi function returning (f, g).
        var_names (List[str]): List of variable names.

    Returns:
        str: The formatted objective function string.
    """
    X = ca.SX.sym("X", expand_f_g.nnz_in())
    f, _ = expand_f_g(X)
    Af = ca.Function("Af", [X], [ca.jacobian(f, X)])
    bf = ca.Function("bf", [X], [f])

    A = Af(0)
    A = ca.sparsify(A)
    b = bf(0)
    b = ca.sparsify(b)

    ind = np.array(A)[0, :]
    objective = []
    for v, c in zip(var_names, ind, strict=True):
        if abs(c) > LP_COEFF_EPSILON:
            objective.extend(["+" if c > 0 else "-", str(abs(c)), v])
    if objective and objective[0] == "-":
        objective[1] = "-" + objective[1]
    if objective:
        objective.pop(0)

    # Add constant term if non-zero (using epsilon threshold for robustness)
    b_val = float(b)
    if abs(b_val) > LP_COEFF_EPSILON:
        objective.extend(["+" if b_val > 0 else "-", str(abs(b_val))])

    objective_str = " ".join(objective)
    # Wrap at word boundaries only (spaces), never breaking tokens like variable names
    wrapped_objective = "\n".join(
        textwrap.wrap(
            "  " + objective_str,
            width=LP_MAX_LINE_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )
    return wrapped_objective


def build_constraints(
    expand_f_g: ca.Function,
    lbg: list,
    ubg: list,
    var_names: list[str],
) -> str:
    """
    Build the LP constraints string.

    Note: Constraints are not wrapped and may exceed the LP format 255-character
    line limit if they contain many variables. Consider using shorter variable names
    for very large problems.

    Args:
        expand_f_g: The expanded CasADi function
        lbg: Lower bounds on constraints
        ubg: Upper bounds on constraints
        var_names: List of variable names

    Returns:
        str: The formatted constraints string
    """
    X = ca.SX.sym("X", expand_f_g.nnz_in())
    _, g = expand_f_g(X)
    Af = ca.Function("Af", [X], [ca.jacobian(g, X)])
    bf = ca.Function("bf", [X], [g])

    A = Af(0)
    A = ca.sparsify(A)
    b = bf(0)
    b = ca.sparsify(b)

    lbg = np.array(ca.veccat(*lbg))[:, 0]
    ubg = np.array(ca.veccat(*ubg))[:, 0]
    A_csc = A.tocsc()
    A_coo = A_csc.tocoo()
    b = np.array(b)[:, 0]

    constraints = [[] for _ in range(A.shape[0])]
    for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data, strict=True):
        if abs(c) > LP_COEFF_EPSILON:
            constraints[i].extend(["+" if c > 0 else "-", str(abs(c)), var_names[j]])

    constraints_str_list = []
    for i, cur_constr in enumerate(constraints):
        lb, ub, b_i = lbg[i], ubg[i], b[i]
        if cur_constr:
            if cur_constr[0] == "-":
                cur_constr[1] = "-" + cur_constr[1]
            cur_constr.pop(0)
        c_str = " ".join(cur_constr)

        # If no variable terms exist, this is a constant constraint (0 on LHS)
        if not c_str:
            c_str = "0"

        if np.isfinite(lb) and np.isfinite(ub) and lb == ub:
            constraint_line = f"{c_str} = {lb - b_i}"
        elif np.isfinite(lb):
            constraint_line = f"{c_str} >= {lb - b_i}"
        elif np.isfinite(ub):
            constraint_line = f"{c_str} <= {ub - b_i}"
        else:
            raise ValueError(
                f"Constraint {i} has no finite bound (lbg={lb}, ubg={ub}, b_i={b_i}). "
                "At least one of lbg or ubg must be finite to write a valid LP constraint."
            )
        constraints_str_list.append(constraint_line)
    constraints_str = "  " + "\n  ".join(constraints_str_list)
    return constraints_str


def build_bounds(var_names: list[str], lbx: list, ubx: list) -> str:
    """
    Build the LP bounds string.

    Args:
        var_names (List[str]): List of variable names.
        lbx (List[Any]): Lower bounds on variables.
        ubx (List[Any]): Upper bounds on variables.

    Returns:
        str: The formatted bounds string.
    """
    bounds_list = []
    for v, lb, ub in zip(var_names, lbx, ubx, strict=True):
        bounds_list.append(f"{lb} <= {v} <= {ub}")
    bounds_str = "  " + "\n  ".join(bounds_list)
    return bounds_str


def sanitize_var_names(indices: dict[str, int | slice], num_total: int) -> list[str]:
    """
    Sanitize and generate variable names compatible with LP solvers.

    Maps decision vector indices to human-readable names, handling both named variables
    (integers or slices) and unassigned slots. Variable names are placed at their exact
    positions in the decision vector to ensure LP coefficients align correctly.

    Naming scheme:
    - Named variables: "{name}__{local_index}" (e.g., "x__0", "u__1")
    - Unassigned slots: "X__{global_index}" (for debugging; indicates no mapping was provided)

    Args:
        indices (Dict[str, Union[int, slice]]): Mapping of variable names to their
            positions in the decision vector. Keys are variable names; values are either:
            - int: single variable at that absolute index
            - slice: contiguous range of variables [start:stop:step]
        num_total (int): Total number of variables in the decision vector.

    Returns:
        List[str]: Variable names indexed by their position in the decision vector.
            Length equals num_total. All slots (including unassigned) are filled.
    """
    # Pre-allocate array so we fill entries at their actual indices from the decision
    # vector, not in iteration order. This is critical because indices dict may not be
    # ordered, and slices represent absolute positions with offsets.
    var_names = [None] * num_total

    for k, v in indices.items():
        if isinstance(v, int):
            # Single index: place name at that exact position
            var_names[v] = f"{k}__{v}"
        else:
            # Slice: iterate over actual index range [start, stop, step]
            # Use the global index (i) in the variable name for consistency
            # with the single-index case
            step = 1 if v.step is None else v.step
            for i in range(v.start, v.stop, step):
                var_names[i] = f"{k}__{i}"

    # Fill remaining unassigned slots with deterministic names for debuggability.
    # These may be intermediate variables (derivatives) or unused slots, but naming them
    # explicitly prevents silent variable mislabeling and aids debugging.
    for i in range(num_total):
        if var_names[i] is None:
            var_names[i] = f"X__{i}"

    # CPLEX does not like [] in variable names
    for i in range(len(var_names)):
        var_names[i] = var_names[i].replace("[", "_I").replace("]", "I_")

    return var_names


def write_lp_file(
    filename: str,
    objective_str: str,
    constraints_str: str,
    bounds_str: str,
    var_names: list[str],
    discrete: list[bool],
    output_folder: str = ".",
) -> None:
    """
    Write the LP file according to the LP format.

    Args:
        filename (str): Base name of the LP file (without directory).
        objective_str (str): The objective function string.
        constraints_str (str): The constraints string.
        bounds_str (str): The bounds string.
        var_names (List[str]): List of variable names.
        discrete (List[bool]): Boolean list indicating discrete variables.
        output_folder (str): Directory where the LP file will be written. Defaults to ".".
    """
    path = os.path.join(output_folder, filename)
    with open(path, "w") as o:
        o.write("Minimize\n")
        o.write(objective_str + "\n")
        o.write("Subject To\n")
        o.write(constraints_str + "\n")
        o.write("Bounds\n")
        o.write(bounds_str + "\n")
        if any(discrete):
            o.write("General\n")
            discrete_vars = "\n".join(
                v for v, is_discrete in zip(var_names, discrete, strict=True) if is_discrete
            )
            o.write(discrete_vars + "\n")
        o.write("End")
