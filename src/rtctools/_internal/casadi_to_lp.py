import os
import textwrap

import casadi as ca
import numpy as np

# The LP file format limits line length to 255 characters.
LP_MAX_LINE_WIDTH = 255

# Threshold for treating coefficients and constants as zero (absorbs floating-point noise)
LP_COEFF_EPSILON = 1e-10

# Note: the LP file is currently used for diagnostics/export only. Using it as a
# solve path — by passing the problem as an in-memory data structure directly to
# the solver (bypassing CasADi's solver interface) — would unlock solver-native
# capabilities such as hierarchical multi-objective optimization (goal programming)
# and lazy constraints with separation oracle callbacks. Both require solver support
# and would likely outperform the current epsilon-constraint approach in rtc-tools
# for large MILP problems.

# The LP format supports several sections not currently implemented here:
#   - Lazy Constraints: hints to the solver that constraints can be added lazily
#   - User Cuts: cutting planes supplied by the user
#   - SOS: special ordered sets (type 1 and 2)
#   - Semi-continuous / Semi-integer: variables that are either zero or within a range
#   - PWLObj: piecewise-linear objective functions
#   - General Constraints: MIN, MAX, ABS, OR, AND, NORM, PWL function constraints
#   - Scenarios: multiple scenario data (objective/constraint/bound changes per scenario)
#   - Multi-objective: weighted/hierarchical objective sections
# See https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html#lp-format


def build_objective(f: ca.SX, x: ca.SX, var_names: list[str]) -> str:
    """
    Build the LP objective string from the symbolic objective and variable names.

    The objective is wrapped to respect the LP format 255-character line limit,
    but only at whitespace boundaries to preserve variable names and coefficients.

    Args:
        f: Symbolic objective expression (affine in x).
        x: Decision variable vector.
        var_names: List of variable names.

    Returns:
        The formatted objective function string.
    """
    A, b = ca.linear_coeff(f, x)
    A = ca.DM(A)
    b = ca.DM(b)

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
    # Emit an explicit zero objective when all terms are below epsilon threshold;
    # some LP parsers reject a blank objective line after "Minimize".
    if not objective_str:
        objective_str = "0"
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
    g: ca.SX,
    x: ca.SX,
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
        g: Symbolic constraint expression (affine in x).
        x: Decision variable vector.
        lbg: Lower bounds on constraints
        ubg: Upper bounds on constraints
        var_names: List of variable names

    Returns:
        The formatted constraints string.
    """
    A, b = ca.linear_coeff(g, x)
    A = ca.sparsify(ca.DM(A))
    b = ca.DM(b)

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


def build_bounds(var_names: list[str], lbx: list, ubx: list, discrete: list[bool]) -> str:
    """
    Build the LP bounds string.

    Binary variables (discrete with bounds [0, 1]) are omitted because the LP
    ``Binary`` section implicitly defines their bounds.

    Args:
        var_names (List[str]): List of variable names.
        lbx (List[Any]): Lower bounds on variables.
        ubx (List[Any]): Upper bounds on variables.
        discrete (List[bool]): Boolean list indicating discrete variables.

    Returns:
        str: The formatted bounds string.
    """
    bounds_list = []
    for v, lb, ub, is_discrete in zip(var_names, lbx, ubx, discrete, strict=True):
        if is_discrete and lb == 0 and ub == 1:
            continue  # Binary section implicitly bounds these to [0, 1]
        bounds_list.append(f"{lb} <= {v} <= {ub}")
    bounds_str = "  " + "\n  ".join(bounds_list)
    return bounds_str


def sanitize_var_names(
    indices_per_member: list[dict[str, int | slice]], num_total: int
) -> list[str]:
    """
    Sanitize and generate variable names compatible with LP solvers.

    Maps decision vector indices to human-readable names for a combined (multi-ensemble)
    decision vector. Controls are shared across ensemble members and appear without a member
    suffix. States differ per member and receive a ``_m{i}`` suffix (e.g., ``x__9_m0``).

    Naming scheme:
    - Shared variables (same indices in all members): ``"{name}__{global_index}"``
    - Per-member variables: ``"{name}__{global_index}_m{member}"``
    - Unassigned slots: ``"__{global_index}"`` (debugging aid)

    Args:
        indices_per_member: List of per-member index dicts. Each dict maps variable names
            to their positions in the combined decision vector as int, slice, or iterable.
        num_total: Total number of variables in the combined decision vector.

    Returns:
        Variable names indexed by their position in the decision vector.
        Length equals num_total. All slots are filled.
    """
    var_names = [None] * num_total

    # Determine which variables are shared (identical indices in all members)
    first = indices_per_member[0]
    shared = {
        k
        for k in first
        if all(indices_per_member[m].get(k) == first[k] for m in range(len(indices_per_member)))
    }

    def _fill(name, v, suffix=""):
        if isinstance(v, int):
            var_names[v] = f"{name}__{v}{suffix}"
        elif isinstance(v, slice):
            step = 1 if v.step is None else v.step
            for i in range(v.start, v.stop, step):
                var_names[i] = f"{name}__{i}{suffix}"
        else:
            for i in v:
                var_names[int(i)] = f"{name}__{int(i)}{suffix}"

    for member, indices in enumerate(indices_per_member):
        for k, v in indices.items():
            suffix = "" if k in shared else f"_m{member}"
            _fill(k, v, suffix)

    for i in range(num_total):
        if var_names[i] is None:
            var_names[i] = f"__{i}"

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
    lbx: list,
    ubx: list,
    output_folder: str = ".",
) -> None:
    """
    Write the LP file according to the LP format.

    Discrete variables with bounds [0, 1] are written to a ``Binary`` section;
    other discrete variables (general integers) go to a ``General`` section.

    Args:
        filename (str): Base name of the LP file (without directory).
        objective_str (str): The objective function string.
        constraints_str (str): The constraints string.
        bounds_str (str): The bounds string.
        var_names (List[str]): List of variable names.
        discrete (List[bool]): Boolean list indicating discrete variables.
        lbx (List): Lower bounds on variables.
        ubx (List): Upper bounds on variables.
        output_folder (str): Directory where the LP file will be written. Defaults to ".".

    """
    binary_vars = []
    general_vars = []
    for v, is_discrete, lb, ub in zip(var_names, discrete, lbx, ubx, strict=True):
        if is_discrete:
            if lb == 0 and ub == 1:
                binary_vars.append(v)
            else:
                general_vars.append(v)

    path = os.path.join(output_folder, filename)
    with open(path, "x") as o:
        o.write("Minimize\n")
        o.write(objective_str + "\n")
        o.write("Subject To\n")
        o.write(constraints_str + "\n")
        o.write("Bounds\n")
        o.write(bounds_str + "\n")
        if general_vars:
            o.write("General\n")
            o.write("\n".join(general_vars) + "\n")
        if binary_vars:
            o.write("Binary\n")
            o.write("\n".join(binary_vars) + "\n")
        o.write("End")
