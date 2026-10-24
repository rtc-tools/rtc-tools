import logging
import os
import textwrap

import casadi as ca
import numpy as np

# The LP file format limits line length to 255 characters.
LP_MAX_LINE_WIDTH = 255

# Threshold for treating coefficients and constants as zero (absorbs floating-point noise)
LP_COEFF_EPSILON = 1e-10

# Tolerance for testing whether a constant constraint row satisfies its bounds. Kept
# separate from LP_COEFF_EPSILON (a coefficient-magnitude threshold) because this absorbs
# floating-point noise in a scalar value-vs-bound comparison, a distinct concern.
LP_FEASIBILITY_EPSILON = 1e-9

# Variable fixed to 0 that stands in for the missing variable term of an infeasible constant row
LP_ZERO_VAR = "__zero"

logger = logging.getLogger("rtctools")

# Note: the LP file is currently used for diagnostics/export only. Using it as a
# solve path — by passing the problem as an in-memory data structure directly to
# the solver (bypassing CasADi's solver interface) — would unlock solver-native
# capabilities such as hierarchical multi-objective optimization (goal programming)
# and lazy constraints with separation oracle callbacks.

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


def _format_lp_bound(val: float) -> str:
    """Format a bound value as an LP-standard string.

    Uses +Inf / -Inf for infinite values (CPLEX LP standard).
    Finite values use :.15g formatting, which avoids Python scientific
    notation while keeping approximately full float precision
    (~15 significant digits).
    """
    if np.isposinf(val):
        return "+Inf"
    if np.isneginf(val):
        return "-Inf"
    return f"{val:.15g}"


def _describe(indices, names: list[str] | None) -> str:
    """List up to ten indices, by name if names are given."""
    indices = list(indices)[:10]
    return ", ".join(names[i] for i in indices) if names is not None else str(indices)


def _check_writable_bounds(
    lb: np.ndarray, ub: np.ndarray, kind: str, names: list[str] | None
) -> None:
    """Raise if a bound has no LP form: NaN, a lower bound of +inf or an upper bound of -inf."""
    bad = np.flatnonzero(np.isnan(lb) | np.isnan(ub) | np.isposinf(lb) | np.isneginf(ub))
    if len(bad):
        raise ValueError(
            f"Cannot export LP file: {len(bad)} {kind}(s) have a NaN bound, a lower bound of "
            f"+inf or an upper bound of -inf: {_describe(bad, names)}."
        )


def _build_objective(f: ca.SX, x: ca.SX, var_names: list[str]) -> str:
    """
    Build the LP objective string from the symbolic objective and variable names.

    The objective is wrapped to respect the LP format 255-character line limit,
    but only at whitespace boundaries to preserve variable names and coefficients.

    Args:
        f: Symbolic objective expression (scalar, affine in x).
        x: Decision variable vector (length must match var_names).
        var_names: Human-readable name for each element of x.

    Returns:
        Indented, line-wrapped objective function string ready to follow
        the ``Minimize`` header in an LP file.
    """
    A, b = ca.linear_coeff(f, x)
    A = ca.DM(A)
    b = ca.DM(b)

    ind = np.array(A)[0, :]
    objective = []
    for v, c in zip(var_names, ind, strict=True):
        if abs(c) > LP_COEFF_EPSILON:
            objective.extend(["+" if c > 0 else "-", f"{abs(c):.15g}", v])
    # Add constant term
    b_val = float(b)
    if abs(b_val) > LP_COEFF_EPSILON:
        objective.extend(["+" if b_val > 0 else "-", f"{abs(b_val):.15g}"])

    # Remove leading sign: "+" is invalid as a leading token;
    # "-" becomes a unary minus by merging it with the following value.
    if objective and objective[0] == "+":
        objective.pop(0)
    elif objective and objective[0] == "-":
        objective[1] = "-" + objective[1]
        objective.pop(0)

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


def _build_constraints(
    g: ca.SX,
    x: ca.SX,
    lbg: list,
    ubg: list,
    var_names: list[str],
) -> str:
    """
    Build the LP constraints string.

    Each constraint row ``g[i]`` is written as one or two inequality/equality
    lines depending on its bounds.  Range constraints (finite lb and ub, lb != ub)
    are split into two separate inequalities; the LP ``Ranges`` section is not used.

    Rows with both bounds infinite are left out. A row without variable terms is left out if
    its constant satisfies its bounds, and otherwise written against ``LP_ZERO_VAR`` (fixed to 0
    by an extra row) so the file stays infeasible.

    Note: constraint lines are not wrapped and may exceed the LP format 255-character
    line limit for problems with many variables. Use shorter variable names if needed.

    Args:
        g: Symbolic constraint vector (affine in x).
        x: Decision variable vector (length must match var_names).
        lbg: Lower bounds on each constraint row (``-inf`` for one-sided upper bounds).
        ubg: Upper bounds on each constraint row (``+inf`` for one-sided lower bounds).
        var_names: Human-readable name for each element of x.

    Returns:
        Indented constraints string ready to follow the ``Subject To`` header
        in an LP file.

    Raises:
        ValueError: If a bound cannot be written (see :func:`_check_writable_bounds`) or a
            row has a non-finite constant term.
    """
    A, b = ca.linear_coeff(g, x)
    A = ca.sparsify(ca.DM(A))
    b = ca.DM(b)

    lbg = np.array(ca.veccat(*lbg))[:, 0]
    ubg = np.array(ca.veccat(*ubg))[:, 0]
    _check_writable_bounds(lbg, ubg, "constraint", None)
    A_csc = A.tocsc()
    A_coo = A_csc.tocoo()
    b = np.array(b)[:, 0]
    nonfinite_rows = np.flatnonzero(~np.isfinite(b))
    if len(nonfinite_rows):
        raise ValueError(
            "Cannot export LP file: constraint rows have a non-finite constant term: "
            f"{_describe(nonfinite_rows, None)}."
        )

    constraints = [[] for _ in range(A.shape[0])]
    for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data, strict=True):
        if abs(c) > LP_COEFF_EPSILON:
            constraints[i].extend(["+" if c > 0 else "-", f"{abs(c):.15g}", var_names[j]])

    constraints_str_list = []
    vacuous_rows = []
    infeasible_constant_rows = []
    for i, cur_constr in enumerate(constraints):
        lb, ub, b_i = lbg[i], ubg[i], b[i]
        if np.isneginf(lb) and np.isposinf(ub):
            vacuous_rows.append(i)
            continue
        if cur_constr:
            if cur_constr[0] == "-":
                cur_constr[1] = "-" + cur_constr[1]
            cur_constr.pop(0)
        c_str = " ".join(cur_constr)

        if not c_str:
            lb_ok = (not np.isfinite(lb)) or b_i >= lb - LP_FEASIBILITY_EPSILON
            ub_ok = (not np.isfinite(ub)) or b_i <= ub + LP_FEASIBILITY_EPSILON
            if lb_ok and ub_ok and lb <= ub:
                continue
            infeasible_constant_rows.append(i)
            c_str = LP_ZERO_VAR

        if np.isfinite(lb) and np.isfinite(ub) and lb == ub:
            constraint_line = f"{c_str} = {lb - b_i:.15g}"
        elif np.isfinite(lb) and np.isfinite(ub):
            # Range constraint: lb <= expr <= ub. We do not use the LP ``Ranges``
            # section; we express this as two separate inequalities instead.
            constraints_str_list.append(f"{c_str} >= {lb - b_i:.15g}")
            constraint_line = f"{c_str} <= {ub - b_i:.15g}"
        elif np.isfinite(lb):
            constraint_line = f"{c_str} >= {lb - b_i:.15g}"
        else:
            constraint_line = f"{c_str} <= {ub - b_i:.15g}"
        constraints_str_list.append(constraint_line)
    if infeasible_constant_rows:
        constraints_str_list.append(f"{LP_ZERO_VAR} = 0")
        logger.warning(
            "LP export: %d constraint row(s) reduce to a constant that violates their bounds, "
            "so the problem is infeasible: %s.",
            len(infeasible_constant_rows),
            _describe(infeasible_constant_rows, None),
        )
    if vacuous_rows:
        logger.info(
            "LP export: left out %d constraint row(s) with both bounds infinite: %s.",
            len(vacuous_rows),
            _describe(vacuous_rows, None),
        )
    if not constraints_str_list:
        return ""
    return "  " + "\n  ".join(constraints_str_list)


def _build_bounds(
    var_names: list[str], lbx: list, ubx: list, discrete: list[bool]
) -> tuple[str, list[str], list[str]]:
    """
    Build the LP bounds string and classify discrete variables.

    Binary variables (discrete with bounds [0, 1]) are omitted from the Bounds
    section because the LP ``Binary`` section implicitly defines their bounds.

    Canonical bound forms emitted:
    - Both infinite: ``name Free``
    - Lower bound only: ``lb <= name``
    - Upper bound only: ``-Inf <= name <= ub`` (without ``-Inf`` the default lower bound is 0)
    - Both finite: ``lb <= name <= ub``

    Args:
        var_names: Human-readable name for each variable.
        lbx: Lower bounds on variables (``-inf`` for unbounded below).
        ubx: Upper bounds on variables (``+inf`` for unbounded above).
        discrete: ``True`` for each variable that must take integer values.

    Returns:
        Tuple ``(bounds_str, binary_vars, general_vars)`` where ``bounds_str``
        is the indented bounds section (empty string when nothing to emit),
        ``binary_vars`` is the list of binary (0/1) variable names, and
        ``general_vars`` is the list of general integer variable names.

    Raises:
        ValueError: If a bound cannot be written (see :func:`_check_writable_bounds`).
    """
    _check_writable_bounds(
        np.asarray(lbx, dtype=float), np.asarray(ubx, dtype=float), "variable", var_names
    )
    bounds_list = []
    binary_vars = []
    general_vars = []
    for v, lb, ub, is_discrete in zip(var_names, lbx, ubx, discrete, strict=True):
        if is_discrete:
            if lb == 0 and ub == 1:
                binary_vars.append(v)
                continue  # Binary section implicitly bounds these to [0, 1]
            else:
                general_vars.append(v)
        if not np.isfinite(lb) and not np.isfinite(ub):
            bounds_list.append(f"{v} Free")
        elif not np.isfinite(lb):
            bounds_list.append(f"-Inf <= {v} <= {_format_lp_bound(ub)}")
        elif not np.isfinite(ub):
            bounds_list.append(f"{_format_lp_bound(lb)} <= {v}")
        else:
            bounds_list.append(f"{_format_lp_bound(lb)} <= {v} <= {_format_lp_bound(ub)}")
    bounds_str = "\n  ".join(bounds_list)
    return ("  " + bounds_str) if bounds_str else "", binary_vars, general_vars


def _sanitize_var_names(
    indices_per_member: list[dict[str, list[int]]], num_total: int
) -> list[str]:
    """
    Build LP-compatible variable names for every slot in the combined decision vector.

    A slot is *shared* when every ensemble member maps it to the same variable and local
    index. Shared slots receive no member suffix; other slots get the ``__m{i}`` suffix of
    the lowest member using them (e.g. ``x__t3__m0``).

    Naming scheme:
    - Shared, multi-slot variable: ``"{name}__t{local_index}"``
    - Shared, single-slot variable: ``"{name}"``
    - Per-member, multi-slot: ``"{name}__t{local_index}__m{member}"``
    - Per-member, single-slot: ``"{name}__m{member}"``
    - Unassigned slot (bug indicator): ``"__unassigned_{global_index}"``

    The local index is the 0-based position within the variable's own slot sequence
    (i.e. the time step for collocated variables).  Single-slot variables such as
    integrated states or scalar parameters receive no ``__t`` suffix.

    Square brackets in names are replaced with ``_I`` / ``I_`` because CPLEX does
    not accept ``[`` or ``]`` in LP variable names.

    Args:
        indices_per_member: Per-member mapping from variable name to its slot indices
            in the combined decision vector (as returned by
            ``_collint_variable_indices_as_lists``).
        num_total: Total number of slots in the combined decision vector.

    Returns:
        List of length ``num_total`` mapping each slot index to its LP variable name.
        Every slot is guaranteed to be filled.
    """
    claims = {}  # slot -> {name without member suffix: members using it}
    for m, indices in enumerate(indices_per_member):
        for name, slots in indices.items():
            slots = np.atleast_1d(slots)
            for local_i, idx in enumerate(slots):
                base = f"{name}__t{local_i}" if len(slots) > 1 else name
                claims.setdefault(int(idx), {}).setdefault(base, []).append(m)

    var_names = [None] * num_total
    conflicting_slots = []
    for idx, by_base in claims.items():
        base, members = next(iter(by_base.items()))
        shared = len(members) == len(indices_per_member)
        var_names[idx] = base if shared else f"{base}__m{members[0]}"
        if len(by_base) > 1:
            conflicting_slots.append(idx)
    if conflicting_slots:
        logger.warning(
            "Decision vector slots %s map to different variable names across ensemble "
            "members; this indicates a bug in discretize_controls() or discretize_states() "
            "(possibly a mixin override). The exported LP file may be incorrect.",
            conflicting_slots[:10],
        )

    for i in range(num_total):
        if var_names[i] is None:
            var_names[i] = f"__unassigned_{i}"

    # CPLEX does not like [] in variable names; replace in one vectorized pass
    arr = np.array(var_names, dtype=str)
    arr = np.char.replace(arr, "[", "_I")
    arr = np.char.replace(arr, "]", "I_")
    return arr.tolist()


def _write_lp_file(
    filename: str,
    objective_str: str,
    constraints_str: str,
    bounds_str: str,
    binary_vars: list[str],
    general_vars: list[str],
    output_folder: str = ".",
) -> None:
    """
    Write the LP file according to the LP format.

    Discrete variables with bounds [0, 1] are written to a ``Binary`` section;
    other discrete variables (general integers) go to a ``General`` section.

    If a file with the same name already exists, a numeric suffix is appended
    (e.g. ``problem_1.lp``, ``problem_2.lp``, …).

    Args:
        filename (str): Base name of the LP file (without directory).
        objective_str (str): The objective function string.
        constraints_str (str): The constraints string (empty string if there are no
            constraints to emit; the ``Subject To`` section is then omitted).
        bounds_str (str): The bounds string (empty string if no bounds to emit).
        binary_vars (List[str]): Names of binary (0/1 discrete) variables.
        general_vars (List[str]): Names of general integer variables.
        output_folder (str): Directory where the LP file will be written. Defaults to ".".

    Raises:
        FileNotFoundError: If ``output_folder`` does not exist.
        PermissionError: If the filesystem denies write access.
        RuntimeError: If 100 counter-suffixed filenames already exist.
    """

    stem, ext = os.path.splitext(filename)
    path = os.path.join(output_folder, filename)
    counter = 1
    _MAX_COUNTER = 100
    while True:
        try:
            with open(path, "x") as o:
                o.write("Minimize\n")
                o.write(objective_str + "\n")
                if constraints_str.strip():
                    o.write("Subject To\n")
                    o.write(constraints_str + "\n")
                if bounds_str:
                    o.write("Bounds\n")
                    o.write(bounds_str + "\n")
                if general_vars:
                    o.write("General\n")
                    o.write("\n".join(general_vars) + "\n")
                if binary_vars:
                    o.write("Binary\n")
                    o.write("\n".join(binary_vars) + "\n")
                o.write("End")
            break
        except FileExistsError:
            if counter > _MAX_COUNTER:
                raise RuntimeError(
                    f"Could not write LP file: {_MAX_COUNTER} counter-suffixed files already "
                    f"exist in {output_folder!r} for base name {filename!r}."
                ) from None
            path = os.path.join(output_folder, f"{stem}_{counter}{ext}")
            counter += 1
        except FileNotFoundError:
            raise FileNotFoundError(
                f"LP export output folder does not exist: {output_folder!r}. "
                "Create it, or configure a valid output folder for the problem."
            ) from None
