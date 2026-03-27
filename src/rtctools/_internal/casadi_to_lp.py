"""Build LP/MILP file representations from CasADi symbolic expressions.

Provides low-level builders (objective, constraints, bounds) and a file writer.
Called by CollocatedIntegratedOptimizationProblem when ``export_lp=True`` is set
in ``solver_options()``.

The LP file is currently used for diagnostics/export only. Using it as a solve path —
by passing the problem as an in-memory data structure directly to the solver (bypassing
CasADi's solver interface) — would unlock solver-native capabilities such as hierarchical
multi-objective optimization (goal programming) and lazy constraints with separation oracle
callbacks.

LP format sections not currently implemented here include: Lazy Constraints, User Cuts,
SOS (type 1 and 2), Semi-continuous/Semi-integer, PWLObj, General Constraints
(MIN/MAX/ABS/OR/AND/NORM/PWL), Scenarios, and Multi-objective.
See https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html#lp-format
"""

import logging
import os
import re
import textwrap

import casadi as ca
import numpy as np

# The LP file format limits line length to 255 characters.
LP_MAX_LINE_WIDTH = 255

# Threshold for treating coefficients and constants as zero (absorbs floating-point noise)
LP_COEFF_EPSILON = 1e-10

# Characters forbidden in LP constraint/variable names (LP format requirement)
_LP_FORBIDDEN_CHARS = " \t\r\n#+-*/^()[]:'\""
_LP_FORBIDDEN_TRANS = str.maketrans(_LP_FORBIDDEN_CHARS, "_" * len(_LP_FORBIDDEN_CHARS))

# Suffixes appended to the two lines emitted for range constraints.
_LP_RANGE_SUFFIXES = ("_lb", "_ub")

# Prefix for dummy variables used when a constraint reduces to a constant expression.
# LP format requires at least one variable term per row; we emit e.g. ``b_i _constant_<name> >= lb``
# with ``_constant_<name>`` fixed to 1 in Bounds, preserving infeasibility detectability.
_LP_CONSTANT_VAR_PREFIX = "_constant_"

# Reserved suffix patterns appended by the LP naming scheme after base names are fixed.
# A user name ending with any of these patterns would produce ambiguous or colliding labels:
#   _d{n}  — deduplication index        e.g. foo_d0, foo_d1
#   _m{n}  — ensemble member suffix     e.g. foo_m0, foo_m1
#   _t{n}  — time-step suffix           e.g. foo_t0, foo_t1
#   _lb / _ub — range-constraint sides  e.g. foo_lb, foo_ub
_LP_RESERVED_SUFFIX_RE = re.compile(r"(_d\d+|_m\d+|_t\d+|_lb|_ub)$")

# Prefixes reserved for auto-generated constraint names in LP export.
# User-provided names that start with these prefixes may cause confusion.
LP_RESERVED_NAME_PREFIXES = (
    "initial_residual_",
    "initial_derivative_",
    "collocation_",
    "delay_",
    "constraint_",
    "path_constraint_",
    "single_pass_objective_",
)

logger = logging.getLogger("rtctools")


def _sanitize_constraint_name(name: str) -> str:
    """Sanitize a constraint name for LP format compatibility.

    Replaces characters that are forbidden in LP constraint names with underscores.
    """
    return name.translate(_LP_FORBIDDEN_TRANS)


def _deduplicate_constraint_names(constraint_names: list[str]) -> list[str]:
    """Deduplicate a constraint name list in-place and return it.

    Returns immediately when all names are unique (fast path). When a name appears
    more than once, ``_d0``, ``_d1``, … suffixes are appended to each occurrence,
    skipping any suffix that already exists as a distinct name. The first occurrence
    is renamed retroactively. Overall complexity is O(n).

    A duplicate in a reserved internal-name prefix is a bug; a warning is emitted
    so it surfaces clearly.
    """
    name_set = set(constraint_names)
    if len(name_set) == len(constraint_names):
        return constraint_names  # all unique, nothing to do
    # Two structures, two responsibilities:
    #   name_set  — all current names in the list; answers "is this candidate taken?"
    #   seen      — per base name: (next_counter, first_occurrence_index); lets the
    #               first occurrence be renamed retroactively without a second O(n) scan.
    seen: dict[str, tuple[int, int]] = {}
    for i, name in enumerate(constraint_names):
        if name in seen:
            if any(name.startswith(p) for p in LP_RESERVED_NAME_PREFIXES):
                logger.warning(
                    "Internal constraint name %r appears more than once; "
                    "this indicates a bug in constraint name generation.",
                    name,
                )
            counter, first_idx = seen[name]
            if first_idx is not None:
                # Rename the first occurrence retroactively.
                while f"{name}_d{counter}" in name_set:
                    counter += 1
                constraint_names[first_idx] = f"{name}_d{counter}"
                name_set.discard(name)
                name_set.add(constraint_names[first_idx])
                counter += 1
                seen[name] = (counter, None)  # first occurrence already renamed
            while f"{name}_d{counter}" in name_set:
                counter += 1
            constraint_names[i] = f"{name}_d{counter}"
            name_set.add(constraint_names[i])
            seen[name] = (counter + 1, None)
        else:
            seen[name] = (0, i)
    return constraint_names


def _build_user_constraint_base_names(user_tuples: list, auto_prefix: str) -> list[str]:
    """Sanitize, validate and deduplicate user-provided constraint base names.

    Extracts the optional name from position 3 of each constraint tuple, falling back
    to ``"{auto_prefix}_{i}"`` when absent. Each name is sanitized (forbidden characters
    replaced with ``_``), validated against reserved prefixes, and renamed with a ``_ren``
    suffix if it ends with a reserved LP suffix (``_d{n}``, ``_m{n}``, ``_t{n}``, ``_lb``,
    ``_ub``). The resulting base names are then deduplicated with ``_d{n}`` indices. Must
    be called before time-index or ensemble-member suffixes are appended.

    Args:
        user_tuples: List of constraint tuples ``(expr, lb, ub[, name])``.
        auto_prefix: Prefix used for auto-generated names (e.g. ``"constraint"`` or
            ``"path_constraint"``).

    Returns:
        List of deduplicated base names, one per tuple.
    """
    base_names = []
    for i, c in enumerate(user_tuples):
        if len(c) > 3 and c[3]:
            raw = c[3]
            name = _sanitize_constraint_name(raw)
            _validate_lp_constraint_name(name)
            if _LP_RESERVED_SUFFIX_RE.search(raw) or _LP_RESERVED_SUFFIX_RE.search(name):
                new_name = name + "_ren"
                logger.warning(
                    "User constraint name %r ends with a reserved LP suffix (_d{n}, _m{n}, "
                    "_t{n}, _lb, _ub); renamed to %r to avoid collision with auto-generated "
                    "labels.",
                    raw,
                    new_name,
                )
                name = new_name
        else:
            name = f"{auto_prefix}_{i}"
        base_names.append(name)
    _deduplicate_constraint_names(base_names)
    return base_names


def _validate_lp_constraint_name(name: str) -> None:
    """Validate a user-provided LP constraint name.

    Emits a warning when the name starts with a prefix reserved for auto-generated
    constraint names, as this may cause confusion in LP output.

    Reserved-suffix detection (``_d{n}``, ``_m{n}``, ``_t{n}``, ``_lb``, ``_ub``) is
    handled separately in :func:`_build_user_constraint_base_names`, which also renames
    the offending name, so it is not repeated here.
    """
    if any(name.startswith(p) for p in LP_RESERVED_NAME_PREFIXES):
        logger.warning(
            "User-provided constraint name %r starts with a prefix reserved for "
            "auto-generated LP constraint names; this may cause confusion in LP output.",
            name,
        )


def _check_nan_bounds(lb: np.ndarray, ub: np.ndarray, names: list[str]) -> None:
    """Raise ValueError if any lower or upper bound is NaN, listing the affected names."""
    nan_bounds_list = [f"{names[i]} (lower)" for i in np.where(np.isnan(lb))[0]] + [
        f"{names[i]} (upper)" for i in np.where(np.isnan(ub))[0]
    ]
    if nan_bounds_list:
        raise ValueError(
            f"NaN bounds found for {nan_bounds_list}; check that all bounds are finite or ±inf."
        )


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
    constraint_names: list[str] | None = None,
) -> tuple[str, str]:
    """
    Build the LP constraints string.

    Note: Constraints are not wrapped and may exceed the LP format 255-character
    line limit if they contain many variables. Consider using shorter variable names
    for very large problems.

    When a constraint reduces to a constant (no variable terms), feasible rows are
    skipped silently. Infeasible ones are represented via a dummy variable
    ``_constant_<name>`` fixed to 1 in Bounds, so solvers can detect the infeasibility
    during presolve. Constraints with both bounds infinite are skipped with a warning
    as they likely indicate a formulation error.

    Args:
        g: Symbolic constraint expression (affine in x).
        x: Decision variable vector.
        lbg: Lower bounds on constraints.
        ubg: Upper bounds on constraints.
        var_names: List of variable names.
        constraint_names: Optional list of constraint names. When provided, each
            constraint is prefixed as ``name: expr op rhs``. Range constraints
            emit two lines with ``_lb`` / ``_ub`` appended. Names are sanitized, guarded
            against range-suffix collisions, and deduplicated before expansion.
            If ``None``, constraints are written without labels.

    Returns:
        Tuple of (constraints_str, extra_bounds_str) where constraints_str is the
        formatted Subject To section content, and extra_bounds_str contains any
        dummy variable bound lines that must be appended to the Bounds section
        (empty string when no constant rows were encountered).
    """
    A, b = ca.linear_coeff(g, x)
    A = ca.sparsify(ca.DM(A))
    b = ca.DM(b)

    lbg = np.array(ca.veccat(*lbg))[:, 0]
    ubg = np.array(ca.veccat(*ubg))[:, 0]

    constraint_labels = (
        constraint_names if constraint_names is not None else [str(i) for i in range(len(lbg))]
    )
    _check_nan_bounds(lbg, ubg, constraint_labels)

    A_csc = A.tocsc()
    A_coo = A_csc.tocoo()
    b = np.array(b)[:, 0]

    constraints = [[] for _ in range(A.shape[0])]
    for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data, strict=True):
        if abs(c) > LP_COEFF_EPSILON:
            constraints[i].extend(["+" if c > 0 else "-", f"{abs(c):.15g}", var_names[j]])

    # Sanitize constraint names once upfront.
    # Callers are expected to have already sanitized, validated reserved suffixes, and
    # deduplicated names (e.g. via _build_user_constraint_base_names). The sanitization
    # here is a safety net for direct callers (e.g. unit tests).
    if constraint_names is not None:
        sanitized_names = []
        for raw in constraint_names:
            sanitized = _sanitize_constraint_name(raw)
            if sanitized != raw:
                invalid_chars = sorted({c for c in raw if c in _LP_FORBIDDEN_CHARS})
                logger.debug(
                    "Constraint name %r contains forbidden characters %r; sanitized to %r.",
                    raw,
                    invalid_chars,
                    sanitized,
                )
            sanitized_names.append(sanitized)
    else:
        sanitized_names = None

    def _prefix(name: str, lb: float, ub: float, side: str) -> str:
        """Return the label prefix for one emitted line, or '' when names are disabled.

        ``side`` is ``"eq"``, ``"lb"``, or ``"ub"``.  For equality and single-sided
        constraints the name is used as-is; for range constraints (both bounds finite)
        the appropriate ``_lb`` / ``_ub`` suffix is appended.
        """
        if sanitized_names is None:
            return ""
        is_range = lb > -np.inf and ub < np.inf and not (np.isfinite(lb) and lb == ub)
        if is_range:
            suffix = _LP_RANGE_SUFFIXES[0] if side == "lb" else _LP_RANGE_SUFFIXES[1]
            label = f"{name}{suffix}" if name else ""
        else:
            label = name
        return f"{label}: " if label else ""

    constraints_str_list = []
    extra_bounds_list = []  # dummy variable bound lines for constant rows
    for i, cur_constr in enumerate(constraints):
        lb, ub, b_i = lbg[i], ubg[i], b[i]
        if cur_constr:
            if cur_constr[0] == "-":
                cur_constr[1] = "-" + cur_constr[1]
            cur_constr.pop(0)
        c_str = " ".join(cur_constr)

        name = sanitized_names[i] if sanitized_names is not None else ""

        if not c_str:
            # All variable coefficients are below epsilon: the expression is a constant b_i.
            # LP format requires at least one variable term per row. Feasible constant rows
            # (lb <= b_i <= ub) are skipped silently. Infeasible ones are represented via a
            # dummy variable fixed to 1, so the solver can detect the infeasibility via presolve.
            lb_ok = (not np.isfinite(lb)) or b_i >= lb - LP_COEFF_EPSILON
            ub_ok = (not np.isfinite(ub)) or b_i <= ub + LP_COEFF_EPSILON
            if lb_ok and ub_ok:
                continue
            # Infeasible — warn and emit using a dummy variable named after the constraint.
            # We use coefficient 1 and move b_i to the RHS (lb - b_i, ub - b_i), matching
            # the normal-row convention and avoiding a zero-coefficient term when b_i == 0.
            label = constraint_labels[i]
            logger.warning(
                "Constraint %r reduces to a constant (%s) that violates bounds [%s, %s]. "
                "The problem is infeasible. Representing via dummy variable in LP export.",
                label,
                b_i,
                lb,
                ub,
            )
            dummy_var = _sanitize_constraint_name(f"{_LP_CONSTANT_VAR_PREFIX}{label}")
            extra_bounds_list.append(f"1 <= {dummy_var} <= 1")
            if np.isfinite(lb) and lb == ub:
                constraints_str_list.append(
                    f"{_prefix(name, lb, ub, 'eq')}{dummy_var} = {_format_lp_bound(lb - b_i)}"
                )
            else:
                if lb > -np.inf:
                    constraints_str_list.append(
                        f"{_prefix(name, lb, ub, 'lb')}{dummy_var} >= {_format_lp_bound(lb - b_i)}"
                    )
                if ub < np.inf:
                    constraints_str_list.append(
                        f"{_prefix(name, lb, ub, 'ub')}{dummy_var} <= {_format_lp_bound(ub - b_i)}"
                    )
            continue

        if not np.isfinite(lb) and not np.isfinite(ub):
            # Both bounds infinite: vacuous constraint, carries no information.
            # Skip but warn — this likely indicates a formulation error.
            logger.warning(
                "Constraint %d has no finite bound (lbg=%s, ubg=%s); skipping in LP export.",
                i,
                lb,
                ub,
            )
            continue

        if np.isfinite(lb) and lb == ub:  # Equality constraint: emit a single = line.
            constraints_str_list.append(
                f"{_prefix(name, lb, ub, 'eq')}{c_str} = {_format_lp_bound(lb - b_i)}"
            )
        else:
            if lb > -np.inf:
                constraints_str_list.append(
                    f"{_prefix(name, lb, ub, 'lb')}{c_str} >= {_format_lp_bound(lb - b_i)}"
                )
            if ub < np.inf:
                constraints_str_list.append(
                    f"{_prefix(name, lb, ub, 'ub')}{c_str} <= {_format_lp_bound(ub - b_i)}"
                )

    constraints_str = "  " + "\n  ".join(constraints_str_list)
    extra_bounds_str = "\n  ".join(extra_bounds_list)
    return constraints_str, ("  " + extra_bounds_str if extra_bounds_str else "")


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
    - Upper bound only: ``name <= ub``
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
    """
    _check_nan_bounds(np.asarray(lbx, dtype=float), np.asarray(ubx, dtype=float), var_names)

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
            bounds_list.append(f"{v} <= {_format_lp_bound(ub)}")
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

    A variable is *shared* when every ensemble member maps it to the exact same
    slot indices.  Shared variables receive no member suffix; per-member variables
    get a ``__m{i}`` suffix (e.g. ``x__t3__m0``).

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
    # Single pass over variables: for each variable, determine whether its slots are shared
    # (identical slot list across all members) or per-member, then write names directly.
    # Sharing is checked at the slot level using numpy array equality, which correctly
    # handles the ControlTreeMixin partial-sharing case (same variable name, different
    # slot lists per member after branching).
    var_names = [None] * num_total
    all_names = sorted(set().union(*(d.keys() for d in indices_per_member)))
    for name in all_names:
        member_slots = {
            m: np.array(indices_per_member[m][name], dtype=np.int32)
            for m in range(len(indices_per_member))
            if name in indices_per_member[m]
        }

        # Check if all members that have this variable share the exact same slots.
        slots_list = list(member_slots.values())
        shared = len(member_slots) == len(indices_per_member) and all(
            np.array_equal(slots_list[0], s) for s in slots_list[1:]
        )

        if shared:
            slots = slots_list[0]
            n_slots = len(slots)
            for local_i, idx in enumerate(slots):
                var_names[int(idx)] = f"{name}__t{local_i}" if n_slots > 1 else name
        else:
            # Per-member or partially shared: warn if different names map to the same slot
            # (indicates a bug in discretize_controls/discretize_states or a mixin override).
            for m, slots in member_slots.items():
                n_slots = len(slots)
                for local_i, idx in enumerate(slots):
                    idx = int(idx)
                    existing = var_names[idx]
                    if existing is not None and not existing.endswith(f"__m{m}"):
                        logger.warning(
                            "Index slot %d maps to different variable names across ensemble "
                            "members; this indicates a bug in discretize_controls() or "
                            "discretize_states() (possibly a mixin override). "
                            "Proceeding with member %d's name (%r); "
                            "the exported LP file may be incorrect.",
                            idx,
                            m,
                            name,
                        )
                    t_part = f"__t{local_i}" if n_slots > 1 else ""
                    var_names[idx] = f"{name}{t_part}__m{m}"

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
        constraints_str (str): The constraints string.
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
                "Create an output_folder or set a valid one in solver_options()."
            ) from None
