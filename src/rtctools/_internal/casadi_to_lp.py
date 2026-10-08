"""Build LP/MILP file representations from CasADi symbolic expressions.

Provides low-level builders (objective, constraints, bounds) and a file writer.
Called by CollocatedIntegratedOptimizationProblem when ``export_lp=True`` is set
in :meth:`~rtctools.optimization.optimization_problem.OptimizationProblem.export_options`.

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
from collections.abc import Callable, Sequence
from typing import Any

import casadi as ca
import numpy as np

# The LP file format limits line length to 255 characters.
LP_MAX_LINE_WIDTH = 255

# Threshold for treating coefficients and constants as zero (absorbs floating-point noise)
LP_COEFF_EPSILON = 1e-10

# Characters forbidden in LP constraint/variable names (LP format requirement)
# (a backslash starts a comment, which would silently drop the rest of the line)
_LP_FORBIDDEN_CHARS = " \t\r\n#+-*/^()[]:'\"\\"
_LP_FORBIDDEN_TRANS = str.maketrans(_LP_FORBIDDEN_CHARS, "_" * len(_LP_FORBIDDEN_CHARS))

# Suffixes appended to the two lines emitted for range constraints.
_LP_RANGE_SUFFIXES = ("_lb", "_ub")

# Prefix for dummy variables used when a constraint reduces to a constant expression b_i.
# LP format requires at least one variable term per row; we emit e.g.
# ``_constant_<name> >= lb - b_i`` with ``_constant_<name>`` fixed to 0 in Bounds, so the row
# reads ``0 >= lb - b_i`` (i.e. ``b_i >= lb``) and stays infeasible for any violation size.
_LP_CONSTANT_VAR_PREFIX = "_constant_"

# Maximum number of row/variable names listed in aggregated warnings and errors.
_LP_MAX_LISTED_NAMES = 10

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

    Replaces characters that are forbidden in LP constraint names with underscores, and
    prefixes a name that starts with a digit or a period with an underscore, since LP
    readers reject such names.
    """
    name = name.translate(_LP_FORBIDDEN_TRANS)
    if name and (name[0].isdigit() or name[0] == "."):
        name = "_" + name
    return name


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
    seen: dict[str, tuple[int, int | None]] = {}
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
            while f"{name}_d{counter}" in name_set:
                counter += 1
            constraint_names[i] = f"{name}_d{counter}"
            name_set.add(constraint_names[i])
            seen[name] = (counter + 1, None)
        else:
            seen[name] = (0, i)
    return constraint_names


def _deduplicate_nonempty_names(names: list[str]) -> list[str]:
    """Deduplicate ``names`` in-place like :func:`_deduplicate_constraint_names`, ignoring
    empty strings (which denote unlabelled rows)."""
    idx = [i for i, n in enumerate(names) if n]
    deduped = _deduplicate_constraint_names([names[i] for i in idx])
    for i, n in zip(idx, deduped, strict=True):
        names[i] = n
    return names


def _build_user_constraint_base_names(user_tuples: list, auto_prefix: str) -> list[str]:
    """Sanitize, validate and deduplicate user-provided constraint base names.

    Extracts the optional name from position 3 of each constraint tuple, falling back
    to ``"{auto_prefix}_{i}"`` when absent. Each name is sanitized (forbidden characters
    replaced with ``_``) and renamed with a ``_ren`` suffix (with a warning) if it starts
    with a prefix reserved for auto-generated names (``LP_RESERVED_NAME_PREFIXES``) or ends
    with a reserved LP suffix (``_d{n}``, ``_m{n}``, ``_t{n}``, ``_lb``, ``_ub``).
    Auto-generated names never end with ``_ren``, so renamed user names cannot collide
    with them. The resulting base names are then deduplicated with ``_d{n}`` indices. Must
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
            if not isinstance(raw, str):
                raise TypeError(
                    f"Constraint name at index {i} must be a str, got {type(raw).__name__}: {raw!r}"
                )
            name = _sanitize_constraint_name(raw)
            if any(name.startswith(p) for p in LP_RESERVED_NAME_PREFIXES):
                new_name = name + "_ren"
                logger.warning(
                    "User constraint name %r starts with a prefix reserved for auto-generated "
                    "LP constraint names (%s); renamed to %r to avoid collision with "
                    "auto-generated labels.",
                    raw,
                    ", ".join(LP_RESERVED_NAME_PREFIXES),
                    new_name,
                )
                name = new_name
            elif _LP_RESERVED_SUFFIX_RE.search(raw) or _LP_RESERVED_SUFFIX_RE.search(name):
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


def _list_names(items: Sequence, fmt: Callable[[Any], str] = repr) -> str:
    """Format up to ``_LP_MAX_LISTED_NAMES`` items for a log/error message.

    ``fmt`` is only called for the listed items.
    """
    shown = items[:_LP_MAX_LISTED_NAMES]
    listing = ", ".join(fmt(item) for item in shown)
    if len(items) > len(shown):
        listing += f", ... ({len(items) - len(shown)} more)"
    return listing


def _describe_bounds(
    lb: np.ndarray, ub: np.ndarray, indices: np.ndarray, name_of: Callable[[int], str]
) -> str:
    """List up to ``_LP_MAX_LISTED_NAMES`` entries as ``name (lower=..., upper=...)``."""

    def describe(i: int) -> str:
        i = int(i)
        return f"{name_of(i)} (lower={_format_lp_bound(lb[i])}, upper={_format_lp_bound(ub[i])})"

    return _list_names(indices, fmt=describe)


def check_bound_conflicts(
    lb: np.ndarray,
    ub: np.ndarray,
    kind: str,
    name_of: Callable[[int], str],
) -> None:
    """Raise :exc:`ValueError` if a bound pair cannot be written to an LP file.

    A lower bound of ``+inf`` or an upper bound of ``-inf`` would need an infinite
    right-hand side, which several LP readers (e.g. CPLEX) reject. Both make the problem
    infeasible. Finite bounds with ``lb > ub`` are not rejected, see
    :func:`warn_crossed_bounds`.

    Args:
        lb, ub: Lower and upper bounds (1-D arrays of equal length). NaNs are ignored.
        kind: Human-readable kind of the bounded entity, e.g. ``"variable"`` or
            ``"constraint"``; used in the error message.
        name_of: Maps an index into ``lb``/``ub`` to a display name. Only called for
            listed entries, so it may be expensive.
    """
    conflicts = np.flatnonzero(np.isposinf(lb) | np.isneginf(ub))
    if len(conflicts) == 0:
        return
    raise ValueError(
        f"Infeasible bounds on {len(conflicts)} {kind}(s): a lower bound of +inf or an upper "
        f"bound of -inf cannot be written to an LP file. Affected: "
        f"{_describe_bounds(lb, ub, conflicts, name_of)}."
    )


def warn_crossed_bounds(
    lb: np.ndarray,
    ub: np.ndarray,
    kind: str,
    name_of: Callable[[int], str],
) -> None:
    """Log a warning if any lower bound is above its upper bound.

    Such bounds are written as given; the LP is infeasible and the solver reports it.
    """
    with np.errstate(invalid="ignore"):
        crossed = np.flatnonzero(lb > ub)
    if len(crossed) == 0:
        return
    logger.warning(
        "LP export: %d %s(s) have a lower bound above the upper bound, so the LP is "
        "infeasible; they are written as given. Affected: %s.",
        len(crossed),
        kind,
        _list_names(crossed, fmt=lambda i: name_of(int(i))),
    )


def _check_nan_bounds(lb: np.ndarray, ub: np.ndarray, names: list[str]) -> None:
    """Raise ValueError if any lower or upper bound is NaN, listing the affected names."""
    nan_bounds_list = [f"{names[i]} (lower)" for i in np.nonzero(np.isnan(lb))[0]] + [
        f"{names[i]} (upper)" for i in np.nonzero(np.isnan(ub))[0]
    ]
    if nan_bounds_list:
        raise ValueError(
            f"NaN bounds found for {_list_names(nan_bounds_list, fmt=str)}; "
            "check that all bounds are finite or ±inf."
        )


# Tolerance for testing whether a constant constraint row satisfies its bounds. Kept
# separate from LP_COEFF_EPSILON (a coefficient-magnitude threshold) because this absorbs
# floating-point noise in a scalar value-vs-bound comparison, a distinct concern.
LP_FEASIBILITY_EPSILON = 1e-9


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
    ``_constant_<name>`` fixed to 0 in Bounds, so solvers can detect the infeasibility
    during presolve.

    No infinite right-hand side is ever written, since several LP readers (e.g. CPLEX)
    reject it. Rows with both bounds infinite (``lb = -inf``, ``ub = +inf``) carry no
    information and are skipped; a single aggregated warning is logged per call. Rows with
    impossible bounds (``lb = +inf`` or ``ub = -inf``) raise a ``ValueError``. Crossed finite
    bounds (``lb > ub``) are written as given and logged as a warning.

    Args:
        g: Symbolic constraint expression (affine in x).
        x: Decision variable vector.
        lbg: Lower bounds on constraints.
        ubg: Upper bounds on constraints.
        var_names: List of variable names.
        constraint_names: Optional list of constraint names. When provided, each
            constraint is prefixed as ``name: expr op rhs``. Range constraints
            emit two lines with ``_lb`` / ``_ub`` appended. Names are sanitized and
            deduplicated, and the emitted labels (after ``_lb`` / ``_ub`` expansion) are
            guaranteed to be unique.
            If ``None``, constraints are written without labels.

    Returns:
        Tuple of (constraints_str, extra_bounds_str) where constraints_str is the
        formatted Subject To section content, and extra_bounds_str contains any
        dummy variable bound lines that must be appended to the Bounds section
        (empty string when no constant rows were encountered).

    Raises:
        ValueError: If ``constraint_names`` has the wrong length, if any bound is NaN,
            if a lower bound is ``+inf`` or an upper bound is ``-inf``, or if a row has a
            non-finite constant term or right-hand side.
    """
    A, b = ca.linear_coeff(g, x)
    A = ca.sparsify(ca.DM(A))
    b = ca.DM(b)

    lbg = np.array(ca.veccat(*lbg))[:, 0]
    ubg = np.array(ca.veccat(*ubg))[:, 0]

    if constraint_names is not None and len(constraint_names) != len(lbg):
        raise ValueError(
            f"constraint_names has {len(constraint_names)} entries but there are "
            f"{len(lbg)} constraint rows; they must match."
        )

    constraint_labels = (
        constraint_names if constraint_names is not None else [str(i) for i in range(len(lbg))]
    )
    _check_nan_bounds(lbg, ubg, constraint_labels)
    check_bound_conflicts(lbg, ubg, "constraint", lambda i: constraint_labels[i])
    warn_crossed_bounds(lbg, ubg, "constraint", lambda i: constraint_labels[i])

    A_csc = A.tocsc()
    A_coo = A_csc.tocoo()
    b = np.array(b)[:, 0]

    nonfinite_b = np.nonzero(~np.isfinite(b))[0]
    if len(nonfinite_b) > 0:
        raise ValueError(
            "Cannot export LP file: constraint row(s) have a non-finite constant term: "
            f"{_list_names([constraint_labels[i] for i in nonfinite_b])}."
        )

    constraints = [[] for _ in range(A.shape[0])]
    for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data, strict=True):
        if abs(c) > LP_COEFF_EPSILON:
            constraints[i].extend(["+" if c > 0 else "-", f"{abs(c):.15g}", var_names[j]])

    # Sanitize and deduplicate constraint names.
    # Callers are expected to have already sanitized, validated reserved prefixes/suffixes,
    # and deduplicated names (e.g. via _build_user_constraint_base_names). This is a safety
    # net for direct callers (e.g. unit tests), where sanitization may itself introduce
    # collisions (e.g. "a b" and "a+b" both become "a_b").
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
        _deduplicate_nonempty_names(sanitized_names)
    else:
        sanitized_names = None

    def _rhs(bound: float, b_i: float, label: str) -> str:
        """Right-hand side ``bound - b_i`` for one emitted row; it must be finite."""
        with np.errstate(over="ignore"):
            rhs = bound - b_i
        if not np.isfinite(rhs):
            raise ValueError(
                f"Cannot export LP file: constraint {label!r} has a non-finite right-hand side "
                f"({bound} - {b_i}); the bound or constant term overflows."
            )
        return _format_lp_bound(rhs)

    def _label(name: str, lb: float, ub: float, side: str) -> str:
        """Return the label for one emitted line, or '' when names are disabled.

        ``side`` is ``"eq"``, ``"lb"``, or ``"ub"``.  For equality and single-sided
        constraints the name is used as-is; for range constraints (both bounds finite)
        the appropriate ``_lb`` / ``_ub`` suffix is appended.
        """
        if not name:
            return ""
        is_range = np.isfinite(lb) and np.isfinite(ub) and lb != ub
        if is_range:
            suffix = _LP_RANGE_SUFFIXES[0] if side == "lb" else _LP_RANGE_SUFFIXES[1]
            return f"{name}{suffix}"
        return name

    # Each emitted line is collected as [label, body]; labels are made unique once all
    # of them are known, because the _lb/_ub expansion of one row can collide with the
    # name of another row (e.g. range row "foo" -> "foo_lb" vs. a row named "foo_lb").
    lines = []
    extra_bounds_list = []  # dummy variable bound lines for constant rows
    vacuous_labels = []
    used_dummy_names = set(var_names)
    for i, cur_constr in enumerate(constraints):
        lb, ub, b_i = lbg[i], ubg[i], b[i]

        if np.isneginf(lb) and np.isposinf(ub):
            # Both bounds infinite: vacuous row, carries no information. Writing it would
            # require an infinite right-hand side, so it is skipped (aggregated warning below).
            vacuous_labels.append(constraint_labels[i])
            continue

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
            # dummy variable fixed to 0, so the solver can detect the infeasibility via presolve.
            lb_ok = (not np.isfinite(lb)) or b_i >= lb - LP_FEASIBILITY_EPSILON
            ub_ok = (not np.isfinite(ub)) or b_i <= ub + LP_FEASIBILITY_EPSILON
            # Crossed bounds (lb > ub) are infeasible even when both tolerance checks pass.
            if lb_ok and ub_ok and not lb > ub:
                continue
            # Infeasible — warn and emit using a dummy variable named after the constraint.
            # With the dummy fixed to 0, "dummy >= lb - b_i" reads "b_i >= lb" (and likewise
            # for <= and =), so every violation stays infeasible regardless of magnitude.
            label = constraint_labels[i]
            logger.warning(
                "Constraint %r reduces to a constant (%s) that violates bounds [%s, %s]. "
                "The problem is infeasible. Representing via dummy variable in LP export.",
                label,
                b_i,
                lb,
                ub,
            )
            dummy_var = _sanitize_constraint_name(f"{_LP_CONSTANT_VAR_PREFIX}{name or i}")
            if dummy_var in used_dummy_names:
                counter = 0
                while f"{dummy_var}_d{counter}" in used_dummy_names:
                    counter += 1
                dummy_var = f"{dummy_var}_d{counter}"
            used_dummy_names.add(dummy_var)
            extra_bounds_list.append(f"0 <= {dummy_var} <= 0")
            c_str = dummy_var

        if np.isfinite(lb) and lb == ub:  # Equality constraint: emit a single = line.
            lines.append(
                [_label(name, lb, ub, "eq"), f"{c_str} = {_rhs(lb, b_i, constraint_labels[i])}"]
            )
        else:
            if np.isfinite(lb):
                lines.append(
                    [
                        _label(name, lb, ub, "lb"),
                        f"{c_str} >= {_rhs(lb, b_i, constraint_labels[i])}",
                    ]
                )
            if np.isfinite(ub):
                lines.append(
                    [
                        _label(name, lb, ub, "ub"),
                        f"{c_str} <= {_rhs(ub, b_i, constraint_labels[i])}",
                    ]
                )

    if vacuous_labels:
        logger.warning(
            "LP export: skipped %d constraint row(s) with both bounds infinite "
            "(lb=-inf, ub=+inf); such rows carry no information. Affected: %s",
            len(vacuous_labels),
            _list_names(vacuous_labels),
        )

    if sanitized_names is not None:
        emitted_labels = [label for label, _ in lines]
        _deduplicate_nonempty_names(emitted_labels)
        for line, label in zip(lines, emitted_labels, strict=True):
            line[0] = label

    constraints_str_list = [f"{label}: {body}" if label else body for label, body in lines]
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
    - Upper bound only: ``-Inf <= name <= ub`` (the explicit ``-Inf`` is required: LP
      readers otherwise apply the default lower bound of 0)
    - Both finite: ``lb <= name <= ub``

    Infinite values only ever appear as ``-Inf`` lower bounds in the Bounds section, which
    is accepted by LP readers (including CPLEX); constraint rows never carry ``Inf``.

    A lower bound above the upper bound is written as given and logged as a warning; the LP
    is then infeasible, which solvers report.

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
        ValueError: If any bound is NaN, or a lower bound is ``+inf`` or an upper bound is
            ``-inf``.
    """
    lbx_arr = np.asarray(lbx, dtype=float)
    ubx_arr = np.asarray(ubx, dtype=float)
    _check_nan_bounds(lbx_arr, ubx_arr, var_names)
    check_bound_conflicts(lbx_arr, ubx_arr, "variable", lambda i: var_names[i])
    warn_crossed_bounds(lbx_arr, ubx_arr, "variable", lambda i: var_names[i])

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
    index. Shared slots receive no member suffix, per-member slots get a ``__m{i}`` suffix
    (e.g. ``x__t3__m0``). The decision is made per slot, so a variable that is shared only
    before a branching time (``ControlTreeMixin``) gets shared names for the shared slots and
    per-member names for the rest.

    Naming scheme:
    - Shared, multi-slot variable: ``"{name}__t{local_index}"``
    - Shared, single-slot variable: ``"{name}"``
    - Per-member, multi-slot: ``"{name}__t{local_index}__m{member}"``
    - Per-member, single-slot: ``"{name}__m{member}"``
    - Slot used by some but not all members (two or more): suffix of the lowest of them.
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
    n_members = len(indices_per_member)

    # For every slot: which (base name, members) claim it. The base name is the name without
    # the member suffix, e.g. "x__t3".
    claims = {}
    for m, indices in enumerate(indices_per_member):
        for name, slots in indices.items():
            slots = np.atleast_1d(np.asarray(slots, dtype=np.int32))
            for local_i, idx in enumerate(slots):
                base = f"{name}__t{local_i}" if len(slots) > 1 else name
                claims.setdefault(int(idx), {}).setdefault(base, []).append(m)

    var_names = [None] * num_total
    conflicts = []
    for idx, by_base in claims.items():
        base, members = list(by_base.items())[-1]
        shared = len(set(members)) == n_members
        var_names[idx] = base if shared else f"{base}__m{min(members)}"
        if len(by_base) > 1:
            conflicts.append((idx, var_names[idx]))

    if conflicts:
        logger.warning(
            "Decision vector slots are claimed by different variable names (slot -> name used): "
            "%s. This indicates a bug in discretize_controls() or discretize_states() "
            "(possibly a mixin override); the exported LP file may be incorrect.",
            _list_names(conflicts, fmt=lambda c: f"{c[0]} -> {c[1]!r}"),
        )

    unassigned = [i for i in range(num_total) if var_names[i] is None]
    if unassigned:
        logger.warning(
            "Decision vector slots %s are not claimed by any variable; this indicates a bug "
            "in discretize_controls() or discretize_states() (possibly a mixin override). "
            "They are named '__unassigned_{index}' and the exported LP file may be incorrect.",
            _list_names(unassigned),
        )
        for i in unassigned:
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
        constraints_str (str): The constraints string (empty, or blank after stripping,
            if there are no constraints to emit; the ``Subject To`` section is then omitted).
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
