import re

# Section keywords (case-insensitive) that HiGHS or CPLEX do not read as a name.
_LP_KEYWORDS = frozenset(
    "bin binaries binary bound bounds end free gen general generals integer integers max "
    "maximize maximum min minimize minimum s.t. semi semis sos st st.".split()
)

# A name that an LP reader parses as (part of) a number:
# - a leading digit or period;
# - a name starting with "inf", which HiGHS reads as infinity;
# - a name starting with e/E followed by a digit, another e/E or nothing (CPLEX reads the first
#   e as an exponent marker of the preceding number or sign, across whitespace: "2 e - 3 z"
#   is read as 2e-3 z, "z + ee" fails). "ex5", "elevation" and "e__t0" are read as names.
_LP_NUMBER_LIKE = re.compile(r"[0-9.]|inf|[eE]([0-9eE]|$)", re.IGNORECASE)


def _lp_safe_name(name: str) -> str:
    """Return ``name``, prefixed with ``_`` if an LP reader would misread it."""
    if name.lower() in _LP_KEYWORDS or _LP_NUMBER_LIKE.match(name):
        return "_" + name
    return name


def _lp_unique_safe_names(names: list[str], reserved: frozenset[str] = frozenset()) -> list[str]:
    """
    Return unique names that LP readers read as names, none of them in ``reserved``.

    A name that ``_lp_safe_name`` leaves alone keeps its spelling over one it prefixes, so
    ``inflow`` becomes ``_inflow__d1`` next to an existing ``_inflow``. Later duplicates get
    a ``__d{n}`` suffix.
    """
    safe = [_lp_safe_name(n) for n in names]
    order = sorted(range(len(names)), key=lambda i: safe[i] != names[i])  # stable
    taken = set(safe) | set(reserved)
    seen = set(reserved)
    next_suffix = {}
    result = [""] * len(names)
    for i in order:
        base = name = safe[i]
        while name in seen or (name != base and name in taken):
            next_suffix[base] = next_suffix.get(base, 0) + 1
            name = f"{base}__d{next_suffix[base]}"
        seen.add(name)
        result[i] = name
    return result
