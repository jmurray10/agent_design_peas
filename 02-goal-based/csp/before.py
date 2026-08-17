"""Constraint satisfaction, classical. Backtracking search over the Australia map.

A CSP is stated declaratively -- variables, domains, constraints -- and a general
purpose algorithm either returns an assignment that satisfies every constraint or
proves that none exists. Nothing in this file knows what a map, a colour, or a
meeting is.

Python 3.10+ standard library only. No API key, no pip install, no network.

after.py imports CSP and backtracking_search from this module instead of copying
them. That import is the whole argument of this example, so the solver lives above
the __main__ guard and only the demo lives below it.
"""

from typing import Any, Callable

# (var1, var2, check) -- check(value_of_var1, value_of_var2) -> bool.
Constraint = tuple[str, str, Callable[[Any, Any], bool]]


class CSP:
    def __init__(
        self,
        variables: list[str],
        domains: dict[str, list[Any]],
        constraints: list[Constraint],
    ) -> None:
        self.variables = variables
        self.domains = domains
        self.constraints = constraints  # (var1, var2, check_func)

    def is_consistent(self, var: str, value: Any, assignment: dict[str, Any]) -> bool:
        for (v1, v2, check) in self.constraints:
            if v1 == var and v2 in assignment:
                if not check(value, assignment[v2]):
                    return False
            if v2 == var and v1 in assignment:
                if not check(assignment[v1], value):
                    return False
        return True


def backtracking_search(
    csp: CSP, assignment: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if assignment is None:
        assignment = {}
    if len(assignment) == len(csp.variables):
        return assignment
    unassigned = [v for v in csp.variables if v not in assignment]
    var = unassigned[0]
    for value in csp.domains[var]:
        if csp.is_consistent(var, value, assignment):
            assignment[var] = value
            result = backtracking_search(csp, assignment)
            if result is not None:
                return result
            # Undo the choice before trying the next value. This is the backtrack.
            del assignment[var]
    return None


if __name__ == "__main__":
    # Map coloring
    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    domains = {v: ['red', 'green', 'blue'] for v in variables}
    constraints = [
        ('WA', 'NT', lambda a, b: a != b),
        ('WA', 'SA', lambda a, b: a != b),
        ('NT', 'SA', lambda a, b: a != b),
        ('NT', 'Q',  lambda a, b: a != b),
        ('SA', 'Q',  lambda a, b: a != b),
        ('SA', 'NSW', lambda a, b: a != b),
        ('SA', 'V',  lambda a, b: a != b),
        ('Q', 'NSW', lambda a, b: a != b),
        ('NSW', 'V', lambda a, b: a != b),
    ]
    solution = backtracking_search(CSP(variables, domains, constraints))
    print(f"Map coloring: {solution}")

    # Re-check the answer against the constraints from scratch. The solver already
    # guarantees this; printing it is what turns the guarantee into something a
    # reader can watch happen.
    print()
    print("Verification (independent of the solver):")
    violations = 0
    for (v1, v2, check) in constraints:
        ok = check(solution[v1], solution[v2])
        violations += 0 if ok else 1
        status = "ok" if ok else "VIOLATED"
        print(f"  [{status}] {v1 + '=' + solution[v1]:<12} != {v2}={solution[v2]}")
    print(f"  {len(constraints) - violations}/{len(constraints)} constraints satisfied")
    print(f"  colours used: {len(set(solution.values()))} of {len(domains['WA'])} available")
    print(f"  T borders nothing, so it takes the first colour in its domain: {solution['T']}")
