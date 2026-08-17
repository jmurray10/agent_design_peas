"""The LLM writes the CSP. The solver that solves it is the one in before.py.

The import two lines below is the entire claim of this example. `CSP` and
`backtracking_search` are not reimplemented here, not subclassed here, not wrapped
here -- they are the same objects before.py defines, and verify_identical.py checks
that mechanically with the ast module.

What the LLM does: turn an English scheduling request into variables, domains and
constraints, and turn the solved assignment back into a sentence.
What the LLM does not do: decide the schedule. Backtracking search does that, and
its completeness guarantee is unchanged because its code is unchanged.

Runs with no API key. See shared/README.md for mock mode.
"""

import json
import sys
from pathlib import Path
from typing import Any

# after.py lives two directories below the repo root, and Python puts the script's
# own directory on sys.path rather than the root. Both entries are added explicitly
# so the file behaves the same whether it is run from here or from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for shared/
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # this directory, for before

from shared.llm import llm_call
from before import CSP, backtracking_search  # noqa: E402  -- the solver, imported not copied


def llm_extract_csp(
    natural_language_input: str, mock_key: str = "csp_extract_scheduling"
) -> CSP:
    """Translate a request in English into a formal CSP.

    `mock_key` only selects which canned response mock mode returns; every real
    backend ignores it. It exists so one file can demonstrate several requests.
    """
    prompt = f"""Extract a Constraint Satisfaction Problem from this request.
Request: "{natural_language_input}"
Return JSON:
- "variables": list of variable names
- "domains": dict of variable -> possible values
- "constraints": list of [var1, var2, "relationship"]
  relationship: "not_equal", "less_than", "not_same_time"
Return valid JSON only."""

    # frontier: this is the one genuinely hard judgement in the file -- ambiguous
    # English ("nobody in two meetings at once", "everyone") has to become an exact
    # formal spec. A missed constraint is not a visible error: the solver will
    # faithfully solve the wrong problem and return a schedule that looks fine.
    response = llm_call(prompt, mock_key=mock_key, tier="frontier")

    text = response.strip()
    if not text.startswith("{"):
        # Fix to the source page, which calls json.loads() on the raw response.
        # Models routinely wrap JSON in a markdown fence or a sentence of preamble.
        text = text[text.index("{"):text.rindex("}") + 1]
    parsed = json.loads(text)

    # Deterministic validation between the model and the solver. The solver's
    # guarantee is a guarantee about the CSP it is handed, so a malformed spec has
    # to be rejected here rather than discovered later as a confident wrong answer.
    for key in ("variables", "domains", "constraints"):
        if key not in parsed:
            raise ValueError(f"extracted CSP has no {key!r}")
    for var in parsed["variables"]:
        if not parsed["domains"].get(var):
            raise ValueError(f"extracted CSP gives variable {var!r} no domain")
    for v1, v2, _rel in parsed["constraints"]:
        for var in (v1, v2):
            if var not in parsed["variables"]:
                raise ValueError(f"constraint names undeclared variable {var!r}")

    print("  variables: " + ", ".join(parsed["variables"]))
    for var in parsed["variables"]:
        print(f"  domain of {var}: {parsed['domains'][var]}")
    for v1, v2, rel in parsed["constraints"]:
        print(f"  constraint: {v1} {rel} {v2}")

    constraint_map = {
        'not_equal': lambda a, b: a != b,
        'less_than': lambda a, b: a < b,
        'not_same_time': lambda a, b: a != b,
    }
    constraints = []
    for v1, v2, rel in parsed['constraints']:
        constraints.append((v1, v2, constraint_map.get(rel, lambda a, b: a != b)))

    return CSP(parsed['variables'], parsed['domains'], constraints)


def llm_format_solution(solution: dict, request: str) -> str:
    prompt = f"""The user asked: "{request}"
Solution: {json.dumps(solution)}
Explain in plain language."""
    # small: nothing is being decided or derived here. The assignment is already
    # solved and already verified, so this call only renders known facts as a
    # sentence -- the cheapest capability tier that can write English will do.
    return llm_call(prompt, mock_key="csp_format_solution", tier="small")


def verify_assignment(csp: CSP, assignment: dict[str, Any]) -> list[str]:
    """Re-check a returned assignment against the CSP, from scratch.

    Deliberately does not call csp.is_consistent: that is the solver's own code,
    and checking the solver with the solver proves nothing. Prints one line per
    check, because "no constraint violated" should be watched rather than trusted,
    and returns the violations it found.
    """
    violations: list[str] = []

    for var in csp.variables:
        if var not in assignment:
            violations.append(f"{var} was never assigned")
            print(f"  [VIOLATED] {var} was never assigned")
        elif assignment[var] not in csp.domains[var]:
            violations.append(f"{var}={assignment[var]} is outside its domain")
            print(f"  [VIOLATED] {var}={assignment[var]} is outside its domain")
        else:
            print(f"  [ok] {var} = {assignment[var]} (in domain)")

    for (v1, v2, check) in csp.constraints:
        if v1 not in assignment or v2 not in assignment:
            continue
        pair = f"{v1}={assignment[v1]!r} vs {v2}={assignment[v2]!r}"
        if check(assignment[v1], assignment[v2]):
            print(f"  [ok] {pair}")
        else:
            violations.append(pair)
            print(f"  [VIOLATED] {pair}")

    return violations


if __name__ == "__main__":
    print(f"solver: backtracking_search from module {backtracking_search.__module__!r} "
          f"({Path(backtracking_search.__code__.co_filename).name}), imported not redefined")

    # ---------------------------------------------------------------------------
    # Request 1: the scheduling request from the source page.
    # ---------------------------------------------------------------------------
    request = """Schedule 3 meetings next week:
- Team standup: Alice and Bob, 30 min
- Design review: Bob and Carol, 1 hour
- Sprint planning: everyone, 1 hour
Nobody in two meetings at once.
Slots: Monday 9am, Monday 2pm, Tuesday 10am, Wednesday 9am"""

    print("\n=== Request 1: three meetings, four slots ===")
    print(request)

    print("\n-- LLM extracts the CSP (this is the part that changed) --")
    csp = llm_extract_csp(request)

    print("\n-- backtracking_search solves it (this is the part that did not) --")
    solution = backtracking_search(csp)  # SAME solver, zero changes
    print(f"  assignment: {solution}")

    print("\n-- Verification, deterministic, no model involved --")
    violations = verify_assignment(csp, solution)

    # An independent reading of the request itself: Bob is in all three meetings, so
    # the three meetings must land in three different slots. This check is written by
    # hand from the English, not derived from what the model extracted, so it also
    # tests the extraction rather than only the solution.
    distinct = len(set(solution.values())) == len(solution)
    print(f"  [{'ok' if distinct else 'VIOLATED'}] independent check: "
          f"{len(solution)} meetings at {len(set(solution.values()))} distinct times")
    if not distinct:
        violations.append("two meetings share a slot")
    print(f"  result: {len(violations)} constraints violated")

    print("\n-- LLM formats the answer (this is the part that changed) --")
    answer = llm_format_solution(solution, request)
    print(answer)

    # ---------------------------------------------------------------------------
    # Request 2: the same meetings, not enough slots. The solver does not guess.
    # ---------------------------------------------------------------------------
    tight_request = """Same three meetings as before, same attendees.
The only slots left are Monday 9am and Monday 2pm."""

    print("\n=== Request 2: three meetings, two slots ===")
    print(tight_request)

    print("\n-- LLM extracts the CSP --")
    tight_csp = llm_extract_csp(tight_request, mock_key="csp_extract_overconstrained")

    print("\n-- backtracking_search solves it --")
    tight_solution = backtracking_search(tight_csp)
    print(f"  assignment: {tight_solution}")
    if tight_solution is None:
        # Completeness: backtracking search searches the whole space, so None is a
        # proof that no assignment exists, not a failure to find one. That property
        # belongs to the algorithm and survives the LLM being anywhere near it.
        print("  no schedule exists. The solver proved that; it did not run out of ideas.")
        print("  A model asked this directly would very likely have answered anyway.")

    # ---------------------------------------------------------------------------
    # Request 3: the model returns a malformed CSP. It never reaches the solver.
    # ---------------------------------------------------------------------------
    bad_request = """Book a kickoff and a retro sometime next week."""

    print("\n=== Request 3: the extraction goes wrong ===")
    print(bad_request)

    print("\n-- LLM extracts the CSP --")
    try:
        llm_extract_csp(bad_request, mock_key="csp_extract_malformed")
    except ValueError as exc:
        print(f"  rejected before the solver ran: {exc}")
        print("  backtracking_search would have raised KeyError on the missing domain.")
        print("  The validation is deterministic, so this failure is a failure and not a schedule.")
