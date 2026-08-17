"""The same A* search, on a stuck order instead of an 8-puzzle.

`before.py` and `after.py` solve the 8-puzzle, because that is the example the source page
shows. It proves the mechanism -- the model reads a board out of prose and A* still
expands the same nodes -- and nobody ships an 8-puzzle.

A* does ship, and routing a van is the version everyone reaches for first. Companies buy
routing software. What they suffer through by hand is the other one: an order that is not
going to arrive on time, and the question of what to do about it.

The stock is scattered. There is some in the regional warehouse, more at the port, a
production run that could be pulled forward, and a substitute SKU nobody wants to offer
unless they have to. Each option costs money and takes days, several can be combined, and
the goal is a fixed date the customer was promised. That is a shortest-path problem over
plans, and `a_star_search` and `SearchProblem` are imported from `before.py` to solve it.

Where the LLM belongs: reading the account manager's note. "They'll take a partial if the
rest lands by month end, but not the substitute -- it failed qualification at their site
last year." That is a set of constraints written as a sentence, and turning it into one is
the model's whole job.

Where it does not: choosing the plan. A* returns the cheapest sequence that meets the
date, or reports that no sequence does -- and the second answer is the valuable one,
because it is the difference between calling the customer today and calling them on the
due date.

Run it:

    python 02-goal-based/search/real_world.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from before import SearchProblem, a_star_search  # noqa: E402

from shared.llm import llm_call  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402

ORDER_UNITS = 500
PROMISED_IN_DAYS = 12

# Every way this company can get units, with what it costs and how long it takes. These
# are operational facts out of the ERP, not judgements, and no model is asked about them.
#   name: (units, cost, days, tag)
SOURCES = {
    "regional_wh":    (180, 0,      1,  "on hand"),
    "port_bonded":    (150, 2_400,  4,  "customs clearance"),
    "pull_production": (250, 9_500,  9,  "expedite the line"),
    "partner_stock":  (120, 6_800,  3,  "buy from a competitor at cost-plus"),
    "air_freight":    (200, 24_000, 2,  "air freight the backorder"),
    "substitute_sku": (500, 1_100,  2,  "ship the substitute part"),
}

ACCOUNT_MANAGER = """Spoke to their ops director. They'll accept a partial shipment as
long as the balance lands before month end, which gives us twelve days total. What they
won't take is the substitute part -- it failed qualification at their site last year and
their quality team won't sign it off again. They're also the reference account for the
tender we're in next quarter, so this is not one to be late on."""


def read_note(text: str) -> dict:
    """The model call. An account manager's note becomes constraints on the search."""
    prompt = f"""Turn this account manager's note into constraints for a fulfilment plan.

Note:
"{text}"

Available sourcing options: {sorted(SOURCES)}

Return JSON only:
{{"days_available": <number>,
  "excluded": ["<option>", ...],
  "partial_shipment_acceptable": true or false}}

excluded lists options the customer or the situation rules out. Use the option names
exactly as listed. Return [] if nothing is excluded."""
    # tier=mid: reading a commercial constraint out of a paragraph is ordinary
    # comprehension, and every field it returns is checked against the option list below
    # before it narrows the search.
    raw = llm_call(prompt, mock_key="fulfilment_note", tier="mid")
    return model_loads(raw)


def build_problem(days_available: int, excluded: set[str]) -> SearchProblem:
    """A shortest-path problem over fulfilment plans.

    A state is (units_secured, days_elapsed, options_used). An action is one sourcing
    option. Cost is money. The same SearchProblem class the 8-puzzle uses.
    """
    usable = [name for name in SOURCES if name not in excluded]

    def actions(state):
        _, days, used = state
        out = []
        for name in usable:
            if name in used:
                continue
            _, _, lead, _ = SOURCES[name]
            # Options run in parallel, so the plan's elapsed time is the slowest leg
            # rather than the sum. A plan that cannot land in time is not generated.
            if max(days, lead) <= days_available:
                out.append(name)
        return out

    def result(state, action):
        units, days, used = state
        got, _, lead, _ = SOURCES[action]
        return (min(units + got, ORDER_UNITS), max(days, lead), used | frozenset([action]))

    def path_cost(cost_so_far, state, action, next_state):
        _, cost, _, _ = SOURCES[action]
        return cost_so_far + cost

    return SearchProblem(
        initial_state=(0, 0, frozenset()),
        goal_test=lambda s: s[0] >= ORDER_UNITS,
        actions=actions,
        result=result,
        path_cost=path_cost,
    )


def heuristic(state) -> float:
    """Cheapest conceivable cost for the units still missing. Never overestimates."""
    units, _, _ = state
    missing = ORDER_UNITS - units
    if missing <= 0:
        return 0.0
    best_per_unit = min(cost / got for got, cost, _, _ in SOURCES.values() if got and cost)
    return missing * best_per_unit


def show(label: str, plan, stats: dict) -> None:
    if plan is None:
        print(f"  {label}: no plan reaches {ORDER_UNITS} units in time")
        print(f"    nodes expanded: {stats.get('nodes_expanded', 0)}")
        return
    total_cost = sum(SOURCES[step][1] for step in plan)
    days = max(SOURCES[step][2] for step in plan)
    units = min(sum(SOURCES[step][0] for step in plan), ORDER_UNITS)
    print(f"  {label}:")
    for step in plan:
        got, cost, lead, tag = SOURCES[step]
        print(f"    {step:<17} {got:>4} units  {cost:>7,} cost  {lead:>2}d   {tag}")
    print(f"    {'':<17} {units:>4} units  {total_cost:>7,} total  {days:>2}d elapsed, "
          f"nodes expanded: {stats.get('nodes_expanded', 0)}")


def main() -> None:
    print("The same A* search, on a stuck order")
    print()
    print("  a_star_search and SearchProblem are imported from before.py, which solves")
    print("  the 8-puzzle with them. A state is a partial plan instead of a board.")
    print()
    print(f"  Order: {ORDER_UNITS} units. Nothing on hand covers it.")
    print()

    print("BEFORE: no constraints, cheapest plan that reaches the quantity")
    print()
    stats: dict = {}
    plan = a_star_search(build_problem(days_available=99, excluded=set()), heuristic, stats)
    show("unconstrained", plan, stats)
    print()
    print("  Cheapest is not the answer. It ships the substitute the customer already")
    print("  rejected once, and nothing in the data says so, because that fact lives in")
    print("  a conversation rather than in the ERP.")
    print()

    print("AFTER: the account manager's note becomes constraints")
    print()
    for line in ACCOUNT_MANAGER.strip().split("\n"):
        print(f"  {line.strip()}")
    print()

    parsed = read_note(ACCOUNT_MANAGER)

    # DETERMINISTIC: the model may only exclude options that exist, and the deadline has
    # to be a number. A hallucinated option name is dropped rather than silently
    # narrowing the search to nothing.
    excluded = {e for e in parsed.get("excluded", []) if e in SOURCES}
    unknown = [e for e in parsed.get("excluded", []) if e not in SOURCES]
    try:
        days_available = int(parsed.get("days_available", PROMISED_IN_DAYS))
    except (TypeError, ValueError):
        days_available = PROMISED_IN_DAYS

    print(f"  model read it as: {days_available} days available, excluded {sorted(excluded)}, "
          f"partial ok = {parsed.get('partial_shipment_acceptable')}")
    if unknown:
        print(f"  dropped, not a sourcing option: {unknown}")
    print()

    stats = {}
    plan = a_star_search(build_problem(days_available, excluded), heuristic, stats)
    show(f"cheapest plan landing within {days_available} days", plan, stats)
    print()

    # The same search on tighter dates, which is where it stops being a nice-to-have.
    for deadline in (4, 2):
        stats_tight = {}
        tight = a_star_search(build_problem(deadline, excluded), heuristic, stats_tight)
        show(f"if they had held us to {deadline} days", tight, stats_tight)
        print()

    print("  What the model did: read one paragraph and produced a deadline and an")
    print("  exclusion. What it did not do: choose a plan, price it, or decide whether a")
    print("  date is reachable at all.")
    print()
    print("  A* did those, and the three searches are the argument. Twelve days is a")
    print("  cheap plan. Four days is the same problem for more than twice the money,")
    print("  which is a number to take into the conversation rather than a feeling. Two")
    print("  days returns nothing -- not an expensive plan, no plan -- and that is a fact")
    print("  worth having today instead of discovering it on the due date.")
    print()
    print("  Asked directly, a model answers all three the same way: confidently, in")
    print("  prose, with a plan for the two-day case.")


if __name__ == "__main__":
    main()
