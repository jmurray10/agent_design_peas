"""The same value iteration, on a maintenance policy instead of a grid world.

`before.py` builds a 4x3 grid world, because that is the example the source page shows.
Value iteration converges to an optimal policy over it and proves the mechanism.

Nobody operates a grid world. Plenty of people operate machines that degrade, and
deciding when to service one is the textbook industrial MDP: a machine sits in some
condition, you can run it, service it, or replace it, each action costs something
different, and running a failing machine risks an unplanned breakdown that costs far more
than the service you skipped.

`value_iteration` and `MDP` are imported from `before.py` rather than reimplemented. What
changes is the model of the world, not the algorithm that solves it.

Where the LLM belongs, and where it does not. The rewards are the numbers a plant manager
argues about: what a breakdown really costs once you count the line stopping, what a
service visit costs, what running one more week is worth. Those are judgements stated in
prose, and turning prose into numbers is the model's job. Choosing the policy is not: value
iteration converges to the optimal one for whatever numbers it is given, and can say so.

Run it:

    python 03-utility-based/value-iteration/real_world.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from before import MDP, value_iteration  # noqa: E402

from shared.llm import llm_call  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402

# Machine condition. A real maintenance MDP has more states; four is enough to show the
# policy change and small enough to print.
STATES = ["new", "good", "worn", "failing"]
ACTIONS = ["run", "service", "replace"]

# How condition moves. Running degrades, servicing recovers some, replacing resets. These
# are the engineering facts, not the money, and no model is asked about them.
TRANSITIONS = {
    ("new", "run"):        [(0.85, "new"), (0.15, "good")],
    ("good", "run"):       [(0.70, "good"), (0.30, "worn")],
    ("worn", "run"):       [(0.55, "worn"), (0.45, "failing")],
    ("failing", "run"):    [(0.60, "failing"), (0.40, "new")],   # 0.40 = it broke and was rebuilt
    ("new", "service"):    [(1.0, "new")],
    ("good", "service"):   [(0.90, "new"), (0.10, "good")],
    ("worn", "service"):   [(0.75, "good"), (0.25, "worn")],
    ("failing", "service"): [(0.60, "good"), (0.40, "worn")],
    ("new", "replace"):    [(1.0, "new")],
    ("good", "replace"):   [(1.0, "new")],
    ("worn", "replace"):   [(1.0, "new")],
    ("failing", "replace"): [(1.0, "new")],
}

PLANT_MANAGER = """Running the line normally earns us about 900 a day in throughput.
A planned service visit costs 1,200 and we lose half a day. Replacing the unit outright is
40,000 and two days down. What actually hurts is an unplanned breakdown -- last time the
line was down four days, we missed two customer commitments, and I would put the true cost
somewhere around 25,000 once you count the lot of it. Running a failing unit is where that
risk lives."""


def read_costs(text: str) -> dict:
    """The model call. A plant manager's prose becomes the reward function."""
    prompt = f"""Turn this into a reward function for a maintenance decision model.

What the plant manager said:
"{text}"

States: {STATES}   Actions: {ACTIONS}

Return the reward for taking each action in each state, as JSON. Rewards are per period.
Earnings are positive, costs are negative. Running a machine in worse condition should
earn less and carry more risk. Do not invent a number the message does not support.

Return JSON only, exactly this shape:
{{"run": {{"new": <n>, "good": <n>, "worn": <n>, "failing": <n>}},
  "service": {{"new": <n>, "good": <n>, "worn": <n>, "failing": <n>}},
  "replace": {{"new": <n>, "good": <n>, "worn": <n>, "failing": <n>}}}}"""
    # tier=mid: reading money out of a paragraph and placing it on a grid of states is
    # ordinary comprehension, and every number is checked for type and sign below before
    # it reaches the solver.
    raw = llm_call(prompt, mock_key="maintenance_rewards", tier="mid")
    try:
        return model_loads(raw)
    except json.JSONDecodeError:
        # The cell-by-cell check in main() is the deterministic guard this file exists to
        # demonstrate, and an unparseable reply used to sail straight past it: the parse
        # raised here, several frames before the guard could refuse anything. A reply
        # that is not JSON at all is the likeliest way a model fails this prompt, so it
        # is the one case the guard most needs to see. Empty means "nothing usable",
        # which main() already knows how to refuse.
        return {}


def build(rewards: dict) -> MDP:
    """An MDP in the shape before.py's class declares.

    Both models are dicts keyed by tuples, not callables: transition_model is
    (state, action, next_state) -> probability, and reward_function the same key ->
    reward. The grid world builds them the same way.
    """
    transition_model = {}
    reward_function = {}
    for (state, action), outcomes in TRANSITIONS.items():
        for probability, next_state in outcomes:
            transition_model[(state, action, next_state)] = probability
            reward_function[(state, action, next_state)] = float(rewards[action][state])

    return MDP(
        states=STATES,
        actions=ACTIONS,
        transition_model=transition_model,
        reward_function=reward_function,
        gamma=0.95,
        initial_state="new",
        # No absorbing state: a machine is always in some condition and the decision
        # never stops. The grid world has terminal states; this problem does not, and
        # value_iteration handles both because it checks rather than assumes.
        terminal_states=set(),
    )


def show_policy(label: str, values, policy) -> None:
    print(f"  {label}")
    for state in STATES:
        print(f"    {state:<9} value {values[state]:>10.1f}   do: {policy[state]}")


def main() -> None:
    print("The same value iteration, on a maintenance policy")
    print()
    print("  value_iteration and MDP are imported from before.py, which solves a 4x3")
    print("  grid world with them. Only the model of the world changed.")
    print()

    # A deliberately naive reward function: someone who has not costed a breakdown.
    naive = {
        "run":     {"new": 900, "good": 900, "worn": 900, "failing": 900},
        "service": {"new": -1200, "good": -1200, "worn": -1200, "failing": -1200},
        "replace": {"new": -40000, "good": -40000, "worn": -40000, "failing": -40000},
    }
    values, policy = value_iteration(build(naive))
    print("BEFORE: rewards guessed without costing a breakdown")
    print()
    show_policy("running earns 900 whatever condition the machine is in", values, policy)
    print()
    print("  The policy runs the machine into the ground, and it is correct to. Nothing")
    print("  in those numbers says a failing machine is worse to run than a new one.")
    print()

    print("AFTER: the plant manager describes the real costs")
    print()
    for line in PLANT_MANAGER.strip().split("\n"):
        print(f"  {line.strip()}")
    print()

    rewards = read_costs(PLANT_MANAGER)

    if not rewards:
        print("  refused: the reply was not JSON, so there is no reward function to check.")
        print("  Falling back to the naive numbers rather than solving a partial model.")
        return

    # DETERMINISTIC: every cell has to be present and numeric before the solver sees it.
    # A missing state is a policy that silently ignores a condition.
    problems = []
    for action in ACTIONS:
        for state in STATES:
            try:
                float(rewards[action][state])
            except (KeyError, TypeError, ValueError):
                problems.append(f"{action}/{state}")
    if problems:
        print(f"  refused: the reward function is missing or non-numeric at {problems}")
        print("  Falling back to the naive numbers rather than solving a partial model.")
        return

    print("  model read the rewards as:")
    for action in ACTIONS:
        cells = "  ".join(f"{s}={float(rewards[action][s]):>9.0f}" for s in STATES)
        print(f"    {action:<8} {cells}")
    print()

    values, policy = value_iteration(build(rewards))
    show_policy("optimal policy for those numbers", values, policy)
    print()
    print("  What the model did: put numbers on sentences. What it did not do: choose")
    print("  when to service the machine. Value iteration did that, it converged, and")
    print("  the policy it returned is optimal for the rewards it was given -- which is")
    print("  a guarantee that survives the model being roughly right rather than exact.")
    print()
    print("  Change the breakdown cost and the policy moves on its own. That is the")
    print("  property worth having: the argument is about the numbers, in the open,")
    print("  rather than about what some model felt was prudent.")


if __name__ == "__main__":
    main()
