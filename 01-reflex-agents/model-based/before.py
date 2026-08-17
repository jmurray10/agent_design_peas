"""Model-based reflex agent -- the classical version.

A two-cell vacuum world. The agent sees one cell at a time, so the world is only
partially observable: it cannot see the other cell and has to remember it. That is the
whole reason this agent exists and the simple reflex agent next door does not suffice.

Three hand-coded pieces do the work:

    update_state    fold a percept into what is already believed
    rule_match      pick an action from the belief, not from the percept
    predict_effect  the world model -- what my own action does to my beliefs

Standard library only. No API key, no network, no install.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Percept:
    data: Dict[str, Any]
    timestamp: float = 0.0


class ModelBasedReflexAgent:
    """Maintains internal state for partial observability."""
    def __init__(self):
        self.state = {}
        self.model = {}

    def agent_function(self, percept: Percept) -> str:
        self.state = self.update_state(self.state, percept)
        rule = self.rule_match(self.state)
        action = rule.get('action', 'no_op')
        self.state = self.predict_effect(self.state, action)
        return action

    def update_state(self, state: dict, percept: Percept) -> dict:
        # A shallow copy, so the returned dict shares the `visited` set with the old
        # one. Safe here because agent_function immediately discards the old state --
        # but it does mean you cannot keep an old state around and expect it frozen.
        new_state = state.copy()
        new_state.update(percept.data)
        if 'visited' not in new_state:
            new_state['visited'] = set()
        if 'location' in percept.data:
            new_state['visited'].add(percept.data['location'])
        return new_state

    def predict_effect(self, state: dict, action: str) -> dict:
        new_state = state.copy()
        if action == 'suck' and 'location' in state:
            new_state[f"{state['location']}_cleaned"] = True
        if action == 'move_right':
            new_state['location'] = 'right'
        if action == 'move_left':
            new_state['location'] = 'left'
        return new_state

    def rule_match(self, state: dict) -> dict:
        loc = state.get('location', '')
        status = state.get('status', '')
        visited = state.get('visited', set())
        if status == 'dirty':
            return {'action': 'suck'}
        if loc == 'left' and 'right' not in visited:
            return {'action': 'move_right'}
        if loc == 'right' and 'left' not in visited:
            return {'action': 'move_left'}
        return {'action': 'no_op'}


def format_state(state: dict) -> str:
    """Render a state dict for a terminal.

    `visited` is a set, which json.dumps refuses to serialize. Sort it into braces on
    the way out rather than changing the agent's data structure to please a printer --
    a set is the right type for "places I have been" and the source page uses one.
    """
    parts = []
    for key, value in state.items():
        if isinstance(value, set):
            value = "{" + ", ".join(sorted(str(item) for item in value)) + "}"
        parts.append(f"{key}={value}")
    return "  ".join(parts)


if __name__ == "__main__":
    agent = ModelBasedReflexAgent()

    # Five percepts across the two cells. The last one deliberately omits `location`:
    # the sensor went quiet, and the agent has to act on what it already believes about
    # where it is standing. A simple reflex agent cannot do that at all.
    percepts = [
        Percept({'location': 'left', 'status': 'dirty'}),
        Percept({'location': 'left', 'status': 'clean'}),
        Percept({'location': 'right', 'status': 'dirty'}),
        Percept({'location': 'right', 'status': 'clean'}),
        Percept({'status': 'dirty'}),
    ]

    print("Model-based reflex agent, two-cell vacuum world.")
    print("Every decision below is a hand-coded conditional. No model is involved.\n")

    for step, percept in enumerate(percepts, start=1):
        action = agent.agent_function(percept)
        print(f"step {step}")
        print(f"  see:   {percept.data}")
        print(f"  do:    {action}")
        print(f"  state: {format_state(agent.state)}  ({len(agent.state)} keys)")
        print()

    print("The agent started with an empty state and ended with "
          f"{len(agent.state)} keys.")
    print("Step 5 had no location in the percept. The action came from memory.")
