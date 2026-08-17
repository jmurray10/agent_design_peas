"""Simple reflex agent, classical version.

A vacuum cleaner robot in a two-cell world. It sees its current location and the status
of that cell, then picks an action out of a hand-written rule table. No memory, no
planning, no model of the world -- one percept in, one action out.

    python before.py
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Percept:
    data: Dict[str, Any]
    timestamp: float = 0.0


class SimpleReflexAgent:
    """Selects action based ONLY on current percept. No history."""

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules

    def agent_function(self, percept: Percept) -> str:
        state = self.interpret_input(percept)
        rule = self.rule_match(state)
        return rule.get('action', 'no_op')

    def interpret_input(self, percept: Percept) -> str:
        # Fix to the source listing. The source returns str(percept.data) alone, but the
        # rule keys 'clean_left' and 'clean_right' never appear as substrings of a dict
        # repr like "{'location': 'left', 'status': 'clean'}", so those two rules could
        # never fire and a clean cell wrongly produced no_op. Appending the canonical
        # "<status>_<location>" token lets the hand-written rules match what they were
        # plainly written to match. The rules dict and rule_match are untouched, and the
        # only percept that still falls through is the one that is genuinely unenumerated.
        status = percept.data.get('status', '')
        location = percept.data.get('location', '')
        return f"{percept.data} {status}_{location}"

    def rule_match(self, state: str) -> Dict:
        for condition, action in self.rules.items():
            if condition in state:
                return {'action': action}
        return {}


# Every rule is hand-coded
rules = {
    'dirty': 'suck',
    'clean_left': 'move_right',
    'clean_right': 'move_left'
}
agent = SimpleReflexAgent(rules)

percepts = [
    Percept({'location': 'left', 'status': 'dirty'}),
    Percept({'location': 'left', 'status': 'clean'}),
    Percept({'location': 'right', 'status': 'dirty'}),
    # Nothing in the rules dict covers 'puddle'. The agent does not crash and it does not
    # improvise: rule_match returns {} and agent_function falls through to no_op. Silent
    # inaction on a percept nobody enumerated is the failure mode this file exists to
    # show. after.py is the same architecture with that one component replaced.
    Percept({'location': 'right', 'status': 'puddle'}),
]
for p in percepts:
    action = agent.agent_function(p)
    print(f"See: {p.data} -> Do: {action}")
