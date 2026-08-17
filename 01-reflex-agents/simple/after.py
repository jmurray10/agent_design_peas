"""Simple reflex agent, LLM version.

Same architecture as before.py. Still no memory, still no planning, still exactly one
action per percept. The only thing that changed is where the condition-action rule comes
from: a hand-written dict became a model call, wrapped in a deterministic check that the
answer is an action this robot actually has.

    python after.py

Runs with no API key. See the mock-mode banner on the first line of output.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# This file is meant to be run from the repo root ("python 01-reflex-agents/simple/
# after.py"). Python puts the script's own directory on sys.path, never the root, so the
# root has to be added by hand before shared/ can be imported. parents[2] is the repo
# root from 01-reflex-agents/simple/after.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.llm import llm_call


@dataclass
class Percept:
    data: Dict[str, Any]
    timestamp: float = 0.0


def mock_key_for(percept: Percept) -> str:
    """Name the canned response mock mode should return for this percept.

    Every real backend ignores mock_key. This exists so that an offline run exercises a
    different model answer per percept instead of returning the same string three times,
    which would make the demo look like a constant rather than a decision.
    """
    status = str(percept.data.get('status', ''))
    slug = re.sub(r'[^a-z0-9]+', '_', status.lower()).strip('_')
    return f"reflex_simple_{slug}"


class LLMSimpleReflexAgent:
    """
    Still a simple reflex agent. Still no memory, no planning.
    The LLM replaces the hand-coded rule table.
    """

    def __init__(self, role: str, available_actions: list):
        self.role = role
        self.available_actions = available_actions
        # What the model actually said, before validation had an opinion about it. Kept
        # only so the demo below can show the difference between an answer that passed
        # and an answer that got collapsed. Nothing in the agent's logic reads it.
        self.last_response: str = ""

    def agent_function(self, percept: Percept) -> str:
        return self.llm_rule_match(percept)

    def llm_rule_match(self, percept: Percept) -> str:
        prompt = f"""You are a {self.role}.
You can ONLY pick from these actions: {self.available_actions}
You see: {percept.data}

Pick one action. Return just the action name, nothing else."""

        # tier="small": the whole job is mapping one short percept onto one label from a
        # four-item list. No multi-step reasoning, no structured output, no synthesis
        # across examples -- the kind of classification a small model does as well as a
        # frontier one. It also sits in the agent's inner loop, once per percept, so
        # per-call latency and cost are the entire budget. Paying frontier prices to
        # choose between "suck" and "move_left" would be the expensive mistake.
        response = llm_call(prompt, mock_key=mock_key_for(percept), tier="small")
        action = response.strip()
        self.last_response = action

        # DETERMINISTIC VALIDATION
        # A model asked for one word can answer with a sentence, or name an actuator this
        # robot does not have. This check is why that cannot turn into an illegal action:
        # anything off the list collapses to no_op, the same floor before.py hit, except
        # here it is a checked boundary rather than a hole in the rule table.
        if action not in self.available_actions:
            return 'no_op'
        return action


agent = LLMSimpleReflexAgent(
    role="vacuum cleaner robot",
    available_actions=["suck", "move_left", "move_right", "no_op"]
)

# Handles inputs it was never explicitly programmed for
percepts = [
    Percept({'location': 'left', 'status': 'dirty'}),
    Percept({'location': 'right', 'status': 'puddle and dirt mixed together'}),
    Percept({'location': 'right', 'status': 'something sticky, maybe glue'}),
]
for p in percepts:
    action = agent.agent_function(p)

    # before.py answers 'no_op' on a percept its rule table does not cover. This agent can
    # answer 'no_op' too, for either of two unrelated reasons, and the distinction is the
    # whole point of the example: the model deliberately declining to vacuum a puddle is
    # not the same event as the model saying something unusable and the validation check
    # catching it. Printed here because both look identical from the outside otherwise.
    if agent.last_response == action:
        why = "chosen by the model"
    else:
        why = f"model said {agent.last_response!r}, which is not an available action"
    print(f"See: {p.data} -> Do: {action}   [{why}]")
