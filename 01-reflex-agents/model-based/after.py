"""Model-based reflex agent -- the LLM version.

The architecture is unchanged from before.py: percept in, state update, rule match,
effect prediction, action out. Three of those five steps are model calls now. The two
that carry the safety guarantees are still ordinary Python:

    action validation   the model's answer must be in available_actions or it is refused
    JSON parse          every parse has a deterministic fallback under it

The second one is the point of this example. A model that returns broken JSON degrades
the agent instead of stopping it: the run keeps going with a hand-coded merge of the
percept into the previous state. This file is wired so that path actually fires.

Runs with no API key. shared/llm.py returns canned responses and prints a banner.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# after.py is run from the repo root as `python 01-reflex-agents/model-based/after.py`,
# and Python puts the script's own directory on sys.path, not the repo root. Two levels
# up from this file is the repo root, which is where `shared` lives.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.llm import llm_call  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402


@dataclass
class Percept:
    data: Dict[str, Any]
    timestamp: float = 0.0


class LLMModelBasedReflexAgent:
    """
    Model-based reflex agent. Maintains internal state.
    LLM handles state interpretation and updates.
    Deterministic code handles validation and execution.
    """
    def __init__(self, role: str, available_actions: list):
        self.role = role
        self.available_actions = available_actions
        self.state = {}
        # Selects which canned response comes back in mock mode, so a three-turn
        # conversation reads like a conversation. Every real backend ignores mock_key,
        # so this counter has no effect on a live run.
        self.step = 0

    def agent_function(self, percept: Percept) -> str:
        self.step += 1
        self.state = self.llm_update_state(self.state, percept)
        action = self.llm_rule_match(self.state)
        if action not in self.available_actions:
            # 'no_op' is not one of the support agent's actions, and that is the right
            # answer: it is a sentinel meaning "the model named something it is not
            # allowed to do", which a caller must treat as a refusal, not an action.
            print(f"  [validation] {action!r} is not an allowed action -> no_op")
            action = 'no_op'
        self.state = self.llm_predict_effect(self.state, action)
        return action

    def llm_update_state(self, state: dict, percept: Percept) -> dict:
        prompt = f"""You are tracking internal state for a {self.role}.
Current state: {json.dumps(state)}
New observation: {json.dumps(percept.data)}
Update the state. Keep relevant history. Return valid JSON only."""
        # tier=mid: structured JSON generation over a state object that grows every
        # turn, with a deterministic merge underneath it. A small model gets the
        # content right and the syntax wrong often enough that the fallback would
        # become the normal path; a frontier model buys nothing once the schema is
        # this narrow and the fallback is this cheap.
        response = llm_call(prompt, mock_key=f"reflex_model_state_{self.step}", tier="mid")
        try:
            return model_loads(response)
        except json.JSONDecodeError:
            updated = state.copy()
            updated.update(percept.data)  # deterministic fallback
            # Print what the merge produced, not just that it happened. The keys are
            # the evidence: the raw percept field lands in state unnormalized, which
            # is exactly the quality the model was supposed to add and did not.
            print(f"  [fallback] llm_update_state got unparseable JSON, ending {response[-18:]!r}")
            print(f"             -> merged the percept in by hand, state keys now: "
                  f"{', '.join(updated)}")
            return updated

    def llm_predict_effect(self, state: dict, action: str) -> dict:
        prompt = f"""Predict the new state after action "{action}".
Current state: {json.dumps(state)}
Return valid JSON only."""
        # tier=mid: structured JSON generation again, projecting one action forward
        # onto the same state object. Wrong-but-parseable output here corrupts beliefs
        # silently rather than raising, so this call wants a model that can hold a
        # schema across turns -- not the cheapest one available.
        response = llm_call(prompt, mock_key=f"reflex_model_effect_{self.step}", tier="mid")
        try:
            return model_loads(response)
        except json.JSONDecodeError:
            print(f"  [fallback] llm_predict_effect got unparseable JSON, starting "
                  f"{response[:34]!r}")
            print(f"             -> state left unchanged, so the effect of {action!r} "
                  f"is not recorded")
            return state

    def llm_rule_match(self, state: dict) -> str:
        prompt = f"""You are a {self.role}.
Your internal state: {json.dumps(state)}
Available actions: {self.available_actions}
Pick the best action. Return just the action name."""
        # tier=small: pick one label from a six-item list, with the state already
        # summarized by the previous call. No free-form generation, and the
        # membership check in agent_function catches anything off-list, so the
        # cheapest model that can follow "return just the name" is enough.
        return llm_call(prompt, mock_key=f"reflex_model_action_{self.step}", tier="small").strip()


if __name__ == "__main__":
    # Handles unstructured, partially observable environments
    agent = LLMModelBasedReflexAgent(
        role="customer support agent",
        available_actions=[
            "reply_to_customer", "escalate_to_manager",
            "check_order_status", "issue_refund",
            "request_more_info", "close_ticket"
        ]
    )

    percepts = [
        Percept({'message': 'My order hasnt arrived and its been 2 weeks'}),
        Percept({'order_lookup': 'Order #4521 - shipped 12 days ago, stuck in transit'}),
        Percept({'message': 'Third time this has happened. I want a refund.'}),
    ]

    print("Model-based reflex agent, customer support ticket.")
    print("Same architecture as before.py. update_state, rule_match and predict_effect")
    print("are model calls now. Validation and JSON parsing are not.\n")

    for percept in percepts:
        print(f"See: {percept.data}")
        action = agent.agent_function(percept)
        print(f"State: {json.dumps(agent.state, indent=2)}")
        print(f"Do: {action}\n")

    print("Any [fallback] line above is a model call that returned unparseable JSON.")
    print("The run continued on hand-coded logic instead of raising.")
