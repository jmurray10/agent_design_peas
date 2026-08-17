# 01 - Reflex Agents: Before and After LLMs

---

## Architecture Overview

**Simple Reflex Agent** -- picks actions based only on the current percept. No memory, no planning. Condition-action rules. Works in fully observable environments.

**Model-Based Reflex Agent** -- maintains internal state to track aspects of the world it cannot currently see. Has a model of how the world evolves and how its own actions affect the world. Handles partially observable environments.

Both are reactive. They do not search or plan ahead.

---

## Simple Reflex Agent

### BEFORE: Classical

A vacuum cleaner robot. Two cells, might have dirt. The agent sees its location and the cell status, then picks from a fixed rule set.

```python
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
        return str(percept.data)

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
]
for p in percepts:
    action = agent.agent_function(p)
    print(f"See: {p.data} -> Do: {action}")
```

**Strengths:** fast, predictable, deterministic.

**Limitations:** needs a rule for every possible percept. If the agent encounters something it was not programmed for ("there is a puddle on the floor"), it returns `no_op`. Real-world percepts are messy and cannot be fully enumerated.

---

### AFTER: LLM-Powered

Same architecture. Still no memory, still no planning. The LLM replaces the rule table so the agent can handle percepts it was never explicitly programmed for.

```python
import json

class LLMSimpleReflexAgent:
    """
    Still a simple reflex agent. Still no memory, no planning.
    The LLM replaces the hand-coded rule table.
    """
    def __init__(self, role: str, available_actions: list):
        self.role = role
        self.available_actions = available_actions

    def agent_function(self, percept: Percept) -> str:
        return self.llm_rule_match(percept)

    def llm_rule_match(self, percept: Percept) -> str:
        prompt = f"""You are a {self.role}.
You can ONLY pick from these actions: {self.available_actions}
You see: {percept.data}

Pick one action. Return just the action name, nothing else."""

        response = llm_call(prompt)
        action = response.strip()

        # DETERMINISTIC VALIDATION
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
    print(f"See: {p.data} -> Do: {action}")
```

### Comparison

| Aspect | Before | After |
|--------|--------|-------|
| Architecture | Simple reflex | Simple reflex (identical) |
| Memory | None | None |
| Rule matching | Hand-coded dict lookup | LLM interprets percept |
| Action validation | N/A | Deterministic check against allowed list |
| Novel percepts | Fails silently | Generalizes |
| Speed | Instant | Network + inference latency |
| Predictability | 100% deterministic | Nondeterministic |

The PEAS spec is identical in both cases. Only the implementation of `rule_match` changed.

### Config

```yaml
agent:
  name: "vacuum-bot"
  architecture: "simple-reflex"
  performance:
    metrics: ["+1 per clean square at time T"]
    eval: "eval/test_cases.json"
  environment:
    type: "fully-observable, deterministic, episodic, static, discrete"
  actuators:
    - name: "suck"
      output_schema: "schemas/action.json"
    - name: "move_left"
      output_schema: "schemas/action.json"
    - name: "move_right"
      output_schema: "schemas/action.json"
    - name: "no_op"
  sensors:
    - name: "location"
      type: "discrete"
      values: ["left", "right"]
    - name: "status"
      type: "text"
      input_schema: "schemas/percept.json"
  prompts:
    system: "prompts/system.md"
```

---

## Model-Based Reflex Agent

### BEFORE: Classical

Tracks internal state. Remembers what it has seen. Predicts consequences of its actions.

```python
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
```

**Strengths:** handles partial observability. Remembers context.

**Limitations:** `update_state`, `predict_effect`, and `rule_match` are all hand-coded. Fine for a two-cell grid. Not feasible for a support agent tracking conversation history, sentiment, and case status across hundreds of interaction patterns.

---

### AFTER: LLM-Powered

Same architecture. Still maintains state. Still uses a world model. The LLM handles the parts that are hard to hand-code: interpreting messy percepts, updating state representation, picking actions from complex state.

```python
import json

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

    def agent_function(self, percept: Percept) -> str:
        self.state = self.llm_update_state(self.state, percept)
        action = self.llm_rule_match(self.state)
        if action not in self.available_actions:
            action = 'no_op'
        self.state = self.llm_predict_effect(self.state, action)
        return action

    def llm_update_state(self, state: dict, percept: Percept) -> dict:
        prompt = f"""You are tracking internal state for a {self.role}.
Current state: {json.dumps(state)}
New observation: {json.dumps(percept.data)}
Update the state. Keep relevant history. Return valid JSON only."""
        response = llm_call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            updated = state.copy()
            updated.update(percept.data)
            return updated  # deterministic fallback

    def llm_predict_effect(self, state: dict, action: str) -> dict:
        prompt = f"""Predict the new state after action "{action}".
Current state: {json.dumps(state)}
Return valid JSON only."""
        response = llm_call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return state

    def llm_rule_match(self, state: dict) -> str:
        prompt = f"""You are a {self.role}.
Your internal state: {json.dumps(state)}
Available actions: {self.available_actions}
Pick the best action. Return just the action name."""
        return llm_call(prompt).strip()


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
for p in percepts:
    action = agent.agent_function(p)
    print(f"See: {p.data}\nState: {json.dumps(agent.state, indent=2)}\nDo: {action}\n")
```

### Comparison

| Aspect | Before | After |
|--------|--------|-------|
| Architecture | Model-based reflex | Model-based reflex (identical) |
| State tracking | Hand-coded dict updates | LLM interprets and updates JSON |
| World model | Hand-coded predict_effect | LLM predicts effects |
| Rule matching | Hand-coded conditionals | LLM picks from action list |
| Fallbacks | None | JSON parse failures fall back to deterministic merge |
| Handles messy input | No | Yes |

The oscillation pattern:
1. **[LLM]** Interpret percept, update state
2. **[LLM]** Pick action from state
3. **[DETERMINISTIC]** Validate action is in the allowed list
4. **[LLM]** Predict effect on state
5. **[DETERMINISTIC]** JSON parse with fallback

### Config

```yaml
agent:
  name: "support-bot"
  architecture: "model-based-reflex"
  performance:
    metrics: ["customer satisfaction", "resolution time", "escalation rate"]
    eval: "eval/test_cases.json"
  environment:
    type: "partially-observable, stochastic, sequential, dynamic"
  actuators:
    - name: "reply_to_customer"
      output_schema: "schemas/action_reply.json"
    - name: "escalate_to_manager"
    - name: "check_order_status"
      output_schema: "schemas/action_lookup.json"
    - name: "issue_refund"
      output_schema: "schemas/action_refund.json"
    - name: "request_more_info"
    - name: "close_ticket"
  sensors:
    - name: "customer_message"
      type: "text"
      input_schema: "schemas/percept_message.json"
    - name: "order_lookup"
      type: "structured"
      input_schema: "schemas/percept_order.json"
    - name: "ticket_history"
      type: "text"
  prompts:
    system: "prompts/system.md"
    reply: "prompts/reply.md"
    escalation: "prompts/escalation.md"
  state:
    schema: "schemas/state.json"
    persistence: "per-conversation"
    fallback: "merge-percept-data"
```
