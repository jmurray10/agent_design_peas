# 03 - Utility-Based Agents: Before and After LLMs

---

## Architecture Overview

A utility-based agent uses a **utility function** that maps states to a degree of desirability. It picks actions that maximize **expected utility**, handling tradeoffs and uncertainty.

**Markov Decision Processes (MDPs)** are the standard framework for sequential decisions under uncertainty. An MDP is defined by:
- Set of states S, set of actions A
- Transition model P(s' | s, a) -- probability of reaching s' from s via action a
- Reward function R(s, a, s')
- Discount factor gamma

The solution is a **policy** (state -> action mapping), not a fixed plan. Solved with **value iteration** (Bellman updates) or **policy iteration** (evaluate then improve).

---

## Utility-Based Agent with Value Iteration

### BEFORE: Classical

The 4x3 grid world. Stochastic transitions (0.8 intended, 0.1 each perpendicular). Terminal states +1 and -1. Living reward -0.04.

```python
from collections import defaultdict
from typing import Dict, List, Set, Any
from dataclasses import dataclass
import random

@dataclass
class MDP:
    states: List[Any]
    actions: List[Any]
    transition_model: Dict
    reward_function: Dict
    gamma: float
    initial_state: Any
    terminal_states: Set[Any]

    def P(self, s_prime, s, a):
        return self.transition_model.get((s, a, s_prime), 0.0)

    def R(self, s, a, s_prime):
        return self.reward_function.get((s, a, s_prime), 0.0)

    def actions_available(self, state):
        if self.terminal_states and state in self.terminal_states:
            return []
        return self.actions

    def successors(self, state, action):
        return [(s, self.P(s, state, action))
                for s in self.states if self.P(s, state, action) > 0]


def value_iteration(mdp: MDP, epsilon=0.001, max_iter=1000):
    """Bellman update until convergence. Returns V* and pi*."""
    V = {s: 0.0 for s in mdp.states}
    for i in range(max_iter):
        V_new, delta = {}, 0
        for s in mdp.states:
            if mdp.terminal_states and s in mdp.terminal_states:
                V_new[s] = V[s]; continue
            max_val = float('-inf')
            for a in mdp.actions_available(s):
                val = sum(p * (mdp.R(s, a, sp) + mdp.gamma * V[sp])
                          for sp, p in mdp.successors(s, a))
                max_val = max(max_val, val)
            V_new[s] = max_val
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < epsilon: break
    policy = {}
    for s in mdp.states:
        if mdp.terminal_states and s in mdp.terminal_states:
            policy[s] = None; continue
        best_a, best_v = None, float('-inf')
        for a in mdp.actions_available(s):
            v = sum(p * (mdp.R(s, a, sp) + mdp.gamma * V[sp])
                    for sp, p in mdp.successors(s, a))
            if v > best_v: best_v, best_a = v, a
        policy[s] = best_a
    return V, policy
```

**Strengths:** provably optimal. Converges to V*. Complexity O(S^2 * A) per iteration.

**Limitations:** requires every state, every transition probability, and every reward to be specified. Real environments rarely provide P(s' | s, a) explicitly. Large or continuous state spaces make value iteration infeasible.

---

### AFTER: LLM-Powered (Three Paths)

**Path 1: Classical MDP + LLM for state estimation**

When the MDP structure is known but real-world percepts are noisy.

```python
import json

class LLMStateEstimator:
    """
    MDP solved classically. LLM maps noisy percepts to MDP states.
    Policy lookup is deterministic.
    """
    def __init__(self, mdp: MDP, policy: Dict):
        self.mdp = mdp
        self.policy = policy

    def act(self, percept: dict) -> str:
        state = self.estimate_state(percept)
        action = self.policy.get(state)
        return action if action else 'no_op'

    def estimate_state(self, percept: dict) -> tuple:
        prompt = f"""Estimate the agent's position in a 4x3 grid.
Wall at (1,1). Terminals: (3,2)=+1, (3,1)=-1.
Sensor reading: {json.dumps(percept)}
Return JSON: {{"x": int, "y": int}}"""
        response = llm_call(prompt)
        try:
            p = json.loads(response)
            return (p['x'], p['y'])
        except (json.JSONDecodeError, KeyError):
            return self.mdp.initial_state
```

**Path 2: LLM as the policy function**

When the state space is too large for value iteration. The LLM approximates pi(s) directly -- acting as a function approximator for the policy.

```python
class LLMPolicyAgent:
    """
    LLM approximates the optimal policy.
    pi(s) = a  becomes  llm_call(state) -> action
    Not provably optimal. Works on state spaces value iteration cannot.
    """
    def __init__(self, role: str, available_actions: list,
                 reward_description: str, gamma: float = 0.9):
        self.role = role
        self.available_actions = available_actions
        self.reward_description = reward_description
        self.gamma = gamma
        self.episode_history = []

    def act(self, state: dict) -> str:
        prompt = f"""You are a {self.role} making sequential decisions.
State: {json.dumps(state)}
Actions: {self.available_actions}
Rewards: {self.reward_description}
Discount: {self.gamma}
Recent history: {self.episode_history[-5:]}

Which action maximizes expected total discounted reward?
Return just the action name."""

        action = llm_call(prompt).strip()
        if action not in self.available_actions:
            action = self.available_actions[0]
        self.episode_history.append({'state': state, 'action': action})
        return action

# Example: content moderation (state space too large for value iteration)
agent = LLMPolicyAgent(
    role="content moderation agent",
    available_actions=["approve", "flag_for_review", "reject", "request_edit"],
    reward_description="""
    +1 correctly approved good content
    +1 correctly rejected harmful content
    -5 approving harmful content (false negative)
    -1 rejecting good content (false positive)
    -0.1 per step (time cost)""",
    gamma=0.95
)
```

**Path 3: LLM exploration with deterministic reward tracking**

When transition probabilities are unknown. LLM picks actions, deterministic code tracks outcomes -- similar to Q-learning but with LLM action selection.

```python
class LLMExplorationAgent:
    """
    LLM explores. Deterministic code tracks rewards.
    Like Q-learning: the agent does not know the transition model
    and must learn from experience.
    """
    def __init__(self, role: str, available_actions: list):
        self.role = role
        self.available_actions = available_actions
        self.reward_history = []
        self.q_estimates = {}

    def act(self, state: str) -> str:
        if random.random() < 0.1:
            return random.choice(self.available_actions)
        prompt = f"""You are a {self.role}.
State: {state}
Actions: {self.available_actions}
Past experience:
{self.format_history()}
Pick the action most likely to give high reward."""
        action = llm_call(prompt).strip()
        if action not in self.available_actions:
            action = random.choice(self.available_actions)
        return action

    def observe_reward(self, state, action, reward, next_state):
        """DETERMINISTIC reward tracking."""
        self.reward_history.append({
            'state': state, 'action': action,
            'reward': reward, 'next_state': next_state
        })
        key = (state, action)
        alpha = 0.1
        prev = self.q_estimates.get(key, 0)
        self.q_estimates[key] = prev + alpha * (reward - prev)

    def format_history(self) -> str:
        return '\n'.join(
            f"  {e['state']} + {e['action']} -> {e['reward']}"
            for e in self.reward_history[-10:]
        ) or '  No experience yet.'
```

---

### Comparison

| Aspect | Classical MDP | LLM-Powered |
|--------|--------------|-------------|
| Optimality | Provably optimal | No guarantee (approximation) |
| State space | Must be finite, enumerable | Handles massive or continuous |
| Transitions | Must know P(s'\|s,a) exactly | Can work with unknown transitions |
| Rewards | Explicit R(s,a,s') for all triples | Natural language reward description |
| Bellman equation | Computed exactly | LLM approximates value reasoning |
| Policy | Extracted from V* | LLM acts as the policy directly |

**Decision rule:** small known MDP -> solve classically, use LLM only for state estimation. Large state space -> LLM approximates policy. Unknown transitions -> LLM explores with deterministic reward tracking.

### Config

```yaml
agent:
  name: "content-moderator"
  architecture: "utility-based"
  performance:
    reward_structure: "schemas/rewards.json"       # externalized reward definitions
    eval: "eval/moderation_cases.json"
  environment:
    type: "partially-observable, stochastic, sequential, dynamic"
  actuators:
    - name: "approve"
      output_schema: "schemas/action_decision.json"
    - name: "flag_for_review"
      output_schema: "schemas/action_decision.json"
    - name: "reject"
      output_schema: "schemas/action_decision.json"
    - name: "request_edit"
      output_schema: "schemas/action_decision.json"
  sensors:
    - name: "content"
      type: "text"
      input_schema: "schemas/percept_content.json"
    - name: "user_history"
      type: "structured"
      input_schema: "schemas/percept_user.json"
    - name: "content_flags"
      type: "structured"
  prompts:
    system: "prompts/system.md"
    moderation: "prompts/moderation.md"
  decision_making:
    strategy: "llm_policy"
    discount_factor: 0.95
    exploration_rate: 0.1
    reward_tracking: "deterministic"
    fallback: "flag_for_review"
```
