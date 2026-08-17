# 04 - Learning Agents: Before and After LLMs

---

## Architecture Overview

A learning agent has four components:

1. **Performance element** -- picks actions (can be any agent type)
2. **Critic** -- evaluates results against the performance measure
3. **Learning element** -- takes feedback from the critic and improves the performance element
4. **Problem generator** -- suggests exploratory actions to gain new experience

The canonical example is **Q-learning**: a model-free RL algorithm that learns Q-values (expected utility of action a in state s) from experience without knowing the transition model. Q-learning is off-policy -- it can learn Q* without ever executing the optimal policy.

---

## Learning Agent with Q-Learning

### BEFORE: Classical

```python
from collections import defaultdict
import random

class QLearningAgent:
    """
    Q-Learning. Model-free reinforcement learning.
    Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.get_best_action(state)

    def get_best_action(self, state):
        best_a, best_v = None, float('-inf')
        for a in self.actions:
            v = self.Q[state][a]
            if v > best_v: best_v, best_a = v, a
        return best_a if best_a else random.choice(self.actions)

    def update(self, state, action, reward, next_state):
        current_q = self.Q[state][action]
        max_next = max((self.Q[next_state][a] for a in self.actions), default=0)
        self.Q[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next - current_q
        )

    def get_policy(self):
        return {s: self.get_best_action(s) for s in self.states}


# Training
states = ['A', 'B', 'C', 'D', 'Goal']
actions = ['left', 'right']
agent = QLearningAgent(states, actions)

for episode in range(500):
    state = 'A'
    for step in range(20):
        if state == 'Goal': break
        action = agent.get_action(state)
        next_map = {'right': {'A':'B','B':'C','C':'D','D':'Goal'},
                    'left': {'D':'C','C':'B','B':'A'}}
        next_state = next_map.get(action, {}).get(state, state)
        reward = 10 if next_state == 'Goal' else -1
        agent.update(state, action, reward, next_state)
        state = next_state

for s in states[:-1]:
    print(f"  {s}: {agent.get_policy()[s]}  "
          f"(Q_left={agent.Q[s]['left']:.1f}, Q_right={agent.Q[s]['right']:.1f})")
```

**Strengths:** converges to Q* with sufficient exploration (proven). Off-policy. No model required.

**Limitations:** discrete, finite state/action spaces. Many episodes to converge. Cannot handle natural language states. Cannot reason about novel situations.

---

### AFTER: LLM-Powered

The four learning agent components map directly:

| Component | Q-Learning | LLM Version |
|-----------|-----------|-------------|
| Performance element | Q-table lookup | LLM picks actions with experience in context |
| Critic | Reward signal | Deterministic metrics + LLM self-evaluation |
| Learning element | Q-update rule | Updated context / few-shot examples |
| Problem generator | Epsilon-greedy | LLM targets gaps in experience |

```python
import json
from collections import defaultdict

class LLMLearningAgent:
    """
    Learning agent architecture implemented with LLM.
    Performance element: LLM (informed by experience)
    Critic: Deterministic code (always)
    Learning element: Experience -> context updates
    Problem generator: LLM suggests exploration
    """
    def __init__(self, role: str, available_actions: list,
                 performance_metrics: list):
        self.role = role
        self.available_actions = available_actions
        self.performance_metrics = performance_metrics

        # Deterministic tracking (the Critic)
        self.experience_log = []
        self.performance_scores = []
        self.action_reward_stats = defaultdict(list)

        # Learning element state
        self.successful_examples = []
        self.failed_examples = []
        self.learned_rules = []

    # -- PERFORMANCE ELEMENT (LLM) --
    def act(self, state: dict) -> str:
        prompt = f"""You are a {self.role}.

Situation: {json.dumps(state)}
Actions: {self.available_actions}

LEARNED FROM EXPERIENCE:
Worked well:
{self._fmt(self.successful_examples[-5:])}

Did NOT work:
{self._fmt(self.failed_examples[-5:])}

Patterns:
{json.dumps(self.learned_rules[-5:])}

Action stats:
{self._fmt_stats()}

Pick the best action. Return just the action name."""

        action = llm_call(prompt).strip()
        if action not in self.available_actions:
            action = self.available_actions[0]
        return action

    # -- CRITIC (Deterministic -- always) --
    def observe_outcome(self, state: dict, action: str, outcome: dict) -> float:
        reward = self._calculate_reward(outcome)
        entry = {'state': state, 'action': action,
                 'outcome': outcome, 'reward': reward}
        self.experience_log.append(entry)
        self.action_reward_stats[action].append(reward)
        self.performance_scores.append(reward)
        return reward

    def _calculate_reward(self, outcome: dict) -> float:
        r = 0.0
        if outcome.get('success'): r += 1.0
        if outcome.get('time_seconds', 999) < 30: r += 0.5
        if outcome.get('error'): r -= 2.0
        if outcome.get('customer_satisfied'): r += 1.5
        return r

    # -- LEARNING ELEMENT (Deterministic sorting + LLM pattern extraction) --
    def learn(self):
        if len(self.experience_log) < 5:
            return
        sorted_exp = sorted(self.experience_log, key=lambda e: e['reward'])
        self.failed_examples = sorted_exp[:3]
        self.successful_examples = sorted_exp[-3:]

        prompt = f"""Analyze these experiences and extract patterns.
Successes: {json.dumps(self.successful_examples[-5:], indent=2)}
Failures: {json.dumps(self.failed_examples[-5:], indent=2)}
What patterns? What to do more, what to avoid?
Return JSON list of rule strings."""
        response = llm_call(prompt)
        try:
            self.learned_rules = json.loads(response)
        except json.JSONDecodeError:
            pass

    # -- PROBLEM GENERATOR (LLM) --
    def suggest_exploration(self, state: dict) -> str:
        prompt = f"""Suggest an experiment for a {self.role}.
Situation: {json.dumps(state)}
Actions: {self.available_actions}
Usage counts: {json.dumps({a: len(r) for a, r in self.action_reward_stats.items()})}
Suggest an underexplored action. Return just the action name."""
        action = llm_call(prompt).strip()
        if action not in self.available_actions:
            counts = {a: len(self.action_reward_stats[a]) for a in self.available_actions}
            action = min(counts, key=counts.get)
        return action

    def _fmt(self, examples):
        if not examples: return "  None yet."
        return '\n'.join(f"  '{e['action']}' -> reward {e['reward']}" for e in examples)

    def _fmt_stats(self):
        lines = []
        for a in self.available_actions:
            rs = self.action_reward_stats[a]
            if rs: lines.append(f"  {a}: avg={sum(rs)/len(rs):.2f} (n={len(rs)})")
            else: lines.append(f"  {a}: no data")
        return '\n'.join(lines)


# Usage
agent = LLMLearningAgent(
    role="email response agent",
    available_actions=["send_template", "write_custom", "escalate", "request_info", "auto_resolve"],
    performance_metrics=["quality", "time_to_resolution", "satisfaction"]
)

for i in range(20):
    state = {'type': 'complaint', 'urgency': 'medium', 'tier': 'premium'}

    if random.random() < 0.15:
        action = agent.suggest_exploration(state)
    else:
        action = agent.act(state)

    outcome = {
        'success': random.random() > 0.3,
        'time_seconds': random.randint(5, 120),
        'customer_satisfied': random.random() > 0.4,
    }

    agent.observe_outcome(state, action, outcome)

    if (i + 1) % 10 == 0:
        agent.learn()
        print(f"After {i+1} interactions:")
        print(f"  Avg reward: {sum(agent.performance_scores[-10:])/10:.2f}")
        print(f"  Rules: {agent.learned_rules[:2]}\n")
```

---

### Comparison

| Component | Q-Learning | LLM Learning Agent |
|-----------|-----------|-------------------|
| Performance element | argmax Q(s,a) | LLM with experience in context |
| Critic | Reward number | Deterministic reward calc (same concept) |
| Learning element | Q-update (Bellman) | Sort experience + LLM pattern extraction |
| Problem generator | Random (epsilon-greedy) | LLM targets gaps |
| State representation | Discrete, enumerable | Natural language, JSON, anything |
| Convergence | Proven | Not proven (improves with experience) |
| Off-policy | Yes | Partial (can learn from logs) |
| Exploration | Random | Intentional |

**Critical design rule:** the critic is always deterministic. If the critic is also an LLM, you lose your ground truth signal.

### Config

```yaml
agent:
  name: "email-responder"
  architecture: "learning"
  performance:
    metrics: ["quality", "time_to_resolution", "satisfaction"]
    reward_function: "schemas/rewards.json"        # externalized reward definitions
    eval: "eval/email_cases.json"
  environment:
    type: "partially-observable, stochastic, sequential, dynamic"
  actuators:
    - name: "send_template"
      output_schema: "schemas/action_reply.json"
    - name: "write_custom"
      output_schema: "schemas/action_reply.json"
    - name: "escalate"
    - name: "request_info"
    - name: "auto_resolve"
  sensors:
    - name: "email_content"
      type: "text"
      input_schema: "schemas/percept_email.json"
    - name: "customer_profile"
      type: "structured"
      input_schema: "schemas/percept_customer.json"
  prompts:
    system: "prompts/system.md"
    response: "prompts/response.md"
    pattern_extraction: "prompts/learn_patterns.md"  # used by learning element
  learning:
    performance_element: "llm"
    critic: "deterministic"
    learning_method: "experience_context"
    exploration_rate: 0.15
    exploration_strategy: "llm_suggested"
    learn_every_n: 10
    max_experience_in_context: 10
    experience_store: "data/experience_log.json"     # persisted experience
```
