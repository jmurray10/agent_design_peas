"""Utility-based agent, LLM-powered: the three paths from the source page.

Runs with no API key. `shared/llm.py` falls back to canned responses so every prompt is
still built, every response still parsed, and every fallback still fires.

    Path 1  LLMStateEstimator     small known MDP -> solve classically, LLM only reads sensors
    Path 2  LLMPolicyAgent        state space too large -> LLM approximates pi(s)
    Path 3  LLMExplorationAgent   transitions unknown -> LLM explores, code tracks reward

Path 1 imports the solved policy from before.py. It does not recompute it and it does not
contain a second copy of value_iteration. That is the point of the whole example: the
optimality guarantee survives because the policy is still the output of the Bellman
updates, and the LLM never touches it.
"""

from __future__ import annotations

import inspect
import json
import random
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for shared/
sys.path.insert(0, str(Path(__file__).resolve().parent))      # this dir, for before.py

from shared.llm import llm_call
from shared.model_json import loads as model_loads

# The classical solver, imported rather than reimplemented.
from before import MDP, build_grid_world, print_policy_grid, value_iteration


# -- Path 1: classical MDP + LLM for state estimation --------------------------------

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
        # tier=mid: turning a noisy sensor dict into a two-field JSON object is structured
        # generation with a hard schema and a fallback already written underneath it. A
        # small model drops the JSON wrapper often enough that the fallback becomes the
        # normal path; a frontier model buys nothing here because there is no ambiguity to
        # resolve once the reading is in hand -- only a format to hit.
        response = llm_call(prompt,
                            mock_key=f"mdp_estimate_{percept.get('reading_id', 'default')}",
                            tier="mid")
        try:
            p = model_loads(response)
            return (p['x'], p['y'])
        except (json.JSONDecodeError, KeyError):
            return self.mdp.initial_state


# -- Path 2: LLM as the policy function ----------------------------------------------

class LLMPolicyAgent:
    """
    LLM approximates the optimal policy.
    pi(s) = a  becomes  llm_call(state) -> action
    Not provably optimal. Works on state spaces value iteration cannot.
    """
    def __init__(self, role: str, available_actions: list,
                 reward_description: str, gamma: float = 0.9,
                 fallback: str | None = None):
        self.role = role
        self.available_actions = available_actions
        self.reward_description = reward_description
        self.gamma = gamma
        self.episode_history = []
        # Not on the source page. See the comment in act() for why it is here.
        self.fallback = fallback

    def act(self, state: dict) -> str:
        prompt = f"""You are a {self.role} making sequential decisions.
State: {json.dumps(state)}
Actions: {self.available_actions}
Rewards: {self.reward_description}
Discount: {self.gamma}
Recent history: {self.episode_history[-5:]}

Which action maximizes expected total discounted reward?
Return just the action name."""

        # tier=mid: the output shape is one label from a list of four, which reads like a
        # small-model job, but the judgment behind it is not. The reward table is
        # asymmetric -- a false negative costs -5 and a false positive costs -1 -- so the
        # model has to weigh an unbalanced tradeoff over ambiguous prose. Small models
        # reliably return a well-formed label and get that tradeoff wrong.
        action = llm_call(prompt,
                          mock_key=f"mdp_moderate_{state.get('content_id', 'default')}",
                          tier="mid").strip()
        if action not in self.available_actions:
            # Source page bug. It falls back to available_actions[0], which for the
            # moderation example below is "approve" -- the single most expensive wrong
            # answer in its own reward table. The source's config.yaml declares
            # fallback: "flag_for_review", so the code and the config disagree. Deferring
            # to the config is the fix; passing no fallback keeps the source behaviour.
            action = self.fallback or self.available_actions[0]
        self.episode_history.append({'state': state, 'action': action})
        return action


# -- Path 3: LLM exploration with deterministic reward tracking ----------------------

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
        # tier=small: the model picks one label from a list of four. Every number it
        # would otherwise have to reason about -- the running reward estimates -- is
        # computed in observe_reward and handed over pre-formatted, so no arithmetic
        # happens inside the model. This is the cheapest call in the file and the one
        # most safely downgraded, because a wrong pick costs one noisy sample and the
        # Q update absorbs it.
        action = llm_call(prompt, mock_key=f"mdp_explore_{state}", tier="small").strip()
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


# -- Path 3 environment ---------------------------------------------------------------

ENV_STATES = ["latency_spiking", "queue_backlog_growing", "error_rate_normal",
              "memory_climbing"]

# The agent never sees this table. That is the premise of Path 3: P(s'|s,a) and R are
# not available to be written down, so there is nothing for value_iteration to consume.
HIDDEN_REWARD = {
    ("latency_spiking", "scale_up"): 1.0,
    ("latency_spiking", "restart_service"): -0.3,
    ("latency_spiking", "rollback"): 0.2,
    ("latency_spiking", "wait"): -0.9,
    ("queue_backlog_growing", "scale_up"): 1.3,
    ("queue_backlog_growing", "restart_service"): -0.6,
    ("queue_backlog_growing", "rollback"): -0.1,
    ("queue_backlog_growing", "wait"): -0.4,
    ("error_rate_normal", "wait"): 0.5,
    ("error_rate_normal", "scale_up"): -0.3,
    ("error_rate_normal", "restart_service"): -0.7,
    ("error_rate_normal", "rollback"): -0.5,
    ("memory_climbing", "restart_service"): 0.9,
    ("memory_climbing", "scale_up"): 0.1,
    ("memory_climbing", "rollback"): 0.3,
    ("memory_climbing", "wait"): -0.7,
}


def step(state: str, action: str) -> tuple[float, str]:
    """The environment. Stochastic, and its model is never handed to the agent."""
    reward = HIDDEN_REWARD.get((state, action), -0.5) + random.gauss(0, 0.05)
    return round(reward, 3), random.choice(ENV_STATES)


# -- demo -----------------------------------------------------------------------------

def banner(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def run_path_1() -> None:
    banner("PATH 1  LLMStateEstimator -- classical MDP, LLM only estimates the state")

    mdp = build_grid_world()
    V, policy = value_iteration(mdp)

    # Acceptance check: the policy this agent uses came out of before.py, not out of a
    # second implementation living in this file. `inspect.getfile` reports where the
    # function object actually lives, which no amount of copy-pasting can fake.
    solver_file = Path(inspect.getfile(value_iteration)).name
    reimplemented_here = solver_file == Path(__file__).name
    print(f"  policy source: {value_iteration.__module__}.value_iteration "
          f"defined in {solver_file}")
    print(f"  reimplemented in after.py: {reimplemented_here}")
    print()
    print("  pi* imported from before.py:")
    print_policy_grid(mdp, policy)
    print()

    estimator = LLMStateEstimator(mdp, policy)

    percepts = [
        {"reading_id": "r1", "beacon_rssi": [-41, -78, -80],
         "note": "strong signal from the south-west beacon"},
        {"reading_id": "r2", "beacon_rssi": [-77, -44, -62],
         "note": "close to the north-east corridor"},
        {"reading_id": "r3", "beacon_rssi": [-88, -87, -89],
         "note": "all beacons weak, reading is degraded"},
        {"reading_id": "r4", "beacon_rssi": [-70, -70, -45],
         "note": "east side, altitude sensor offline"},
        {"reading_id": "r5", "beacon_rssi": [-90, -39, -55],
         "note": "at the goal marker"},
        {"reading_id": "r6", "beacon_rssi": [-62, -61, -63],
         "note": "dead centre, possibly inside the obstacle"},
    ]

    # Demo narration only -- the agent never sees this. Each reading is chosen to drive
    # one branch of estimate_state and act.
    exercises = {
        "r1": "well-formed JSON, non-terminal state",
        "r2": "well-formed JSON, non-terminal state",
        "r3": "prose instead of JSON -> JSONDecodeError -> mdp.initial_state",
        "r4": "JSON missing the 'y' key -> KeyError -> mdp.initial_state",
        "r5": "the +1 terminal -> policy holds None -> 'no_op'",
        "r6": "the wall, which is not a state at all -> lookup misses -> 'no_op'",
    }

    for percept in percepts:
        rid = percept["reading_id"]
        # Called twice on purpose: once to show the state the LLM produced, once to show
        # the action the deterministic lookup produced from it.
        state = estimator.estimate_state(percept)
        action = estimator.act(percept)
        print(f"  {rid}: estimated {str(state):<7} -> action {action!r:<8} "
              f"[{exercises[rid]}]")

    print()
    print("  The LLM chose the state. It did not choose the action. Every action above")
    print("  is a dictionary lookup into a policy produced by the Bellman updates, so")
    print("  the optimality guarantee is intact for every percept that was read")
    print("  correctly -- and a misread percept degrades to a defined fallback, not to")
    print("  an arbitrary move.")


def run_path_2() -> None:
    banner("PATH 2  LLMPolicyAgent -- the LLM is the policy")

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
        gamma=0.95,
        # config.yaml: decision_making.fallback
        fallback="flag_for_review",
    )

    print("  There is no transition table here and there never will be. The state is a")
    print("  post plus an author history plus a flag set -- unbounded, unenumerable.")
    print("  value_iteration has nothing to consume, so the LLM approximates pi(s).")
    print()

    queue = [
        {"content_id": "c1", "text": "Our Q3 numbers are up 12 percent. Full deck inside.",
         "reports": 0, "account_age_days": 1420, "flags": []},
        {"content_id": "c2", "text": "[redacted -- explicit threat against a named person]",
         "reports": 14, "account_age_days": 2, "flags": ["violence", "targeted"]},
        {"content_id": "c3", "text": "This supplement cured my condition in four days.",
         "reports": 3, "account_age_days": 90, "flags": ["health_claim"]},
        {"content_id": "c4", "text": "Ambiguous sarcasm about a public figure, quoted.",
         "reports": 6, "account_age_days": 800, "flags": ["harassment?"]},
    ]

    for state in queue:
        action = agent.act(state)
        note = ""
        if state["content_id"] == "c4":
            note = "  <- model returned an action outside the actuator set; fallback fired"
        print(f"  {state['content_id']}: reports={state['reports']:>2} "
              f"flags={state['flags']} -> {action!r}{note}")

    print()
    print(f"  episode_history length: {len(agent.episode_history)}")
    print("  No optimality guarantee survives this path. What survives is the action")
    print("  set: the agent cannot emit an action the actuators do not implement,")
    print("  because the membership test is deterministic code, not a prompt.")
    print("  Source-page fix: its fallback is available_actions[0], which here is")
    print("  'approve' -- the -5 outcome. config.yaml says 'flag_for_review'. The")
    print("  config wins.")


def run_path_3() -> None:
    banner("PATH 3  LLMExplorationAgent -- LLM explores, code keeps the books")

    # act() draws from `random` for epsilon-greedy exploration and for the invalid-action
    # fallback, and the environment draws for reward noise and the next state. Seeding
    # once here makes the whole path reproducible.
    random.seed(13)

    agent = LLMExplorationAgent(
        role="incident response agent",
        available_actions=["restart_service", "scale_up", "rollback", "wait"],
    )

    print("  Transition probabilities are unknown. The agent finds out by acting.")
    print("  Seeded with random.seed(13), so this run reproduces exactly.")
    print("  Two branches bypass the model entirely: the 10 percent epsilon-greedy roll,")
    print("  and any response that is not one of the four actions. Both fire below.")
    print()

    state = "latency_spiking"
    for t in range(12):
        action = agent.act(state)
        reward, next_state = step(state, action)
        agent.observe_reward(state, action, reward, next_state)
        print(f"  t={t:>2}  {state:<22} {action:<16} reward={reward:>6.3f}")
        state = next_state

    print()
    print("  Q estimates after 12 steps, computed by observe_reward -- arithmetic only,")
    print("  no model involved:")
    for (s, a), q in sorted(agent.q_estimates.items(), key=lambda kv: -kv[1]):
        print(f"    {s:<22} {a:<16} {q:+.4f}")

    print()
    print(f"  reward_history entries: {len(agent.reward_history)}")
    print("  The LLM decided what to try. Every number above came from Python. Swap the")
    print("  model and the ledger is unchanged; swap the ledger for a model and there is")
    print("  no ledger.")


def main() -> None:
    run_path_1()
    run_path_2()
    run_path_3()
    print()
    print("Decision rule from the source page: small known MDP -> solve classically and")
    print("use the LLM only for state estimation. Large state space -> LLM approximates")
    print("the policy. Unknown transitions -> LLM explores, deterministic code tracks.")


if __name__ == "__main__":
    main()
