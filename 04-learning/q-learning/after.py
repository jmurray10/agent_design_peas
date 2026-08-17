"""The same four-component learning agent with three components swapped for a model.

    Performance element   Q-table lookup      ->  LLM, experience in context
    Critic                environment reward  ->  UNCHANGED. Deterministic arithmetic.
    Learning element      Bellman update      ->  sort experience + LLM pattern extraction
    Problem generator     epsilon-greedy      ->  LLM targets gaps in experience

Three of the four moved. The critic did not, and the block around _calculate_reward
below says why at length. That asymmetry is the whole example.

Runs with no API key:

    python after.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# after.py is run from the repo root ("python 04-learning/q-learning/after.py"), which
# puts this file's own directory on sys.path but not the root. parents[2] is the root:
# parents[0] = q-learning, parents[1] = 04-learning, parents[2] = repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.llm import llm_call  # noqa: E402


def _extract_json_array(text: str) -> str:
    """Return the outermost bracketed span of `text`, or `text` itself if there is none.

    The source page called json.loads straight on the model response and swallowed the
    JSONDecodeError with a bare `pass`. In a learning agent that is worse than it looks:
    a fenced or prose-wrapped reply is the single most common thing a model returns, and
    swallowing it silently discards the entire cycle's learning with no error, no log,
    and an agent that reports it has learned. Slicing to the outermost brackets turns
    the common case back into a parse. See the README.
    """
    start, end = text.find('['), text.rfind(']')
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start:end + 1]


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

    # ==================================================================================
    # COMPONENT 1 of 4 -- PERFORMANCE ELEMENT (LLM)
    # Picks the action. Was argmax over a Q-table; is now a model reading the same
    # statistics the Q-table held, plus rules and examples the learning element wrote.
    # ==================================================================================
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

        # tier=mid: this is not "pick one label from a short list", which would be small.
        # The list is five items long but the decision is conditioned on running reward
        # averages per action, three worked/three failed examples, and the rules from the
        # last learn() cycle. Weighing supplied evidence to produce one bounded choice is
        # the mid tier's job. small would read the action names and ignore the context
        # underneath them, which is the whole input. frontier buys open-ended reasoning
        # that a five-way choice against explicit statistics does not need.
        #
        # mock_key tracks whether the learning element has produced rules yet, so the
        # canned response can show the performance element consuming what the learning
        # element wrote. Every real backend ignores mock_key.
        mock_key = "qlearn_act_learned" if self.learned_rules else "qlearn_act_cold"
        action = llm_call(prompt, mock_key=mock_key, tier="mid").strip()

        # Deterministic guard. The actuator list is a contract, not a suggestion, and
        # the model does not get to widen it by returning something not on it.
        if action not in self.available_actions:
            print(f"  [performance element] model returned {action[:40]!r}, not an "
                  f"available action; falling back to {self.available_actions[0]!r}")
            action = self.available_actions[0]
        return action

    # ==================================================================================
    # ==================================================================================
    ##
    ##   COMPONENT 2 of 4 -- CRITIC.  DETERMINISTIC.  NEVER AN LLM.  NOT EVER.
    ##
    ##   This is the one component in the architecture that must not be a model call,
    ##   and it is the reason this example exists.
    ##
    ##   The critic produces the reward. The reward is the only ground truth in the
    ##   loop: the learning element sorts on it, rewrites the agent's context from it,
    ##   and the performance element then acts on that context. Put a model here and
    ##   the agent is scoring its own work with the same faculty that produced it, and
    ##   the error has nowhere to go but around again -- generous score, wrong rule,
    ##   worse action, generous score. Nothing outside the loop is left to contradict
    ##   it, and the failure is silent, because a drifting agent with a drifting critic
    ##   reports improving numbers the whole way down.
    ##
    ##   Everything between this banner and the next one is arithmetic over facts that
    ##   were observed rather than judged: did it succeed, how many seconds, was there
    ##   an error, was the customer satisfied. No llm_call appears anywhere in it. One
    ##   should never be added. If a reward genuinely cannot be computed, that is a
    ##   signal to go find a measurable proxy, not a license to ask a model.
    ##
    # ==================================================================================
    # ==================================================================================
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
    # ==================================================================================
    # END OF THE CRITIC. Everything below this line may call a model.
    # ==================================================================================

    # ==================================================================================
    # COMPONENT 3 of 4 -- LEARNING ELEMENT (deterministic sorting + LLM extraction)
    # Was one Bellman update per experience. Is now: sort the log by the critic's
    # reward (deterministic), then ask a model what the extremes have in common. The
    # sort is what keeps this honest -- the model is handed the best and worst runs as
    # measured, not asked which runs it thinks went well.
    # ==================================================================================
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

        # tier=frontier: synthesis across examples into a formal output. The model has
        # to read six full experience records, find what separates the top three from
        # the bottom three, and state it as rules general enough to apply to a case it
        # has not seen -- then emit them as a JSON array. That is the frontier tier's
        # description almost word for word. It is also the highest-leverage call in the
        # agent: these rules go into every act() prompt afterwards, so a shallow answer
        # here degrades every decision the agent makes for the rest of the run. Ten
        # frontier calls per hundred interactions is a cheap place to spend.
        mock_key = "qlearn_learn_refined" if self.learned_rules else "qlearn_learn_first"
        response = llm_call(prompt, mock_key=mock_key, tier="frontier")

        payload = _extract_json_array(response)
        if payload != response.strip():
            print("  [learning element] response was not bare JSON; "
                  "recovered the array from it")
        try:
            rules = json.loads(payload)
        except json.JSONDecodeError:
            rules = None

        # Validation, not decoration. learned_rules is pasted into every subsequent
        # act() prompt, so a dict, a nested list, or a list of objects here would
        # quietly corrupt every later decision. Anything that is not a list of strings
        # is refused and the previous rules stand.
        if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
            print(f"  [learning element] fallback: unusable pattern output, keeping the "
                  f"{len(self.learned_rules)} rule(s) already held")
            return
        self.learned_rules = rules

    # ==================================================================================
    # COMPONENT 4 of 4 -- PROBLEM GENERATOR (LLM)
    # Was epsilon-greedy: with probability epsilon, do something random. Is now a model
    # asked to name the gap in the agent's own experience. The deterministic fallback
    # underneath it is the epsilon-greedy idea reduced to its useful core -- pick the
    # action with the least data behind it.
    # ==================================================================================
    def suggest_exploration(self, state: dict) -> str:
        prompt = f"""Suggest an experiment for a {self.role}.
Situation: {json.dumps(state)}
Actions: {self.available_actions}
Usage counts: {json.dumps({a: len(r) for a, r in self.action_reward_stats.items()})}
Suggest an underexplored action. Return just the action name."""

        # tier=mid: same five-way choice as act(), but against a different question --
        # which action is least understood, rather than which is best. That is a read of
        # the usage counts plus a judgment about which gap is worth paying for, so it is
        # bounded judgment again rather than a label pick, and mid is the floor. It is
        # deliberately not frontier: the deterministic fallback below is a perfectly
        # good answer, so the model is only ever buying an improvement on it, never a
        # result the agent could not otherwise reach.
        action = llm_call(prompt, mock_key="qlearn_explore", tier="mid").strip()

        if action not in self.available_actions:
            counts = {a: len(self.action_reward_stats[a]) for a in self.available_actions}
            fallback = min(counts, key=counts.get)
            print(f"  [problem generator] model returned {action[:40]!r}..., not an "
                  f"available action; deterministic fallback picked the least-used, "
                  f"{fallback!r}")
            action = fallback
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


if __name__ == "__main__":
    # Seeded so the exploration schedule and the simulated outcomes are the same on
    # every machine. The canned mock responses are fixed strings, so with the seed
    # pinned the entire run is reproducible.
    random.seed(7)

    # Usage
    agent = LLMLearningAgent(
        role="email response agent",
        available_actions=["send_template", "write_custom", "escalate", "request_info",
                           "auto_resolve"],
        performance_metrics=["quality", "time_to_resolution", "satisfaction"]
    )

    print("20 interactions. learn() fires every 10.\n")
    for i in range(20):
        state = {'type': 'complaint', 'urgency': 'medium', 'tier': 'premium'}

        if random.random() < 0.15:
            source = "problem generator"
            action = agent.suggest_exploration(state)
        else:
            source = "performance element"
            action = agent.act(state)

        # The environment, simulated. 'error' is not in the source's usage block, which
        # means the -2.0 branch of the critic never fires there; without it a reader
        # sees three quarters of the reward function and has to take the fourth on
        # faith. Noted in the README.
        outcome = {
            'success': random.random() > 0.3,
            'time_seconds': random.randint(5, 120),
            'customer_satisfied': random.random() > 0.4,
            'error': 'timeout' if random.random() < 0.15 else None,
        }

        reward = agent.observe_outcome(state, action, outcome)
        print(f"  {i + 1:>2} [{source:<19}] {action:<14} -> reward {reward:5.2f}")

        if (i + 1) % 10 == 0:
            agent.learn()
            print(f"\nAfter {i+1} interactions:")
            print(f"  Avg reward: {sum(agent.performance_scores[-10:])/10:.2f}")
            print("  Rules:")
            for rule in agent.learned_rules:
                print(f"    - {rule}")
            print()

    # This repo's whole argument is that the code should be checkable, which cuts both
    # ways: the two average-reward numbers above sit right next to two rounds of
    # extracted rules and invite exactly one reading, and that reading is wrong.
    print("What the numbers above are and are not:")
    print("  Each outcome is drawn at random and does NOT depend on the action chosen,")
    print("  so the change in average reward between the two cycles is noise. It is not")
    print("  evidence that the agent learned anything. Showing that this loop improves")
    print("  behavior needs an environment where the action changes the outcome and a")
    print("  ground truth the agent never sees -- a separate experiment, not this demo.")
    print("  What this demo does show is that the architecture runs: all four components")
    print("  execute, and the deterministic ones catch the model when it goes off-list.")
    print()

    # The component table, printed, so the file's central claim is in its output and
    # not only in its comments.
    print("Components, and what each one ran on:")
    print("  performance element  LLM, tier=mid       experience and stats in context")
    print("  critic               no model at all     arithmetic over observed outcomes")
    print("  learning element     LLM, tier=frontier  pattern extraction over sorted log")
    print("  problem generator    LLM, tier=mid       deterministic fallback underneath")
