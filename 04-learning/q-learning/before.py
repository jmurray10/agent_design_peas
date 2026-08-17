"""Q-learning: the classical learning agent, all four components as arithmetic.

The four components of a learning agent are all here, and none of them is a model:

    Performance element   argmax over the Q-table          (get_best_action)
    Critic                the environment's reward signal  (the `reward` line below)
    Learning element      the Bellman update               (update)
    Problem generator     epsilon-greedy                   (get_action)

Standard library only. No API key, no network, no install.

    python before.py
"""

from __future__ import annotations

import random
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning. Model-free reinforcement learning.
    Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, states: list[str], actions: list[str], alpha: float = 0.1,
                 gamma: float = 0.9, epsilon: float = 0.1) -> None:
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    # -- PROBLEM GENERATOR: epsilon of the time, ignore what is known and go look --
    def get_action(self, state: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.get_best_action(state)

    # -- PERFORMANCE ELEMENT: greedy lookup, no search, no model --
    def get_best_action(self, state: str) -> str:
        best_a, best_v = None, float('-inf')
        for a in self.actions:
            v = self.Q[state][a]
            if v > best_v: best_v, best_a = v, a
        # `is not None` rather than the source's truth test: an action named "" or 0
        # would be discarded by a bare `if best_a` and replaced with a random pick,
        # which is a silent wrong answer rather than a crash. Noted in the README.
        return best_a if best_a is not None else random.choice(self.actions)

    # -- LEARNING ELEMENT: one Bellman step per experience --
    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        current_q = self.Q[state][action]
        max_next = max((self.Q[next_state][a] for a in self.actions), default=0)
        self.Q[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next - current_q
        )

    def get_policy(self) -> dict[str, str]:
        return {s: self.get_best_action(s) for s in self.states}


if __name__ == "__main__":
    # Seeded so the policy, the Q-values, and the convergence trace are identical on
    # every machine. Q-learning's convergence guarantee is about the limit, not about
    # any single run, and a run a reader cannot reproduce proves nothing.
    random.seed(42)

    # Training
    states = ['A', 'B', 'C', 'D', 'Goal']
    actions = ['left', 'right']
    agent = QLearningAgent(states, actions)

    # Hoisted out of the step loop, where the source rebuilds it 10,000 times. Same
    # contents. 'left' has no entry for 'A' and 'right' none for 'Goal', so those
    # moves fall through to the `state` default below and the agent stays put.
    next_map = {'right': {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'Goal'},
                'left': {'D': 'C', 'C': 'B', 'B': 'A'}}

    episode_returns: list[float] = []
    policy_history: list[tuple[str, ...]] = []

    print("Convergence trace, 50-episode blocks:")
    print("  episodes    avg return   Q(A,right)")
    for episode in range(500):
        state = 'A'
        episode_return = 0.0
        for step in range(20):
            if state == 'Goal': break
            action = agent.get_action(state)
            next_state = next_map.get(action, {}).get(state, state)
            reward = 10 if next_state == 'Goal' else -1
            agent.update(state, action, reward, next_state)
            state = next_state
            episode_return += reward
        episode_returns.append(episode_return)
        policy_history.append(tuple(agent.get_policy()[s] for s in states[:-1]))

        if (episode + 1) % 50 == 0:
            block = episode_returns[-50:]
            print(f"  {episode - 48:>3}-{episode + 1:<7} {sum(block) / len(block):9.2f} "
                  f"{agent.Q['A']['right']:12.2f}")

    # Two things converge here at very different speeds, and a return column on its
    # own hides that. On a four-state chain a single -1 is enough to push the agent
    # off a bad action, so the greedy policy is right almost immediately and the
    # returns plateau inside the first block. The Q-values are the slow part: alpha
    # moves each estimate a tenth of the way, and the +10 at the goal has to travel
    # backwards one state per visit. Ranking the actions correctly is cheap; knowing
    # what they are worth is not.
    final_policy = policy_history[-1]
    settled_at = max((i for i, p in enumerate(policy_history) if p != final_policy),
                     default=-1) + 2
    print(f"\nGreedy policy reached its final form at episode {settled_at} and held it "
          f"for the remaining {500 - settled_at + 1} episodes.")

    print("\nLearned policy:")
    for s in states[:-1]:
        print(f"  {s}: {agent.get_policy()[s]}  "
              f"(Q_left={agent.Q[s]['left']:.1f}, Q_right={agent.Q[s]['right']:.1f})")

    # The four components again, named, so the after.py comparison has something to
    # point at. Every one of them is arithmetic over observed numbers.
    print("\nComponents used, none of them a model:")
    print("  performance element  argmax Q(s,a)")
    print("  critic               reward from the environment: +10 at Goal, -1 per step")
    print("  learning element     Q(s,a) <- Q(s,a) + alpha * TD error")
    print(f"  problem generator    epsilon-greedy, epsilon={agent.epsilon}")
