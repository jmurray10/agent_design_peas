"""Utility-based agent, classical: value iteration on the 4x3 grid world.

Standard library only. No API key, no network, no install.

The source page gives the `MDP` dataclass and `value_iteration` but never builds an
environment for them to solve, so this file adds the textbook 4x3 world: stochastic
moves (0.8 intended, 0.1 to each side), a wall at (1, 1), terminals at (3, 2) = +1 and
(3, 1) = -1, and a -0.04 living reward on every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple


@dataclass
class MDP:
    states: List[Any]
    actions: List[Any]
    transition_model: Dict
    reward_function: Dict
    gamma: float
    initial_state: Any
    terminal_states: Set[Any]

    def P(self, s_prime: Any, s: Any, a: Any) -> float:
        return self.transition_model.get((s, a, s_prime), 0.0)

    def R(self, s: Any, a: Any, s_prime: Any) -> float:
        return self.reward_function.get((s, a, s_prime), 0.0)

    def actions_available(self, state: Any) -> List[Any]:
        if self.terminal_states and state in self.terminal_states:
            return []
        return self.actions

    def successors(self, state: Any, action: Any) -> List[Tuple[Any, float]]:
        return [(s, self.P(s, state, action))
                for s in self.states if self.P(s, state, action) > 0]


def value_iteration(mdp: MDP, epsilon: float = 0.001,
                    max_iter: int = 1000) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
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
        if delta < epsilon:
            # Not in the source page. Printed because the iteration count is the whole
            # point of the "provably converges" claim -- a reader should see the number,
            # not take it on faith. after.py imports this function, so the line also
            # shows up there as proof the policy was solved classically.
            print(f"  value iteration converged after {i + 1} sweeps "
                  f"(delta={delta:.6f} < epsilon={epsilon})")
            break
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


# -- the 4x3 grid world --------------------------------------------------------------

WIDTH, HEIGHT = 4, 3
WALL = (1, 1)
GOAL = (3, 2)
PIT = (3, 1)
LIVING_REWARD = -0.04
TERMINAL_REWARD = {GOAL: 1.0, PIT: -1.0}

ACTIONS = ["up", "down", "left", "right"]
DELTA = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
PERPENDICULAR = {"up": ["left", "right"], "down": ["left", "right"],
                 "left": ["up", "down"], "right": ["up", "down"]}
ARROW = {"up": "^", "down": "v", "left": "<", "right": ">"}

INTENDED_PROB = 0.8
SIDE_PROB = 0.1


def move(state: Tuple[int, int], action: str) -> Tuple[int, int]:
    """Where a noiseless `action` lands. A move into the wall or off the grid stays put."""
    dx, dy = DELTA[action]
    candidate = (state[0] + dx, state[1] + dy)
    if candidate == WALL:
        return state
    if not (0 <= candidate[0] < WIDTH and 0 <= candidate[1] < HEIGHT):
        return state
    return candidate


def build_grid_world(gamma: float = 1.0) -> MDP:
    """Assemble the full transition and reward tables the MDP dataclass expects.

    Every P(s'|s,a) and every R(s,a,s') has to be written down before value iteration
    can run. That requirement is the classical algorithm's real cost, and building the
    tables by hand is the honest way to show it.
    """
    states = [(x, y) for x in range(WIDTH) for y in range(HEIGHT) if (x, y) != WALL]
    terminal_states = {GOAL, PIT}
    transition_model: Dict[Any, float] = {}
    reward_function: Dict[Any, float] = {}

    for s in states:
        if s in terminal_states:
            continue
        for a in ACTIONS:
            outcomes = [(move(s, a), INTENDED_PROB)]
            outcomes += [(move(s, side), SIDE_PROB) for side in PERPENDICULAR[a]]
            for s_prime, p in outcomes:
                key = (s, a, s_prime)
                # Two outcomes can land on the same square -- bouncing off a wall in two
                # different directions, for instance -- so probabilities accumulate.
                transition_model[key] = transition_model.get(key, 0.0) + p
                # Reward sits on the transition, not the state: every step pays the
                # living reward, and entering a terminal pays its payoff on top. Written
                # this way the terminals keep V = 0 (value_iteration never updates them)
                # while every non-terminal value still matches the textbook figure.
                reward_function[key] = LIVING_REWARD + TERMINAL_REWARD.get(s_prime, 0.0)

    return MDP(
        states=states,
        actions=ACTIONS,
        transition_model=transition_model,
        reward_function=reward_function,
        gamma=gamma,
        initial_state=(0, 0),
        terminal_states=terminal_states,
    )


# -- display -------------------------------------------------------------------------

def _grid(mdp: MDP, cell) -> None:
    """Draw the world with y increasing upward, the way the textbook figure is laid out."""
    for y in reversed(range(HEIGHT)):
        row = "".join(cell((x, y)).rjust(9) for x in range(WIDTH))
        print(f"  y={y} |{row}")
    print("        " + "-" * (9 * WIDTH))
    print("        " + "".join(f"x={x}".rjust(9) for x in range(WIDTH)))


def print_value_grid(mdp: MDP, V: Dict[Any, float]) -> None:
    """Print V*(s) as a grid."""
    def cell(s):
        if s == WALL:
            return "####"
        if s in mdp.terminal_states:
            return f"{TERMINAL_REWARD[s]:+.2f}"
        return f"{V[s]:.3f}"
    _grid(mdp, cell)
    print("  terminal cells show their entry reward; V is 0 there because no further")
    print("  reward accrues once the agent has arrived.")


def print_policy_grid(mdp: MDP, policy: Dict[Any, Any]) -> None:
    """Print pi*(s) as arrows."""
    def cell(s):
        if s == WALL:
            return "####"
        if s in mdp.terminal_states:
            return f"{TERMINAL_REWARD[s]:+.0f}"
        return ARROW[policy[s]]
    _grid(mdp, cell)


def main() -> None:
    mdp = build_grid_world()
    print("4x3 grid world")
    print(f"  states: {len(mdp.states)} (wall at {WALL} is not a state)")
    print(f"  actions: {mdp.actions}")
    print(f"  transitions specified: {len(mdp.transition_model)}")
    print(f"  gamma: {mdp.gamma}   living reward: {LIVING_REWARD}")
    print(f"  move noise: {INTENDED_PROB} intended, {SIDE_PROB} each perpendicular")
    print()

    V, policy = value_iteration(mdp)
    print()

    print("Value function V*")
    print_value_grid(mdp, V)
    print()

    print("Optimal policy pi*")
    print_policy_grid(mdp, policy)
    print()

    print(f"V* at the start state {mdp.initial_state}: {V[mdp.initial_state]:.3f}")
    print(f"pi* at the start state {mdp.initial_state}: {policy[mdp.initial_state]}")
    print("Note (2,0): the policy steers left, away from the -1 pit, rather than taking")
    print("the shorter route up the right-hand column. That is the stochastic move noise")
    print("being priced in, not a bug.")


if __name__ == "__main__":
    main()
