"""Goal-based agent, LLM-powered: the same A* from before.py, fed by a language model.

Source: reference/02-goal-based-agents-before-after.md, the AFTER section.

The search machinery is imported from before.py, not restated. Nothing about A* changes
here. What changes is where its inputs come from: a person types prose, the LLM turns that
prose into an initial state and a goal, and deterministic code decides whether the result
is solid enough to hand to a search algorithm.

Two requests run below. One formalizes and goes to A*. One does not and falls back to the
LLM planning directly. The difference between those two paths is the whole point.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Run from the repo root as "python 02-goal-based/search/after.py". Python puts the
# SCRIPT's directory on sys.path, not the repo root, so both entries are needed:
# parents[2] reaches shared/, and the script's own directory reaches before.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.llm import llm_call
from shared.model_json import loads as model_loads
from before import (
    SearchProblem,
    a_star_search,
    actions,
    goal,
    goal_test,
    initial,
    manhattan_distance,
    render,
    result,
)


# Key names that tend to carry the start board and the goal board in a model response.
# Hints for traversal order only -- the permutation check still decides what is legal.
STATE_KEYS = ("initial", "current", "start", "tiles", "board")
GOAL_KEYS = ("goal", "target", "final", "solved")


class LLMGoalBasedAgent:
    """
    Goal-based agent. Uses search when possible, LLM when not.
    LLM: interprets goal, parses state from unstructured input.
    Deterministic: search, validation, execution.
    """
    def __init__(self, role: str, available_actions: list, mock_key_prefix: str = "search_puzzle"):
        self.role = role
        self.available_actions = available_actions
        self.state = None
        self.goal = None
        # Only mock mode reads mock_key_prefix; it lets one class drive two different demo
        # requests without either of them seeing the other's canned responses. Every real
        # backend ignores mock_key entirely.
        self.mock_key_prefix = mock_key_prefix
        # agent_function returns a single action, which is the correct agent contract but
        # hides the plan A* actually produced. These keep the last one around so the demo
        # below can print it and compare it against before.py.
        self.last_solution: Optional[list] = None
        self.last_stats: dict = {}

    def agent_function(self, percept: dict) -> str:
        if self.goal is None:
            self.goal = self.llm_interpret_goal(percept)

        self.state = self.llm_parse_state(percept)

        # Can we formalize as a search problem?
        formalizable = self.is_formalizable()  # held in a local only so both branches can print it
        print(f"  is_formalizable()      -> {formalizable}")
        if formalizable:
            print("  PATH TAKEN: A* SEARCH    -- deterministic, optimal, no LLM in the loop")
            problem = self.build_search_problem()
            self.last_stats = {}
            solution = a_star_search(problem, self.get_heuristic(), self.last_stats)
            self.last_solution = solution
            if solution:
                return solution[0]
            print("  A* proved no plan exists from this state; falling through to the LLM")

        # Fallback: LLM plans directly
        print("  PATH TAKEN: LLM FALLBACK -- one plausible action, no optimality guarantee")
        return self.llm_plan_next_action()

    def llm_interpret_goal(self, percept: dict) -> dict:
        prompt = f"""You are a {self.role}.
The user said: {percept.get('user_request', '')}

What is the formal goal? Return JSON:
- "goal_state": target end state
- "constraints": solution constraints
- "optimize": what to minimize/maximize"""
        # frontier: turning ambiguous prose into a formal goal is the hardest judgment in
        # this file. A wrong goal does not produce a visible error -- A* still returns a
        # provably optimal plan, for the wrong problem, which is worse than no plan.
        response = llm_call(prompt, mock_key=f"{self.mock_key_prefix}_goal", tier="frontier")
        try:
            parsed = model_loads(response)
            print(f"  llm_interpret_goal()   -> {json.dumps(parsed)}")
            return parsed
        except json.JSONDecodeError:
            print("  llm_interpret_goal()   -> not JSON; falling back to the raw request as the goal")
            return {"goal_state": percept.get('user_request', '')}

    def llm_parse_state(self, percept: dict) -> dict:
        prompt = f"""Parse this into structured state.
Input: {json.dumps(percept)}
Return valid JSON with relevant state variables."""
        # mid: filling a fixed schema from prose is bounded extraction, not interpretation,
        # and the json.loads below plus the permutation check in _puzzle_tuple catch a bad
        # answer. A frontier model would buy accuracy the validation already guarantees.
        response = llm_call(prompt, mock_key=f"{self.mock_key_prefix}_state", tier="mid")
        try:
            parsed = model_loads(response)
            print(f"  llm_parse_state()      -> {json.dumps(parsed)}")
            return parsed
        except json.JSONDecodeError:
            print("  llm_parse_state()      -> not JSON; falling back to the raw percept")
            return percept

    def is_formalizable(self) -> bool:
        return (isinstance(self.state, dict) and isinstance(self.goal, dict)
                and 'goal_state' in self.goal and self._has_known_transitions())

    def _has_known_transitions(self) -> bool:
        """The 8-puzzle override of the source's `return False  # override per domain`.

        This is the gate that decides which path the agent takes, and it is deliberately
        not a judgment call. A transition model exists only if nine integers came back and
        they form a real permutation of 0..8 -- not because the model used the word
        "puzzle" anywhere in its answer.
        """
        tiles = self._find_puzzle_tuple(self.state, STATE_KEYS)
        target = self._find_puzzle_tuple(self.goal, GOAL_KEYS)
        if tiles is None:
            print("  no transition model: parsed state holds no valid 3x3 tile permutation")
        elif target is None:
            print("  no transition model: goal_state is not a valid 3x3 tile permutation")
        return tiles is not None and target is not None

    def build_search_problem(self) -> SearchProblem:
        """Assemble the six-part search problem from LLM-supplied endpoints.

        The endpoints come from the model. Everything else -- the action set, the
        transition model, the unit cost function -- is imported unchanged from before.py.
        """
        initial_state = self._find_puzzle_tuple(self.state, STATE_KEYS)
        goal_state = self._find_puzzle_tuple(self.goal, GOAL_KEYS)
        return SearchProblem(
            initial_state,
            lambda s: s == goal_state,
            actions,                          # from before.py, unchanged
            result,                           # from before.py, unchanged
            lambda c, s, a, s2: c + 1,
        )

    def get_heuristic(self):
        """Manhattan distance measured against the goal the LLM returned.

        before.py's `manhattan_distance` closes over its module-level `goal`. Here the goal
        arrives at runtime, so the identical formula is computed against that tuple. Still
        admissible: one move relocates one tile by one square, so the sum of per-tile
        distances can never exceed the true remaining cost, which is what keeps A* optimal.
        """
        goal_state = self._find_puzzle_tuple(self.goal, GOAL_KEYS)
        position = {tile: i for i, tile in enumerate(goal_state)}

        def heuristic(state) -> int:
            distance = 0
            for i, tile in enumerate(state):
                if tile != 0:
                    goal_idx = position[tile]
                    distance += abs(i//3 - goal_idx//3) + abs(i % 3 - goal_idx % 3)
            return distance

        return heuristic

    def llm_plan_next_action(self) -> str:
        prompt = f"""You are a {self.role}.
State: {json.dumps(self.state)}
Goal: {json.dumps(self.goal)}
Actions: {self.available_actions}
Best next action? Return just the action name."""
        # small: pick one label from a short list. The membership test two lines down is
        # the real safety property here, so paying for a stronger model would buy politeness
        # rather than correctness -- a wrong label is rejected either way.
        action = llm_call(prompt, mock_key=f"{self.mock_key_prefix}_action", tier="small").strip()
        if action not in self.available_actions:
            print(f"  rejected {action!r}: not in the action vocabulary; "
                  f"using {self.available_actions[0]!r}")
            return self.available_actions[0]
        return action

    @staticmethod
    def _puzzle_tuple(value) -> Optional[tuple]:
        """Validate an LLM-supplied grid. Returns a 9-tuple or None.

        Only an exact permutation of 0..8 gets through. Prose, a short list, a repeated
        tile, a string of digits -- all return None, and the agent takes the fallback path
        rather than handing A* a state space whose transition model does not apply.
        """
        if not isinstance(value, (list, tuple)) or len(value) != 9:
            return None
        # bool is a subclass of int, so True would otherwise pass as the tile 1.
        if not all(isinstance(t, int) and not isinstance(t, bool) for t in value):
            return None
        if sorted(value) != list(range(9)):
            return None
        return tuple(value)

    @classmethod
    def _find_puzzle_tuple(cls, value, prefer: tuple = ()) -> Optional[tuple]:
        """Find the one 9-tile permutation inside a parsed model response.

        `_puzzle_tuple` above decides whether a candidate is a legal board. This decides
        where to look for candidates, and it exists because asking a model for "structured
        state" and expecting a particular key is a bet that does not pay. Against
        claude-opus-5 the board came back as `initial_state.grid` shaped `[[7,2,4],...]`,
        having been asked for nothing more specific than valid JSON. Reading only
        `state["tiles"]` found nothing, `is_formalizable` returned False, and A* -- the
        entire subject of this example -- never ran outside of mock mode.

        Normalizing a correct answer that arrived in an unexpected shape is deterministic
        work, and it belongs on this side of the boundary. What is emphatically not
        relaxed is the check itself: a candidate still has to be an exact permutation of
        0..8, so a hallucinated board is rejected here exactly as before.

        `prefer` biases the traversal by key name, because a single response often carries
        both the start board and the goal board and picking the wrong one would hand A* a
        solved puzzle.
        """
        for candidate in cls._candidate_grids(value, prefer):
            tiles = cls._puzzle_tuple(candidate)
            if tiles is not None:
                return tiles
        return None

    @classmethod
    def _candidate_grids(cls, value, prefer: tuple):
        """Yield every list nested anywhere in `value`, preferred keys visited first."""
        if isinstance(value, dict):
            ordered = sorted(
                value,
                key=lambda k: (0 if any(p in str(k).lower() for p in prefer) else 1, str(k)),
            )
            for key in ordered:
                yield from cls._candidate_grids(value[key], prefer)
        elif isinstance(value, list):
            for candidate in (value,
                              # A 3x3 nested grid flattens to the same nine tiles in
                              # reading order.
                              [tile for row in value if isinstance(row, list) for tile in row]):
                yield candidate
                yield cls._blank_as_zero(candidate)
            for item in value:
                yield from cls._candidate_grids(item, prefer)

    @staticmethod
    def _blank_as_zero(candidate: list) -> list:
        """Rewrite a null or empty-string blank as 0, this puzzle's encoding for it.

        `0` means "the empty square" to before.py and to nothing else. The prompt asks for
        structured state and never says which integer stands for a hole, so a model is
        free to write `null` -- and claude-sonnet-5 did, returning
        `[[7, 2, 4], [5, null, 6], [8, 3, 1]]`, which is the board, correctly, in a
        notation this code had not been told to expect. The permutation check then
        rejected it for containing a non-integer and the agent fell through to the
        planning path, so the offline run of this example never reached A* at all.

        Translating a blank into the local encoding is deterministic work and belongs
        here. The check itself is untouched: the result still has to be an exact
        permutation of 0..8, so this admits a different spelling of a valid board and not
        a wider class of board.
        """
        return [0 if tile is None or tile == "" else tile for tile in candidate]


PUZZLE_REQUEST = (
    "The kids scrambled the sliding tile puzzle again. Top row reads 7, 2, 4. "
    "Middle row is 5, then the empty square, then 6. Bottom row is 8, 3, 1. "
    "Get it back to 1 through 8 in order with the empty square bottom right, "
    "in as few moves as possible."
)

TRAVEL_REQUEST = (
    "Help me plan two weeks in Europe in September. I would rather take trains than "
    "planes, one stop should be somewhere my partner has never been, and I do not want "
    "to spend much more than four thousand dollars."
)


def run_request(agent: LLMGoalBasedAgent, request: str, header: str) -> str:
    """Send one natural-language request through agent_function and narrate the result."""
    print("=" * 78)
    print(header)
    print("=" * 78)
    print(f'user: "{request}"')
    print()

    action = agent.agent_function({"user_request": request})

    if agent.last_solution:
        print()
        print(f"  A* returned a complete plan, {len(agent.last_solution)} moves:")
        print(f"    {agent.last_solution}")
        print(f"    nodes expanded   : {agent.last_stats['nodes_expanded']}")
        print(f"    nodes generated  : {agent.last_stats['nodes_generated']}")
        print(f"    path cost        : {agent.last_stats['solution_cost']}")
    print()
    print(f"  action handed to the actuators: {action!r}")
    print()
    return action


if __name__ == "__main__":
    puzzle_agent = LLMGoalBasedAgent(
        role="sliding tile puzzle solver",
        available_actions=["up", "down", "left", "right"],
        mock_key_prefix="search_puzzle",
    )
    travel_agent = LLMGoalBasedAgent(
        role="travel planner",
        available_actions=["research_destinations", "compare_rail_routes",
                           "draft_itinerary", "book_lodging"],
        mock_key_prefix="search_travel",
    )

    run_request(puzzle_agent, PUZZLE_REQUEST,
                "REQUEST 1 of 2  --  formalizable: a state space with a transition model")
    run_request(travel_agent, TRAVEL_REQUEST,
                "REQUEST 2 of 2  --  not formalizable: no state space to search")

    # The claim on the tin: the LLM changed how A* got its inputs, not what A* returned.
    # before.py's problem is rebuilt here from before.py's own module-level definitions and
    # solved in this process, then compared move for move against what the agent produced.
    print("=" * 78)
    print("VERIFY  --  path 1 returned exactly what before.py returns")
    print("=" * 78)

    reference_stats: dict = {}
    reference_problem = SearchProblem(initial, goal_test, actions, result,
                                      lambda c, s, a, s2: c + 1)
    reference_solution = a_star_search(reference_problem, manhattan_distance, reference_stats)
    # Same search the agent itself uses. This line read state["tiles"] directly until a
    # live model answered with the board under `initial_state`, nested as three rows --
    # at which point the agent found the board, ran A*, returned the right plan, and
    # this check reported "same starting state: False" about a state it had looked for
    # in the wrong place. A verification that does not use the same lookup as the thing
    # it verifies is checking something else.
    agent_initial = LLMGoalBasedAgent._find_puzzle_tuple(puzzle_agent.state, STATE_KEYS)

    print(f"  before.py initial state : {initial}")
    print(f"  after.py  initial state : {agent_initial}   (parsed from prose by the LLM)")
    print(f"  same starting state     : {agent_initial == initial}")
    print()

    if puzzle_agent.last_solution is None:
        # Reached only if request 1 never made it down the search path -- an edited
        # request, a model that returned a grid the validator rejected. Saying so beats
        # crashing on a comparison there is nothing to compare.
        print("  request 1 did not take the search path, so there is nothing to compare")
    else:
        print(f"  before.py : {len(reference_solution)} moves, "
              f"{reference_stats['nodes_expanded']} nodes expanded")
        print(f"  after.py  : {len(puzzle_agent.last_solution)} moves, "
              f"{puzzle_agent.last_stats['nodes_expanded']} nodes expanded")
        print(f"  move lists identical    : {puzzle_agent.last_solution == reference_solution}")
        print(f"    {reference_solution}")
        print()

        # Replaying the agent's plan on before.py's transition model, from before.py's
        # initial state, has to land on before.py's goal. Anything less is a plan on
        # paper only.
        state = initial
        for move in puzzle_agent.last_solution:
            state = result(state, move)
        print("  replaying the agent's plan from before.py's initial state:")
        print(render(state))
        print(f"  reaches before.py's goal : {state == goal}")
