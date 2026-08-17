# 02 - Goal-Based Agents: Before and After LLMs

---

## Architecture Overview

A goal-based agent has explicit goals and uses **search/planning** to find action sequences that reach them. Unlike reflex agents, it plans ahead -- considering sequences of actions that form a path to a goal state.

A search problem is defined by:
1. A set of states S
2. An initial state
3. Actions applicable per state
4. A transition model: Result(s, a) -> s'
5. A path cost function
6. A goal test

Search algorithms (BFS, DFS, A*, UCS) operate on this definition. A related structure is the **Constraint Satisfaction Problem** (CSP), where the goal is not a path but an assignment of values to variables that satisfies all constraints.

---

## Goal-Based Agent with A* Search

### BEFORE: Classical

The 8-puzzle. Fully observable, deterministic, discrete. A* with Manhattan distance heuristic -- optimal when the heuristic is admissible (never overestimates).

```python
from dataclasses import dataclass
from typing import List, Optional, Any
import heapq

@dataclass
class SearchProblem:
    initial_state: Any
    goal_test: callable
    actions: callable
    result: callable
    path_cost: callable

class SearchNode:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def solution(self):
        node, actions = self, []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

def a_star_search(problem: SearchProblem, heuristic) -> Optional[List]:
    """A* search. f(n) = g(n) + h(n). Optimal with admissible heuristic."""
    node = SearchNode(problem.initial_state)
    frontier = [(0 + heuristic(node.state), node)]
    explored = set()

    while frontier:
        _, node = heapq.heappop(frontier)
        if problem.goal_test(node.state):
            return node.solution()
        if node.state in explored:
            continue
        explored.add(node.state)

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored:
                child_cost = problem.path_cost(
                    node.path_cost, node.state, action, child_state
                )
                child = SearchNode(child_state, node, action, child_cost)
                priority = child.path_cost + heuristic(child.state)
                heapq.heappush(frontier, (priority, child))
    return None


# 8-puzzle
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def goal_test(state):
    return state == goal

def actions(state):
    moves = []
    blank = state.index(0)
    row, col = blank // 3, blank % 3
    if row > 0: moves.append('up')
    if row < 2: moves.append('down')
    if col > 0: moves.append('left')
    if col < 2: moves.append('right')
    return moves

def result(state, action):
    s = list(state)
    blank = s.index(0)
    row, col = blank // 3, blank % 3
    swap = {'up': (row-1)*3+col, 'down': (row+1)*3+col,
            'left': row*3+(col-1), 'right': row*3+(col+1)}
    idx = swap[action]
    s[blank], s[idx] = s[idx], s[blank]
    return tuple(s)

def manhattan_distance(state):
    distance = 0
    for i, tile in enumerate(state):
        if tile != 0:
            goal_idx = goal.index(tile)
            distance += abs(i//3 - goal_idx//3) + abs(i%3 - goal_idx%3)
    return distance

initial = (7, 2, 4, 5, 0, 6, 8, 3, 1)
problem = SearchProblem(initial, goal_test, actions, result, lambda c,s,a,s2: c+1)
solution = a_star_search(problem, manhattan_distance)
print(f"Solution in {len(solution)} moves: {solution}")
```

**Strengths:** guaranteed optimal with admissible heuristic. Well-understood complexity.

**Limitations:** requires formal definitions for every piece -- states, actions, transitions, goal test, heuristic. Feasible for puzzles and grid navigation. Not feasible for "help me plan a trip to Europe" where the state space cannot be formally specified.

---

### AFTER: LLM-Powered

The LLM handles what is hard to formalize (interpreting goals, parsing messy inputs). Classical search still does the actual planning when possible.

```python
import json

class LLMGoalBasedAgent:
    """
    Goal-based agent. Uses search when possible, LLM when not.
    LLM: interprets goal, parses state from unstructured input.
    Deterministic: search, validation, execution.
    """
    def __init__(self, role: str, available_actions: list):
        self.role = role
        self.available_actions = available_actions
        self.state = None
        self.goal = None

    def agent_function(self, percept: dict) -> str:
        if self.goal is None:
            self.goal = self.llm_interpret_goal(percept)

        self.state = self.llm_parse_state(percept)

        # Can we formalize as a search problem?
        if self.is_formalizable():
            problem = self.build_search_problem()
            solution = a_star_search(problem, self.get_heuristic())
            if solution:
                return solution[0]

        # Fallback: LLM plans directly
        return self.llm_plan_next_action()

    def llm_interpret_goal(self, percept: dict) -> dict:
        prompt = f"""You are a {self.role}.
The user said: {percept.get('user_request', '')}

What is the formal goal? Return JSON:
- "goal_state": target end state
- "constraints": solution constraints
- "optimize": what to minimize/maximize"""
        response = llm_call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"goal_state": percept.get('user_request', '')}

    def llm_parse_state(self, percept: dict) -> dict:
        prompt = f"""Parse this into structured state.
Input: {json.dumps(percept)}
Return valid JSON with relevant state variables."""
        response = llm_call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return percept

    def is_formalizable(self) -> bool:
        return (isinstance(self.state, dict) and isinstance(self.goal, dict)
                and 'goal_state' in self.goal and self._has_known_transitions())

    def _has_known_transitions(self) -> bool:
        return False  # override per domain

    def build_search_problem(self) -> SearchProblem:
        raise NotImplementedError

    def get_heuristic(self):
        raise NotImplementedError

    def llm_plan_next_action(self) -> str:
        prompt = f"""You are a {self.role}.
State: {json.dumps(self.state)}
Goal: {json.dumps(self.goal)}
Actions: {self.available_actions}
Best next action? Return just the action name."""
        action = llm_call(prompt).strip()
        if action not in self.available_actions:
            return self.available_actions[0]
        return action
```

### The Key Pattern

The LLM does not replace A*. It sets up the inputs A* needs:
- Initial state (LLM parses from natural language)
- Goal test (LLM interprets from ambiguous request)
- Heuristic (LLM can suggest one for novel domains)

Classical search runs deterministically and provides optimality guarantees the LLM alone never could. When the state space is too messy for formal search, the LLM falls back to direct planning -- but that path has no optimality guarantee.

### Config

```yaml
agent:
  name: "puzzle-solver"
  architecture: "goal-based"
  performance:
    metrics: ["solution optimality", "time to solve"]
    eval: "eval/test_cases.json"
  environment:
    type: "fully-observable, deterministic, episodic, discrete"
  actuators:
    - name: "up"
      output_schema: "schemas/action_move.json"
    - name: "down"
      output_schema: "schemas/action_move.json"
    - name: "left"
      output_schema: "schemas/action_move.json"
    - name: "right"
      output_schema: "schemas/action_move.json"
  sensors:
    - name: "puzzle_description"
      type: "text"
    - name: "puzzle_grid"
      type: "structured"
      input_schema: "schemas/percept_grid.json"
  prompts:
    system: "prompts/system.md"
    goal_interpretation: "prompts/interpret_goal.md"
  planning:
    strategy: "a_star"
    fallback: "llm_reasoning"
    heuristic: "manhattan_distance"
```

---

## Constraint Satisfaction Problems

### BEFORE: Classical

A CSP has variables, domains, and constraints. The solver uses backtracking search, optionally combined with constraint propagation (AC-3). The power of the approach: specify the problem declaratively and a general-purpose algorithm solves it.

```python
class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints  # (var1, var2, check_func)

    def is_consistent(self, var, value, assignment):
        for (v1, v2, check) in self.constraints:
            if v1 == var and v2 in assignment:
                if not check(value, assignment[v2]):
                    return False
            if v2 == var and v1 in assignment:
                if not check(assignment[v1], value):
                    return False
        return True

def backtracking_search(csp, assignment=None):
    if assignment is None:
        assignment = {}
    if len(assignment) == len(csp.variables):
        return assignment
    unassigned = [v for v in csp.variables if v not in assignment]
    var = unassigned[0]
    for value in csp.domains[var]:
        if csp.is_consistent(var, value, assignment):
            assignment[var] = value
            result = backtracking_search(csp, assignment)
            if result is not None:
                return result
            del assignment[var]
    return None

# Map coloring
variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
domains = {v: ['red', 'green', 'blue'] for v in variables}
constraints = [
    ('WA', 'NT', lambda a, b: a != b),
    ('WA', 'SA', lambda a, b: a != b),
    ('NT', 'SA', lambda a, b: a != b),
    ('NT', 'Q',  lambda a, b: a != b),
    ('SA', 'Q',  lambda a, b: a != b),
    ('SA', 'NSW', lambda a, b: a != b),
    ('SA', 'V',  lambda a, b: a != b),
    ('Q', 'NSW', lambda a, b: a != b),
    ('NSW', 'V', lambda a, b: a != b),
]
solution = backtracking_search(CSP(variables, domains, constraints))
print(f"Map coloring: {solution}")
```

**Strengths:** general-purpose, correct, complete.

**Limitations:** requires formal specification of variables, domains, and constraints. Real-world problems arrive in natural language.

---

### AFTER: LLM Extracts the CSP, Solver Solves It

The solver is identical. The LLM translates natural language into the formal CSP.

```python
import json

def llm_extract_csp(natural_language_input: str) -> CSP:
    prompt = f"""Extract a Constraint Satisfaction Problem from this request.
Request: "{natural_language_input}"
Return JSON:
- "variables": list of variable names
- "domains": dict of variable -> possible values
- "constraints": list of [var1, var2, "relationship"]
  relationship: "not_equal", "less_than", "not_same_time"
Return valid JSON only."""

    response = llm_call(prompt)
    parsed = json.loads(response)

    constraint_map = {
        'not_equal': lambda a, b: a != b,
        'less_than': lambda a, b: a < b,
        'not_same_time': lambda a, b: a != b,
    }
    constraints = []
    for v1, v2, rel in parsed['constraints']:
        constraints.append((v1, v2, constraint_map.get(rel, lambda a, b: a != b)))

    return CSP(parsed['variables'], parsed['domains'], constraints)

def llm_format_solution(solution: dict, request: str) -> str:
    prompt = f"""The user asked: "{request}"
Solution: {json.dumps(solution)}
Explain in plain language."""
    return llm_call(prompt)


request = """Schedule 3 meetings next week:
- Team standup: Alice and Bob, 30 min
- Design review: Bob and Carol, 1 hour
- Sprint planning: everyone, 1 hour
Nobody in two meetings at once.
Slots: Monday 9am, Monday 2pm, Tuesday 10am, Wednesday 9am"""

csp = llm_extract_csp(request)
solution = backtracking_search(csp)  # SAME solver, zero changes
answer = llm_format_solution(solution, request)
print(answer)
```

### Comparison

| Aspect | Before | After |
|--------|--------|-------|
| Solver | Backtracking + AC-3 | Backtracking + AC-3 (identical) |
| Problem spec | Hand-coded | LLM extracts from natural language |
| Output | Raw dict | LLM formats to plain language |
| Correctness | Complete + correct | Complete + correct (solver unchanged) |
| Handles NL input | No | Yes |

### Config

```yaml
agent:
  name: "scheduling-agent"
  architecture: "goal-based"
  performance:
    metrics: ["all constraints satisfied", "minimize schedule duration"]
    eval: "eval/scheduling_cases.json"
  environment:
    type: "fully-observable, deterministic, episodic, static, discrete"
  actuators:
    - name: "solve_csp"
      type: "deterministic"
      output_schema: "schemas/csp_solution.json"
    - name: "present_schedule"
      type: "output"
  sensors:
    - name: "scheduling_request"
      type: "text"
      preprocessing: "llm_extract_csp"
      output_schema: "schemas/csp_definition.json"   # schema for the extracted CSP
  prompts:
    system: "prompts/system.md"
    extract_csp: "prompts/extract_csp.md"             # prompt template for CSP extraction
    format_solution: "prompts/format_solution.md"     # prompt template for NL output
  behavior:
    decision_strategy: "parse request -> extract CSP -> solve -> present"
    solver: "backtracking_with_ac3"
    fallback: "llm_suggest_schedule"
```
