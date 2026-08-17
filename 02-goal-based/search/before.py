"""Goal-based agent, classical: A* search on the 8-puzzle.

Source: reference/02-goal-based-agents-before-after.md, the A* section.

Standard library only. No API key, no pip install, no network. The point of this file is
that the planning is done by an algorithm with a proof attached: A* with an admissible
heuristic returns an optimal solution or reports that none exists. Nothing here guesses.
"""

from dataclasses import dataclass
from typing import Any, List, Optional
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
        # heapq compares the second element of a (priority, node) tuple whenever two
        # priorities tie, and f-value ties are the common case in the 8-puzzle. Without
        # this method every tie would raise TypeError.
        return self.path_cost < other.path_cost

    def solution(self):
        node, actions = self, []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))


def a_star_search(problem: SearchProblem, heuristic, stats: Optional[dict] = None) -> Optional[List]:
    """A* search. f(n) = g(n) + h(n). Optimal with admissible heuristic.

    `stats` is the one addition to the listing in the article. If a dict is passed it is
    filled with the search's own accounting -- nodes expanded, nodes generated, the cost
    of the returned path. Without a node count a reader has no way to see that A* did any
    work, and "the LLM could have just done this" becomes unfalsifiable.
    """
    node = SearchNode(problem.initial_state)
    frontier = [(0 + heuristic(node.state), node)]
    explored = set()
    expanded = 0
    generated = 1

    def record(cost):
        if stats is not None:
            stats.update(
                nodes_expanded=expanded,
                nodes_generated=generated,
                frontier_remaining=len(frontier),
                solution_cost=cost,
            )

    while frontier:
        _, node = heapq.heappop(frontier)
        if problem.goal_test(node.state):
            record(node.path_cost)
            return node.solution()
        # A state can be pushed onto the frontier several times before it is popped;
        # the first pop is the cheapest, so later ones are dropped here.
        if node.state in explored:
            continue
        explored.add(node.state)
        expanded += 1

        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored:
                child_cost = problem.path_cost(
                    node.path_cost, node.state, action, child_state
                )
                child = SearchNode(child_state, node, action, child_cost)
                priority = child.path_cost + heuristic(child.state)
                heapq.heappush(frontier, (priority, child))
                generated += 1

    record(None)
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


def render(state: tuple) -> str:
    """Display only. Three rows, a dot for the blank."""
    rows = []
    for r in range(3):
        rows.append(" ".join("." if t == 0 else str(t) for t in state[r*3:(r+1)*3]))
    return "\n".join("    " + row for row in rows)


if __name__ == "__main__":
    print("8-puzzle, A* with Manhattan distance")
    print()
    print("  initial state")
    print(render(initial))
    print()
    print("  goal state")
    print(render(goal))
    print()

    problem = SearchProblem(initial, goal_test, actions, result, lambda c, s, a, s2: c+1)
    stats: dict = {}
    solution = a_star_search(problem, manhattan_distance, stats)

    print(f"Solution in {len(solution)} moves: {solution}")
    print()
    print(f"  heuristic at start : {manhattan_distance(initial)} "
          f"(a lower bound on the true cost, which is why A* is optimal here)")
    print(f"  nodes expanded     : {stats['nodes_expanded']}")
    print(f"  nodes generated    : {stats['nodes_generated']}")
    print(f"  frontier left over : {stats['frontier_remaining']}")
    print(f"  path cost          : {stats['solution_cost']}")
    print()

    # Replaying the moves proves the returned list is a real plan and not a label.
    state = initial
    for action in solution:
        state = result(state, action)
    print(f"  replaying the moves reaches the goal: {state == goal}")
