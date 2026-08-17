"""Adversarial search: minimax with alpha-beta pruning, tic-tac-toe, full depth.

Classical competitive multi-agent search. Two agents, opposed performance measures,
perfect information. Minimax computes the optimal strategy assuming the opponent also
plays optimally; alpha-beta pruning cuts branches that provably cannot change the
answer.

The point of this file is the node count. Plain minimax and alpha-beta are written as
two separate functions so the comparison is honest -- same game, same starting position,
same depth, same move ordering. They return the same move and the same value. Alpha-beta
just refuses to look at most of the tree.

Standard library only. No API key, no network.
"""

from __future__ import annotations

import time

# Nodes visited, by search. Module-level so both search functions keep exactly the
# signatures the article shows -- a node-count comparison between two functions the
# reader has not seen before would not be worth much.
NODES = {"minimax": 0, "alpha_beta": 0}


def minimax(
    state: tuple[str, ...],
    depth: int,
    is_maximizing: bool,
    game: "TicTacToe",
) -> tuple[float, int | None]:
    """Plain minimax. Identical to alpha_beta below with the two cutoffs removed."""
    NODES["minimax"] += 1

    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state), None

    if is_maximizing:
        best_val, best_act = float('-inf'), None
        for action in game.get_actions(state):
            val, _ = minimax(game.result(state, action), depth-1, False, game)
            if val > best_val: best_val, best_act = val, action
        return best_val, best_act
    else:
        best_val, best_act = float('inf'), None
        for action in game.get_actions(state):
            val, _ = minimax(game.result(state, action), depth-1, True, game)
            if val < best_val: best_val, best_act = val, action
        return best_val, best_act


def alpha_beta(
    state: tuple[str, ...],
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    game: "TicTacToe",
) -> tuple[float, int | None]:
    """Minimax with alpha-beta pruning.

    alpha is the best value the maximizer can already force, beta the best the
    minimizer can already force. Once beta <= alpha the rest of this node's children
    cannot affect the value that propagates upward, so they are never generated.
    """
    NODES["alpha_beta"] += 1

    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state), None

    if is_maximizing:
        best_val, best_act = float('-inf'), None
        for action in game.get_actions(state):
            val, _ = alpha_beta(game.result(state, action),
                                depth-1, alpha, beta, False, game)
            if val > best_val: best_val, best_act = val, action
            alpha = max(alpha, best_val)
            if beta <= alpha: break
        return best_val, best_act
    else:
        best_val, best_act = float('inf'), None
        for action in game.get_actions(state):
            val, _ = alpha_beta(game.result(state, action),
                                depth-1, alpha, beta, True, game)
            if val < best_val: best_val, best_act = val, action
            beta = min(beta, best_val)
            if beta <= alpha: break
        return best_val, best_act


class TicTacToe:
    """The environment. A state is a 9-tuple of 'X', 'O', and '.'."""

    def is_terminal(self, state: tuple[str, ...]) -> bool:
        return self.check_winner(state) is not None or len(self.get_actions(state)) == 0

    def get_actions(self, state: tuple[str, ...]) -> list[int]:
        return [i for i in range(9) if state[i] == '.']

    def result(self, state: tuple[str, ...], action: int) -> tuple[str, ...]:
        # Whose turn it is comes from the board itself, so a state carries everything
        # the search needs and nothing has to be threaded through the recursion.
        s = list(state)
        s[action] = 'X' if state.count('X') == state.count('O') else 'O'
        return tuple(s)

    def check_winner(self, state: tuple[str, ...]) -> str | None:
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if state[a] == state[b] == state[c] != '.': return state[a]
        return None

    def evaluate(self, state: tuple[str, ...]) -> float:
        w = self.check_winner(state)
        if w == 'X': return 1
        if w == 'O': return -1
        return 0


def render(state: tuple[str, ...]) -> str:
    """Three rows of three, for printing."""
    return '\n'.join('  ' + ' '.join(state[i:i+3]) for i in range(0, 9, 3))


if __name__ == "__main__":
    game = TicTacToe()
    empty = tuple('.' * 9)

    # Full depth: 9 plies fills the board, so the depth limit never binds and the
    # evaluation function is only ever asked about finished games. Nothing here is an
    # estimate.
    start = time.perf_counter()
    value, action = alpha_beta(empty, 9, float('-inf'), float('inf'), True, game)
    ab_seconds = time.perf_counter() - start

    print(f"Best first move: {action}, value: {value}")
    print(render(game.result(empty, action)))
    print()

    start = time.perf_counter()
    plain_value, plain_action = minimax(empty, 9, True, game)
    mm_seconds = time.perf_counter() - start

    print("Same position, same depth, same move ordering:")
    print(f"  plain minimax -> move {plain_action}, value {plain_value}, "
          f"{NODES['minimax']} nodes")
    print(f"  alpha-beta    -> move {action}, value {value}, "
          f"{NODES['alpha_beta']} nodes")

    agree = (plain_value, plain_action) == (value, action)
    print(f"  same answer   -> {agree}")

    pruned = NODES["minimax"] - NODES["alpha_beta"]
    share = 100.0 * pruned / NODES["minimax"]
    print(f"  never visited -> {pruned} nodes ({share:.1f}% of the tree)")
    print()

    # Node counts are a property of the algorithm and reproduce anywhere. Seconds are
    # not: they are what this machine did on this run.
    print(f"Wall clock on this machine, this run: "
          f"alpha-beta {ab_seconds:.2f}s, plain minimax {mm_seconds:.2f}s")
    print("Node counts are exact and reproducible; the seconds are not.")
