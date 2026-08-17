"""The same alpha-beta search, with the evaluation function swapped for a model.

What is identical to before.py: the game, the tree, the alternation of maximizer and
minimizer, the alpha-beta cutoffs, and the exact evaluation of finished games.

What changed: one method. Positions that the search reaches at its depth limit are not
finished games, so no exact answer exists for them. Classically that is where you
hand-write a heuristic. Here the model supplies the estimate instead.

The ledger printed at the end is the argument. Every position with an exact answer is
answered by code. The model is asked only about positions where code has nothing to say.

Runs with no API key. With none set, every evaluation comes back as a canned response
and the search, the parsing, and the fallback all execute exactly as they would live.
"""

from __future__ import annotations

import random
import sys
import zlib
from pathlib import Path

# parents[2] is the repo root: adversarial -> 05-multi-agent -> peas. Needed because
# Python puts this script's directory on sys.path, not the directory it was launched
# from. The second insert is this example's own directory, so `from before import ...`
# resolves no matter which launcher is used.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from before import TicTacToe, render  # the same environment, unmodified
from shared.llm import llm_call

# How many distinct recorded evaluations the transcript holds for this example.
MOCK_EVAL_RESPONSES = 6


def mock_eval_key(state: tuple[str, ...]) -> str:
    """Choose which canned evaluation this position gets in mock mode.

    Mock-mode plumbing only -- every real backend ignores mock_key. It exists because
    a single key would hand every position the identical score, the search would have
    nothing to discriminate on, and the offline run would appear to work for a reason
    that has nothing to do with the code being demonstrated. Bucketing on the board
    varies the canned scores the way a live model's would vary, and the bucket is a
    pure function of the position, so the offline game is reproducible.

    crc32 rather than the builtin hash: hash() of a string is salted per process, which
    would make the offline run irreproducible. An arithmetic bucket on the raw character
    codes looks simpler and is worse -- 'X', 'O' and '.' are 42 and 33 apart, so a
    modulus that shares a factor with either silently makes most of the canned responses
    unreachable, and the run then exercises less of the parsing code than it appears to.
    """
    digest = zlib.crc32(''.join(state).encode())
    return f"adversarial_eval_{digest % MOCK_EVAL_RESPONSES}"


class LLMGameAgent:
    """
    Alpha-beta search with LLM-powered evaluation.
    Deterministic: tree structure, pruning, terminal evaluation.
    LLM: non-terminal position evaluation (the hard part).
    """
    def __init__(self, game: TicTacToe, max_depth: int = 4):
        self.game = game
        self.max_depth = max_depth

        # The evaluation ledger. code_evals are facts, model_evals are estimates, and
        # the whole claim of this example is that the second number buys something the
        # first cannot supply while the first stays untouched.
        self.code_evals = 0
        self.model_evals = 0
        self.parse_fallbacks = 0
        self.clamped = 0
        self.sample_code: tuple[str, float] | None = None
        self.sample_model: tuple[str, str, float] | None = None
        self.sample_fallback: str | None = None

    def get_action(self, state: tuple[str, ...]) -> int | None:
        _, action = self.search(state, self.max_depth,
                                float('-inf'), float('inf'), True)
        return action

    def search(
        self,
        state: tuple[str, ...],
        depth: int,
        alpha: float,
        beta: float,
        is_max: bool,
    ) -> tuple[float, int | None]:
        if depth == 0 or self.game.is_terminal(state):
            return self.evaluate(state), None
        if is_max:
            best_v, best_a = float('-inf'), None
            for a in self.game.get_actions(state):
                v, _ = self.search(self.game.result(state, a),
                                   depth-1, alpha, beta, False)
                if v > best_v: best_v, best_a = v, a
                alpha = max(alpha, best_v)
                if beta <= alpha: break
            return best_v, best_a
        else:
            best_v, best_a = float('inf'), None
            for a in self.game.get_actions(state):
                v, _ = self.search(self.game.result(state, a),
                                   depth-1, alpha, beta, True)
                if v < best_v: best_v, best_a = v, a
                beta = min(beta, best_v)
                if beta <= alpha: break
            return best_v, best_a

    def evaluate(self, state: tuple[str, ...]) -> float:
        # Terminal: deterministic, exact. The game is over, so the result is a fact and
        # asking a model about it would be strictly worse than reading it off the board.
        if self.game.is_terminal(state):
            self.code_evals += 1
            score = self.game.evaluate(state)
            if self.sample_code is None:
                self.sample_code = (render(state), score)
            return score

        # Non-terminal: LLM estimates
        board = '\n'.join(' '.join(state[i:i+3]) for i in range(0, 9, 3))
        # "Return just a number" is not an instruction a chatty model obeys. Asked exactly
        # that, claude-sonnet-5 reasoned about the position until it hit the token cap --
        # every call truncated, every parse failed, every position silently scored 0.0, and
        # a depth-2 game became unusably slow. The extra two lines below are the cheapest
        # fix available and they belong in the prompt rather than in a max_tokens argument,
        # because the search needs one scalar per position and nothing else at any tier.
        prompt = f"""Evaluate this game position from -1.0 to 1.0.
+1.0 = X winning, -1.0 = O winning, 0.0 = even.
{board}
Answer with the number alone. No reasoning, no explanation, no units.
Example of a valid answer: -0.25"""
        self.model_evals += 1
        # tier=mid: bounded numeric judgment inside a hard range, with a deterministic
        # parse-and-clamp check behind it. It is not a label from a fixed list, which a
        # small model would handle, and it is not a synthesis problem -- the search
        # supplies all the structure and asks only for one scalar per position.
        raw = llm_call(prompt, mock_key=mock_eval_key(state), tier="mid")

        try:
            score = float(raw.strip())
        except (ValueError, AttributeError):
            # The source catches ValueError, which is the common failure: the model
            # answered in prose. AttributeError is added for a backend that returns
            # something that is not a string at all, where .strip() fails before float()
            # is ever reached. Either way this is a failed call rather than a genuine
            # reading of "even", so it is counted instead of quietly averaged into the
            # search.
            self.parse_fallbacks += 1
            if self.sample_fallback is None:
                self.sample_fallback = raw.strip()
            return 0.0

        if score < -1.0 or score > 1.0:
            self.clamped += 1
        score = max(-1.0, min(1.0, score))
        if self.sample_model is None:
            self.sample_model = (render(state), raw.strip(), score)
        return score


def play_game(agent: LLMGameAgent, game: TicTacToe, seed: int = 7) -> tuple[str, ...]:
    """Play a full game: the agent is X, a random opponent is O. Returns the final state."""
    random.seed(seed)
    state = tuple('.' * 9)
    turn = 1

    while not game.is_terminal(state):
        agent_to_move = state.count('X') == state.count('O')

        if agent_to_move:
            before_code, before_model = agent.code_evals, agent.model_evals
            action = agent.get_action(state)
            code_here = agent.code_evals - before_code
            model_here = agent.model_evals - before_model
            print(f"turn {turn}: X plays {action}   "
                  f"[{code_here} positions by code, {model_here} by model]")
        else:
            action = random.choice(game.get_actions(state))
            print(f"turn {turn}: O plays {action}   [random opponent, no search]")

        state = game.result(state, action)
        print(render(state))
        print()
        turn += 1

    return state


if __name__ == "__main__":
    # Optional depth on the command line. Depth is the cost dial: every extra ply
    # multiplies the number of positions the model is asked about, and in real mode each
    # of those is one API call.
    #
    # The default is 2 rather than 3 because of what that dial actually costs. A depth-3
    # game asks the model about roughly 150 positions; depth 2 asks about 49, and shows
    # the same three things -- the code/model split in the ledger, the parse fallback, and
    # a search whose structure never changes. Defaulting to the setting that costs a
    # reader three times as much to reproduce, in a repository where running it live is
    # now the point, would be choosing a bigger number over a useful one. Pass a depth to
    # raise it; the ledger prints how many calls it cost either way.
    try:
        depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    except ValueError:
        # A visitor's first instinct is --help, and a traceback is a poor answer to it.
        raise SystemExit(f"usage: python {Path(__file__).name} [search_depth]   (default 2)")
    if depth < 1:
        # A depth-0 search evaluates the root and returns no action, so there would be
        # nothing to play. Fail loudly rather than move at random and call it a search.
        raise SystemExit("depth must be at least 1")

    game = TicTacToe()
    agent = LLMGameAgent(game, max_depth=depth)

    print(f"Alpha-beta, depth limit {depth}. Agent is X, random opponent is O (seed 7).")
    print()

    final = play_game(agent, game, seed=7)

    winner = game.check_winner(final)
    print(f"Result: {winner + ' wins' if winner else 'draw'}")
    print()

    total = agent.code_evals + agent.model_evals
    print("Who evaluated what, over the whole game")
    print(f"  by CODE  (terminal, exact)       {agent.code_evals:6d}"
          f"   {100.0 * agent.code_evals / total:5.1f}%")
    print(f"  by MODEL (non-terminal, estimate){agent.model_evals:6d}"
          f"   {100.0 * agent.model_evals / total:5.1f}%")
    print(f"  total positions evaluated        {total:6d}")
    print()
    print("Deterministic checks applied to every model answer")
    print(f"  unparseable -> fell back to 0.0  {agent.parse_fallbacks:6d}")
    print(f"  outside [-1, 1] -> clamped       {agent.clamped:6d}")
    print()

    if agent.sample_code:
        board, score = agent.sample_code
        print(f"A position CODE evaluated, exactly ({score}):")
        print(board)
        print()
    if agent.sample_model:
        board, raw, score = agent.sample_model
        print(f"A position the MODEL evaluated (returned {raw!r}, used {score}):")
        print(board)
        print()
    if agent.sample_fallback:
        print(f"A model answer that failed the parse: {agent.sample_fallback!r}")
        print("The search used 0.0 for that position and counted the failure.")
        print()

    print(f"In real mode this game is {agent.model_evals} API calls, one per "
          f"model-evaluated position.")
    print("Raising the depth limit raises that number. The search structure does not "
          "change.")
