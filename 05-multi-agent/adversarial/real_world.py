"""The same alpha-beta search, on a price war instead of tic-tac-toe.

`before.py` plays tic-tac-toe, because that is the example the source page shows. It proves
the mechanism -- alpha-beta prunes most of the tree and returns exactly the move minimax
returns -- and nobody ships a tic-tac-toe engine.

Adversarial search does ship, in the place it has always belonged: two parties with
opposed objectives, moving in turn, each having to assume the other will respond well. A
competitor pricing against you is that game, and it is one companies currently play by
watching a dashboard and reacting.

The structure is exactly minimax. You cut, they see it within a day and can cut back, and
the position two moves from now is what determines whether cutting today was worth it. A
price move that looks good in isolation and loses badly once matched is the whole reason
to search rather than react, and reacting is what a dashboard makes you do.

`alpha_beta` is imported from `before.py` rather than reimplemented. What changes is the
game: a state is a pair of prices rather than marks on a board.

Where the LLM belongs. In tic-tac-toe a leaf is +1, 0 or -1 and `evaluate()` is
arithmetic. A market position has no such number -- whether "we are at 89 and they are at
85" is a good place to be depends on brand, switching costs and how much of the category
is on promotion. That judgement is what the model supplies, one score per position, with
no knowledge that a search is happening.

Where it does not. It never chooses the move, never looks ahead, and never assumes the
competitor sits still. Alpha-beta does that, and the floor -- never price below cost -- is
code that no score can talk its way through.

Run it:

    python 05-multi-agent/adversarial/real_world.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from before import NODES, alpha_beta  # noqa: E402

from shared.llm import llm_call  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402

# A state is (our_price, their_price, whose_move). The turn lives in the state for the same
# reason it lives on a board: alpha_beta calls get_actions(state) with nothing else.
OPENING = (89, 89, "us")

UNIT_COST = 62          # below this we are selling at a loss, whatever the share gain
FLOOR = 65              # the lowest price commercial policy allows anyone to set
CEILING = 95

# Price moves either side can make. Three rungs, deliberately: a real ladder has more and
# the search does not care, but every reachable position has to be scored, and asking one
# model call for eighty-one judgements returned nothing at all the first time this was
# written. The tree grows faster than the prompt should.
MOVES = [("hold", 0), ("cut_4", -4), ("cut_8", -8)]

MARKET = """We're the incumbent in this category with about 55 percent share. Their brand
is thinner but they've been buying share with promotions all year and retail buyers have
noticed. Switching costs for the customer are close to zero -- it's a shelf decision. Our
unit cost is 62 and we've told the street we'll hold category margin this year, so a long
war at 70 is worse for us than losing a few points of share. They're private and burning
cash, which means they can go lower than us for longer than is comfortable."""


class PriceWar:
    """The interface alpha_beta expects: is_terminal, get_actions, result, evaluate.

    The same four methods TicTacToe provides in before.py, a different game behind them.
    """

    def __init__(self, leaf_scores: dict):
        self.leaf_scores = leaf_scores

    def is_terminal(self, state) -> bool:
        ours, theirs, _ = state
        return ours <= FLOOR or theirs <= FLOOR

    def get_actions(self, state):
        if self.is_terminal(state):
            return []
        return MOVES

    def result(self, state, action):
        _, delta = action
        ours, theirs, turn = state
        if turn == "us":
            return (max(FLOOR, min(ours + delta, CEILING)), theirs, "them")
        return (ours, max(FLOOR, min(theirs + delta, CEILING)), "us")

    def evaluate(self, state) -> float:
        """What this position is worth to us. Supplied by the model, floored by code."""
        ours, _, _ = state
        # DETERMINISTIC: no leaf score can make selling below cost attractive. The model
        # is asked about market position, not about whether losing money is acceptable.
        if ours < UNIT_COST:
            return -1000.0
        return float(self.leaf_scores.get(position_key(state), 0.0))


def position_key(state) -> str:
    return f"us{state[0]}_them{state[1]}"


def reachable_positions(depth: int) -> list[tuple[int, int]]:
    """Every price pair the search can reach, so the model scores each exactly once."""
    game = PriceWar({})
    seen_states, pairs, frontier = {OPENING}, {(OPENING[0], OPENING[1])}, [OPENING]
    for _ in range(depth):
        nxt = []
        for state in frontier:
            for move in game.get_actions(state):
                child = game.result(state, move)
                if child not in seen_states:
                    seen_states.add(child)
                    pairs.add((child[0], child[1]))
                    nxt.append(child)
        frontier = nxt
    return sorted(pairs)


BATCH = 5


def score_positions(pairs: list[tuple[int, int]]) -> dict:
    """The model call. One judgement per price pair, from our point of view.

    Scored in small batches rather than all at once, and that is a finding rather than a
    style choice. Asked to judge fifteen positions in a single call the model reasoned
    past the 4096-token cap and returned no text at all -- a `thinking` block and nothing
    else. The shim said so, loudly, which is the whole reason `_warn_if_truncated` exists:
    a truncated response is indistinguishable from a malformed one by the time anything
    parses it.

    Smaller asks leave room to answer. The tree is unchanged; only how many judgements
    are requested per call.
    """
    scores: dict = {}
    for start in range(0, len(pairs), BATCH):
        chunk = pairs[start:start + BATCH]
        listed = "\n".join(
            f"  us{ours}_them{theirs}: we are at {ours}, they are at {theirs}"
            for ours, theirs in chunk)
        prompt = f"""Score market positions in a price war, from our point of view.

Market context:
{MARKET.strip()}

Score each position from -100 to 100. Positive is a good place for us to be over the next
two quarters, negative is bad. Weigh margin against share: being cheaper wins volume, a
matched cut leaves both of us poorer at the same relative position, and our unit cost is
{UNIT_COST}.

Positions:
{listed}

Answer with the JSON object and nothing else. Do not explain.
{{"<key>": <score>, ...}}"""
        # tier=mid: judging whether a price position is a good place to be, given brand,
        # switching costs and a competitor's cash position, is exactly the reading a small
        # model flattens. Not frontier: it is one bounded judgement repeated, and the
        # search is what turns those judgements into a decision.
        raw = llm_call(prompt, mock_key="price_positions", tier="mid")
        scores.update(model_loads(raw))
    return scores


def main() -> None:
    print("The same alpha-beta search, on a price war")
    print()
    print("  alpha_beta is imported from before.py, which plays tic-tac-toe with it.")
    print("  A state is a pair of prices instead of marks on a board.")
    print()
    print(f"  We are at {OPENING[0]}. They are at {OPENING[1]}. Unit cost {UNIT_COST}.")
    print()

    depth = 3
    pairs = reachable_positions(depth)
    print(f"BEFORE: {len(pairs)} price positions reachable in {depth} moves")
    print()
    print("  Nothing to search without a way to value one. Tic-tac-toe gets +1, 0 or -1")
    print("  from arithmetic; a market position has no such number, and supplying it is")
    print("  the only thing the model is asked for.")
    print()

    scores = score_positions(pairs)

    # DETERMINISTIC: every reachable position needs a numeric score, or the tree has
    # holes and every comparison against them is a comparison against zero.
    missing = [position_key((a, b, "us")) for a, b in pairs
               if not isinstance(scores.get(position_key((a, b, "us"))), (int, float))]
    if missing:
        print(f"  refused: {len(missing)} position(s) unscored, first few {missing[:3]}")
        print("  A tree with holes searches fine and means nothing.")
        return

    ranked = sorted(pairs, key=lambda p: scores[position_key((p[0], p[1], "us"))], reverse=True)
    print("AFTER: the model scored every position, the search chose the move")
    print()
    for a, b in ranked[:2]:
        print(f"    best for us    we {a} / them {b}   "
              f"{scores[position_key((a, b, 'us'))]:>6.0f}")
    for a, b in ranked[-2:]:
        print(f"    worst for us   we {a} / them {b}   "
              f"{scores[position_key((a, b, 'us'))]:>6.0f}")
    print()

    game = PriceWar(scores)
    NODES["alpha_beta"] = 0
    value, best = alpha_beta(OPENING, depth, float("-inf"), float("inf"), True, game)
    searched = NODES["alpha_beta"]
    print(f"  move to make now: {best[0] if best else 'none'}")
    print(f"  value if they respond as well as we do: {value:.0f}")
    print(f"  nodes visited: {searched}")
    print()

    # The comparison that makes the point: what a dashboard would tell you to do.
    greedy = max(MOVES, key=lambda m: scores.get(
        position_key((max(FLOOR, min(OPENING[0] + m[1], CEILING)), OPENING[1], "them")), 0))
    print(f"  what reacting would do: {greedy[0]}")
    print("    that is the move with the best score one step out, which is what you pick")
    print("    when you are looking at a dashboard rather than a tree.")
    print()

    if greedy[0] != (best[0] if best else None):
        print("  They differ, and the difference is the whole argument. The one-step move")
        print("  looks better until the competitor answers it, and the search is what")
        print("  sees the answer coming.")
    else:
        print("  They agree here. Worth knowing rather than assuming -- on this board the")
        print("  greedy move survives the response, and on a different one it will not.")
    print()
    print("  What the model did: score positions, one at a time. What it did not do:")
    print("  choose the move, look ahead, or assume the competitor sits still.")
    print("  Alpha-beta did that, and the floor -- never below unit cost -- is code that")
    print("  no score can argue with.")


if __name__ == "__main__":
    main()
