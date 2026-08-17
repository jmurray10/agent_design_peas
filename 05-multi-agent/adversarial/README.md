# Adversarial search

**Source:** reference/05-multi-agent-systems-before-after.md

## The claim

Replacing the evaluation function with a language model does not cost you the search.
Alpha-beta's tree, its cutoffs, and its exact treatment of finished games all survive
intact; the model is asked only about positions where no exact answer exists. Both
claims print a number you can check: `before.py` reports how much of the game tree
pruning skipped, `after.py` reports how many positions were answered by code and how
many by the model.

## Run it

    python before.py
    python after.py 2        # depth 2, which is what the shipped transcript covers
    python after.py          # depth 2 by default
    python 05-multi-agent/adversarial/real_world.py

`before.py`:

    Best first move: 0, value: 0
      X . .
      . . .
      . . .

    Same position, same depth, same move ordering:
      plain minimax -> move 0, value 0, 549946 nodes
      alpha-beta    -> move 0, value 0, 18297 nodes
      same answer   -> True
      never visited -> 531649 nodes (96.7% of the tree)

Two more lines follow those, giving wall-clock seconds for both searches and saying in the
same breath that the seconds are not reproducible and the node counts are. They are not
quoted here for that reason: they measure this machine on one run.

`after.py 2`:

    Alpha-beta, depth limit 2. Agent is X, random opponent is O (seed 7).

    [replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
    turn 1: X plays 0   [0 positions by code, 16 by model]
      X . .
      . . .
      . . .
    ...
    turn 5: X plays 1   [1 positions by code, 4 by model]
      X X X
      O . .
      O . .

    Result: X wins

    Who evaluated what, over the whole game
      by CODE  (terminal, exact)            1     2.4%
      by MODEL (non-terminal, estimate)    40    97.6%
      total positions evaluated            41

    Deterministic checks applied to every model answer
      unparseable -> fell back to 0.0       0
      outside [-1, 1] -> clamped            0

    A position the MODEL evaluated (returned '0.3', used 0.3):
      X O .
      . . .
      . . .

    In real mode this game is 40 API calls, one per model-evaluated position.
    Raising the depth limit raises that number. The search structure does not change.

`before.py`'s node counts are exact and reproduce anywhere. The evaluation counts depend
on the scores the model returned, so they move when the model does. The transcript here is
40 `claude-sonnet-5` responses from 2026-08-04, one per non-terminal position the depth-2
search reached. On this recording none of them came back unparseable and none landed
outside `[-1, 1]`, so both counters read zero.

An earlier recording had two empty responses among 72 positions, and this file used to
spend two paragraphs on them. They are gone, and the counters now report that honestly
rather than describing a failure that no longer happens. Both checks are in the code
either way; what changed is what the model did, not what the example was arranged to
show, and a counter reading zero is the only trustworthy kind.

`python after.py 3` searches a depth the recording does not cover, reaches positions with
three pieces on the board, and stops on the first one. The no-argument run defaults to
depth 2 and completes -- this README said it defaulted to 3 and halted, which was true of
an earlier version of the script and made the whole section unreproducible:

    LookupError: No recorded response for this prompt in shared/transcripts/05_multi_agent__adversarial__after.json.
      prompt digest: a3df40322122b533fded2ecd
      This happens when a prompt changed, or when the example is new. Record it:
        ANTHROPIC_API_KEY=... LLM_RECORD=1 python <the script you just ran>
      Nothing is faked in its place on purpose -- a stand-in answer here is how a
      demo keeps printing plausible output after it stopped meaning anything.

That is the design working, not a crash to route around. The recording was taken at depth
2, depth 3 asks about a different set of boards, and every prompt is keyed by its own
content — so the alternative to halting is replaying an answer to a question about a
different position. Record depth 3 with a key and it runs; until then, the honest run is
depth 2.

The prompt is worth reading before you record anything. Asked exactly "return just a
number", `claude-sonnet-5` reasoned about the position until it hit the token cap. Every
call truncated, every parse failed, every position silently scored 0.0, and a depth-2 game
became unusably slow. Two extra lines fixed it — "Answer with the number alone. No
reasoning, no explanation, no units." and an example of a valid answer — and they belong
in the prompt rather than in a `max_tokens` argument, because the search needs one scalar
per position and nothing else.

### The same search, on a price war

`real_world.py` imports `alpha_beta` and runs it over competitive pricing: we move, they
see it within a day and can answer, and the position two moves out is what decides whether
moving today was worth it. Companies play this game by watching a dashboard and reacting.

In tic-tac-toe a leaf is +1, 0 or -1 and `evaluate()` is arithmetic. A market position has
no such number, and scoring one -- given brand, switching costs and a competitor burning
cash -- is the judgement the model supplies, with no knowledge that a search is happening.

On the shipped recording the search says cut by 4 and reacting to the best one-step score
says cut by 8. The deeper cut looks better until it is matched, and the search is what sees
the answer coming. The floor -- never below unit cost -- is code that no score can argue
with.

That pair is one outcome, not the outcome, and the script is careful about this where an
earlier version of this paragraph was not: it compares the two moves and prints either
"They differ, and the difference is the whole argument" or "They agree here. Worth knowing
rather than assuming." Six live runs against `claude-sonnet-5` gave four readable pairs --
`cut_8/cut_8`, `cut_4/cut_4`, `cut_8/cut_4`, `cut_8/cut_4` -- so two agreed and two did
not, and the recorded `cut_4/cut_8` was not among them. Read the recording as one draw.
What survives every draw is the structure: the search is what looks a move ahead, the
model supplies a judgement it has no way to derive, and the floor holds regardless.

Writing it produced a finding too. Asked to judge fifteen positions in one call, the model
reasoned past the token cap and returned a `thinking` block with no text at all. The shim
warned on stderr and then handed the empty string downstream, which is half a guard: a
truncated response is indistinguishable from a malformed one by the time anything parses
it, so every fallback fires correctly and for the wrong reason. `_call_anthropic` now
raises on empty text rather than warning about it, the same way `_call_gemini` does. The
scoring runs in
small batches now.

## What changed

The evaluation function, and nothing else. `before.py` searches to full depth, where
every leaf is a finished game, so it never estimates anything. `after.py` stops at a
depth limit, where most leaves are unfinished, and asks the model for a score in
[-1, 1] rather than hand-coding a heuristic. Still deterministic: the rules, move
generation, the tree, the alpha-beta cutoffs, terminal scoring, the parse, and the
clamp. `after.py` imports `TicTacToe` from `before.py` so the environment cannot
quietly differ between the two runs.

Two fixes to the source's `evaluate()`. It now catches `AttributeError` as well as
`ValueError`, because a backend that returns something other than a string fails at
`.strip()` before `float()` is reached. And the clamp is applied after the parse rather
than inside it, so out-of-range answers can be counted instead of vanishing into a
`min`/`max`.

## What it costs

Optimality and speed. Alpha-beta is only as good as what it evaluates, so the
guarantee that the move is best is gone the moment a leaf is a guess. A model answer
that fails to parse scores 0.0, which is a real position treated as an even one. None of
the 40 recorded answers were empty on this run, and an earlier recording had two that
were, which is the failure mode rather than a fixed property. Depth is a bill: at depth 2
one game asked the model about 40 positions, each an API call with its own latency,
against `before.py`
solving the entire game in 18,297 nodes with no network at all.
