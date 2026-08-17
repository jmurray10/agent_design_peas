# A* and the 8-puzzle

**Source:** reference/02-goal-based-agents-before-after.md

## The claim

An LLM sitting in front of A* does not change what A* returns, and the gate deciding
whether A* runs at all is deterministic code that never consults the model's confidence.
`after.py` prints which path it took and, on the search path, compares the result to
`before.py` move for move and node for node. The recording shipped here does both. The
first request formalises: the model reads a board out of prose, the permutation check
accepts it, A* expands 282 nodes and returns the same 20-move plan `before.py` returns
from a tuple. The second does not: the request describes a state that is not a legal
board, `is_formalizable()` refuses it, and the agent falls back to LLM planning — one
action, no plan, no optimality guarantee.

Both paths in one run is the honest version of this example. The check is deterministic
and never consults the model's confidence, so which path a request takes is decided by
the request rather than by how sure anything sounds.

## Run it

    python before.py
    python after.py
    python 02-goal-based/search/real_world.py

`before.py`:

    8-puzzle, A* with Manhattan distance

      initial state
        7 2 4
        5 . 6
        8 3 1

    Solution in 20 moves: ['down', 'right', 'up', 'left', 'left', 'up', ...]

      heuristic at start : 14 (a lower bound on the true cost, ...)
      nodes expanded     : 282
      nodes generated    : 461
      path cost          : 20

      replaying the moves reaches the goal: True

`after.py`, abbreviated to the two path decisions and the comparison:

    REQUEST 1 of 2  --  formalizable: a state space with a transition model
    user: "The kids scrambled the sliding tile puzzle again. Top row reads 7, 2, 4. ..."

    [replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
      llm_interpret_goal()   -> {"puzzle": "3x3 sliding tile puzzle (8-puzzle)", "initial_state": [[7, 2, 4], [5, 0, 6], [8, 3, 1]], "blank_symbol": 0, ...}
      llm_parse_state()      -> {"task_type": "sliding_tile_puzzle", "initial_state": {"grid": [[7, 2, 4], [5, null, 6], [8, 3, 1]], ...}, ...}
      is_formalizable()      -> True
      PATH TAKEN: A* SEARCH    -- deterministic, optimal, no LLM in the loop

      A* returned a complete plan, 20 moves:
        ['down', 'right', 'up', 'left', 'left', 'up', 'right', 'right', 'down', 'left', ...]
        nodes expanded   : 282
        nodes generated  : 461
        path cost        : 20

      action handed to the actuators: 'down'

    REQUEST 2 of 2  --  not formalizable: no state space to search
    user: "Help me plan two weeks in Europe in September. ..."

      no transition model: parsed state holds no valid 3x3 tile permutation
      is_formalizable()      -> False
      PATH TAKEN: LLM FALLBACK -- one plausible action, no optimality guarantee

      action handed to the actuators: 'research_destinations'

    VERIFY  --  path 1 returned exactly what before.py returns
      before.py initial state : (7, 2, 4, 5, 0, 6, 8, 3, 1)
      after.py  initial state : (7, 2, 4, 5, 0, 6, 8, 3, 1)   (parsed from prose by the LLM)
      same starting state     : True

      before.py : 20 moves, 282 nodes expanded
      after.py  : 20 moves, 282 nodes expanded
      move lists identical    : True

`before.py`'s node counts are what this code did on this problem, not a published
benchmark. They are deterministic — the same numbers on any machine — because they count
heap operations, not seconds.

Request 1 is worth reading closely, because getting it to this point took two fixes and
an earlier recording did not reach A* at all. In that one the model wrote the blank square
as `null` rather than `0`; `[[7, 2, 4], [5, null, 6], [8, 3, 1]]` is the puzzle, in the
right order, with the blank in the right cell, and `_puzzle_tuple` refused it because it
accepts only an exact permutation of 0..8 and `null` is not `0`. A correct board in an
unexpected encoding bought the fallback path rather than a search, which is the check doing
its job on the one input a human would wave through, and the reason `_blank_as_zero`
exists.

Two earlier fixes are why the gate is the only thing standing between this run and A*, and
both were found by running live rather than offline. The model returned its JSON inside a
```json fence, which a bare `json.loads` rejects, so the parse failed and there was no
board at all — `shared/model_json.py` now unwraps the fence first. Then the parse succeeded
and the board still went unfound, because the model had answered with `initial_state.grid`
shaped `[[7,2,4],...]` rather than the flat `tiles` key the example had been reading. Both
are correct answers to a prompt that asked for neither shape, and reading one fixed key was
a bet on an address the prompt never specified. `_find_puzzle_tuple` now searches the
parsed response for the board.

What did not get relaxed is the check. A candidate still has to be an exact permutation of
0..8, so every fix was to where the code looks, never to what it accepts — `_blank_as_zero`
maps the `null` to the blank before the permutation test, and the test is the same test.

With those three in place the shipped recording formalises: `is_formalizable() -> True`,
282 nodes expanded, the same 20 moves `before.py` prints, and no model in the loop after
the parse. That is what request 1 above does. Whether a given day's model returns a shape
the finder can reach is still a property of that day — three separate live runs found
three ways for it not to be, which is why the gate is worth having and why request 2 is in
the file.

### The same search, on a stuck order

Routing a van is the version everyone reaches for, and companies buy routing software.
What they suffer through by hand is an order that will not arrive on time.

`real_world.py` imports `a_star_search` and `SearchProblem` and searches over fulfilment
plans instead of board positions. Stock is scattered across a warehouse, a bonded port, a
production run that could be pulled forward and a substitute nobody wants to offer. Each
option costs money and takes days, several combine, and the goal is a promised date.

The model reads the account manager's note -- "they'll take a partial if the balance lands
by month end, but not the substitute, it failed qualification at their site" -- and turns
it into a deadline and an exclusion.

Three searches, and the third is the one that earns its place. Twelve days is a cheap
plan. Four days is the same order for more than twice the money, which is a number to take
into the conversation. Two days returns no plan at all -- not an expensive one, none --
and that is a fact worth having today rather than on the due date.

## What changed

One component. The LLM replaced the step where a person hand-writes
`initial = (7, 2, 4, 5, 0, 6, 8, 3, 1)` and the goal beside it. In `after.py` both come
out of a sentence about a scrambled puzzle.

Everything downstream is the same code: `a_star_search`, `SearchProblem`, `actions` and
`result` are imported from `before.py`. The heuristic is Manhattan distance, computed
against the goal tuple the model returned instead of a module constant.

The gate between the two paths is deterministic. `_puzzle_tuple` accepts a grid only if it
is an exact permutation of 0..8, and `_has_known_transitions` returns True only when state
and goal both pass. The model's confidence is never consulted. `llm_plan_next_action` is
clamped to `available_actions` the same way, which fires in request 1, where a model asked
for one action name returned a BFS trace.

`config.yaml` is the source page's puzzle-solver PEAS block, verbatim.

## What it costs

The search path pays two LLM calls and their latency before A* starts, and inherits their
failure modes. A plausible but wrong grid yields a provably optimal plan for a puzzle
nobody owns: the permutation check catches malformed input, not confidently wrong input.

The fallback costs more. It returns one action with no plan behind it, no optimality
claim, no completeness claim, and no way to distinguish a good action from a merely legal
one. `before.py` can prove that no solution exists. The fallback can only ever say
something.
