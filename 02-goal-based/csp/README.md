# CSP scheduling

**Source:** reference/02-goal-based-agents-before-after.md

## The claim

`after.py` contains no CSP solver. It imports `CSP` and `backtracking_search` from
`before.py`, so the algorithm that schedules three meetings from a paragraph of English is
the same object, in the same file, as the one that colours the map of Australia.
`verify_identical.py` proves it two ways — object identity at runtime and a digest of the
one copy of the solver source — and exits non-zero the moment either stops holding.

## Run it

    python before.py
    python after.py
    python verify_identical.py

`before.py` — the hand-written CSP:

    Map coloring: {'WA': 'red', 'NT': 'green', 'SA': 'blue', 'Q': 'red', ...}

    Verification (independent of the solver):
      [ok] WA=red       != NT=green
      ...
      9/9 constraints satisfied
      colours used: 3 of 3 available
      T borders nothing, so it takes the first colour in its domain: red

`after.py` — the same solver, given a CSP the model wrote. Three requests: one solvable,
one with no solution, one that was meant to be malformed and no longer is.

    solver: backtracking_search from module 'before' (before.py), imported not redefined

    === Request 1: three meetings, four slots ===
    -- LLM extracts the CSP (this is the part that changed) --
    [replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
      variables: team_standup, design_review, sprint_planning
      constraint: team_standup not_same_time design_review
      ...
    -- backtracking_search solves it (this is the part that did not) --
      assignment: {'team_standup': 'Monday 9am', 'design_review': 'Monday 2pm', 'sprint_planning': 'Tuesday 10am'}
    -- Verification, deterministic, no model involved --
      [ok] team_standup='Monday 9am' vs design_review='Monday 2pm'
      ...
      [ok] independent check: 3 meetings at 3 distinct times
      result: 0 constraints violated

    === Request 2: three meetings, two slots ===
      assignment: None
      no schedule exists. The solver proved that; it did not run out of ideas.
      A model asked this directly would very likely have answered anyway.

    === Request 3: the extraction goes wrong ===
      variables: kickoff, retro
      domain of kickoff: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
      domain of retro: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
      constraint: kickoff not_equal retro

Request 3 prints nothing after that, and the heading is currently a lie. It was written to
force a malformed extraction through `mock_key`, so that the validation would reject a
variable with no domain before the solver saw it. Every live backend ignores `mock_key`,
and so does replay — prompts are matched by their own content now. The model, asked to
book a kickoff and a retro, returned a perfectly well-formed CSP, the validation had
nothing to reject, and the demonstration quietly stopped happening. That is the same
defect `05-multi-agent/orchestration/` found and fixed by injecting the bad payload
directly instead of asking a backend to produce one. It is not fixed here yet.

`verify_identical.py` — the claim, checked:

    Static check: after.py defines no solver of its own
      [ok] after.py does not bind 'CSP' itself
      [ok] after.py does not bind 'backtracking_search' itself
    ...
    Runtime check: the objects are the same objects
      [ok] after.CSP is before.CSP
      [ok] Python attributes after.backtracking_search to before.py -- before.py

    PASS: after.py does not contain a solver. It imports the one in before.py.
          solver source: 38 lines in .../02-goal-based/csp/before.py
          sha256: 322c5be60b616616f91016ca883cae7242d7bfd3f13cb1a01c9a9bf20ba3fcc7
          One copy of that text exists in this directory. This is its digest.

`grep -rn --include="*.py" "def backtracking_search"` over this directory returns exactly
one line, in `before.py`. The bare word `backtracking` appears in every file here, this one
included, because they all talk about the solver constantly — which is exactly why the
check parses syntax and compares objects instead of matching text.

## What changed

The LLM replaced the problem specification. In `before.py` a human wrote out seven
variables, three domains and nine constraints by hand. In `after.py`, `llm_extract_csp`
produces that same structure from a paragraph of English, and `llm_format_solution` turns
the solved dict back into a sentence.

Everything between those two calls is deterministic and unchanged: `CSP`,
`CSP.is_consistent`, `backtracking_search`, the validation that rejects a malformed
extraction before the solver ever sees it, and the verification that re-checks the returned
assignment against every constraint and every domain.

One fix to the source page: it calls `json.loads` on the raw response, which crashes when a
model wraps its JSON in a markdown fence. `after.py` strips the fence first.

## What it costs

The solver guarantees that the assignment satisfies the constraints it was handed. It
cannot guarantee those were the right constraints. A dropped attendee becomes a dropped
constraint, and the schedule is then verifiably correct about the wrong problem — which is
why `after.py` validates the extraction and re-checks the result. Two calls of latency
arrive where `before.py` had none, and extraction is not reproducible run to run.

`config.yaml` is the source page's spec, verbatim; nothing loads it. It names
`backtracking_with_ac3`, and neither the source solver nor this one has AC-3.
