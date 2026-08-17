# Simple reflex agent

**Source:** reference/01-reflex-agents-before-after.md

## The claim

A simple reflex agent is only as complete as its rule table, and no rule table enumerates
the real world. `before.py` meets a percept nobody wrote a rule for and does nothing,
silently, with no error. `after.py` is the same agent -- same loop, one percept in, one
action out, still no memory and no planning -- with the table swapped for a model call,
and it acts on percepts it was never programmed for.

## Run it

    python before.py
    python after.py
    python 01-reflex-agents/simple/real_world.py

`before.py`:

```
See: {'location': 'left', 'status': 'dirty'} -> Do: suck
See: {'location': 'left', 'status': 'clean'} -> Do: move_right
See: {'location': 'right', 'status': 'dirty'} -> Do: suck
See: {'location': 'right', 'status': 'puddle'} -> Do: no_op
```

The fourth line is the whole point. No key in the rules dict matches `puddle`, so
`rule_match` returns nothing and `agent_function` falls through to `no_op`. The agent does
not crash and does not warn. It just stops being useful.

`after.py`:

```
[replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
See: {'location': 'left', 'status': 'dirty'} -> Do: suck   [chosen by the model]
See: {'location': 'right', 'status': 'puddle and dirt mixed together'} -> Do: suck   [chosen by the model]
See: {'location': 'right', 'status': 'something sticky, maybe glue'} -> Do: suck   [chosen by the model]
```

Neither file needs an API key or a `pip install`. With no backend configured, `after.py`
replays what `claude-haiku-4-5` returned to these three exact prompts on 2026-08-04. The
prompt is still built, the response is still parsed, and the validation block still runs;
the only thing that does not happen is the network call.

Read the second line against `before.py`'s fourth. `before.py` said `no_op` about a puddle
because nothing in its table matched -- a hole, printed in the same format as a decision.
Here the model was asked about a puddle and chose `suck`. The bracket exists so those two
events cannot be confused, and it earns its keep on the runs where the model also says
`no_op`.

Which it usually does. Asked four times about the puddle, `claude-haiku-4-5` answered
`no_op` three times and `suck` once, all four on its own action list. The recording caught
the fourth. That is worth knowing before you read a replay as the model's answer: it is
one sample from one date, not a distribution, and not what the model would say today.

Two notes on the code. First, `interpret_input` in `before.py` differs from the source
listing by one line: the source returns `str(percept.data)`, and the rule keys
`clean_left` and `clean_right` never occur as substrings of a dict repr, so those two
rules could never fire and a clean cell also produced `no_op`. The fix appends a
`"<status>_<location>"` token so the hand-written rules match what they were written to
match, leaving the rules dict and `rule_match` untouched. Second, `config.yaml` is the
PEAS spec for this agent, copied from the source page. `00-config-runtime/` loads it:
`agents/uptime-triage/agent.yaml` is this file plus a `behavior` block, and
`python 00-config-runtime/demo.py` runs this agent from it, through a generic runtime that
contains no vacuum-bot code at all.

### The same agent, on the coding table an AP team maintains

The vacuum world is the source page's example and it is a toy. `real_world.py` is the same
`SimpleReflexAgent`, imported rather than reimplemented, on accounts payable coding.

Every invoice needs a general ledger account before it can be paid, and most finance teams
do that with a vendor table. It is fast, auditable, free, and right for the suppliers you
pay every month.

The tail is the cost centre. A vendor nobody has coded before falls through `rule_match`
and lands in a human queue, where an accountant reads the line item to work out whether
"professional services" means legal, consulting or contract engineering. Run it and the
three first-time vendors are exactly that case: all three read as professional services
from the name, and all three belong in different accounts.

The model is asked about the fall-through only. Every vendor the table knows is answered
by the table at the same cost as before, and an account outside the chart holds the
invoice rather than posting it somewhere that does not exist.

## What changed

`rule_match` -- a substring scan over a hand-written dict -- became `llm_rule_match`, one
small-tier model call mapping a percept onto an action name. Nothing else moved: the
`Percept` dataclass, `agent_function` running once per percept, no state between percepts,
no lookahead.

What stayed deterministic is the action set and the check that enforces it. The model
proposes a string; `if action not in self.available_actions` decides whether it counts.
A sentence, an invented actuator, or an empty string collapses to `no_op` before anything
reaches hardware. The legal actions live in code, not in the prompt. The prompt names them
too, but a prompt is a request and the check is the guarantee.

The check does not fire on these three percepts, because the model answered with a bare
action name all three times. `09-model-portability/fallback_report.py` counts how often it
does fire, and on which backend.

## What it costs

Determinism. `before.py` is a table lookup: same percept, same action, forever, auditable
by reading three lines of a dict. `after.py` can answer differently on two identical runs
-- the puddle recording is the proof -- and the rule behind any given action is written
down nowhere. A loop that was free and instant now carries network latency and a
per-percept cost. The validation block bounds what the agent can do. It cannot bound
whether the choice was any good.
