# Model-based reflex agent

**Source:** reference/01-reflex-agents-before-after.md

## The claim

A model-based reflex agent survives having three of its five steps replaced by an LLM,
because the two steps that carry its guarantees are not model calls. Every `json.loads` in
`after.py` sits under a hand-coded fallback, and every action name is checked against the
actuator list before it runs. The recorded run parses cleanly from end to end, and that is
itself the finding: what used to count as a parse failure here was a fence, not a fault.

## Run it

    python before.py
    python after.py
    python 01-reflex-agents/model-based/real_world.py

`before.py` — five percepts over a two-cell vacuum world. State accumulates. The last
percept has no `location` in it, so the action comes from memory:

    Model-based reflex agent, two-cell vacuum world.
    Every decision below is a hand-coded conditional. No model is involved.

    step 1
      see:   {'location': 'left', 'status': 'dirty'}
      do:    suck
      state: location=left  status=dirty  visited={left}  left_cleaned=True  (4 keys)
    ...
    step 5
      see:   {'status': 'dirty'}
      do:    suck
      state: location=right  status=dirty  visited={left, right}  left_cleaned=True  right_cleaned=True  (5 keys)

    The agent started with an empty state and ended with 5 keys.
    Step 5 had no location in the percept. The action came from memory.

`after.py` — the same architecture on a support ticket. The state blocks it prints are
long, so they are cut here at the fields that move:

    Model-based reflex agent, customer support ticket.
    Same architecture as before.py. update_state, rule_match and predict_effect
    are model calls now. Validation and JSON parsing are not.

    See: {'message': 'My order hasnt arrived and its been 2 weeks'}
    [replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
    State: {
      "customer": {
        "issue_type": "order_not_arrived",
        "status": "pending_customer_response",
        ...
    Do: request_more_info

    See: {'order_lookup': 'Order #4521 - shipped 12 days ago, stuck in transit'}
    State: {
      "customer": {
        "order_info": { "order_id": "4521", "shipping_status": "stuck in transit", ... },
        "status": "status_checked",
        ...
    Do: check_order_status

    See: {'message': 'Third time this has happened. I want a refund.'}
    State: {
      "customer": {
        "sentiment": "angry",
        "status": "escalated_to_manager",
        ...
    Do: escalate_to_manager

    Any [fallback] line above is a model call that returned unparseable JSON.
    The run continued on hand-coded logic instead of raising.

There is no `[fallback]` line above it. Every state response in this recording is real
`claude-sonnet-5` output from 2026-08-04, and every one of them arrived wrapped in a
```json fence. A bare `json.loads` rejects all of them. `shared/model_json.py` unwraps the
fence before the parse sees it, so they succeed, and the hand-coded merge path stays where
it belongs — underneath, unused on a good run.

That distinction was not free. This example originally treated a fence as a failure, and
against `claude-sonnet-5` that produced a 100 percent fallback rate — six calls, six
"unparseable JSON" lines, an agent completing the ticket entirely on its merge path while
appearing to work. Across the twenty-case suite in `08-production-patterns/evaluation/`
the same bug reported a 74 percent fallback rate and a 30 percent success rate. Unwrapping
the fence first took the fallback rate on that suite to 0.9 percent.

A fallback that fires on everything is not a safety net. It is the agent, and the model is
decoration.

The fallback is still reachable, and nothing here is arranged so that it fires. Truncated
output still raises — `model_json` unwraps, it does not repair — which is what a response
cut off at the token cap looks like. Entries are keyed by the SHA-256 of the prompt, so
editing any prompt in this file misses its recording and raises rather than replaying an
answer to a question it no longer asks.

### The same agent, on a loan file

`real_world.py` runs `ModelBasedReflexAgent` over a loan application instead of a floor.
Documents arrive one at a time over days and nobody ever sees the whole file at once,
which is the partially observable problem the vacuum stands in for.

The model replaces `update_state`: a processing note becomes a change to what is on file.
The policy over that state stays a rule table anyone can read.

Watch the passport. It is identity evidence, it is the document the file is waiting for,
and it is expired -- so it verifies nothing and the agent keeps waiting. A rule table
keyed on "passport received" marks the requirement satisfied and sends an incomplete file
to underwriting.

## What changed

`update_state`, `rule_match` and `predict_effect` became model calls. Nothing else moved:
the agent still folds a percept into state, matches a rule against state, then predicts
the effect of its own action. Two things stayed deterministic and both are load-bearing.
The action returned by `llm_rule_match` is checked against `available_actions` before
anything runs, and a name that is not on the list becomes `no_op` — a refusal, not an
action. Every `json.loads` sits under a hand-coded fallback: `llm_update_state` merges the
raw percept into the previous state, `llm_predict_effect` keeps the state it already had.
`config.yaml` is the support-bot PEAS spec from the source page, verbatim.
`00-config-runtime/` drives this agent from it, `state` block included — that block is what
tells a runtime knowing nothing about support tickets to keep state between turns.

## What it costs

Three model calls per percept instead of three dict operations, so a turn costs latency
and tokens where it cost microseconds. State stops being reproducible: the same percept
sequence can yield different state, which makes a failed ticket hard to replay. The
fallbacks keep the run alive but they degrade it — after a parse failure the state holds a
raw percept string where a normalized field belonged, and the effect of an action can go
unrecorded. Broken JSON stops being a crash and becomes silent drift.
