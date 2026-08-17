# Cooperative orchestration

**Source:** reference/05-multi-agent-systems-before-after.md

## The claim

Replacing hand-coded parsers with LLM agents does not by itself produce a system. `before.py`
breaks on the second document it is shown — the same two facts in a different layout — and
`after.py` puts that same document through all three orchestration patterns. Then it breaks on
purpose: the extractor returns a record that parses as valid JSON and looks entirely reasonable,
with the amount as `"$2,450.00"` rather than a number, and the pipeline halts at the extractor
before the validator is ever called. What stops it is a deterministic schema check between the
two agents, not a model noticing.

## Run it

    python before.py
    python after.py

`before.py` — document one parses, document two does not:

```
Document 1 -- the format the parser was written for
    Name: John Smith
    Amount: $1500.00
    Date: 2024-01-15
  extracted: {'name': 'John Smith', 'amount': 1500.0}
  errors:    []
  report:    Processed: John Smith for $1500.0

Document 2 -- the same two facts, a different layout
    INVOICE
    Bill to: Acme Manufacturing LLC
    Total amount: $2,450.00
    Terms: Net 30
  extract_data raised: ValueError: could not convert string to float: '2,450.00'
  The pipeline stopped at the first stage. No report was produced.
```

`after.py` — four runs over that second document:

```
schema validator: jsonschema (installed)

=== pattern: workflow -- sequential, contract checked at every hand-off ===
  gate  pipeline input -> extractor   contract ok
  call  extractor (tier=frontier)
[replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
  gate  extractor -> validator   contract ok
  call  validator (tier=mid)
  gate  validator -> formatter   contract ok
  call  formatter (tier=small)
  status: complete
  final:  {"action": "format_json", "output": {"status": "approve", "name": "Acme Manufacturing LLC", "amount": 2450.0}}

=== pattern: parallel -- three independent reads of the same document ===
  extractor  {"name": "Acme Manufacturing LLC", "amount": 2450.0, "confidence": 0.97}
  no gate ran: a fan-out has no agent N+1 to hold to a contract

=== pattern: routing -- classify, then dispatch to one handler ===
  routed_to: extractor

=== failure run: the extractor emits output the validator cannot accept ===
  gate  pipeline input -> extractor   contract ok
  call  extractor (tier=frontier)
  note  extractor output injected by the demo, no model call
  gate  extractor -> validator   CONTRACT VIOLATION
          amount: '$2,450.00' is not of type 'number'
          root: 'confidence' is a required property
  halted at:    extractor
  message:      extractor produced output that validator cannot accept
  never called: validator, formatter
```

Without `jsonschema` installed the built-in checker runs instead and the same two violations
read `amount: expected number, got string` and `confidence: required property is missing`. The
first line of output says which one you got.

The three successful patterns above are replays of `claude-opus-5`, `claude-sonnet-5` and
`claude-haiku-4-5` responses recorded on 2026-08-04, one model per tier. The failure run is
not a replay at all: its payload is injected by the demo, which is why it prints
`no model call` and why it halts identically with or without a key.

Three things had to change before this file survived a real model, and all three ran green
offline first.

`_evaluate` on the source page compares `result.get('confidence', 0) > 0.8`. That holds for
as long as the model returns a number, and `claude-opus-5` returned the string `"0.93"`,
which raises `TypeError` and takes the pipeline down. Note where: inside the agent's own
performance measure, which fires *before* the contract gate — the gate is exactly the thing
that would have rejected a string where a number belongs, and it was one call too late to
help.

Then the workflow halted on its first hand-off with `'name' is a required property`. The
extractor had returned a sensible record with the fields nested a level down, because the
prompt said "Return JSON" and never said which JSON. That is a contract failure
manufactured by not stating the contract. Each agent is now handed the next agent's input
schema and names the required fields in its prompt. The gate is unchanged — the difference
is that the model is now wrong when it misses a field, rather than unlucky.

Last, the failure run only failed offline. It selected bad extractor output through
`mock_key`, which every real backend ignores — and which replay ignores too, since prompts
are now matched by their own content. With a key set the extractor behaved itself and the
demonstration quietly stopped happening. It now injects the payload directly and halts
identically either way.

## What changed

The LLM replaced `extract_data`, the parser. Nothing else moved: sequencing, fan-out, the
dispatch table, the per-agent performance measures and the checks between stages are all
ordinary code.

The source config declares `validation_between_agents: true` and the source code never
implements it. Here `SCHEMAS` holds one input schema per agent and `_workflow` checks agent N's
output against agent N+1's schema before spending a token on the call. `jsonschema` when it is
installed, a built-in required-key and type check when it is not; the run prints which one ran,
and neither is a required dependency.

Two fixes to the source listing: `process` awaits on a worker thread so `_parallel` genuinely
overlaps instead of serialising behind a blocking client, and `run` raises on an unknown pattern
instead of returning `None`.

`config.yaml` is verbatim from the source page. Nothing loads it yet — see `00-config-runtime/`.

## What it costs

Three model calls where `before.py` made none, and the extractor runs at the most expensive
tier. Latency moves from microseconds to seconds. The output stops being reproducible: the same
invoice can extract as `2450.0` today and `"$2,450.00"` tomorrow, which is precisely the failure
run. The schema gate does not prevent that. It converts corrupt output into a halt, which is
cheaper to debug and more annoying to a user. A halt is all you get — there is no repair path
here beyond the `retry_once_then_fail` the config names.
