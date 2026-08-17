# Context engineering

**Source:** reference/08-support-context-tools-production.md

## The claim

Context management is a policy problem, not a model problem. Both scripts here run
a deterministic policy — a budget check, a fixed retention window, a schema, a
character cap — and hand the LLM exactly one job inside it: writing prose short
enough to keep. Run them and you can watch the policy operate on real model output:
a summary is accepted only after code has confirmed it is shorter than what it
replaced, and a finding is written into the state document only after code has
clamped it to 120 characters.

This directory has no `before.py` and `after.py` pair. There is no classical
algorithm being upgraded here; these are the two context strategies the source page
describes, built so the numbers behind them can be checked.

Neither script needs an API key. With no backend configured they replay what
`claude-sonnet-5` returned to these exact prompts on 2026-08-04, from
`shared/transcripts/`. The summaries and notes below are a model's work, recorded.
They are also a single run: rerun them live and the model will write different prose,
the estimates will move, and the ratio will not be the one printed here.

## Run it

    python 08-production-patterns/context/compaction.py
    python 08-production-patterns/context/structured_notes.py

`compaction.py` grows a research agent's history two tool results at a time and
compacts whenever the estimated token count crosses the budget:

    cycle  5  before: 10 entries, est.  1266 tokens
              after:  10 entries, est.  1266 tokens   under budget, untouched
    cycle  6  before: 12 entries, est.  1522 tokens
              after:   6 entries, est.  1220 tokens   COMPACTED
              recent window preserved verbatim: True
              summary entry: **Summary of Search Results:**

    1. **Loss Runs (AUTO-2024-001, 3yr):** 3 claims over 3 y...
    ...
    cycle 11  before: 12 entries, est.  1630 tokens
              after:   6 entries, est.  1178 tokens   COMPACTED
              recent window preserved verbatim: True

    Compactions fired: 3
    Final history: 8 entries, est. 1428 tokens, budget 1500
    Without compaction the same run would have ended at est. 3026 tokens.

`structured_notes.py` carries the same five-step ACORD batch job twice, once as the
`<agent_state>` document from the source page and once as raw accumulated history,
and counts both after every step:

      step   state doc   raw history   raw / state
         1         171           298          1.7x
         2         268           560          2.1x
         3         347           836          2.4x
         4         405          1008          2.5x
         5         577          1190          2.1x

Every token number either script prints is an estimate from a character-count
approximation — characters divided by four — not a tokenizer. It runs low on code,
JSON, and identifiers. Both sides of the comparison are measured the same way, which
is what makes the ratio worth reading even though the absolute figures are
approximate. The ratio itself is a real measurement of these two representations of
this run on this machine, and nothing more: the tool outputs are invented for the
demo, and writing terser ones would shrink the gap. So would a terser model. The
model that produced the state document above wrote thirteen findings by step five,
and every one of them is a line somebody has to keep paying for.

## Where the deterministic layer is visible in this run

`structured_notes.py` prints the 120-character clamp doing its job. The model's
eighth finding arrives longer than the cap and is stored truncated:

    <finding>HOME-2024-003 loc_address mismatch: declarations 'Unit 4B' vs application 'Unit 4D', unresolved since 2025-02-11 clai...</finding>

Code decided that, not the model. The document is rendered from a validated dict, so
it cannot be malformed or lose a section however the model answers.

`compaction.py`'s equivalent guard — reject a summary that is empty or no shorter
than the entries it replaces, and substitute a deterministic manifest — did not fire
on this recording. All three summaries came back shorter, and all three were kept. The
old hand-written mock responses included one that was deliberately useless so that the
fallback ran on every offline run; that was a person demonstrating a code path, not a
model failing. The guard still runs on every compaction. There is now no offline run
in which you can watch it catch something, which is a fair trade for the summaries
being real.

## Context editing

The source page's third strategy is API-side. Anthropic's Messages API can clear
stale tool results automatically as a session approaches the context limit:

    context_editing = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"input_tokens": 100000},
        "keep": 5,
        "clear_at_least": {"input_tokens": 15000},
        "exclude_tools": ["get_risk_factors"],
    }

There is no script for it here on purpose. A local reimplementation would be a
different mechanism wearing the same name — the real thing runs server-side, decides
against the provider's own tokenizer, and mutates the conversation the API holds.
Reimplementing that locally teaches the wrong lesson. Read it as the same policy as
`compaction.py` moved behind the API boundary: a trigger threshold, a keep-recent
window, and an exclusion list, all deterministic, with no summarization step at all.

**On the 84 percent figure.** Anthropic publishes an 84 percent token reduction for
long sessions using context editing. That number is quoted in the source page for
this directory and is theirs, not ours. This repository did not measure it, cannot
measure it offline, and nothing in this directory reproduces it. Keep it separate
from the `raw / state` ratio above, which this directory does measure, on your
machine, with the approximate estimator described above.

## What changed

In `compaction.py` the LLM replaced one component: the summarizer. The budget check,
the retention window, the decision to compact, and the check that the summary is
genuinely shorter than what it replaced stay deterministic.

In `structured_notes.py` the LLM replaced the note writer — deciding which line of a
900-character tool dump is worth carrying forward. It never touches the document.
Code renders the XML from a validated dict, so the document cannot be malformed or
lose a section. The model proposes a JSON patch; code validates the schema, clamps
every entry to 120 characters, and applies it.

Fix to the source: `compact_context` had no guard for a history shorter than the
recent window, where `older` is empty and the function grows the context it was
asked to shrink. It now returns unchanged.

## What it costs

Both scripts add a network round trip on the exact turn the agent is already at its
context limit — the worst moment to be waiting. Both are lossy by design, and the
loss is silent: nothing tells you later which finding the summarizer dropped, or
what was on the far side of the 120-character clamp. Both are non-deterministic, so
two identical runs can retain different facts, which makes a failure that depends on
a dropped detail very hard to reproduce. Replay hides that last cost, because a
recording retains the same facts every time; a key does not. The deterministic
fallbacks bound the damage; they do not undo it.
