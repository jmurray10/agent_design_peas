# Component tier routing

**Source:** reference/04-learning-agents-before-after.md (the agent), shared/providers.yaml (the tiers)

## The claim

A tier is chosen per component, not per agent. `mixed_agent.py` takes the four-component
learning agent from `04-learning/q-learning/after.py` unmodified, runs the same seeded
scenario three times -- everything on frontier, everything on small, and each component on
the tier it asked for -- and prints where the tokens went in each. One row of that table
never moves: the critic runs on every interaction, asks for no tier, reaches no model, and
costs nothing under any price assumption, because it is arithmetic over observed outcomes.
That zero is read out of the agent's own source at runtime, so it is checkable rather than
claimed.

## Run it

    python 09-model-portability/tier-routing/mixed_agent.py

The script opens by counting model calls in the agent's source, which takes no model to do,
so this part runs anywhere:

    Agent:    LLMLearningAgent, loaded unmodified from 04-learning/q-learning/after.py
    Scenario: 20 interactions, random seed 7, identical in all three runs
    Provider: replay   (selected by shared/llm.py, not by this script)

    Where each component reaches a model -- counted in the agent's source, not asserted here:

      component            method(s) scanned                       llm_call( found
      ------------------------------------------------------------------------------
      performance element  act                                                   1
      critic               observe_outcome, _calculate_reward                    0   <-- NO MODEL AT ALL
      learning element     learn                                                 1
      problem generator    suggest_exploration                                   1

    MEASURED on this run, and safe to quote:
      - which tier each component asked for, taken from the calls it actually made
      - how many times each component ran, and how many of those reached a model
      - prompt and reply sizes, estimated at 4 characters per token (see estimate_tokens)
      - the number of llm_call( occurrences in each component's source

    ASSUMED, and not safe to quote as a price:
      - the cost units column. It is token counts times TIER_PRICE_UNITS, which this
        file assumes to be small=1, mid=5, frontier=25 per 1000 tokens. Those three
        numbers are stand-ins chosen for readable arithmetic, not a vendor quote, and
        they price input and output the same, which no provider does. No figure below
        is a bill.

    NOT MEASURED, and deliberately absent:
      - answer quality. Nothing below scores an answer. See the last section.

The three configurations then run, each printing a per-component table -- runs, model
calls, prompt and reply tokens, cost units -- with the critic's row reading `nothing` for
the tier it asked for and `NO MODEL` for the tier it was routed to, and zero everywhere
after that. That gap is closed. The recording exists, `python 09-model-portability/tier-routing/mixed_agent.py` runs offline to completion, and the critic row reads `NO MODEL / 20 runs / 0 calls / 0.00 cost` in every configuration -- which is the claim this directory makes and can now show rather than describe.

Two things to hold on to. Cost units are arithmetic over token counts and three assumed
weights; set the weights equal and all three configurations cost the same, so the spread
in that column is the assumption, not a finding.

And the replay does compare three models, which was not true when this was written.
Transcripts were keyed by prompt alone, so every tier read the same entry and all three
configurations replayed identical responses -- the paragraph here used to say so, and used
to conclude that a tier comparison is only meaningful live. Each configuration now records
into its own transcript and `shared/transcript.py` raises when a requested tier does not
match the recorded one, so the three configurations replay three models' answers. The
script reports `Steps that matched the first configuration: 11 of 40`, and the token totals
differ across the three rows because the models returned different text.

The separation took two goes, and the second was found by the check rather than by
reading. Every entry recorded before 2026-08-14 carried a null tier, and the tier check
skips a null rather than guessing -- so it had been silent on 1,466 entries. Backfilling
the tier from the model each entry already recorded made it live, and it immediately
refused this script: one prompt recorded at `mid` and replayed at `frontier`, which is a
tier comparison comparing one model against itself.

What has not changed is what that is worth: one run of one scenario is not a quality
measurement, live or replayed. Scoring a routing configuration needs a labelled eval set, repeated
runs, and a metric fixed before the run, none of which is in this file.

## What changed

Nothing in the agent. `04-learning/q-learning/after.py` is loaded by path and left exactly
as it is; `mixed_agent.py` installs a router in place of the name `llm_call` inside the
loaded module, which is how a call gets attributed to a component and re-routed to another
tier without touching the file. Attribution is exact rather than guessed -- when the router
runs, the frame underneath it is the component method itself. That indirection is also why
this example needs a transcript of its own. What stays deterministic is everything that was
already deterministic: the critic, the reward sort feeding the learning element, the
allowed-action guard, the least-used fallback under the problem generator, and the seeded
scenario that makes the three runs comparable at all.

## What it costs

Routing per component means the agent's behaviour now depends on three model choices
instead of one, and a regression in any of them looks the same from outside: slightly worse
answers. Pinning each component to the cheapest tier that passes today is how an agent
quietly gets worse when a provider ships a new small model. The mixed configuration is also
the hardest of the three to reason about in an incident, because no single model is
responsible for the output. The critic is the exception in this file as well: it has no
tier to regress.
