# Cross-model comparison

**Source:** reference/08-support-context-tools-production.md (the suite), shared/providers.yaml (the backends)

## The claim

If the model is a config field, then changing it is a change to `LLM_PROVIDER` and
`shared/providers.yaml` and to nothing else -- and the same twenty-case suite should run
against any backend that answers. `compare_models.py` runs Task 13's suite once per
configured backend and prints one row each. `fallback_report.py` attributes every fallback
to the deterministic layer that caught it, counted from the lines the agent itself prints,
so the counts are triggers rather than estimates. The number that moves between backends is
the fallback rate. The architecture around it does not move at all.

There is no `before.py` and `after.py` pair here. This directory measures the examples that
have them.

## Run it

    python 09-model-portability/compare_models.py
    python 09-model-portability/fallback_report.py
    python 09-model-portability/same_percept_three_tiers.py

Also `--provider`, `--case c01 c04 c16 c17` to keep a live run cheap, `--timeout`, and
`--json out.json` on either script. `fallback_report.py --all` shows every case rather than
only the ones a layer touched.

With no key, no server and no pip install, one row replays the recorded run and the other
four say why they are not available:

    $ python 09-model-portability/compare_models.py
    Cross-model comparison: 08-production-patterns/evaluation/run_eval.py
    Same suite, same agent, same expectations. Only the backend changes.

      will run  replay              always available; recorded real responses from shared/transcripts/, no key, no network
      not run   ollama              nothing is listening on http://localhost:11434
      not run   anthropic           ANTHROPIC_API_KEY is not set
      not run   gemini              GEMINI_API_KEY is not set
      not run   openai_compatible   OPENAI_COMPATIBLE_BASE_URL is not set

    running replay ...

    provider          model                         success fb rate  invalid  parse   p50 ms  tok/case
    --------------------------------------------------------------------------------------------------
    replay            (recorded; see shared/transcripts/)   9/20 = 45%    1.9%        2      0      2.8      2841
    ollama            not run                            --      --       --     --       --        --
    anthropic         not run                            --      --       --     --       --        --
    gemini            not run                            --      --       --     --       --        --
    openai_compatible not run                            --      --       --     --       --        --
    --------------------------------------------------------------------------------------------------

Set one key and the table has something to compare. Against Google, with
`GEMINI_API_KEY` set and nothing else changed:

    provider          model                         success fb rate  invalid  parse   p50 ms  tok/case
    --------------------------------------------------------------------------------------------------
    replay            (recorded; see shared/transcripts/)   9/20 = 45%    1.9%        2      0      2.8      2841
    gemini            2 models, listed below     9/20 = 45%    0.0%        0      0  10043.8      1402

Two rows, two vendors, one unedited suite, and the same success rate on both -- which is
worth exactly as much as the four-point gap it replaced, because twenty cases cannot
resolve either. The columns that do separate them are the other ones: the recorded run
needed a deterministic guard on 1.9 percent of calls and had two actions refused for not
being on the actuator list, and this Gemini run had neither. That is the number worth
watching, and it is the one the fallback report breaks down case by case.

`compare_models.py` used to ask the shim for a backend named `mock`. The zero-setup path
had been renamed `replay` and `shared/providers.yaml` has no `mock` entry, so that row
failed before it started and the table came back with no runnable row at all -- while
still exiting 0, which is how it survived. The script has been taught the new name and the
replay row is populated in the block above. What it did right even while broken is worth
keeping in mind: it printed the error and no numbers, rather than a row of zeroes.

The suite underneath it is the same command the table issues:

    $ python 08-production-patterns/evaluation/run_eval.py
    THE SIX METRICS
                                                                   maps to (PEAS)
      task success rate     9 / 20 = 45.0%                         performance measure
      tokens per task       2841 mean, 56822 total (estimated)     cost and scalability
      tool calls per task   1.80 mean, 36 total                    actuator efficiency
      tool error rate       2 / 36 = 5.6%                          actuator reliability
      latency (end to end)  26.9 ms mean, 11.8 ms p50              performance measure
      escalation rate       6 / 20 = 30.0%                         confidence calibration

      model calls            108
      fallbacks fired        2
        action_not_allowed          2

And the layer attribution, abbreviated:

    $ python 09-model-portability/fallback_report.py
      layer                         fired     of     rate
      action_not_allowed                2     36     5.6%   per action choices
      update_state_json_parse           0     36     0.0%   per state updates
      predict_effect_json_parse         0     36     0.0%   per effect predictions
      output_schema_validation        n/a    n/a      n/a   not on this agent's path
      all layers                        2    108     1.9%   per model call

      c02  FAIL  strong/hard  Second replacement arrives damaged, same courier and
            actions: reply > REFUSED > reply
            step 2: caught by action_not_allowed
              -> substitutes the no_op sentinel; the action is never executed

                   guard fired    no guard fired
      passed                 0                 9
      failed                 2                 9

Both of those runs replayed `shared/transcripts/08_production_patterns__evaluation__run_eval.json`,
which holds 108 prompts answered by `claude-haiku-4-5` and `claude-sonnet-5`, stamped
2026-08-04 UTC.

This directory compares backends over one twenty-case suite. For the same comparison over
every script in the repository -- 48 of 48 against Anthropic, 46 against the Hugging Face
router, 44 against Gemini, all at one commit on one day -- see "The same suite, three
vendors, one commit" in the root README.

## The post this feeds had its premise inverted

The tidy story would be: a weaker model returns more unusable output, the
deterministic layer catches it, and the agent never crashes. Running this suite against a
live Anthropic backend on 2026-08-03 did not tell that story.

The first live run scored 6 of 20 with a 74.1 percent fallback rate -- 10 refused actions
and 70 responses that would not parse as JSON, across 108 model calls. That reads as a
capability limit and is not one. The model was returning correct JSON wrapped in a markdown
code fence, and `json.loads` does not eat fences. With one deterministic unwrap step added
in front of the parse, the same suite against the same backend scored 9 of 20 with a 0.9
percent fallback rate: 1 refused action, 0 parse failures.

Both of those are single live runs on one day, not benchmarks. But the shape of the
correction is the finding. The deterministic layer earned its keep and did not earn it the
way the draft assumed: it was not compensating for a weak model, it was absorbing a
formatting habit, quietly, at a rate high enough to look like the model's fault. A fallback
rate is a property of the seam between two components. It is not a score for either one of
them, and read as one it will point at the wrong component.

## What these numbers are, and are not

Read this before quoting anything above.

With nothing configured, the suite replays recorded responses. **A real model chose every
action in that run, on the date stamped in the transcript.** It is not canned, not
authored, and not a simulation: the file holds the exact prompt, the model name, the date
and what came back. That is the difference between this repository and the version of it
that shipped hand-written responses, where the headline numbers were authored by whoever
wanted the demo to come out a particular way.

A replay is still not the same as a run. It re-serves recorded responses in the order they
were recorded, so it reproduces the responses rather than the run -- anything that shifts
the sequence of prompts shifts what comes back next. The replay above and a live Anthropic
run both read 9 of 20, and that agreement is worth no more than the disagreement it
replaced: neither is the other's re-run, and one recorded run is one sample on one day. Providers
move models under fixed names. **Run it yourself, against the model you actually plan to
ship, before you believe any of it.** The reproduce command is printed under every row for
exactly that reason:

    LLM_PROVIDER=anthropic python 08-production-patterns/evaluation/run_eval.py --json out.json
    PowerShell: $env:LLM_PROVIDER='anthropic'; python 08-production-patterns/evaluation/run_eval.py

Because transcript entries are keyed by the SHA-256 of the prompt, editing a prompt misses
its recording and raises instead of replaying. That is deliberate. A harness that answers a
question you stopped asking is worse than one that stops.

Backends this machine is not configured for are not skipped quietly and are not filled in
from memory. They print `not run` with the reason and the one thing that would fix it. **No
row in this repository was ever produced by a backend that did not run.** A missing number
is a finding; an invented one is a lie with a column header.

Two more limits. Tokens are estimated at four characters each because a real tokenizer
would be a dependency and this repo installs nothing; only the comparisons between rows,
which share one estimator, are sound. Latency is wall clock from this machine, so on replay
it measures a dictionary lookup and some Python, and on a live row it includes the network
and the provider's queue. It is not a property of the model either way.

## Where the fallback counts come from

Nothing here simulates a failure. Every layer counted existed before this task did, in code
this task did not modify:

| layer | lives in | catches |
|---|---|---|
| `action_not_allowed` | `01-reflex-agents/model-based/after.py`, `agent_function` | an action name that is not on the actuator list |
| `update_state_json_parse` | same file, `llm_update_state` | a state update that will not parse as JSON |
| `predict_effect_json_parse` | same file, `llm_predict_effect` | an effect prediction that will not parse as JSON |
| `output_schema_validation` | `00-config-runtime/runtime.py`, `05-multi-agent/orchestration/after.py` | the shape of a structured action -- **not on this agent's path** |

The agent prints a line when a guard fires. Task 13's harness captures those lines. This
script sorts them. The fourth layer is reported as absent rather than as zero, because a
guard that is not installed and a guard that never fired are indistinguishable in a total
and mean opposite things.

The two JSON-parse rows read 0.0 percent on the current recording, and that zero is the
whole story of the section above. Before the fence unwrap those two layers carried 70 of
the 80 fallbacks in the live run. After it they carry none. The layer that survives is
`action_not_allowed`, which fired twice on 36 action choices -- the model naming something
that is not on the actuator list, which no amount of parsing tidiness fixes.

The last section of `fallback_report.py` is the one the post is built on:

      passed with a guard fired    a model call came back unusable and the case still
                                   met its expected outcome
      failed with no guard fired   the output was well formed, on the actuator list,
                                   and wrong

Ten of the twenty cases in the run above are that second kind. No validation layer in this
agent catches them, because nothing in the code knows what correct is. They are caught by
the expected-outcome comparison in the eval suite -- a test a person wrote in advance,
which production does not have. **A fallback rate measures how often a model returns
unusable output, not how often it is wrong.** Two different numbers, and only one of them
has a guard behind it.

### The same percept, three models

`compare_models.py` asks which backend is better. `same_percept_three_tiers.py` asks the
question the thesis actually rests on: how much of the answer came from the model, and how
much from the harness around it.

It takes real percepts out of three agents' evaluation suites and runs each one through the
same agent three times, changing only the capability tier. Same prompt, same sensor
schemas, same actuator list, same validation on both sides.

    16 of 17 percepts produced the same action from claude-haiku-4-5,
    claude-sonnet-5 and claude-opus-5.

On those sixteen the architecture decided the outcome. The percept, the actuator list and
the prompt had already narrowed the question to one defensible answer, and the cheapest
model found it. The remaining one is where capability bought something, and knowing which
one is the point -- it is a shorter list than it feels like.

This read 13 of 17 until four eval percepts were rewritten. Each of those four carried a
detail that argued for a second action at the same time the case claimed to isolate a
first, and measured live each split its own recorded samples three-to-two or four-to-one.
The rewrite took the fighting detail out rather than moving the expectation to match the
model. Cross-tier agreement rising from 13 to 16 is the same fact seen from another angle:
three of the four disagreements between a cheap model and an expensive one were the
percept being ambiguous, not the cheap model being wrong. That is a result about the
harness, which is what this script is for.

No row can show a model choosing an action that does not exist, or returning arguments
that fail an actuator schema. Not because the models are good, but because the
deterministic halves refuse it before a caller sees it, identically for all three.

One implementation detail that turned into a finding. Transcript entries are keyed by
prompt content and hold one model each, so running the same prompt at three tiers made
each recording overwrite the last, and the agents afterwards replayed a tier they never
asked for. The tier check in `shared/transcript.py` caught it rather than serving it,
which is exactly the failure that check was written for. Each tier now records into its
own transcript.

## What changed

Nothing in the agent, nothing in the eval suite, nothing in the shim. `compare_models.py`
runs `08-production-patterns/evaluation/run_eval.py --json` once per backend in its own
process -- a fresh process because the shim resolves its provider once and caches it, and
because a row produced by a command a reader can type by hand is a row a reader can check.
Model names are resolved through `shared/providers.yaml` for the tiers the run actually
asked for, rather than typed into this directory, so the table cannot claim a model the
shim would not have used. Backend detection asks the same questions `shared/llm.py` asks:
is anything listening on the Ollama port, is `ANTHROPIC_API_KEY` set and the SDK
importable, is `OPENAI_COMPATIBLE_BASE_URL` set.

Neither script contains an `llm_call`. The comparison harness measures models; it does not
consult one.

## What it costs

A live comparison is n backends times twenty cases times three model calls, paid in tokens
and wall clock every time you want a current number -- the argument against running it in
CI on every commit. Replay costs nothing and buys less: it is one recorded day. Subprocess
isolation costs a process launch per row. The bigger cost is what the table cannot tell
you: two backends with the same fallback rate can differ entirely in how sensible their
permitted answers were, and that shows up only in the success column, which is only as good
as the twenty cases behind it.
