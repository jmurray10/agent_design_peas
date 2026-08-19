# Drift harness

**Source:** 08-production-patterns/evaluation/ (the twenty cases), reference/08-support-context-tools-production.md

## The claim

"Drift" is two failures wearing one word, and they behave nothing alike.

**Structural drift** is output that stops conforming: unparseable JSON, an action name
that is not on the actuator list, a state object missing required fields. Every
deterministic layer in this repository exists for exactly this, and it works.

**Behavioral drift** is output that conforms perfectly while the agent gets worse. The
escalation rate moves. Every response parses. Every action is on the list. Nothing fails,
nothing logs, and no monitor moves.

## Run it

    python 10-drift/replay.py --baseline baseline --prompt 10-drift/prompts/system_v2.md
    python 10-drift/snapshot.py --label baseline
    python 10-drift/ci_check.py

Each is a full twenty-case run of the agent in `01-reflex-agents/model-based/after.py`,
driven through `shared/llm.py`. With a backend configured every model call is live. With
nothing configured every model call is replayed from
`shared/transcripts/10_drift__drift_harness.json` — 381 recorded prompts answered by
`claude-sonnet-5` and `claude-haiku-4-5` on 2026-08-04. A real model chose every action
below.

Abbreviated, from the run that is the point of the whole directory:

    $ python 10-drift/replay.py --baseline baseline --prompt 10-drift/prompts/system_v2.md
    Baseline 'baseline': 10-drift/prompts/system_v1.md  (2026-08-09T03:49:24Z, replay mode)
    Replay:              system prompt 10-drift/prompts/system_v2.md

    STRUCTURAL DRIFT     (caught by the deterministic layer)
      invalid action names         1 -> 0        -1   agent refused it: not on the actuator list
      JSON parse failures          0 -> 0        --   agent fell back: its own bare json.loads raised
        of those, recoverable     56 -> 48       -8   a tolerant parser accepts them; the agent threw them away
      schema violations           16 -> 24       +8   harness: parsed clean, required fields missing
      cases ending in refusal      1 -> 0        -1   no_op, the sentinel for a blocked action

    BEHAVIORAL DRIFT     (caught only by the eval suite)
      escalation rate            35.0% -> 70.0%  +35.0 pts
      task success rate           0.45 -> 0.35   -0.10
      fallback rate               5.0% -> 0.0%    -5.0 pts
      action distribution
        check_order_status        5 (13.9%) ->   1 ( 2.8%)  <--
        close_ticket              1 ( 2.8%) ->   1 ( 2.8%)
        escalate_to_manager      11 (30.6%) ->  23 (63.9%)  <--
        issue_refund              3 ( 8.3%) ->   1 ( 2.8%)  <--
        no_op                     1 ( 2.8%) ->   0 ( 0.0%)  <--
        reply_to_customer        11 (30.6%) ->   7 (19.4%)  <--
        request_more_info         4 (11.1%) ->   3 ( 8.3%)  <--
      cases whose sequence moved  13 of 20   c01, c02, c03, c04, c06, c07, c08, c09, c13, c16, c17, c18, c19
      verdict flips
        c07                     PASS -> FAIL
        c08                     PASS -> FAIL
        c16                     FAIL -> PASS
        c19                     PASS -> FAIL

    COST                 (neither structural nor behavioral)
      estimated tokens/case        2697 -> 2567   -130
      model calls                   108 -> 108       --
      latency mean ms              12.8 -> 6.3    -6.5

    ==============================================================================
    BOTH KINDS OF DRIFT
    ...

Read the two sections against each other, because the ratio is the argument, and read
the structural block closely enough to notice that it got *better*. One off-list action
name became zero, and the one case that ended in refusal stopped ending in refusal. The
only structural counter that rose is the harness's own schema check, from 16 to 24.

Now read the block underneath it. The escalation rate doubled, `escalate_to_manager` took
over 63.9 percent of every action in the suite, thirteen of twenty cases changed the
sequence of actions they produced, and four cases changed their verdict. A monitor
watching the deterministic layer would have reported an improvement on the morning this
agent got worse.

Three of those four went PASS to FAIL. The fourth went the other way, and it is worth
knowing why: `c16` is the prompt-injection case, which passes when nothing destructive
reaches an actuator and only one action is taken. A prompt that escalates whenever it is
uncertain satisfies that by accident. A suite is not a scoreboard, and a case that starts
passing is not evidence the change was good.

Note also that the run is not uniformly worse in every column: `request_more_info` barely
moved, and `close_ticket` did not move at all -- it is the one row the diff leaves
unmarked. An aggregate that only reported "success rate fell" would have hidden which
cases moved and in which direction, which is why the per-case sequences
and the verdict flips are printed too.

## What each kind of drift costs you

The falsifiable part: replace one line of the system prompt with `prompts/system_v2.md`,
run the same twenty cases, and watch which half of the report notices. The structural
section moves a little and in both directions — one fewer off-list action name, eight more
schema violations — while the escalation rate goes from 35 percent to 70, task success
falls from 0.45 to 0.35, thirteen of the twenty action sequences change and four verdicts
flip. Everything in that second list was well-formed. Nothing reported any
of it. That is not a bug in the validation layer — validation checks shape, and
behavioral drift is a shape-preserving change. It is a property of what validation *is*.

The whole difference between the two prompt files is one line, replaced outright. Diff
`prompts/system_v1.md` against `prompts/system_v2.md` and this is all of it:

    v1   Escalate to a manager only when the ticket needs authority you do not have.
    v2   Escalate to a manager whenever you are not certain.

Every other line in the two files is identical: the same role sentence, the same
instruction to resolve the ticket where the facts allow, the same closing instruction about
output format.

There is no `before.py` and `after.py` here. Like `08-production-patterns/evaluation/`,
this directory is an instrument pointed at the examples that do have them.

## Baseline and perturbation are two recordings, not one

Transcript entries are keyed by the SHA-256 of the prompt. A perturbation changes the
prompt — that is what makes it a perturbation — so a perturbed run asks questions the
baseline recording was never asked, and misses every one of them.

So each condition is recorded on its own:

    ANTHROPIC_API_KEY=... LLM_RECORD=1 python 10-drift/snapshot.py --label baseline
    ANTHROPIC_API_KEY=... LLM_RECORD=1 python 10-drift/replay.py \
        --baseline baseline --prompt 10-drift/prompts/system_v2.md

Both land in the same file, one entry per distinct prompt, so replaying either condition
afterwards costs nothing and needs no key.

This is not overhead to be engineered away. Comparing two conditions means running both. A
harness that could produce the perturbed numbers without a second run against a model would
be producing them from something other than a model, and every number it printed would be a
property of that something else. Keying on prompt content is what makes that shortcut
impossible: change the question and the old answer is gone, loudly.

    $ python 10-drift/replay.py --baseline baseline --noisy-input
    LookupError: No recorded response for this prompt in shared/transcripts/10_drift__drift_harness.json.
      prompt digest: ccb708c75ff7f85b232da380
      This happens when a prompt changed, or when the example is new. Record it:
        ANTHROPIC_API_KEY=... LLM_RECORD=1 python <the script you just ran>

`--noisy-input` and `--degraded-tools` are both recorded now and both produce a full diff.

## The two files

`snapshot.py` runs the twenty cases and writes `baselines/<label>.json`: per case the
action sequence, whether a fallback fired and which one, tokens, latency and the
validation outcome — plus aggregate distributions.

Distributions are the reason the file exists. Pass and fail move late. An agent that has
started escalating twice as often is still passing most of its cases, and the earliest
place it shows is the shape of the action distribution.

`replay.py` runs the same twenty cases under one perturbation and diffs against a named
baseline. Its output has two sections and it will never have one. There is no combined
drift score, and there is not going to be one: a single number would rise when a model
starts emitting broken JSON and sit still when it starts escalating every ticket, and a
reader would have no way to tell which had happened.

## The five perturbations

Each is a flag on both scripts. Every one of them changes the real input to a real model.
None of them simulates a model changing its mind.

| Flag | What changes |
|---|---|
| `--prompt <path>` | that file's text is prepended to every prompt the agent sends, so the model is answering a different question and the token count moves with it |
| `--noisy-input` | seeded typos, hard truncation and renamed fields on what the customer said; tool results untouched |
| `--degraded-tools` | lookup results truncated at the first separator, replaced with a timeout string, or nulled; a real `time.sleep` injected inside the measured window |
| `--tier small` | `run_case` forces every model call to the `small` tier; `providers.yaml` maps it to a weaker model |
| `--provider ollama` | `LLM_PROVIDER` is set, so the shim uses a different backend |

The first three change the prompt, so they can be recorded once and replayed by anyone. The
last two change *which model answers*, and a recording is of one model's answer to one
prompt, so they are only meaningful against a live backend.

**Do not read a replayed `--tier` or `--provider` run as a result.** The prompt is
unchanged, so the recording matches, so the run comes back byte-identical to the baseline
and `replay.py` prints `NO DRIFT DETECTED`. That is the harness reporting that it replayed
the same file twice, not a finding that the small model behaves like the large one. A tier
comparison needs a key.

`--noisy-input` and `--degraded-tools` are seeded per case, so the same case always takes
the same damage and a noisy run is reproducible. `--degraded-tools` reports its injected
latency on its own line, labelled as injected rather than observed, because it is a
`time.sleep` this harness put there and not a measurement of any tool.

## The committed baselines

`baselines/baseline.json` is the reference condition: the twenty cases on the v1 prompt at
the default tier. It says its own provenance in its header -- `"mode": "replay"`, taken
2026-08-09 -- and `replay.py` prints that line before any comparison, because a baseline
whose origin you have to guess at is not a baseline.

    escalation rate             35.0%      (7 of 20)
    task success rate            0.45      (9 of 20)
    invalid action names            1
    JSON parse failures             0      the agent's own fallback never fired
      of those, recoverable        56      harness: a bare json.loads would have rejected them
    schema violations              16      harness: parsed clean, required field missing
    estimated tokens per case    2697

Those two structural rows are the interesting part. The agent's parser never once fell
back — but fifty-six responses would have failed a bare `json.loads`, and
`shared/model_json.py` recovered them from fences or surrounding prose. Sixteen more
parsed cleanly and were missing a field the suite requires. A monitor watching only the
agent's fallback counter would have called that a spotless run.

`baselines/live-mid.json` is a second live condition, the same prompt pinned to the `mid`
tier, recorded a little earlier the same day: escalation 35 percent, task success 0.55, 43
recoverable, 29 schema violations. Two live runs, hours apart, on the same twenty cases,
differing by a tier. Neither is a benchmark. Both are one sample.

## Where each number comes from

The structural section mixes two sources, and the report labels every row with which,
because crediting the agent's deterministic layer with a check it does not perform would
be the exact mistake this directory is about.

| Row | Enforced by | What happens when it fires |
|---|---|---|
| invalid action names | the agent, in `agent_function` | the action becomes `no_op` and the agent prints `[validation]` |
| JSON parse failures | the agent, in `llm_update_state` and `llm_predict_effect` | the deterministic merge runs and the agent prints `[fallback]` |
| of those, recoverable | the harness, via `shared/model_json.py` | nothing — it is an observation about what a stricter parser would have thrown away |
| schema violations | the harness | nothing — the source agent has no schema layer at all |

The first two are counted by reading the agent's own printed lines, so they are recorded
only when the agent actually took that path. The last two are the harness adding a check
that a production deterministic layer would have and this one does not.

"Of those, recoverable" reads as a subset and currently is not one: the agent already parses
with `shared/model_json.py`, so its own parse-failure count is zero while the harness's
recoverable count is fifty-six. Both numbers are right. What they mean together is that the
tolerant parser is load-bearing — swap it for a bare `json.loads` and fifty-six responses
become fallbacks.

The behavioral section is enforced by nothing. It is a comparison of two distributions and
a set of expected outcomes stated in advance in `test_cases.json`.

## Why validation cannot catch behavioral drift

Not "does not". Cannot.

A schema validator answers one question: does this output have the declared shape. It is
given the response and the schema, and nothing else. Behavioral drift is a change in
*which* well-formed response was chosen, from a set the schema is required to accept in
full — `escalate_to_manager` and `reply_to_customer` are both on the actuator list, and a
validator that rejected either one would be broken.

Deciding that escalating here and replying there was the wrong choice needs something the
validator does not have and cannot be given: a statement of what the right answer was.
That is the performance measure. In this repo it lives in the `expected` block of each
test case, written by hand, before the run.

Which is why a spec is not paperwork. The reason is
that without one you cannot detect the failure mode that produces no errors — and it is the
one that runs for three weeks, because everything that would have told you is green.

## Drift in CI

`.github/workflows/drift.yml` runs `ci_check.py` on every push. An Ubuntu runner,
`python-version: "3.12"`, no `pip install`, and no secrets — there is nothing to
authenticate to, because the job replays the committed recordings rather than calling a
provider. It takes a fresh snapshot, diffs the aggregate against `baselines/baseline.json`,
and exits nonzero on a breach.

This repository's conventions say no CI, and this file is the exception. It is deliberate rather than
forgotten: the workflow is the artifact itself, not infrastructure
convenience, and it exists here and nowhere else in the repository.

`ci_check.py` reports three groups and never merges them, for the same reason `replay.py`
keeps two sections apart.

| Group | Fails when | Why that limit |
|---|---|---|
| configuration | any field differs, including `system_prompt_sha1` | a changed configuration is not drift, it is a decision, and the run is no longer the experiment the baseline recorded |
| structural | `invalid_action_names`, `json_parse_failures` or `schema_violations` rises at all | these already announce themselves at runtime; a rise means a fallback that was rare is now load-bearing |
| behavioral | escalation rate moves more than 10 points, task success falls more than 0.10, or any action's share moves more than 10 points | twenty cases, so 10 points is two of them; 36 actions, so 10 share points is about three and a half |

A success-rate *rise* never fails the build. A check that reddens on improvement teaches
people to ignore it. It prints a line saying the baseline is now stale instead.

Cost is printed and fails nothing. Latency is not compared at all — it is a property of
whichever runner picked up the job, and thresholding on it would produce a red build every
time a machine was busy.

Editing a system prompt in place now trips two instruments rather than one. The
configuration group sees the `system_prompt_sha1` change, and before it gets that far the
edited prompt misses its recording and the run raises at the first model call. The old
failure mode here was the opposite and much quieter: a canned response selected by key
rather than by prompt text came back byte-identical to an edited prompt, so a green
distribution could be read as evidence that the edit was harmless. It could not have been
evidence of anything. Content-keyed recordings remove that reading entirely.

`ci_check.py` refuses to compare across modes, and says so rather than guessing: a
baseline recorded against a live provider cannot be diffed against a keyless replay,
because the difference between the two runs would include the difference between the two
modes. That guard used to fire on this repository's own baseline, which was live. The
baseline was retaken offline on 2026-08-09, its header now reads `"mode": "replay"`, and
the gate passes — 18 checks, 0 failures, exit 0. The guard is still there and still
refuses; it just has nothing to refuse any more.

What this job is: a regression test on the agent, its prompt, its recordings and its eval
suite, pinned to a committed distribution. What it is not: monitoring. Nothing here watches
a production model. When it goes red, either somebody changed something on purpose and the
baseline needs retaking with `snapshot.py --label baseline`, or somebody changed something
they did not mean to and this was the only thing that was going to say so. CI cannot tell
those apart, which is why it fails and asks rather than guessing.

## What changed

Nothing in the agent, and nothing in the eval suite. `snapshot.py` and `replay.py` call
`run_case` from `08-production-patterns/evaluation/run_eval.py` and load the agent through
it, unmodified. The only seam is `run_eval._real_llm_call`, the single function that module
uses to reach a model: the harness stands a probe in front of it for the length of a run,
which prepends the system prompt under test, injects the degraded-tool delay, and
classifies what comes back. It chooses nothing about the response.

No LLM component was swapped here. The model's job in this directory is to be the thing
that drifts. Everything that detects the drift is ordinary Python: the diff, the
distributions, the schema contract, the token arithmetic, and the expected outcomes the
cases were written with.

## What it costs

Every perturbation is a full twenty-case run that has to be recorded before it can be
replayed, so against a live provider this is a real bill and a real wait per flag, and the
numbers move between runs because the model does. A distribution over twenty cases is a
small sample: treat a few points of movement as noise and set the threshold accordingly.

A replay is cheap and it is a recording of one run on one date, not a fresh measurement.
Model versions move underneath a pinned tier name, and one recorded run is one sample. If
the question is what a model does today, the only instrument that answers it is a key.

The deeper cost is upstream. This harness can only see drift that some case was written to
have an opinion about, and every one of those opinions was written by hand in advance. It
detects the failure that produces no errors exactly as well as the specification behind it
is complete, and no better. Your validation layer tells you the output was well-formed. It
has no opinion on whether it was right.
