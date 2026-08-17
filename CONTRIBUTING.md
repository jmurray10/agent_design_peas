# Contributing

This repository is MIT-licensed and contributions are welcome, but it is not a
library and the useful contributions are a different shape from the usual ones.

Its whole value is that every claim it makes ships with code you can run to check.
So the most valuable thing you can send is evidence that a claim does not hold.

## The best contribution: a claim that does not reproduce

Every example states a falsifiable claim in its README and gives you the command
that demonstrates it. If you run one and get something different, that is worth an
issue more than anything else here.

That is not a formality. Several defects in this repository were found exactly that
way, and each one is recorded in the commit that fixed it:

- Two agents declared a state schema and had no way to write to it, so both behaved
  like reflex agents while their configs said otherwise. Nothing failed.
- The evaluation suite reset state once per suite rather than once per case, so a
  stateful agent was being scored on the order its neighbours sat in the file.
- Three sequence tests asserted an answer their own inputs did not determine, and
  passed or failed by sampling.

If you find the next one, please say so.

## Other things worth sending

- **A model that behaves differently.** Every recorded response in
  `shared/transcripts/` is one model, on one date. If a current model, or a
  different provider through `shared/providers.yaml`, produces a materially
  different answer to the same prompt, that is a finding about the claim rather
  than about your setup.
- **A number that has moved.** Timings and success rates are measured on the
  machine and model that ran them. If a figure in a README no longer matches what
  the script prints, say which figure and what you got.
- **A defect in the deterministic half.** The algorithms are meant to be
  unchanged from their source pages. If A star expands a different number of nodes,
  or the CSP solver is not the same object `before.py` uses, that breaks the
  central argument and is the most serious kind of bug here.
- **Corrections to the prose.** Overclaiming is the failure mode this repository is
  most exposed to. If a sentence says more than its evidence supports, that is a
  bug.

## Things that will probably be declined

- **A new framework or abstraction layer.** There is deliberately no framework
  here. Twenty examples that each run on their own are the point.
- **A UI for the before-and-after pairs.** Their entire demonstration is two
  commands printing next to each other, and a web dependency breaks the rule that
  every `before.py` runs on a fresh machine with nothing installed.
- **Mock or stub responses.** Canned responses were removed for a reason and the
  reason is written up in the README. There are two modes: a live call, or a replay
  of a recorded real response. Nothing invents a third.

## The rules any change has to keep

1. **Every `before.py` runs with zero setup.** Python 3.10 or newer, standard
   library only, no key, no network, no install. Check it in the container:
   `docker run --rm --network none peas python <path>/before.py`.
2. **Nothing invents a model response.** Backend configured means a live call.
   Nothing configured means replaying a recording. No recording means raising
   loudly with instructions. There is no third branch.
3. **One entry point per kind of model call.** Text goes through
   `shared/llm.py`, embeddings through `shared/embeddings.py`. Nobody writes their
   own client, and model names live in `shared/providers.yaml` and nowhere else.
4. **Never present a cited number as something measured here.** Published figures
   are attributed to their source. Measured figures say what ran them and when. If
   you cannot verify a number, leave it out.
5. **No emoji.**

## Before you open a pull request

    docker build -t peas .
    docker compose run --rm verify      # every example, offline, no key
    docker compose run --rm ci          # the drift gate

If you touched a model call, a guard or a parse, run it against a live model too:

    LLM_PROVIDER=gemini GEMINI_API_KEY=... docker compose run --rm verify-live

That is the one that catches things, because offline every model call replays a
recording and so agrees with the code that reads it. Nothing about a live run is a
gate -- it costs money and a model may answer differently twice -- so read its
failures rather than counting them.

`verify` should report every script passing. If you changed a prompt, a schema or
a percept, the recordings for it no longer match and you will need to re-record
against a live model:

    ANTHROPIC_API_KEY=... LLM_RECORD=1 python <the script you changed>

That failure is deliberate. Recordings are keyed by the content of the prompt, so
a changed prompt misses its recording rather than silently replaying an answer to
a question it no longer asks.

Tell us what you ran and what it printed. A claim about behaviour without the
output that supports it is the thing this repository exists to argue against.
