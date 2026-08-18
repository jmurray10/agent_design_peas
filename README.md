# Classical AI Agents, and What the LLM Actually Replaced

[![drift gate](https://github.com/jmurray10/agent_design_peas/actions/workflows/drift.yml/badge.svg)](https://github.com/jmurray10/agent_design_peas/actions/workflows/drift.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#run-it)
[![before.py: no key, no install](https://img.shields.io/badge/before.py-no%20key%2C%20no%20install-brightgreen.svg)](#on-your-machine-with-nothing-installed)
[![agents](https://img.shields.io/badge/agents-9%20across%205%20architectures-8a2be2.svg)](#the-five-architectures-each-with-a-live-endpoint)

Classical agent architectures are not obsolete. An LLM upgrades one component inside them
and leaves the architecture standing. Every claim in this repository ships with both
halves running side by side, so you can check it rather than take it.

Nothing here is a framework. It is twenty examples, each one an argument you can run.

## Start here

    git clone https://github.com/jmurray10/agent_design_peas
    cd agent_design_peas
    python 01-reflex-agents/simple/before.py

That is a classical reflex agent, running. No API key, no `pip install`, no network, no
virtualenv -- Python 3.10 or newer and nothing else.

    python 01-reflex-agents/simple/after.py

The same architecture with one component swapped for a model call. It also runs with no
key: where a model would be asked, the repository replays what a real model actually
answered on a recorded date.

Open the two files side by side. The difference between them is the entire argument, and
it is about forty lines.

Or, if you would rather click than read -- nine agents as nine HTTP services, each with a
page you can send real requests from:

    docker compose up -d peas
    # open http://localhost:8079

That needs no key either. Everything below is detail.

## What is in here

Every numbered directory is one architecture or one production concern, drawn from a
source page kept unchanged in `reference/`. Most contain a `before.py` and an `after.py`:
the classical algorithm, and the same architecture with exactly one component swapped for
a model call.

Five of those pairs use the textbook problem the source page shows -- a vacuum world, an
8-puzzle, a grid world, tic-tac-toe -- because readers arrive having seen that code and it
is the clearest way to watch the mechanism. Each of those five now has a `real_world.py`
beside it: the same imported algorithm on a problem someone is paid to solve. A card
invoice coding queue, a loan file, a stuck order, a maintenance schedule, a price war.
The toy explains the mechanism and the companion shows it is not a toy.

| Directory | What it demonstrates | Source page |
|---|---|---|
| `shared/` | One LLM entry point, one for embeddings, five backends, tiers as capability hints | — |
| `00-config-runtime/` | Nine agents, all five architectures, one runtime with no agent-specific code | `reference/00-overview-classical-to-llm-agents.md` |
| `01-reflex-agents/simple/` | The rule table becomes a model call. Nothing else moves, and `real_world.py` runs the same class on an AP coding queue | `reference/01-reflex-agents-before-after.md` |
| `01-reflex-agents/model-based/` | State tracking survives because the parse has a fallback, and `real_world.py` carries a loan file instead of a floor | `reference/01-reflex-agents-before-after.md` |
| `02-goal-based/search/` | A\* returns the same plan whether the state came from a tuple or from prose, and `real_world.py` replans a stuck order | `reference/02-goal-based-agents-before-after.md` |
| `02-goal-based/csp/` | The solver is imported, not reimplemented, and a script proves it | `reference/02-goal-based-agents-before-after.md` |
| `03-utility-based/value-iteration/` | Three ways to put a model near an MDP, one of which keeps the guarantee, and `real_world.py` solves a maintenance policy | `reference/03-utility-agents-before-after.md` |
| `04-learning/q-learning/` | Four components, and the one that must never be a model | `reference/04-learning-agents-before-after.md` |
| `05-multi-agent/adversarial/` | Alpha-beta prunes the same tree; the model only scores leaves, and `real_world.py` prices against a competitor | `reference/05-multi-agent-systems-before-after.md` |
| `05-multi-agent/orchestration/` | The contract between agents is the architecture | `reference/05-multi-agent-systems-before-after.md` |
| `06-parallelization/` | Parallelism has a floor, measured on the machine you run it on | `reference/06-support-gpu-parallelization.md` |
| `06-parallelization/hf-space/` | The same floor on real CUDA, as a Hugging Face Space | — |
| `07-nlp-foundations/` | Similarity, retrieval and routing by hand, then against real embeddings | `reference/07-support-nlp-foundations.md` |
| `08-production-patterns/context/` | Compaction and structured notes, with the token counts | `reference/08-support-context-tools-production.md` |
| `08-production-patterns/tools/` | The five tool principles, implemented twice | `reference/08-support-context-tools-production.md` |
| `08-production-patterns/evaluation/` | Twenty cases, six metrics, and the judge/critic distinction | `reference/08-support-context-tools-production.md` |
| `08-production-patterns/permissions/` | Authorization is not a prompt problem | — |
| `09-model-portability/` | The same suite across backends, counting which layer caught what | — |
| `09-model-portability/tier-routing/` | Per-component tiers, and a critic row reading "no model" | — |
| `10-drift/` | Structural drift versus behavioral drift | — |
| `10-drift/critic-experiment/` | A central claim tested as a measurement instead of asserted | — |
| `colab/` | Three notebooks: the claim itself, a GPU measurement, the nine agents | — |
| `.github/workflows/` | The drift snapshot as a build gate, no secrets, 36 lines | — |

`reference/` holds the nine source pages the code is drawn from, unchanged.

### The five architectures, each with a live endpoint

`00-config-runtime/` is the spine. An agent there is a directory of YAML, prompts and JSON
Schema; `runtime.py` contains no agent-specific code, and `demo.py` proves it by parsing
that file for every name belonging to a specific agent. Adding an agent adds a
directory, not a branch.

| Architecture | Agents |
|---|---|
| simple reflex | `claims-intake`, `data-contract`, `safety-signal`, `uptime-triage` |
| model-based reflex | `aml-alert`, `support-bot` |
| goal-based | `trial-scheduler` -- a CSP, where the solver proves a request unsatisfiable rather than answering anyway |
| utility-based | `claim-reserve` -- where under-reserving and over-reserving both cost money, asymmetrically |
| learning | `triage-tuner` -- which has an explore action, and whose critic is arithmetic rather than a model call |

All nine are real working domains: AML alert disposition, insurance reserving,
insurance first-notice-of-loss triage, data contract review, adverse event intake, support
ticket routing, clinical trial monitoring visits, customer support enquiries, and on-call
alert triage. Each
carries the constraint its domain imposes. `aml-alert` can recommend a suspicious activity
report and has no actuator that files one. `safety-signal` routes a report for human review
and has none that assesses causality. `claims-intake` has none that pays money.
`trial-scheduler` proposes an assignment and books nothing. That boundary is the actuator
list, which is deterministic code, rather than a sentence in a prompt asking the model to
behave, and each agent's generated documentation states it.

The ninth is `uptime-triage`, on-call alert triage, and it is the honest case for the
cheapest model in the repository. The decision runs on every alert of every day, so a
frontier model on each one is not a design anyone can afford -- which is a constraint of
the domain rather than a compromise, and the reason its config asks for the `small` tier.

The textbook vacuum-world agent used to sit in that slot. Its before-and-after pair is
still in `01-reflex-agents/simple/`, where the source page's code belongs, and it is still
the clearest demonstration of what an LLM changes and what it leaves alone. What it could
not demonstrate is why anyone runs a reflex agent in production, which is volume.

## Run it

### On your machine, with nothing installed

    git clone https://github.com/jmurray10/agent_design_peas
    cd agent_design_peas
    python 01-reflex-agents/simple/before.py

No API key. No `pip install`. No network. No virtualenv. Python 3.10 or newer and nothing
else, and that is true of every `before.py` in the repository.

    python 01-reflex-agents/simple/after.py

Same architecture, same loop, one component swapped for a model call. It also runs with no
key, because of recorded transcripts.

### In a browser

Three notebooks in `colab/`. Each opens in Colab and clones this repository in its first
cell, so they need nothing installed on your machine.

    colab/oscillation.ipynb   the claim itself: before.py and after.py side by side
                              across five architectures. No key, no GPU.
    colab/gpu_floor.ipynb     the parallelization floor on whatever GPU Colab gives you.
                              No key, no model call. Arithmetic, timed.
    colab/agents_live.ipynb   nine agents through one runtime, replaying by default and
                              live if you supply your own key.

The GPU notebook exists because this repository cannot otherwise check its own most
quotable material: the CUDA figures it quotes are published ones, and the machine it was
written on has no GPU. `06-parallelization/hf-space/` runs the identical module as a hosted
Space if you would rather click than run.

### In a container

    docker build -t peas .
    docker run --rm --network none peas

`--network none` is the point: the container has no network interface, so the offline path
is demonstrated rather than claimed.

    docker compose run --rm replay      offline, nine agents through one runtime
    docker compose run --rm live        the same command against a real model
    docker compose run --rm verify      every example, offline, as a smoke test
    docker compose run --rm verify-live the same set against a live model
    docker compose run --rm ci          the drift gate, exactly as CI runs it

`verify` reports 49/49 offline inside the container. `verify-live` runs the same scripts
against whatever `LLM_PROVIDER` names, and is the one that finds things: it needs a key,
takes tens of minutes, and costs real tokens, because the suite is roughly 1,700 model
calls. It is separate from `verify` rather than a flag on it because the two answer
different questions, and only the offline one can be a gate.

### The same suite, three vendors, one commit

Measured on 2026-08-14, at commit `a57fb52`, from one `--no-cache` image, on one machine.
`10-drift/ci_check.py` is excluded from the denominator: it refuses a live backend without
`--allow-real` on every vendor, by design, because the drift gate exists to replay and
pointing it at a model is a category error rather than a failure.

| backend | passed | wall clock | model behaviour | transport |
|---------|--------|------------|-----------------|-----------|
| `anthropic` | **48 / 48** | 89.2 min | 0 | 0 |
| `openai_compatible` | 46 / 48 | 39.3 min | 2 | 0 |
| `gemini` | 44 / 48 | 58.9 min | 2 | 2 |

Read this as portability, not as a ranking, and there are three reasons why. The prompts
here were written and iterated against Claude, which is most of the top row. The two
`gemini` transport failures are TLS resets from the provider, nothing to do with the model
or the code. And a vendor is three models, not one -- each backend maps `small`, `mid` and
`frontier` in `shared/providers.yaml`, so the `openai_compatible` row is Llama 3.1 8B, Llama
3.3 70B and DeepSeek V3 answering different calls in the same run.

What the four model-behaviour failures actually are is the interesting part, and none of
them is a crash. Both non-Claude backends fail the two sequence cases, where the claim is
that conversation history changes the answer and those models answer the primed arm exactly
as they answer the cold one. Llama fails `orchestration/after.py` because a contract check
rejected a hand-off, which is the guard working. Gemini fails `tools/compare.py` by spending
its whole token budget reasoning and returning no answer text, which the shim now says in
those words instead of raising an IndexError.

None of that is visible offline, and neither were the three defects these sweeps found on
earlier commits -- written up under "What running it live changed" below. Offline every
model call replays a recording and therefore agrees with the code reading it, which is what
makes the offline suite a usable gate and a poor test.

One number here is a measurement of the network rather than of anything in this repository:
an earlier attempt at the `openai_compatible` row returned 21 of 49 because the router's
upstream rate-limited a burst and answered with HTML error pages. Run alone rather than
alongside the other two sweeps, the same commit returned 46. A live suite measures the day
it ran on, which is the argument for keeping the offline one as the gate. The image carries the optional
dependencies -- `anthropic`, `pyyaml`, `jsonschema`, `numpy` -- so it exercises the real
path rather than the fallbacks; every `before.py` still needs none of them. An API key is
passed at run time and never enters a layer, which is why `.env` is in `.dockerignore`
alongside everything else that would be recoverable from a published image.

### As nine HTTP services

    docker compose up -d peas

Then open <http://localhost:8079>. That is the index: every agent grouped by architecture,
what each one decides, and a link to its documentation. It is generated from the same
`agent.yaml` files the runtime reads, so an agent that is added appears on it without
anyone editing a page. `:8079/agents.json` is the same thing for a script.

    :8079 the index       :8080 aml-alert       :8081 claim-reserve
    :8082 claims-intake   :8083 data-contract   :8084 safety-signal
    :8085 support-bot     :8086 triage-tuner    :8087 trial-scheduler
    :8088 uptime-triage

One container named `peas`, one process per agent, supervised so a dead agent fails the
container rather than leaving it serving less than it advertises. The OpenAPI document
behind each `/docs` page is generated from that agent's `agent.yaml` at start-up -- sensors
become the request schema, actuators become the action enum, and the request examples are
the agent's own eval cases -- so the documentation cannot drift from the agent. See
`00-config-runtime/` for the mapping, and for why a rejected percept is a 422 while a
rejected model answer is a 502.

## Arriving from one of the write-ups

Each claim in the series has one file that backs it. Every command below runs with no API
key and no install, replaying what a real model actually returned.

| The claim | Run this |
|---|---|
| A* does not go anywhere. The LLM sets up the search; the search still runs | `python 02-goal-based/search/after.py` -- prints its own comparison against `before.py`: same plan, same nodes expanded, from prose instead of a tuple |
| The CSP solver is unchanged, not reimplemented | `python 02-goal-based/csp/verify_identical.py` -- proves it by object identity and a source digest |
| Alpha-beta still prunes the same tree | `python 05-multi-agent/adversarial/before.py` -- node counts, identical on any machine because they count operations |
| The contract between agents is the architecture | `python 05-multi-agent/orchestration/after.py` -- the last run halts on valid, plausible JSON that a schema check refuses |
| Parallelism has a floor | `python 06-parallelization/agent_floor.py` -- finds the crossover on your machine and prints its own caveats |
| A kernel-only speedup is not a speedup you can spend | `colab/gpu_floor.ipynb` -- the same measurement on real CUDA; counting transfers, the element-wise crossover never arrives |
| An embedding buys coordinates, not arithmetic | `python 07-nlp-foundations/real_embeddings.py` -- the hand-written cosine from `similarity.py`, run over real vectors, ranking a control pair the opposite way |
| Most of the answer was the harness, not the model | `python 09-model-portability/same_percept_three_tiers.py` -- the same percept through haiku, sonnet and opus; 13 of 17 produced the same action |
| A new agent is a new directory, not new code | `python 00-config-runtime/demo.py` -- nine agents through one class, and a check that the runtime names none of them |
| A single-percept suite cannot test whether an agent learns | `python 00-config-runtime/sequence_eval.py` -- the same percept twice, once cold and once after a preamble |
| Validation catches malformed output, not wrong output | `python 10-drift/replay.py --baseline baseline --prompt 10-drift/prompts/system_v2.md` -- structural and behavioural drift reported separately |
| A critic must be deterministic | `10-drift/critic-experiment/analysis.md` -- tested twice, and the result does not support the claim as originally stated |

The last row is not a typo. The experiment contradicted the article it was built to
support, and the analysis says so.

## Recorded, not mocked

With no credentials configured, `shared/llm.py` replays what a real model actually
returned to that exact prompt, from `shared/transcripts/`. Each entry carries the prompt,
the model name, the date, and the response. A one-line banner says which mode you are in.
`shared/embeddings.py` does the same for vectors.

There is no third mode. If nothing is configured and no recording exists for a prompt,
`llm_call` raises and tells you how to record one. Nothing here invents a response.

    ANTHROPIC_API_KEY=... LLM_RECORD=1 python 01-reflex-agents/simple/after.py

records while running live. Running it again with no key replays that recording.

This replaced hand-written canned responses, and the reason is the argument of the whole
repository: a canned string is not a model, so an example built on one demonstrates
everything except the claim it is making. That was not hypothetical here -- see "What
running it live changed" below.

A replay is a recording of an agent, not a simulation of one. It is still not the same as
running your own. Model versions move, and one recorded run is one sample.

## About the numbers

**Published figures, not measured here.** The GPU speedups in
`reference/06-support-gpu-parallelization.md` (33x SAXPY, 437x tiled matrix multiply, and
the rest), the token cost multipliers (~4x single agent, ~15x multi-agent), the 90 percent
prompt-caching cost reduction, and the 84 percent context-editing reduction are all cited
from their sources. Those numbers are quoted in prose and attributed, never printed by a
script as though it had produced them, and nothing here reproduces any of them.

**Measured here, and hardware-dependent.** `06-parallelization/` finds a real crossover on
whatever machine runs it, and prints the CPU count and caveats in its own output. On the
machine this was developed on the element-wise crossover fell between 10,000 and 25,000
elements, and the agent-layer crossover between 3 and 4 concurrent tasks. Yours will
differ. That is the point of shipping the script rather than the number.

There is now a GPU half to that, in `06-parallelization/hf-space/` and `colab/`. One run on
a Blackwell MIG slice found no element-wise crossover at all once both transfers are counted
-- the CPU finished first at every size tested, on the same rows where the kernel-only
column reads 32x to 76x -- and a matrix-multiply crossover at n = 1024. Those are that
run's numbers, and they are not the published figures above, which remain cited and
unreproduced. What they explain is what kind of number a kernel-only speedup is.

**Measured here, and model-dependent.** `09-model-portability/compare_models.py` reports
one run, of one model version, on one day. It marks replayed rows as replays and refuses to
print a row for a backend that did not run. One run is not a benchmark. The same applies to
the agent eval standings: 49 of 49 across nine agents, every declared actuator reached
by at least one case. That is a suite small enough to read rather than a benchmark, and
the cases were argued with rather than tuned -- several expectations were corrected
because the agent read its domain better than the case did.

**Deterministic regardless of machine.** Node counts, expansion counts, and the value
iteration grid are exact and reproducible anywhere, because they count operations rather
than seconds.

**A result that argues with the thesis it was built to support.** `10-drift/critic-experiment/` tests
a central claim rather than asserting it: two identical learning agents, one with
a deterministic critic and one with a model call, both scored against a ground truth
neither can see. Over 200 interactions against `claude-sonnet-5`, the agent with the LLM
critic did not degrade — and the experiment was then run a second time, days later, and did
not degrade again. Its critic was miscalibrated in both runs, harsh rather than generous,
and preserved rank order 95.5 and 95.9 percent of the time. Rank order is the only property
of a critic that reaches behavior, because the learning element sorts on it. The hypothesis
is in that directory's README, written before the first run; `analysis.md` reports what
happened and what it does and does not license. Two runs on one seed, one task, one model,
and a reward that was computable to begin with. It is not the last word. It is the reason
the claim about deterministic critics is stated narrowly here.

## What running it live changed

Every example passed offline, on hand-written canned responses, before any of it met a real
model. Seven defects survived that. Each is written up where it happened; these are the
ones that shaped the repository:

- Responses were being truncated at the token cap, so downstream parses failed, every
  fallback fired as designed, and the examples printed plausible answers produced without
  the model. The validation layer reported malformed output, which was true and useless.
- Fenced JSON was being discarded as unparseable. Against a live model this put the
  fallback rate on the twenty-case suite at 74 percent and the success rate at 30. With
  `shared/model_json.py` unwrapping the fence first, the same suite reports a 0.9 percent
  fallback rate. A fallback that fires on everything is not a safety net; it is the agent,
  and the model is decoration.
- Non-ASCII model output crashed the process on a Windows console, which the canned
  responses never hit because they were ASCII.
- A search example never reached its own subject: the model returned the board under a key
  the code did not read, so A\* never ran outside the offline path.
- The source page's own scoring code raised `TypeError` when a model returned a confidence
  of `"0.93"` as a string — inside the agent's performance measure, which runs before the
  contract gate that would have caught it.
- A "deliberate failure" demonstration only failed when no key was set.

Two agents later turned out to declare a state schema with no way to write to it, and an
eval suite turned out to be scoring each case on the ones before it. Both ran green
offline. Both were found by running the same code against a live model.

Sweeping every script against three vendors added three more, and the first two are the
same shape -- a demo that crashed where it had a correct answer already written:

- `value-iteration/real_world.py` checks every cell of the model's reward function and
  falls back to the naive numbers if any is missing. That guard is the point of the file
  and it was unreachable: the JSON parse raised several frames earlier, so a reply that was
  not JSON at all -- the likeliest way a model fails that prompt -- went straight past the
  check written for it.
- `agent_floor.py` asserted that its sequential and concurrent paths return identical
  routing. Offline that assertion could not fail, because both paths replay one recording
  per prompt. Live it fires, because two calls to a model may answer the same ticket
  differently. It now counts the disagreements instead: 26 of 108 on one backend, 0 of 108
  on another, both exiting cleanly.
- `openai_compatible`'s `small` tier named a model the provider does not serve, so every
  call at that tier was an HTTP 400 out of urllib rather than an agent -- and urllib's
  default message is the status line, so the provider's own explanation of what was wrong
  was being discarded. The shim now re-raises with the body attached.

Running offline is what makes this repository usable with no setup. It was never evidence
that anything worked, which is why the canned responses are gone and the offline path now
replays real ones.

## Contact

Built by **Jeff Murray** ([@jmurray10](https://github.com/jmurray10)).

- Email: **jeff.murray@alumni.upenn.edu**
- LinkedIn: [linkedin.com/in/jeff-murray-ai](https://www.linkedin.com/in/jeff-murray-ai)
- GitHub: [@jmurray10](https://github.com/jmurray10)

## License

MIT -- see [LICENSE](LICENSE). The examples are yours to take apart, and the point of
shipping the scripts rather than the numbers is that you can disagree with them.

## References

- Russell & Norvig, *Artificial Intelligence: A Modern Approach* (4th ed.)
- Anthropic, "Building Effective Agents"
- Anthropic, "Context Engineering"
- Kirk & Hwu, *Programming Massively Parallel Processors* (4th ed.)
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.)
