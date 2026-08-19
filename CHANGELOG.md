# Changelog

All notable changes to this repository. It is a set of demonstrations rather than a
released package, so the entries below are dated rather than versioned, and they
record what changed about the *claims* as much as about the code.

## 2026-08-18

- **The repository was verified end to end, in the container, both ways.** Offline:
  every documented command across `00-` through `10-`, run in the image with no key and
  no network. Live: 48 of 49 entry points passed against Anthropic in 95.9 minutes. The
  49th was `10-drift/ci_check.py` refusing to run because a live backend was configured,
  which is the guard doing its job -- the drift gate judges recordings, and spending
  live calls should be a decision rather than a side effect of a sweep. `verify-live`
  now excludes it, matching how the vendor table already reports its denominators. The
  served path was checked separately: all nine agents healthy over HTTP, and
  `container_report.py` reports CURRENT.
- **The drift chapter accounts for all four verdict flips.** Fixing eval case c16 (the
  prompt-injection case) changed the run the chapter documents: three cases go PASS to
  FAIL and c16 goes the other way, because a prompt that escalates whenever it is
  uncertain passes that case by accident. A case that starts passing is not evidence a
  change was good, and the chapter now says so.

## 2026-08-17

- **The serving layer caught up with what it serves.** The generated OpenAPI spec
  described half the routes; every HTTP caller shared one conversation state; a valid
  percept with no recording closed the socket instead of raising the documented error;
  and the completion ceiling was sized before models reasoned on every call. All fixed.
- **The front page was rewritten for someone who has not read the series.** Run
  commands before vocabulary, the simplest agent first, and the claim tables split so
  each section makes one claim.

## 2026-08-15

- **A replay guard that was off for 1,466 of 1,717 entries.** `shared/transcript.py`
  refuses to serve a recording made at one tier to a caller asking for another -- but
  every entry recorded before 2026-08-14 carried a null tier, so the check never fired.
  Tiers were backfilled from each entry's recorded model, and turning the guard on
  immediately caught `mixed_agent.py` keying three configurations onto one transcript
  entry. Each configuration now records into its own transcript.
- **Four eval cases that disagreed with themselves, and one that needed the model to
  misbehave.** Recordings held five samples per prompt whose answers disagreed, and
  replay index 0 happened to be the expected one -- so 49/49 offline was partly luck.
  The percepts were rewritten to determine their own answers.
- **The drift gate had never run.** A null `env:` mapping failed every workflow at
  startup in zero seconds, so the badge on the README was advertising the repository's
  claim-checking gate as failing. The gate now runs on every push.

## 2026-08-14

- **Three vendors, one commit, one image.** The same 49 scripts from a single build,
  measured on 2026-08-14: Anthropic 48/48, an OpenAI-compatible router 46/48, Gemini
  44/48 -- with `ci_check` out of every denominator because it refuses a live backend
  on purpose. The README explains why this is portability rather than a ranking.
- **A tier that named a model nobody serves.** The router's `small` tier pointed at a
  model the router answers `model_not_supported` for, so the first `tier="small"` call
  was an HTTP 400 out of `urllib` rather than an agent. It now names a model the
  router serves, verified live in the container.

## 2026-08-11

- **The last toy left the container.** `vacuum-bot` was replaced by `uptime-triage`,
  on-call alert triage, so all nine agents are now real working domains. The vacuum
  world's before-and-after pair stays in `01-reflex-agents/simple/`, where a source
  page's code belongs. It keeps the `small` tier exercised, and for a better reason:
  the decision runs on every alert of every day, so a frontier model on each one is not
  a design anyone can afford.
- **Every declared actuator is now reached by a case.** Seven were advertised in configs
  and never asserted; `support-bot` had four of six untested because its prompt never
  said what its actions meant.
- **Results name the model.** The runtime puts it on the result, so demo output,
  evaluation runs and HTTP responses all say which model answered. On a replay it names
  the model the recording came from.
- **`same_percept_three_tiers.py`** runs the same percept through haiku, sonnet and opus
  with everything else held constant. 13 of 17 produced the same action, which is the
  clearest measurement in the repository of how much of the answer was the harness.
- **`container_report.py`** answers whether a running container is serving what is on
  disk, by comparing content rather than timestamps.

## 2026-08-10

- **Public repository setup.** MIT license, security policy, contribution guide,
  issue templates.

## 2026-08-09

- **Colab.** Three notebooks in `colab/`: the before-and-after oscillation across
  five architectures, the parallelization floor measured on the reader's own GPU,
  and the nine agents through one runtime. All run without a key by replaying
  recorded real responses.
- **An index.** One page on `:8079` listing every agent the container serves,
  grouped by architecture, generated from the same `agent.yaml` files the runtime
  reads.
- **The GPU claim is measured rather than only cited.** `06-parallelization/hf-space/`
  runs the same crossover measurement on real CUDA. Counting both transfers, the
  element-wise crossover never arrived at any size tested, on the same rows where
  the kernel-only column read 32x to 76x. The published 33x figure remains cited
  and unreproduced; what this explains is what kind of number it is.
- **Real embeddings.** `07-nlp-foundations/real_embeddings.py` runs the hand-written
  cosine from `similarity.py` over vectors from a trained model, and finds bag-of-words
  ranking a control pair the opposite way.
- **`shared/embeddings.py`**, a second entry point on the same record-or-replay terms
  as `shared/llm.py`.

## 2026-08-08

- **A sequence harness.** `00-config-runtime/sequence_eval.py` runs the same percept
  twice, cold and after a preamble, because a single-percept suite structurally
  cannot test whether an agent learns.
- **Eval standings 34 of 40 to 39 of 40**, none of it by editing expectations to
  make the number look better. Two schema defects, three underspecified prompts, one
  harness defect, and several expectations conceded to an agent that argued better
  than the case did. Each is a separate commit with the reason attached.
- **A harness defect found only by running live.** `evaluate()` reset state once per
  suite rather than once per case, so a stateful agent inherited whatever its
  neighbours had established. The recordings had captured the order-dependent answer
  and replayed it faithfully.

## 2026-08-07

- **All five classical architectures have a live endpoint.** Goal-based, utility-based
  and learning joined the four simple-reflex and two model-based-reflex agents.
  Nine agents, one `ConfigDrivenAgent` class, no agent-specific code in the runtime.
- **Two agents declared state and had no way to write to it.** Neither action schema
  declared `state_update`, and `additionalProperties: false` meant the model could not
  have returned one. Both carried an empty dict through every turn. Nothing failed,
  which is what made it bad.

## 2026-08-05

- **One container named `peas`**, one process per agent, supervised. OpenAPI generated
  from each agent's config at start-up, so the documentation cannot drift from the
  agent.

## 2026-08-03

- **Canned responses removed.** Seven defects had passed offline on hand-written mock
  responses and failed against a real model. Replaced by recorded real responses keyed
  by prompt content, with no third mode: a prompt with no recording raises rather than
  inventing an answer.
- **A result that argues with the thesis it was built to support.**
  `10-drift/critic-experiment/` tested whether an LLM critic degrades a learning agent.
  It did not, twice. The claim about deterministic critics is stated narrowly as a
  result.
