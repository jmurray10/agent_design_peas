# Changelog

All notable changes to this repository. It is a set of demonstrations rather than a
released package, so the entries below are dated rather than versioned, and they
record what changed about the *claims* as much as about the code.

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
