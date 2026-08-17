# Critic experiment

**Source:** reference/04-learning-agents-before-after.md

**Read this before reading anything else here.** Every number in this directory comes from
one live run: 200 interactions per arm, Arm B's critic on `claude-sonnet-5`, 2026-08-03,
preserved verbatim in `run-2026-08-03-sonnet5.log` and read in `analysis.md`. A real model
wrote every Arm B score. With no backend configured the run is replayed from
`shared/transcripts/`, which returns the same numbers because they are the same responses --
a recording of the run, not a re-measurement of it. What limits this result is not that
anything was invented. It is that it is one task, one model, one seed, and a reward that
happens to be computable.

## The claim

A learning agent rewrites its own context every cycle from whatever its critic says, so
an error in the critic does not stay where it was made -- it is compounded by the next
learning cycle, and the next. The falsifiable version: given a reward that is genuinely
computable, an agent whose critic is a model call will earn a lower true reward than an
identical agent whose critic is arithmetic, while reporting a higher score for itself.

**The hypothesis, stated before the experiment was run:** Arm B's self-reported score
rises while its true reward stagnates or declines, and the gap between them widens as the
learning element compounds on a signal nobody is checking.

The result is reported either way. If Arm B tracks ground truth, the claim has to soften
to "the critic must be deterministic where the reward is computable, and LLM critics are
viable where it genuinely is not" -- a narrower claim, and a more honest one. `analysis.md`
was written after the run and reports what happened, including the parts that did not go
the way the hypothesis said.

**It went that way.** Against `claude-sonnet-5`, over 200 interactions per arm, run twice,
the hypothesis was not borne out either time. Arm B's true reward rose rather than
degraded and finished level with Arm A. The critic was miscalibrated in the opposite
direction from the prediction -- harsh, not generous -- and it preserved rank order to
within half a point of 96 percent, which is the only property of a critic that reaches
behaviour here.

The numbers below are the second run, because that is the one whose transcript ships:
running the script with no key reproduces them exactly. `analysis.md` puts both runs in
one table and is the thing to read before quoting this directory. The paragraph above the
bold is the prediction, not the finding.

Two arms, same agent class, same percepts, same number of interactions:

- **Arm A** -- `LLMLearningAgent` from `04-learning/q-learning/after.py`, unmodified. Its
  critic is `_calculate_reward`.
- **Arm B** -- the same class, the same prompts, the same fallbacks. `observe_outcome`
  asks a model to score the outcome instead.

Both arms are scored every interaction by the same ground-truth function, which is
`_calculate_reward` called from one shared instance. There is no second implementation of
the reward anywhere in `run_experiment.py`, which is what makes "computed identically for
both arms" checkable rather than asserted. Arm A's own critic is that same inherited
method, so Arm A's recorded reward equals ground truth by construction, and the harness
asserts that on every interaction. That is the definition of the control arm, not a
result.

Arm B never sees ground truth. It is not in Arm B's prompt, not stored on the Arm B
agent, and not derivable from anything Arm B holds. That separation is the experiment.

One thing is disclosed to Arm B on purpose and should not be mistaken for a leak: its
prompt states the scale endpoints, -2.0 to 3.0, which happen to be the range of the true
reward function. Without a shared scale the required "gap between self-report and ground
truth" would be a comparison of two arbitrary units. The endpoints say nothing about
which of the four facts carries weight, which is the entire question.

## Run it

    python 10-drift/critic-experiment/run_experiment.py
    ANTHROPIC_API_KEY=... LLM_RECORD=1 python 10-drift/critic-experiment/run_experiment.py
    LLM_PROVIDER=ollama python 10-drift/critic-experiment/run_experiment.py

Default run: seed 21, 200 interactions per arm, `learn()` every 10, Arm B's critic at tier
`mid`. Live, that is roughly 640 model calls; the recorded run took 42 minutes of wall
clock. The recording is in `shared/transcripts/`, so it replays to completion with no key,
no network and no cost, and prints this:

    Seed 21. 200 interactions per arm. learn() every 10.
      Arm A critic: deterministic (_calculate_reward)
      Arm B critic: LLM (_llm_critic_score)

      Arm A recorded reward == ground truth on 200/200 interactions (by construction -- it is the same method).
      Arm B recorded reward == ground truth on 113/200 interactions.

    Block means, one row per learn() cycle

      block   A true   B true   B self   B gap   A action (block mode)   B action (block mode)
      -----   ------   ------   ------   -----   ---------------------   ---------------------
         10     1.65     1.60     1.20   -0.40   escalate         4/10   escalate         4/10
         20     1.70     2.00     1.75   -0.25   request_info     8/10   write_custom     7/10
        ...
        130     2.50     2.00     1.90   -0.10   write_custom    10/10   write_custom     6/10
        ...
        200     2.15     1.85     1.50   -0.35   write_custom     6/10   write_custom     8/10

    First quarter vs last quarter

      Arm A true reward           1.73 ->   1.88
      Arm B true reward           1.81 ->   1.90
      Arm B self-reported         1.43 ->   1.57
      Arm B gap (self - true)    -0.38 ->  -0.33
      A minus B, true reward     -0.08 ->  -0.02

    Arm B's critic against ground truth, on Arm B's own interactions

      mean self-reported score    1.61
      mean ground truth           1.93
      mean gap                   -0.31
      pairwise rank agreement    95.9% (11232 concordant, 481 discordant, 11713 ordered pairs)

    Was the hypothesis borne out?

      self-reported score rose        True
      true reward stagnated or fell   False
      gap widened                     False
      all three                       False

    Arm B critic parsing
      responses recovered from fenced or prose wrapping   0
      scores coerced from string to float                 0
      scores clamped into range                           0
      responses that could not be parsed at all           0

The two arms diverge from the first block, not from a fixed warm-up point: a real critic
scores the same outcome differently from the arithmetic one immediately, and the arms are
choosing from different tables by interaction ten. Rank agreement is the number that
reaches behaviour. `learn()` sorts the experience log by reward and `act()` reads
per-action averages, so a critic that is uniformly wrong by a constant produces the same
sort. Only reordering changes what the agent does, and this critic reordered 506 pairs out
of 11,339.

`--replicates 10` runs consecutive seeds and summarises across them. That is the first
thing to do before this is relied on: one seed is one sample. `--mock-critic` is
retired and changes nothing in any mode; the script says so if you pass it.

## What changed

One component. Arm B's `observe_outcome` calls a model instead of `_calculate_reward`,
which is still inherited, still correct, and never called. Everything else is Task 6's
code running unmodified: the same `act()` prompt, the same `learn()` sort and JSON
recovery, the same `suggest_exploration()` fallback, the same off-list action guard.

Deterministic and identical across both arms: the ground-truth reward function, the
percept sequence, the environment, the exploration rate, and the random stream. The
environment contains one deliberate trap -- a fast automated path that closes tickets
reliably and leaves the customer cold. It looks best to anything scoring how an
interaction reads, and computes to 1.385 against `write_custom`'s 2.160, because
`customer_satisfied` carries the largest positive weight and `error` the largest negative
one. Both facts are in the critic's prompt, so it is not information-disadvantaged.

## What it costs

Arm A pays four comparisons per interaction. Arm B pays a network call to buy a reward
that was already available. Here that purchase cost nothing -- but its failure mode is
invisible from inside: every response validated, every score landed in range, nothing
raised and nothing logged. The only thing that can tell you an agent stopped improving is
a number it is not allowed to see: the performance measure itself. If you do not
have one, this experiment cannot be run on your system, which is itself the finding.
