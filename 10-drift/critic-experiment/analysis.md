# Analysis

Written after the run. The hypothesis was in `README.md` before `run_experiment.py` was
executed and has not been edited since.

---

## The live run: the hypothesis was not borne out

    ANTHROPIC_API_KEY=... python 10-drift/critic-experiment/run_experiment.py \
        --interactions 200 --block 10 --critic-tier mid

200 interactions per arm, one seed, Arm B's critic on `claude-sonnet-5`, run 2026-08-03,
42 minutes wall clock. Arm A's critic is the deterministic `_calculate_reward` from
`04-learning/q-learning/after.py`. Both arms are scored against the same ground truth,
which Arm B never sees.

The hypothesis was that Arm B's self-reported score would rise while its true reward
stagnated or fell, with the gap widening as the learning element compounded on an
unchecked signal. One of those three conditions held.

|                          | first quarter | last quarter |
|--------------------------|---------------|--------------|
| Arm A true reward        | 1.89          | 1.90         |
| Arm B true reward        | 1.77          | **1.94**     |
| Arm B self-reported      | 1.52          | 1.60         |
| Arm A minus Arm B (true) | 0.12          | **-0.04**    |

Arm B did not degrade. It improved, and finished marginally ahead of the arm with the
deterministic critic. The difference is well inside the noise of a single seed, so the
honest reading is that the two arms were indistinguishable, not that the LLM critic won.

## It happened twice

The experiment was run again on 2026-08-04 while recording a transcript, which made it an
independent second execution rather than a rerun of the first: same seed, same code, a
model free to answer differently, and it did. That run is what
`shared/transcripts/10_drift__critic_experiment__run_experiment.json` replays, so running
the script with no key reproduces the second column below, not the first.

|                            | 2026-08-03    | 2026-08-04    |
|----------------------------|---------------|---------------|
| Arm A true reward          | 1.89 -> 1.90  | 1.73 -> 1.88  |
| Arm B true reward          | 1.77 -> 1.94  | 1.81 -> 1.90  |
| Arm A minus Arm B, last    | -0.04         | -0.02         |
| Arm B critic calibration   | -0.27 (harsh) | -0.31 (harsh) |
| pairwise rank agreement    | 95.5%         | 95.9%         |
| hypothesis borne out       | no            | no            |

Every load-bearing number reproduced. The critic was harsh both times rather than
generous, it preserved rank order both times to within half a percentage point, and the
arm with the model critic failed to degrade both times. The two runs disagree about
details that a single sample was never entitled to claim -- which quarter Arm A starts
from, the exact final gap -- and agree about everything the conclusion rests on.

This does not make it a benchmark. It is two runs, one task, one model, one reward that
happens to be computable. It does mean the first result was not a fluke of one sequence of
outcomes, which was the largest objection available to it and is the reason "one seed" is
no longer the first entry under limitations.

## Why it did not degrade, which is the useful part

The LLM critic was **miscalibrated and it did not matter.** It scored a mean of 1.69
against a true mean of 1.96 — systematically harsh by 0.27, and harsh rather than lenient,
which is the opposite of the failure mode the hypothesis assumed.

What it preserved was order. Pairwise rank agreement with ground truth was **95.5 percent**
(10,833 concordant against 506 discordant, over 11,339 ordered pairs).

Order is the only property of the critic that reaches behavior in this architecture.
`learn()` sorts the experience log by reward and hands the top and bottom slices to the
pattern extractor; `act()` reads per-action averages. A critic that is uniformly wrong by
a constant produces the same sort and the same ranking of averages as a perfect one. Only
*reordering* changes what the agent does.

That is a sharper claim than the original, and this run supports it:

> A critic does not have to be accurate. It has to be monotonic in the thing you care
> about. An LLM critic fails when it reorders outcomes, not when it is badly calibrated.

## What this does not license

One task. One model. Two runs on one seed. 200 interactions each. A reward that is *computable* — the
ground truth here is arithmetic over an outcome dict, which is precisely the case where
the original claim is right that you should just compute it. Nothing here says an LLM critic is
safe on a task where the reward is genuinely subjective, which is the only case where
anyone reaches for one.

One seed is weak evidence even run twice: a second execution varies the model's answers
but not the situations it is answering about. `--replicates` runs consecutive seeds and
summarises across them, and the finding should be re-tested that way before it is relied
on.
The result also lives one prompt change away from being different: a critic asked to score
on a rubric it interprets inconsistently would reorder, and reordering is the failure.

## What the claim should be now

The blanket form — *your critic cannot be an LLM* — is not supported by this run and is
overclaiming. The defensible version:

- Where the reward is computable, compute it. Not because an LLM critic necessarily
  degrades the agent, but because a deterministic critic is free, instant, reproducible,
  and cannot reorder.
- Where the reward is not computable, an LLM critic is viable, and the property to test
  is rank agreement against whatever ground truth you can assemble — not calibration.
- The failure mode to instrument is reordering. A critic that is uniformly generous is
  harmless here; a critic that swaps two outcomes is not.

That is a narrower claim than the original and a better supported one, and this repository
contains the run that earned it.


## What this document can and cannot claim

Every number above comes from a live run against `claude-sonnet-5`, 200 interactions per
arm. The first is preserved verbatim in `run-2026-08-03-sonnet5.log`; the second is
preserved as the transcript that `run_experiment.py` replays. A real model produced every
Arm B score in both. Nothing here is authored.

What that does not make it is a benchmark. Two runs on one seed, one task, one model, and
a reward chosen to be computable. The rest of this document is about which parts would
survive changing any of those.

An earlier version of this file analysed a run in which Arm B's critic was a canned string
this repository wrote, and reported the opposite result: Arm B ending on a worse true
reward in ten seeds out of ten, and rank agreement of 74.9 percent. That analysis has been
deleted rather than relabelled. It described a judge that no longer exists, and its
conclusion was a property of how the canned judge had been written -- which is the whole
reason this repository stopped shipping canned responses. Keeping it alongside the live
result would have left two contradictory findings in one document, each citing the other's
subject.

## Setup

Two `LLMLearningAgent` instances from `04-learning/q-learning/after.py`. Arm A's critic is
the inherited `_calculate_reward`. Arm B's `observe_outcome` calls a model instead; the
same method is still inherited, still correct, and never called.

Ground truth is one shared `_calculate_reward` instance applied to both arms every
interaction. There is no second implementation of the reward anywhere in
`run_experiment.py`, which is what makes "computed identically for both arms" checkable.
Arm A's recorded reward equals ground truth by construction and the harness asserts that
on all 200 interactions -- that is the definition of the control, not a result.

Arm B never sees ground truth. It is not in Arm B's prompt, not stored on the Arm B agent,
and not derivable from anything Arm B holds. One thing is disclosed on purpose: Arm B's
prompt states the scale endpoints, -2.0 to 3.0, because without a shared scale the
"gap between self-report and truth" would compare two arbitrary units. The endpoints say
nothing about which of the four outcome facts carries weight, which is the entire question.

## Limitations, in descending order of how much they matter

1. **Two runs, one seed.** Both executions used seed 21, so the environment handed both
   arms the same sequence of situations; what differed was the model's answers. That is
   enough to rule out a one-off fluke of model output and not enough to rule out something
   peculiar to this seed's sequence of outcomes. `--replicates 10` runs consecutive seeds
   and summarises across them, and it is still the thing to do before this is
   relied on.
2. **The reward is computable.** Ground truth here is arithmetic over an outcome dict,
   which is exactly the case where the original claim is right that you should just compute
   it. This says nothing about a genuinely subjective reward, which is the only case where
   anyone reaches for an LLM critic in the first place.
3. **A real judge is not deterministic.** Asked the same outcome twice it may not answer
   the same way twice, and that variance is not measured by one pass. Variance would
   produce more misranking, not less, so a single-pass result is the optimistic one.
4. **The environment contains a deliberate trap.** `auto_resolve` is designed to look
   better on the surface than it is. That is what gives a critic something to get wrong,
   and it is also why the result does not extrapolate to environments where surface
   features and true value agree -- there, a critic has no opportunity to misrank.
5. **`learn()`'s top-three sort is a weak estimator.** It ranks single draws, so in a
   stochastic environment the extremes tend to be the luckiest samples of the
   highest-variance action rather than the best action. That is a property of Task 6's
   code, not something introduced here, and it is left alone.
6. **The critic ran at tier `mid`.** `--critic-tier frontier` and `--critic-tier small`
   both exist, because "would a better judge have done better, would a worse one have
   failed" is the obvious next question and it should be settled by rerunning rather than
   by arguing. Note that a tier change alters no prompt, so a replay of this recording
   returns the same answers whatever tier is requested; comparing tiers requires a key.

## What this does to the original claim

The original claim was that a critic cannot be an LLM. This run does not support that, and
the honest replacement is narrower and more useful:

- Where the reward is computable, compute it. Not because an LLM critic necessarily
  degrades the agent -- this one did not -- but because a deterministic critic is free,
  instant, reproducible, and cannot reorder.
- Where it is not computable, an LLM critic is viable, and the property to test is rank
  agreement against whatever ground truth can be assembled. Not calibration.
- The failure mode to instrument is reordering. A uniformly generous or uniformly harsh
  critic changes nothing that reaches behaviour.

The experiment was specified, the hypothesis was committed to the repository before the
run, and the result contradicted it. Reporting that is the point of having built it.
