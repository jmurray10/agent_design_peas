# Q-learning

**Source:** reference/04-learning-agents-before-after.md

## The claim

A learning agent has four components, and exactly one of them cannot be a model. Three
of the four -- performance element, learning element, problem generator -- swap to LLM
calls and the architecture still runs end to end. The critic does not, and the claim is
checkable rather than argued: `_calculate_reward` in `after.py` is five arithmetic lines
with no `llm_call` in them, sitting above a comment that says END OF THE CRITIC, and every
number the learning element sorts on comes out of it.

## Run it

    python before.py
    python after.py

`before.py` -- 500 episodes on the five-state chain, seeded:

    Convergence trace, 50-episode blocks:
      episodes    avg return   Q(A,right)
        1-50           5.96         3.11
       51-100          6.80         4.54
      101-150          6.56         4.58
      ...
      451-500          6.74         4.58

    Greedy policy reached its final form at episode 12 and held it for the remaining 489 episodes.

    Learned policy:
      A: right  (Q_left=2.5, Q_right=4.6)
      B: right  (Q_left=2.7, Q_right=6.2)
      C: right  (Q_left=4.1, Q_right=8.0)
      D: right  (Q_left=5.8, Q_right=10.0)

Two things converge at very different speeds, which is why the trace carries both
columns. One `-1` is enough to push the agent off a bad action, so the policy is correct
by episode 12 and the return column plateaus in the first block. The values take until
roughly episode 150 to settle on 4.58, 6.2, 8.0, 10.0 -- the exact optima for this chain.
Ranking the actions is cheap; knowing what they are worth is not.

`after.py` -- 20 interactions, `learn()` every 10:

    20 interactions. learn() fires every 10.

    [replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
       1 [performance element] write_custom   -> reward  0.00
       2 [problem generator  ] escalate       -> reward -1.00
       3 [performance element] write_custom   -> reward  1.50
       ...
      10 [performance element] write_custom   -> reward  0.50
      [learning element] response was not bare JSON; recovered the array from it

    After 10 interactions:
      Avg reward: 0.85
      Rules:
        - For complaint + medium urgency + premium tier: default to write_custom — it produced all 3 positive outcomes (rewards 2.0, 2.0, 2.5) out of 4 attempts (75% hit rate).
        - Treat elapsed time as the dominant risk factor: every reward >= 2.0 finished in <= 64s, while every negative-or-zero reward took 69-88s.
        ...

      19 [problem generator  ] request_info   -> reward  3.00
      20 [performance element] write_custom   -> reward  3.00
      [learning element] response was not bare JSON; recovered the array from it

    After 20 interactions:
      Avg reward: 1.60
      Rules:
        - The strongest signal is latency, not action choice: every success completed in <=41s (21, 26, 41) and every failure took >=50s (50, 69, 85).
        - Do more: request_info in this state. It is the single best observation (reward 3.0, fastest at 21s) and has no recorded failures - but n=1, so treat it as a promising hypothesis to test, not an established best action.
        ...

    Components, and what each one ran on:
      performance element  LLM, tier=mid       experience and stats in context
      critic               no model at all     arithmetic over observed outcomes
      learning element     LLM, tier=frontier  pattern extraction over sorted log
      problem generator    LLM, tier=mid       deterministic fallback underneath

One deterministic guard fires twice in that run and one does not fire at all. The learning
element asked for "JSON list of rule strings" and got a list wrapped in prose both times,
which the recovery step unwraps -- `[learning element] response was not bare JSON`. The
guard under the problem generator stays quiet, because every action it named this time
(`escalate`, `request_info`) was on the actuator list. Both were recorded on 2026-08-04;
the off-list case is real and has been seen, it is simply not in this recording, which is
what one sample of one run buys you.

The two average-reward numbers are noise, and `after.py` prints a paragraph saying so.
Each simulated outcome is drawn independently of the action chosen, so nothing in this
demo could show the loop improving behavior even if it did. Read the model's own rules
above with that in mind: they are lucid, specific, quantified, and describing a pattern
that is not there. What the run does establish is that all four components execute and
that the reward every one of those rules was derived from is arithmetic.

## What changed

Three of the four components became model calls. The performance element was `argmax`
over a Q-table; now a model reads the same statistics plus last cycle's rules. The
learning element was one Bellman update per experience; now a deterministic sort by
reward feeds a model call naming what separates the best runs from the worst. The problem
generator was epsilon-greedy; now a model names the gap in the agent's own experience.

The critic did not move. Neither did the reward sort feeding the learning element, the
guard rejecting actions off the actuator list, or the least-used fallback beneath the
problem generator.

Three source fixes, marked in the code: `get_best_action` used a truth test where it
meant `is not None`; `learn()` swallowed a JSON parse failure with a bare `pass`; and the
demo's simulated outcome now sets `error`, without which the critic's penalty branch
never fires.

## What it costs

Q-learning converges to Q* given enough exploration. That is a theorem. Nothing in
`after.py` converges to anything, and no quantity of experience will make it. The trade
is a proof for reach: states you could never enumerate, in natural language. It also
costs three arithmetic operations per step, replaced by three network calls, and a policy
readable off a table, replaced by one you can only observe. What survives is the reward
-- the same number computed the same way in both files, and the last thing left that can
tell you the agent got worse.
