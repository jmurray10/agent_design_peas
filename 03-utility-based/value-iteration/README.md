# MDP value iteration

**Source:** reference/03-utility-agents-before-after.md

## The claim

Value iteration returns a provably optimal policy for the 4x3 grid world, and putting an
LLM in front of it does not weaken that guarantee as long as the LLM never picks the
action. `after.py` imports the solved policy from `before.py` rather than recomputing it,
so the guarantee is inherited, not re-argued. The two paths that do let the LLM choose
actions -- policy approximation and exploration -- give the guarantee up, and the code
says so at the point where it happens.

## Run it

    python before.py
    python after.py
    python 03-utility-based/value-iteration/real_world.py

`before.py`:

    4x3 grid world
      states: 11 (wall at (1, 1) is not a state)
      transitions specified: 96
      gamma: 1.0   living reward: -0.04
      move noise: 0.8 intended, 0.1 each perpendicular

      value iteration converged after 19 sweeps (delta=0.000691 < epsilon=0.001)

    Value function V*
      y=2 |    0.812    0.868    0.918    +1.00
      y=1 |    0.762     ####    0.660    -1.00
      y=0 |    0.705    0.655    0.611    0.387

    Optimal policy pi*
      y=2 |        >        >        >       +1
      y=1 |        ^     ####        ^       -1
      y=0 |        ^        <        <        <

`after.py`:

    PATH 1  LLMStateEstimator -- classical MDP, LLM only estimates the state
      value iteration converged after 19 sweeps (delta=0.000691 < epsilon=0.001)
      policy source: before.value_iteration defined in before.py
      reimplemented in after.py: False
    [replay] No backend configured. Replaying recorded responses from shared/transcripts/. These are real model outputs, not invented ones -- see shared/README.md.
      r1: estimated (1, 2)  -> action 'right'  [well-formed JSON, non-terminal state]
      r2: estimated (3, 3)  -> action 'no_op'  [well-formed JSON, non-terminal state]
      r3: estimated (1, 3)  -> action 'no_op'  [prose instead of JSON -> JSONDecodeError -> mdp.initial_state]
      r4: estimated (3, 2)  -> action 'no_op'  [JSON missing the 'y' key -> KeyError -> mdp.initial_state]
      r5: estimated (3, 2)  -> action 'no_op'  [the +1 terminal -> policy holds None -> 'no_op']
      r6: estimated (2, 2)  -> action 'right'  [the wall, which is not a state at all -> lookup misses -> 'no_op']

    PATH 2  LLMPolicyAgent -- the LLM is the policy
      c1: reports= 0 flags=[] -> 'approve'
      c2: reports=14 flags=['violence', 'targeted'] -> 'reject'
      c3: reports= 3 flags=['health_claim'] -> 'flag_for_review'
      c4: reports= 6 flags=['harassment?'] -> 'flag_for_review'  <- model returned an action outside the actuator set; fallback fired

    PATH 3  LLMExplorationAgent -- LLM explores, code keeps the books
      t= 6  queue_backlog_growing  rollback         reward=-0.123
      queue_backlog_growing  scale_up         +0.2416
      latency_spiking        scale_up         +0.0957

Path 1 needs a warning label, because the bracketed notes are a hardcoded dict written
against the responses this example was originally built on, and four of the six no longer
describe what the recording returns. Every recorded estimate here is well-formed JSON
inside a ```json fence, `claude-sonnet-5`, 2026-08-04. Nothing returned prose, nothing
dropped a key, so `JSONDecodeError` and `KeyError` never fire.

What fires instead is more interesting than what the labels promise. The grid has rows
`y=0..2`, and for `r2` and `r3` the model answered `y: 3` — a coordinate outside the world
it was told the size of. That is not a parse failure and no `except` catches it. The
policy is a dict keyed by state, the lookup misses, and the agent emits `no_op`. `r4` and
`r5` both land on `(3, 2)`, the +1 terminal, where the policy holds `None` and the same
`no_op` comes out for an entirely different reason. Only `r1` and `r6` reach a real
non-terminal state and get a real action.

Which is the claim holding rather than failing. Four of six percepts were misread, and no
misread percept produced a wrong move — it produced no move. The LLM chose the state; a
dictionary lookup into the Bellman-derived policy chose the action, every time.

### The same solver, on a maintenance policy

`real_world.py` imports `value_iteration` and `MDP` and runs them on when to service a
machine. The transition model is engineering fact and no model is asked about it. The
rewards are what a plant manager argues about -- what a breakdown really costs once the
line stops -- and turning that paragraph into numbers is the model's whole job.

Run it and the policy changes when the breakdown cost arrives, because the numbers
changed, not because anything felt more prudent. That is the property worth having: the
argument is about the rewards, in the open, and the solver is optimal for whatever it is
given.

## What changed

Path 1 replaces the sensor model only. The LLM turns a noisy beacon reading into a grid
coordinate; `value_iteration`, the Bellman updates, and the policy lookup stay exactly
where they were, imported from `before.py`. Path 2 replaces the policy function itself,
because a moderation queue has no enumerable state space for value iteration to consume.
Path 3 replaces action selection under unknown transitions; `observe_reward` still does
every Q update in Python.

Two things the source left out are added here. The grid world itself -- the source gives
the solver but never an environment -- and a fix to `LLMPolicyAgent`: the source falls
back to `available_actions[0]`, which in its own moderation example is `approve`, the -5
outcome in its own reward table. Its `config.yaml` declares `fallback: "flag_for_review"`.
The config wins.

## What it costs

Path 1 costs one model call of latency per percept and adds a failure mode the classical
agent did not have: a misread percept produces a legal action for the wrong state. Paths
2 and 3 cost the optimality guarantee outright. Nothing in an LLM policy converges, and
nothing bounds its regret. What remains is the action set -- membership is a Python `in`
test, so the agent cannot emit an action the actuators do not implement -- and, in Path 3,
a reward ledger that is arithmetic rather than judgment.
