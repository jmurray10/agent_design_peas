# What the two suites here test

`test_cases.json` judges each percept on its own. `evaluate()` starts from empty state, so
a case is measured on what it contains rather than on what ran before it. That is what the
HTTP contract promises a caller: send this percept, get a defensible action.

`sequences.json` tests the claim a single-percept suite structurally cannot. A learning
agent is worth building only if what happened earlier changes what it does now, and the
only way to demonstrate that is to run the same percept twice with different histories in
front of it:

    python 00-config-runtime/sequence_eval.py 00-config-runtime/agents/triage-tuner

Each test declares a control action, a preamble, and a primed action. A pass needs the two
arms to differ in the declared direction. Two arms that agree mean either state that
nothing reads or a preamble that carries nothing, and the harness prints both answers so
you can tell which.

Writing these found three things worth recording. The first control expectation was wrong:
cold, with no outcomes at all, this agent explores rather than guessing confidently, which
is what its own prompt tells it to do. The second preamble carried one outcome and the
agent did not move, which is the correct reading of a single data point -- it took three
before the area counted as known. And `support-bot` was carrying state that said an
inquiry was resolved without saying which order it was about, so a follow-up asked a
question the previous turn had already answered.

The longer-horizon version of the same claim, over hundreds of interactions rather than
two, is in `04-learning/q-learning/after.py` and measured in `10-drift/critic-experiment/`.
