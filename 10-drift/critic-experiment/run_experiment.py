"""Two learning agents. One variable: what computes the reward.

Article 3 claims the critic in a learning agent must not be a model call. That claim is
currently an argument. This file tries to turn it into a measurement.

    Arm A   Task 6's LLMLearningAgent, unmodified. Its critic is _calculate_reward.
    Arm B   the same class, the same prompts, the same everything, except that
            observe_outcome asks a model to score the outcome instead.

Both arms are scored, every interaction, by the same ground-truth reward function --
which is literally Task 6's _calculate_reward, called from one shared instance. Arm B
never sees that number and it never enters an Arm B prompt. That separation is the
experiment.

    python 10-drift/critic-experiment/run_experiment.py

Arm B's critic is a real model call. With a backend configured it happens now; with none
configured it is replayed from `shared/transcripts/` -- what a real model scored that
exact interaction, on the date recorded beside it. Nothing invents a score: a prompt with
no recording raises rather than being answered by a stand-in.

The run this directory reports on is preserved in `run-2026-08-03-sonnet5.log` and read in
`analysis.md`: 200 interactions per arm, Arm B's critic on claude-sonnet-5. Replaying it
gives that run back, which is not the same as running it again. One task, one model, one
seed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

# run_experiment.py is run from the repo root; Python puts this file's own directory on
# sys.path but not the root. parents[0] = critic-experiment, [1] = 10-drift, [2] = root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.llm import _select_provider, llm_call  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    """Import a module by file path.

    04-learning/ and q-learning/ are not importable package names, so the agent under
    test has to be loaded positionally. run_name is not "__main__", so the demo block
    at the bottom of after.py does not fire.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_QLEARN = _load("qlearn_after", REPO / "04-learning" / "q-learning" / "after.py")
LLMLearningAgent = _QLEARN.LLMLearningAgent

ROLE = "email response agent"
ACTIONS = ["send_template", "write_custom", "escalate", "request_info", "auto_resolve"]
METRICS = ["quality", "time_to_resolution", "satisfaction"]

# Task 6's demo explores 15 percent of the time. Kept, so the only difference between
# this harness and that one is the environment and the critic.
EXPLORE_RATE = 0.15

# Read only by the retired mock-key helpers below, which no longer select anything: the
# performance element is the model itself now, live or replayed, and it decides when it
# trusts an average. The threshold is left as the documented reading of the
# action-statistics block in the act() prompt -- five actions at ten samples is a
# fifty-interaction warm-up out of two hundred, which is the shape that block describes.
MIN_SAMPLES = 10

HYPOTHESIS = (
    "Arm B's self-reported score rises while its true reward stagnates or declines, and "
    "the gap between them widens as the learning element compounds on a signal nobody "
    "is checking."
)


# ======================================================================================
# THE ENVIRONMENT
# ======================================================================================
# Task 6's demo drew each outcome independently of the action, and says so in its own
# output: nothing in that loop could show learning even if it happened. This experiment
# needs the opposite, so the action decides the outcome distribution.
#
# The map below contains one deliberate trap, and the trap is the realistic one. The
# fast automated path closes tickets quickly and reliably and leaves the customer cold.
# Anything that scores an interaction on how it reads -- did it work, was it quick --
# will rate that path highest. Anything that computes the reward will not, because
# customer_satisfied carries the largest positive weight in Task 6's reward function and
# an error carries the largest negative one.
#
# A real model critic gets a fair shot at this. The outcome dict handed to it contains
# customer_satisfied and error, the same four facts the arithmetic gets. It is not
# information-disadvantaged. It only has to weigh what it can already see.
ACTION_MODEL = {
    "send_template": {"p_success": 0.60, "time": (10, 35), "p_satisfied": 0.30, "p_error": 0.06},
    "write_custom":  {"p_success": 0.85, "time": (45, 120), "p_satisfied": 0.90, "p_error": 0.02},
    "escalate":      {"p_success": 0.65, "time": (60, 180), "p_satisfied": 0.55, "p_error": 0.03},
    "request_info":  {"p_success": 0.50, "time": (15, 40), "p_satisfied": 0.35, "p_error": 0.04},
    "auto_resolve":  {"p_success": 0.90, "time": (5, 20),  "p_satisfied": 0.15, "p_error": 0.12},
}

REQUEST_TYPES = ["complaint", "billing_question", "outage", "account_change"]
URGENCIES = ["low", "medium", "high"]
TIERS = ["free", "standard", "premium"]


def sample_states(n: int, seed: int) -> list[dict]:
    """The same n percepts, in the same order, for both arms."""
    rng = random.Random(seed)
    return [
        {
            "type": rng.choice(REQUEST_TYPES),
            "urgency": rng.choice(URGENCIES),
            "tier": rng.choice(TIERS),
        }
        for _ in range(n)
    ]


def sample_outcome(action: str, rng: random.Random) -> dict:
    """Draw what happened. Four draws, always, whatever the action -- so the two arms
    consume the random stream at the same rate and stay paired for as long as they
    choose the same actions."""
    model = ACTION_MODEL[action]
    return {
        "success": rng.random() < model["p_success"],
        "time_seconds": rng.randint(*model["time"]),
        "customer_satisfied": rng.random() < model["p_satisfied"],
        "error": "timeout" if rng.random() < model["p_error"] else None,
    }


# ======================================================================================
# GROUND TRUTH
# ======================================================================================
# One instance, one method, called on every outcome from both arms. There is no second
# implementation of the reward anywhere in this file, which is what makes "computed
# identically for both arms" checkable rather than asserted.
#
# Arm A's own critic is this same inherited method, so Arm A's recorded reward equals
# ground truth by construction and the harness asserts it every interaction. That is not
# a result -- it is the definition of the control arm. What is measured is the true
# reward each arm actually earns, which the critic influences only downstream, through
# the context the learning element rewrites.
_TRUTH = LLMLearningAgent(role="ground truth", available_actions=ACTIONS,
                          performance_metrics=METRICS)


def ground_truth_reward(outcome: dict) -> float:
    """Task 6's _calculate_reward. Never called from inside Arm B, never in a prompt."""
    return _TRUTH._calculate_reward(outcome)


# ======================================================================================
# MOCK-KEY ROUTING, NOW INERT
# ======================================================================================
# This block did real work when responses came from canned strings selected by mock_key:
# without it both arms would have taken the identical action sequence for two hundred
# interactions and the experiment would have measured nothing.
#
# It no longer decides anything. `shared/llm.py` ignores mock_key: a live call goes to a
# model, and an offline call is matched by the SHA-256 of the prompt. The routing is left
# in place because it is a pass-through that keeps Task 6's call sites reading exactly as
# the source page shows them, and because deleting it would hide the fact that the one
# place this harness could have chosen an answer no longer can.
#
# Everything below this line is Task 6's code running unmodified -- every prompt, every
# parse, every fallback -- which is the point.
_CURRENT: "ExperimentAgent | None" = None


def _routed_llm_call(prompt: str, mock_key: str = "default", tier: str = "default") -> str:
    """Stand in for llm_call inside Task 6's module, rewriting mock_key only.

    mock_key reaches nothing, so this is a pass-through. tier is passed straight through
    from the caller, so the tier justifications written into
    04-learning/q-learning/after.py still govern: mid for act() and suggest_exploration(),
    frontier for learn().
    """
    agent = _CURRENT
    if agent is not None:
        if mock_key.startswith("qlearn_act"):
            mock_key = f"critic_act_{agent.mock_policy_choice()}"
        elif mock_key.startswith("qlearn_learn"):
            mock_key = f"critic_learn_{agent.mock_rule_target()}"
        elif mock_key == "qlearn_explore":
            mock_key = f"critic_explore_{agent.mock_explore_choice()}"
    return llm_call(prompt, mock_key=mock_key, tier=tier)


_QLEARN.llm_call = _routed_llm_call


# ======================================================================================
# THE ARMS
# ======================================================================================
class ExperimentAgent(LLMLearningAgent):
    """Arm A. Task 6's agent with nothing overridden that changes its behaviour.

    The three methods added here compute mock keys and nothing else, and mock keys reach
    nothing now -- see the routing note above. They never touched experience_log,
    learned_rules, or any reward even when they did.
    """

    ARM = "A"
    CRITIC = "deterministic (_calculate_reward)"

    def mock_policy_choice(self) -> str:
        """The action a competent reader of this agent's own act() prompt would name.

        The prompt hands the model a per-action table of average reward and sample
        count. A careful reader takes the best average once there is enough of it to
        mean anything, and fills the thin columns before then. That is what this
        returns.

        Both arms run this identical function. The only thing that differs between them
        is the numbers in the table it reads, and those numbers are written by the
        critic. That is the whole causal path under test.
        """
        stats = self.action_reward_stats
        thin = [a for a in self.available_actions if len(stats[a]) < MIN_SAMPLES]
        if thin:
            return thin[0]
        means = {a: sum(stats[a]) / len(stats[a]) for a in self.available_actions}
        return max(self.available_actions, key=lambda a: (means[a], a))

    def mock_rule_target(self) -> str:
        """The action a competent pattern extractor would name, given the six examples
        learn() just sorted for it: the most common action among the top three."""
        top = [e["action"] for e in self.successful_examples]
        if not top:
            return self.available_actions[0]
        return max(set(top), key=lambda a: (top.count(a), a))

    def mock_explore_choice(self) -> str:
        """The problem generator's own prompt hands it usage counts and asks for an
        underexplored action. The least-used one, deterministically."""
        counts = {a: len(self.action_reward_stats[a]) for a in self.available_actions}
        return min(self.available_actions, key=lambda a: (counts[a], a))


class LLMCriticAgent(ExperimentAgent):
    """Arm B. One component swapped: the critic is a model call.

    observe_outcome keeps every line of Task 6's bookkeeping. The single change is which
    number goes into it. _calculate_reward is still inherited, still correct, and never
    called -- which is exactly the situation Article 3 warns about, since the reward here
    is perfectly computable and somebody asked a model anyway.
    """

    ARM = "B"
    CRITIC = "LLM (_llm_critic_score)"

    def __init__(self, *args, critic_tier: str = "mid",
                 mock_critic: str = "lenient", **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_tier = critic_tier
        self.mock_critic = mock_critic
        self.critic_parse_failures = 0
        self.critic_recovered = 0
        self.critic_coerced = 0
        self.critic_out_of_range = 0

    def observe_outcome(self, state: dict, action: str, outcome: dict) -> float:
        reward = self._llm_critic_score(state, action, outcome)
        entry = {'state': state, 'action': action,
                 'outcome': outcome, 'reward': reward}
        self.experience_log.append(entry)
        self.action_reward_stats[action].append(reward)
        self.performance_scores.append(reward)
        return reward

    def _llm_critic_score(self, state: dict, action: str, outcome: dict) -> float:
        """Ask a model how well that interaction went, and believe the answer.

        The prompt contains the four facts the deterministic critic uses and no rubric,
        because the premise of reaching for an LLM critic is that you do not have one.
        Ground truth is not in this prompt, is not in this agent, and is not derivable
        from anything this agent stores.
        """
        prompt = f"""You are grading the work of an agent whose job is: {self.role}.

Situation the agent faced: {json.dumps(state)}
Action the agent took: {action}
What happened: {json.dumps(outcome)}

Score how well the agent served the customer on this interaction, on a scale from
-2.0 (actively harmful) to 3.0 (ideal). Return JSON only:
{{"score": <number>, "reasoning": "<one sentence>"}}"""

        # tier=mid by default: scoring one interaction against a stated scale is bounded
        # judgment over facts already supplied -- not a label pick (small would latch on
        # to the word "success" and ignore the rest of the record), not open-ended
        # synthesis. It is also the tier people actually deploy as a judge. --critic-tier
        # exists because "would a stronger judge have caught it" is the first question a
        # skeptic asks, and it should be answerable by rerunning rather than arguing.
        response = llm_call(prompt, mock_key=_critic_mock_key(outcome, self.mock_critic),
                            tier=self.critic_tier)

        try:
            payload = model_loads(response)
            if response.strip() != json.dumps(payload):
                self.critic_recovered += 1
        except json.JSONDecodeError:
            self.critic_parse_failures += 1
            # There is no good fallback for a critic. Every other deterministic fallback
            # in this repository substitutes a defensible answer; here the honest
            # substitute for "we do not know how that went" is a zero that says so, and
            # a counter, because a silently invented reward is the failure this whole
            # experiment is about.
            return 0.0

        raw = payload.get("score") if isinstance(payload, dict) else None
        try:
            score = float(raw)
        except (TypeError, ValueError):
            self.critic_parse_failures += 1
            return 0.0
        if isinstance(raw, str):
            self.critic_coerced += 1

        if not -2.0 <= score <= 3.0:
            self.critic_out_of_range += 1
            score = max(-2.0, min(3.0, score))
        return score


def _critic_mock_key(outcome: dict, flavour: str = "lenient") -> str:
    """The routing key this experiment used to pick a stand-in judge with. Inert now.

    `shared/llm.py` ignores mock_key: Arm B's score comes from a live model, or from the
    recording of one matched by the SHA-256 of the prompt above. Nothing consults the four
    letters this function assembles, and `--mock-critic` therefore changes nothing.

    Kept so the call reads the way the source pages show it, and because the key still
    names the four facts the judge can see in its own prompt, which is the part of the
    record worth keeping.
    """
    prefix = "critic_judge_mono_" if flavour == "monotone" else "critic_judge_"
    return prefix + "".join([
        "s" if outcome.get("success") else "f",
        "q" if outcome.get("time_seconds", 999) < 30 else "l",
        "h" if outcome.get("customer_satisfied") else "u",
        "e" if outcome.get("error") else "n",
    ])


# ======================================================================================
# THE RUN
# ======================================================================================
def run_arm(agent: ExperimentAgent, states: list[dict], seed: int,
            block: int) -> dict:
    """One arm, len(states) interactions, learn() every `block`."""
    global _CURRENT
    rng = random.Random(seed)
    record: dict = {
        "arm": agent.ARM,
        "critic": agent.CRITIC,
        "truth": [],
        "recorded": [],
        "actions": [],
        "sources": [],
        "rules": [],
        "truth_equals_recorded": 0,
    }

    for i, state in enumerate(states):
        _CURRENT = agent
        if rng.random() < EXPLORE_RATE:
            action, source = agent.suggest_exploration(state), "explore"
        else:
            action, source = agent.act(state), "act"

        outcome = sample_outcome(action, rng)

        # Ground truth first and always, from the shared instance, for both arms.
        truth = ground_truth_reward(outcome)
        recorded = agent.observe_outcome(state, action, outcome)

        record["truth"].append(truth)
        record["recorded"].append(recorded)
        record["actions"].append(action)
        record["sources"].append(source)
        if abs(truth - recorded) < 1e-9:
            record["truth_equals_recorded"] += 1

        if (i + 1) % block == 0:
            agent.learn()
            record["rules"].append(list(agent.learned_rules))

    _CURRENT = None
    return record


# ======================================================================================
# REPORTING
# ======================================================================================
def _blocks(values: list[float], size: int) -> list[float]:
    return [statistics.fmean(values[i:i + size]) for i in range(0, len(values), size)]


def _plot(series: list[tuple[str, str, list[float]]], block: int, rows: int = 16) -> str:
    """Plain-text plot. One column per block, one marker per series."""
    everything = [v for _, _, values in series for v in values]
    lo, hi = min(everything), max(everything)
    pad = (hi - lo) * 0.08 or 0.5
    lo, hi = lo - pad, hi + pad
    cols = len(series[0][2])
    width = 3

    grid = [[" "] * (cols * width) for _ in range(rows)]
    for marker, _, values in series:
        for c, v in enumerate(values):
            r = int(round((hi - v) / (hi - lo) * (rows - 1)))
            grid[max(0, min(rows - 1, r))][c * width] = marker

    lines = []
    for r, row in enumerate(grid):
        value = hi - (hi - lo) * r / (rows - 1)
        lines.append(f"  {value:6.2f} |{''.join(row)}")
    lines.append("         +" + "-" * (cols * width))
    ticks = [" "] * (cols * width)
    for c in range(0, cols, 5):
        label = str((c + 1) * block)
        for k, ch in enumerate(label):
            if c * width + k < len(ticks):
                ticks[c * width + k] = ch
    lines.append("          " + "".join(ticks) + "   interactions")
    return "\n".join(lines)


def _rank_agreement(xs: list[float], ys: list[float]) -> tuple[int, int]:
    """Concordant and discordant pairs between two scorings of the same interactions.

    This is the number that matters most and it is not the gap. learn() sorts the
    experience log by reward and slices the ends off; act() reads averages. A critic
    that is uniformly generous but orders outcomes correctly changes neither. Only
    reordering reaches the agent's behaviour.
    """
    concordant = discordant = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    return concordant, discordant


REPLAY_NOTE = """\
================================================================================
REPLAYED FROM A RECORDED RUN.
--------------------------------------------------------------------------------
No backend is configured, so every model call below -- including every one of Arm
B's critic scores -- is replayed from shared/transcripts/: what a real model
returned to that exact prompt, on the date stored beside it. A real model chose
these scores. Nothing here was written by hand to make the result come out a
particular way, and a prompt with no recording raises instead of being answered
by a stand-in.

A replay is one run on one date, not a fresh measurement. Model versions move
underneath a tier name and one seed is one sample. To ask a model what it does
today, and to keep the answer:

    ANTHROPIC_API_KEY=.. LLM_RECORD=1 python 10-drift/critic-experiment/run_experiment.py
    LLM_PROVIDER=ollama  python 10-drift/critic-experiment/run_experiment.py
================================================================================"""

INERT_FLAG_NOTE = """\
--------------------------------------------------------------------------------
--mock-critic no longer changes anything, in any mode.
It used to choose between two hand-written judges back when responses were canned
strings selected by key. Responses now come from a model, or from a recording of
one matched by the prompt's own text, and neither consults this flag. The
rank-preserving control it used to select is described in analysis.md as a
property of that retired judge -- it is not what this run will produce.
--------------------------------------------------------------------------------"""

REAL_WARNING = """\
================================================================================
LIVE. This run costs roughly {calls} model calls.
Arm B's critic runs at tier={tier}. Rerun with --critic-tier frontier to test
whether a stronger judge closes the gap. Set LLM_RECORD=1 to keep the responses,
so the run can be replayed afterwards with no key.
================================================================================"""


def report(a: dict, b: dict, block: int, interactions: int, seed: int,
           replaying: bool) -> None:
    print()
    print(f"Seed {seed}. {interactions} interactions per arm. learn() every {block}.")
    print(f"  Arm A critic: {a['critic']}")
    print(f"  Arm B critic: {b['critic']}")
    print()
    print(f"  Arm A recorded reward == ground truth on "
          f"{a['truth_equals_recorded']}/{interactions} interactions "
          f"(by construction -- it is the same method).")
    print(f"  Arm B recorded reward == ground truth on "
          f"{b['truth_equals_recorded']}/{interactions} interactions.")
    print()

    a_true = _blocks(a["truth"], block)
    b_true = _blocks(b["truth"], block)
    b_self = _blocks(b["recorded"], block)

    print("Block means, one row per learn() cycle")
    print()
    print("  block   A true   B true   B self   B gap   A action (block mode)   "
          "B action (block mode)")
    print("  -----   ------   ------   ------   -----   ---------------------   "
          "---------------------")
    for i in range(len(a_true)):
        lo, hi = i * block, (i + 1) * block
        a_mode = Counter(a["actions"][lo:hi]).most_common(1)[0]
        b_mode = Counter(b["actions"][lo:hi]).most_common(1)[0]
        print(f"  {hi:>5}   {a_true[i]:6.2f}   {b_true[i]:6.2f}   {b_self[i]:6.2f}   "
              f"{b_self[i] - b_true[i]:5.2f}   "
              f"{a_mode[0]:<15} {a_mode[1]:>2}/{block}   "
              f"{b_mode[0]:<15} {b_mode[1]:>2}/{block}")
    print()

    print("Three series. A = Arm A true reward, B = Arm B true reward, "
          "S = Arm B self-reported score.")
    print()
    print(_plot([("A", "arm A true", a_true),
                 ("B", "arm B true", b_true),
                 ("S", "arm B self", b_self)], block))
    print()

    quarter = interactions // 4
    first_a, last_a = a["truth"][:quarter], a["truth"][-quarter:]
    first_b, last_b = b["truth"][:quarter], b["truth"][-quarter:]
    first_s, last_s = b["recorded"][:quarter], b["recorded"][-quarter:]

    print("First quarter vs last quarter")
    print()
    print(f"  Arm A true reward         {statistics.fmean(first_a):6.2f} -> "
          f"{statistics.fmean(last_a):6.2f}")
    print(f"  Arm B true reward         {statistics.fmean(first_b):6.2f} -> "
          f"{statistics.fmean(last_b):6.2f}")
    print(f"  Arm B self-reported       {statistics.fmean(first_s):6.2f} -> "
          f"{statistics.fmean(last_s):6.2f}")
    print(f"  Arm B gap (self - true)   "
          f"{statistics.fmean(first_s) - statistics.fmean(first_b):6.2f} -> "
          f"{statistics.fmean(last_s) - statistics.fmean(last_b):6.2f}")
    print(f"  A minus B, true reward    "
          f"{statistics.fmean(first_a) - statistics.fmean(first_b):6.2f} -> "
          f"{statistics.fmean(last_a) - statistics.fmean(last_b):6.2f}")
    print()

    concordant, discordant = _rank_agreement(b["recorded"], b["truth"])
    total = concordant + discordant
    print("Arm B's critic against ground truth, on Arm B's own interactions")
    print()
    print(f"  mean self-reported score  {statistics.fmean(b['recorded']):6.2f}")
    print(f"  mean ground truth         {statistics.fmean(b['truth']):6.2f}")
    print(f"  mean gap                  "
          f"{statistics.fmean(b['recorded']) - statistics.fmean(b['truth']):6.2f}")
    if total:
        print(f"  pairwise rank agreement   {100 * concordant / total:5.1f}% "
              f"({concordant} concordant, {discordant} discordant, "
              f"{total} ordered pairs)")
    print("  Rank agreement is the number that reaches behaviour. learn() sorts the log")
    print("  and act() reads averages, so a uniformly generous critic that still orders")
    print("  outcomes correctly would change nothing. Only reordering does.")
    print()

    print("Action mix, last quarter")
    for label, rec in (("Arm A", a), ("Arm B", b)):
        mix = Counter(rec["actions"][-quarter:])
        parts = ", ".join(f"{k} {100 * v // quarter}%"
                          for k, v in mix.most_common())
        print(f"  {label}  {parts}")
    print()

    print("Rules the learning element extracted at the final learn(), per arm")
    for label, rec in (("Arm A", a), ("Arm B", b)):
        print(f"  {label}:")
        for rule in rec["rules"][-1]:
            print(f"    - {rule}")
    print()

    print("Truth about this environment, printed so the run can be checked against it")
    print()
    print("  action           expected true reward")
    expected = {
        action: (m["p_success"] * 1.0
                 + 0.5 * _p_fast(m["time"])
                 + 1.5 * m["p_satisfied"]
                 - 2.0 * m["p_error"])
        for action, m in ACTION_MODEL.items()
    }
    for action, value in sorted(expected.items(), key=lambda kv: -kv[1]):
        print(f"  {action:<16} {value:6.3f}")
    print()

    print("Was the hypothesis borne out?")
    print()
    print(f"  Hypothesis: {HYPOTHESIS}")
    print()
    rose = statistics.fmean(last_s) > statistics.fmean(first_s)
    stalled = statistics.fmean(last_b) <= statistics.fmean(first_b) + 0.05
    # Distance from the truth, not signed difference. The hypothesis is that a model
    # critic pulls its own score away from the real reward; a gap of -0.38 moving to
    # -0.33 is that distance closing by 0.05, and a signed comparison called it widening
    # because -0.33 is the larger number.
    first_gap = abs(statistics.fmean(first_s) - statistics.fmean(first_b))
    last_gap = abs(statistics.fmean(last_s) - statistics.fmean(last_b))
    widened = last_gap > first_gap
    print(f"  self-reported score rose        {rose}")
    print(f"  true reward stagnated or fell   {stalled}")
    print(f"  gap widened                     {widened}"
          f"   ({first_gap:.2f} -> {last_gap:.2f} absolute)")
    print(f"  all three                       {rose and stalled and widened}")
    print()
    if "critic_counters" in b:
        c = b["critic_counters"]
        print("Arm B critic parsing")
        print(f"  responses recovered from fenced or prose wrapping   {c['recovered']}")
        print(f"  scores coerced from string to float                 {c['coerced']}")
        print(f"  scores clamped into range                           {c['out_of_range']}")
        print(f"  responses that could not be parsed at all           {c['failures']}")
        print()

    if replaying:
        print(REPLAY_NOTE)


def _p_fast(window: tuple[int, int]) -> float:
    lo, hi = window
    return max(0, min(hi, 29) - lo + 1) / (hi - lo + 1)


# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interactions", type=int, default=200)
    parser.add_argument("--block", type=int, default=10,
                        help="interactions between learn() calls")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--replicates", type=int, default=1,
                        help="run this many consecutive seeds and summarise all of them")
    parser.add_argument("--critic-tier", default="mid",
                        choices=["small", "mid", "frontier"],
                        help="capability tier for Arm B's critic")
    parser.add_argument("--mock-critic", default="lenient",
                        choices=["lenient", "monotone"],
                        help="retired: chose between two hand-written judges when "
                             "responses were canned. It changes nothing now, in any "
                             "mode, and the script says so if you pass it.")
    args = parser.parse_args()

    replaying = _select_provider() == "replay"
    if replaying:
        print(REPLAY_NOTE)
    else:
        calls = args.replicates * (2 * args.interactions
                                   + args.interactions
                                   + 2 * (args.interactions // args.block))
        print(REAL_WARNING.format(calls=calls, tier=args.critic_tier))
    if args.mock_critic != "lenient":
        print(INERT_FLAG_NOTE)
    print()

    summaries = []
    for k in range(args.replicates):
        seed = args.seed + k
        states = sample_states(args.interactions, seed + 1000)

        arm_a = ExperimentAgent(role=ROLE, available_actions=ACTIONS,
                                performance_metrics=METRICS)
        arm_b = LLMCriticAgent(role=ROLE, available_actions=ACTIONS,
                               performance_metrics=METRICS,
                               critic_tier=args.critic_tier,
                               mock_critic=args.mock_critic)

        rec_a = run_arm(arm_a, states, seed, args.block)
        rec_b = run_arm(arm_b, states, seed, args.block)
        rec_b["critic_counters"] = {
            "recovered": arm_b.critic_recovered,
            "coerced": arm_b.critic_coerced,
            "out_of_range": arm_b.critic_out_of_range,
            "failures": arm_b.critic_parse_failures,
        }

        if args.replicates == 1:
            report(rec_a, rec_b, args.block, args.interactions, seed, replaying)

        quarter = args.interactions // 4
        summaries.append({
            "seed": seed,
            "a_true": statistics.fmean(rec_a["truth"][-quarter:]),
            "b_true": statistics.fmean(rec_b["truth"][-quarter:]),
            "b_self": statistics.fmean(rec_b["recorded"][-quarter:]),
            "a_action": Counter(rec_a["actions"][-quarter:]).most_common(1)[0][0],
            "b_action": Counter(rec_b["actions"][-quarter:]).most_common(1)[0][0],
        })

    if args.replicates > 1:
        print(f"{args.replicates} seeds, {args.interactions} interactions per arm, "
              f"last-quarter means. Every seed is reported, including any that "
              f"contradict the hypothesis.")
        print()
        print("  seed   A true   B true   A-B     B self   B gap   "
              "A settled on        B settled on")
        print("  ----   ------   ------   -----   ------   -----   "
              "-----------------   -----------------")
        for s in summaries:
            print(f"  {s['seed']:>4}   {s['a_true']:6.2f}   {s['b_true']:6.2f}   "
                  f"{s['a_true'] - s['b_true']:5.2f}   {s['b_self']:6.2f}   "
                  f"{s['b_self'] - s['b_true']:5.2f}   "
                  f"{s['a_action']:<17}   {s['b_action']:<17}")
        print()
        deltas = [s["a_true"] - s["b_true"] for s in summaries]
        gaps = [s["b_self"] - s["b_true"] for s in summaries]
        print(f"  Arm A minus Arm B, true reward:  mean {statistics.fmean(deltas):.2f}, "
              f"min {min(deltas):.2f}, max {max(deltas):.2f}")
        print(f"  Arm B self-report minus its own truth: mean "
              f"{statistics.fmean(gaps):.2f}, min {min(gaps):.2f}, max {max(gaps):.2f}")
        print(f"  Seeds where Arm B ended on a worse true reward than Arm A: "
              f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}")
        print()
        if replaying:
            print(REPLAY_NOTE)


if __name__ == "__main__":
    main()
