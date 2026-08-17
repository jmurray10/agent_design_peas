"""One agent, four components, three tier-routing configurations.

The agent is `LLMLearningAgent`, loaded unmodified from
`04-learning/q-learning/after.py`. It has four components with genuinely different
requirements, which is what makes it worth routing:

    performance element   pick an action against supplied statistics   -> mid
    critic                arithmetic over observed outcomes            -> NO MODEL
    learning element      synthesise rules across sorted experience    -> frontier
    problem generator     name the least-understood action             -> mid

The same seeded 20-interaction scenario runs three times -- everything to frontier,
everything to small, and each component on the tier it asked for -- and the script
prints where the tokens went and what the three routings would cost under a stated
price assumption.

WHAT THIS SCRIPT DOES NOT DO: compare answer quality. In mock mode `shared/llm.py`
returns the same canned string for a given mock_key no matter which tier was
requested, so all three configurations produce byte-identical agent behaviour. The
script checks that and says so. It never scores an answer.

Runs with no API key, from the repo root:

    python 09-model-portability/tier-routing/mixed_agent.py
"""

from __future__ import annotations

import importlib.util
import inspect
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Run from the repo root, so this file's directory is on sys.path but the root is not.
# parents[0] = tier-routing, parents[1] = 09-model-portability, parents[2] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared import llm as shared_llm  # noqa: E402
from shared.llm import llm_call  # noqa: E402

QLEARN_PATH = REPO_ROOT / "04-learning" / "q-learning" / "after.py"


def _load(name: str, path: Path):
    """Import a module by file path.

    `04-learning` is not a legal package name, so the agent cannot be imported the
    normal way. Loading it under a private module name also means its demo block does
    not fire: that block is guarded by `__name__ == "__main__"` and this name is not
    that.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qlearn = _load("qlearn_after_tier_routing", QLEARN_PATH)


# -- the price assumption -------------------------------------------------------------
#
# ASSUMPTION. NOT A MEASUREMENT AND NOT A QUOTE FROM ANY VENDOR.
#
# Cost units per 1000 tokens, with the small tier defined as 1.0. These three numbers
# are stand-ins picked because the arithmetic is easy to follow, not because a price
# list says so. Every "cost units" figure this script prints is these numbers times a
# token count -- change them and every cost figure changes with them. To get a number
# that means something about your bill, replace them with your provider's published
# per-token prices for the models named in shared/providers.yaml, and note that real
# providers charge different rates for input and output tokens, which this model does
# not.
TIER_PRICE_UNITS = {"small": 1.0, "mid": 5.0, "frontier": 25.0}

CHARS_PER_TOKEN = 4

SEED = 7
INTERACTIONS = 20

# Component names, and the agent method that each one lives in. The router attributes
# a model call by looking at which of these methods called it.
PERFORMANCE = "performance element"
CRITIC = "critic"
LEARNING = "learning element"
PROBLEM_GEN = "problem generator"

COMPONENTS = [PERFORMANCE, CRITIC, LEARNING, PROBLEM_GEN]

COMPONENT_BY_METHOD = {
    "act": PERFORMANCE,
    "learn": LEARNING,
    "suggest_exploration": PROBLEM_GEN,
}

# The methods each component is made of, used for the source scan below. The critic is
# the only entry with more than one, and the only one whose scan is the point.
METHODS_BY_COMPONENT = {
    PERFORMANCE: ["act"],
    CRITIC: ["observe_outcome", "_calculate_reward"],
    LEARNING: ["learn"],
    PROBLEM_GEN: ["suggest_exploration"],
}


def estimate_tokens(text: str) -> int:
    """Four characters per token.

    An estimator, not a tokenizer. A real count needs the model's own vocabulary,
    which would be a dependency, and this repo installs nothing. The same estimator
    runs over every string in every configuration, so the comparisons between columns
    are on equal footing even though the absolute numbers are approximations.
    """
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


@dataclass
class ComponentUsage:
    """What one component did during one run. Every field is counted, none assumed."""

    component: str
    runs: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    reply_tokens: int = 0
    cost_units: float = 0.0
    tiers_requested: list[str] = field(default_factory=list)
    tiers_routed: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.reply_tokens

    def _label(self, tiers: list[str], empty: str) -> str:
        if not tiers:
            return empty
        return "/".join(sorted(set(tiers)))

    @property
    def requested_label(self) -> str:
        return self._label(self.tiers_requested, "nothing")

    @property
    def routed_label(self) -> str:
        return self._label(self.tiers_routed, "NO MODEL")


@dataclass
class Configuration:
    label: str
    override: str | None  # None means "honour whatever tier the component asked for"
    blurb: str


@dataclass
class Run:
    config: Configuration
    usage: dict[str, ComponentUsage]
    trace: list[tuple[int, str, str, str, float]]
    reward_total: float

    def total(self, attr: str):
        return sum(getattr(u, attr) for u in self.usage.values())


class TierRouter:
    """Sits between the agent's components and `llm_call`, and keeps the receipts.

    Installed in place of the name `llm_call` inside the loaded agent module, which is
    how each component gets routed and measured without editing the agent. That also
    makes attribution exact rather than guessed: when this object is called, the frame
    underneath it is the component method itself.
    """

    def __init__(self, config: Configuration):
        self.config = config
        self.usage = {name: ComponentUsage(name) for name in COMPONENTS}

    def record_run(self, component: str) -> None:
        """Note that a component executed. Says nothing about whether it called a model.

        The critic only ever gets counted here, which is the entire point of the
        example: it runs on every single interaction and reaches the model on none.
        """
        self.usage[component].runs += 1

    def __call__(self, prompt: str, mock_key: str = "default", tier: str = "default") -> str:
        caller = "unknown"
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller = frame.f_back.f_code.co_name
        component = COMPONENT_BY_METHOD.get(caller, f"unattributed ({caller})")
        usage = self.usage.setdefault(component, ComponentUsage(component))

        routed = self.config.override or tier
        response = llm_call(prompt, mock_key=mock_key, tier=routed)

        prompt_tokens = estimate_tokens(prompt)
        reply_tokens = estimate_tokens(response)
        usage.model_calls += 1
        usage.prompt_tokens += prompt_tokens
        usage.reply_tokens += reply_tokens
        usage.tiers_requested.append(tier)
        usage.tiers_routed.append(routed)
        usage.cost_units += (prompt_tokens + reply_tokens) / 1000.0 * TIER_PRICE_UNITS[routed]
        return response


def run_configuration(config: Configuration, verbose: bool) -> Run:
    """Run the 20-interaction scenario once under one routing configuration.

    Each configuration records into its own transcript, for the same reason
    `same_percept_three_tiers.py` gives one to each tier. All three runs send byte-identical
    prompts -- that is the design, since only the routed tier is meant to vary -- so a
    single transcript keys all three onto one entry, and whichever ran last wins. The next
    replay then serves one model's answer under a tier it never asked for, which turns a
    tier comparison into a model compared against itself.

    The tier check in `shared/transcript.py` refuses to serve that, and refusing is how this
    was found: it had been silent because every entry recorded before 2026-08-14 carried a
    null tier, and the check skips a null rather than guessing.
    """
    # Seeded per configuration, not per process, so all three runs see the same
    # exploration schedule and the same simulated outcomes. Without this the token
    # counts would differ between configurations for a reason that has nothing to do
    # with tiers.
    random.seed(SEED)

    agent = qlearn.LLMLearningAgent(
        role="email response agent",
        available_actions=["send_template", "write_custom", "escalate", "request_info",
                           "auto_resolve"],
        performance_metrics=["quality", "time_to_resolution", "satisfaction"],
    )

    router = TierRouter(config)
    original = qlearn.llm_call
    qlearn.llm_call = router
    trace: list[tuple[int, str, str, str, float]] = []
    try:
        for i in range(INTERACTIONS):
            state = {'type': 'complaint', 'urgency': 'medium', 'tier': 'premium'}

            # Same draw order as the demo in 04-learning/q-learning/after.py, so the
            # scenario is the same one and the traces can be compared against it.
            if random.random() < 0.15:
                component = PROBLEM_GEN
                action = agent.suggest_exploration(state)
            else:
                component = PERFORMANCE
                action = agent.act(state)
            router.record_run(component)

            outcome = {
                'success': random.random() > 0.3,
                'time_seconds': random.randint(5, 120),
                'customer_satisfied': random.random() > 0.4,
                'error': 'timeout' if random.random() < 0.15 else None,
            }

            reward = agent.observe_outcome(state, action, outcome)
            router.record_run(CRITIC)

            routed = router.usage[component].tiers_routed[-1]
            trace.append((i + 1, component, routed, action, reward))
            if verbose:
                print(f"  {i + 1:>2}  {component:<19}  tier={routed:<8}  {action:<14}"
                      f"  critic reward {reward:5.2f}")

            if (i + 1) % 10 == 0:
                agent.learn()
                router.record_run(LEARNING)
    finally:
        # Leave the agent module exactly as it was found.
        qlearn.llm_call = original

    return Run(config, router.usage, trace, sum(agent.performance_scores))


def scan_source_for_model_calls() -> list[tuple[str, str, int]]:
    """Count `llm_call(` in the source of each component's own methods.

    This is a measurement of the agent file on disk, not a claim made by this file.
    The critic's row is the one that matters and it is not hardcoded anywhere -- if
    somebody adds a model call to the critic tomorrow, this scan reports it.
    """
    rows = []
    for component in COMPONENTS:
        methods = METHODS_BY_COMPONENT[component]
        count = 0
        for method in methods:
            source = inspect.getsource(getattr(qlearn.LLMLearningAgent, method))
            count += source.count("llm_call(")
        rows.append((component, ", ".join(methods), count))
    return rows


def print_component_table(run: Run) -> None:
    print(f"  {'component':<19}  {'asks for':<9}  {'routed':<9}  {'runs':>4}  {'calls':>5}"
          f"  {'prompt tok':>10}  {'reply tok':>9}  {'cost units':>10}")
    print("  " + "-" * 88)
    # Any component the router saw but this file did not expect gets a row too, so the
    # rows always add up to the total line underneath them.
    names = COMPONENTS + [name for name in run.usage if name not in COMPONENTS]
    for name in names:
        usage = run.usage[name]
        # The critic's row carries a marker rather than blending into the table. It is
        # the row the example exists to show.
        lead = "> " if name == CRITIC else "  "
        print(f"{lead}{usage.component:<19}  {usage.requested_label:<9}  {usage.routed_label:<9}"
              f"  {usage.runs:>4}  {usage.model_calls:>5}  {usage.prompt_tokens:>10}"
              f"  {usage.reply_tokens:>9}  {usage.cost_units:>10.2f}")
    print("  " + "-" * 88)
    print(f"  {'total':<19}  {'':<9}  {'':<9}  {run.total('runs'):>4}"
          f"  {run.total('model_calls'):>5}  {run.total('prompt_tokens'):>10}"
          f"  {run.total('reply_tokens'):>9}  {run.total('cost_units'):>10.2f}")


CONFIGURATIONS = [
    Configuration("all frontier", "frontier",
                  "every model call routed to the frontier tier"),
    Configuration("all small", "small",
                  "every model call routed to the small tier"),
    Configuration("mixed", None,
                  "each component gets the tier its own job asked for"),
]


if __name__ == "__main__":
    # Read-only. shared/llm.py resolves the backend once and caches it; asking here
    # only resolves it earlier. The script needs to know, because what it is allowed
    # to claim about the three runs depends on whether they hit the same canned
    # strings or three different models.
    provider = shared_llm._select_provider()

    print("=" * 92)
    print("Component tier routing -- one agent, four components, three routings")
    print("=" * 92)
    print()
    print("Agent:    LLMLearningAgent, loaded unmodified from 04-learning/q-learning/after.py")
    print(f"Scenario: {INTERACTIONS} interactions, random seed {SEED}, identical in all three runs")
    print(f"Provider: {provider}   (selected by shared/llm.py, not by this script)")
    print()

    print("Where each component reaches a model -- counted in the agent's source, not asserted here:")
    print()
    print(f"  {'component':<19}  {'method(s) scanned':<38}  {'llm_call( found':>15}")
    print("  " + "-" * 78)
    for component, methods, count in scan_source_for_model_calls():
        tail = "   <-- NO MODEL AT ALL" if component == CRITIC else ""
        print(f"  {component:<19}  {methods:<38}  {count:>15}{tail}")
    print()
    print("  The critic computes the reward from observed facts -- did it succeed, how many")
    print("  seconds, was there an error, was the customer satisfied. There is no tier to")
    print("  choose for it, because there is nothing to choose a model for. It is the one")
    print("  component that no routing configuration below can change.")
    print()

    print("-" * 92)
    print("MEASURED on this run, and safe to quote:")
    print("  - which tier each component asked for, taken from the calls it actually made")
    print("  - how many times each component ran, and how many of those reached a model")
    print("  - prompt and reply sizes, estimated at 4 characters per token (see estimate_tokens)")
    print("  - the number of llm_call( occurrences in each component's source")
    print()
    print("ASSUMED, and not safe to quote as a price:")
    print("  - the cost units column. It is token counts times TIER_PRICE_UNITS, which this")
    print(f"    file assumes to be small={TIER_PRICE_UNITS['small']:.0f}, "
          f"mid={TIER_PRICE_UNITS['mid']:.0f}, frontier={TIER_PRICE_UNITS['frontier']:.0f} "
          "per 1000 tokens. Those three")
    print("    numbers are stand-ins chosen for readable arithmetic, not a vendor quote, and")
    print("    they price input and output the same, which no provider does. No figure below")
    print("    is a bill.")
    if provider == "mock":
        print("  - the reply half of every token count. In mock mode the replies are the canned")
        print("    responses in shared/transcripts/, so their length is a recording. The prompt half is")
        print("    built by the agent from its own state and is observed either way.")
    print()
    print("NOT MEASURED, and deliberately absent:")
    print("  - answer quality. Nothing below scores an answer. See the last section.")
    print("-" * 92)
    print()

    runs = []
    for config in CONFIGURATIONS:
        verbose = config.override is None  # the trace is printed once; see the check below
        print("=" * 92)
        print(f"CONFIGURATION: {config.label}  --  {config.blurb}")
        print("=" * 92)
        if verbose:
            print()
        else:
            print("  Per-interaction trace suppressed, for length only. It is printed once,")
            print("  under the mixed configuration below. Do not read the suppression as the")
            print("  runs being identical -- they are not, and the last section counts how")
            print("  many steps actually matched rather than asserting anything here.")
        # One transcript per configuration. See run_configuration's docstring: all three
        # send byte-identical prompts, so a shared transcript keys them onto one entry and
        # the last recording wins.
        slug = config.label.replace(" ", "_")
        with shared_llm.transcript_source(
                f"09_model_portability__tier_routing__mixed_agent__{slug}"):
            run = run_configuration(config, verbose=verbose)
        runs.append(run)
        print()
        print_component_table(run)
        print()
        critic = run.usage[CRITIC]
        print(f"  The critic ran {critic.runs} times and called a model {critic.model_calls} times.")
        print("  Zero tokens, zero cost, no tier, in this configuration and in every other one.")
        print()

    print("=" * 92)
    print("THE THREE CONFIGURATIONS SIDE BY SIDE")
    print("=" * 92)
    print()
    cheapest = min(r.total("cost_units") for r in runs)
    print(f"  {'configuration':<15}  {'model calls':>11}  {'est. tokens':>11}"
          f"  {'cost units':>10}  {'relative':>8}")
    print("  " + "-" * 62)
    for run in runs:
        total_tokens = run.total("prompt_tokens") + run.total("reply_tokens")
        cost = run.total("cost_units")
        ratio = cost / cheapest if cheapest else 0.0
        print(f"  {run.config.label:<15}  {run.total('model_calls'):>11}  {total_tokens:>11}"
              f"  {cost:>10.2f}  {ratio:>7.1f}x")
    print()

    token_totals = {r.total("prompt_tokens") + r.total("reply_tokens") for r in runs}
    if len(token_totals) == 1:
        print("  Read the token column before the cost column. It is identical across all three")
        print("  rows: the same components sent the same prompts and got back the same replies.")
        print("  The entire spread in the cost column therefore comes from TIER_PRICE_UNITS,")
        print("  which this file assumes. Set those three weights equal and the three")
        print("  configurations cost exactly the same.")
    else:
        print("  The token column differs between rows, so the models returned different text.")
        print("  That is a difference in verbosity, which is not a difference in quality.")
    print()
    print("  What the mixed row buys is not a smaller bill than all-small. It is a smaller bill")
    print("  than all-frontier while the one call that actually needs synthesis -- the learning")
    print("  element, whose rules get pasted into every later act() prompt -- still runs on the")
    print("  frontier tier.")
    print()

    print("=" * 92)
    print("THE CRITIC ROW")
    print("=" * 92)
    print()
    critic_runs = runs[0].usage[CRITIC].runs
    total_runs = runs[0].total("runs")
    print(f"  {critic_runs} of the {total_runs} component executions in each run were the critic.")
    print("  Model calls made by the critic, in all three configurations: "
          f"{sum(r.usage[CRITIC].model_calls for r in runs)}.")
    print()
    print("  Every other row on those tables moved when the routing changed. The critic's row")
    print("  is the same in all three, and it would be the same under any price assumption,")
    print("  because it is arithmetic. The reward it produces is the only ground truth in the")
    print("  loop: the learning element sorts on it, and the performance element then acts on")
    print("  what that sort produced. A component that scores the work cannot be the same kind")
    print("  of thing that produced the work.")
    print()
    print("  'Which tier for this component' is the wrong first question. 'Does this component")
    print("  need a model at all' is the first question, and for one component in four here the")
    print("  answer is no.")
    print()

    print("=" * 92)
    print("OUTCOME QUALITY")
    print("=" * 92)
    print()
    baseline = runs[0]
    identical_steps = 0
    for run in runs[1:]:
        for a, b in zip(baseline.trace, run.trace):
            # Compare component, action and reward. The routed tier is expected to
            # differ -- that is the independent variable.
            if (a[1], a[3], a[4]) == (b[1], b[3], b[4]):
                identical_steps += 1
    comparisons = len(baseline.trace) * (len(runs) - 1)
    rewards = [f"{r.reward_total:.2f}" for r in runs]

    print(f"  Steps that matched the first configuration: {identical_steps} of {comparisons}")
    print(f"  Reward totals, one per configuration: {', '.join(rewards)}")
    print()
    if provider == "replay":
        print("  Every response above came out of shared/transcripts/, and transcripts are")
        print("  keyed by tier, so the three configurations really did replay three different")
        print("  models' answers. The divergence is a real one between real models. What it")
        print("  is not is a measurement of today: it is the difference those models produced")
        print("  on the date recorded beside each entry, and it will not move until someone")
        print("  records again.")
        print()
        print("  This script reports no quality comparison either way, and the cheapest column")
        print("  above is not an argument for the small tier. Nothing here scores an answer.")
    else:
        print(f"  Provider is {provider}, so the three configurations really were answered by")
        print("  different models just now, and any difference above is a real difference. It")
        print("  is still one sample of one scenario, which is not a quality measurement.")
        print("  Scoring a routing configuration needs a labelled eval set, repeated runs, and")
        print("  a metric fixed before the run -- none of which is in this file.")
    print()
    print("=" * 92)
