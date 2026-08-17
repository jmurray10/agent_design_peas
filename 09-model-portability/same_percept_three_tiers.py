"""The same percept, three models, one harness.

    python 09-model-portability/same_percept_three_tiers.py
    python 09-model-portability/same_percept_three_tiers.py --agent claims-intake

Every other comparison in this repository asks which model is better. This one asks a
narrower question that the thesis actually depends on: **how much of the answer was the
model, and how much was the harness around it.**

Take a real percept from an agent's evaluation suite and put it through the same agent
three times, changing nothing but the capability tier -- `small` to `claude-haiku-4-5`,
`mid` to `claude-sonnet-5`, `frontier` to `claude-opus-5`. Same prompt, same sensor
schemas, same actuator list, same validation on both sides.

Two outcomes and both are informative:

    all three agree     The architecture decided this, not the model. The percept, the
                        actuator list and the prompt narrowed it to one defensible
                        answer, and a cheaper model reaches it. These are the decisions
                        you do not need to spend on.

    they disagree       This is where model capability is actually purchasing something,
                        and it is worth knowing which decisions those are rather than
                        assuming it is all of them.

What no row can show is a model choosing an action that does not exist, or returning
arguments that fail the actuator schema. That is not because the models are good. It is
because the deterministic halves reject it before the caller ever sees it, and they reject
it identically for all three.

Costs real calls: one per tier per case. With no backend configured it replays, and a
replay of three tiers is three recordings that were made against three models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "00-config-runtime"))

from runtime import ActionRejected, ConfigDrivenAgent, PerceptRejected  # noqa: E402

TIERS = ["small", "mid", "frontier"]
AGENTS_ROOT = ROOT / "00-config-runtime" / "agents"

# Agents whose decisions are judgement calls rather than lookups, so agreement across
# tiers is worth something. A goal-based agent would agree trivially -- its answer is
# decided by a solver -- and that is a different claim, made in 02-goal-based/csp/.
DEFAULT_AGENTS = ["claims-intake", "claim-reserve", "safety-signal"]


def cases_for(agent_dir: Path, config: dict) -> list[dict]:
    raw = json.loads((agent_dir / config["performance"]["eval"]).read_text(encoding="utf-8"))
    return raw["cases"] if isinstance(raw, dict) else raw


def run_one(agent_dir: Path, percept: dict, tier: str) -> tuple[str, str | None]:
    """One turn at a forced tier. Returns the action and the model behind it."""
    import shared.llm as llm_module

    agent = ConfigDrivenAgent(agent_dir)
    # The only thing that changes between rows. Everything else -- prompt, schemas,
    # actuator list, validation -- is the same object.
    agent.tier = tier
    # Each tier's answers go in their own transcript. A transcript entry is keyed by
    # prompt content and holds one model, so without this the three tiers would record
    # over each other and the agent would later replay whichever ran last -- under a tier
    # it never asked for. The tier check in shared/transcript.py catches that rather than
    # serving it, which is how this was found.
    agent.transcript_suffix = f"__tier_{tier}"
    try:
        result = agent.run(percept)
        return result["action"], result.get("model")
    except (PerceptRejected, ActionRejected) as refusal:
        return f"refused ({refusal.__class__.__name__})", llm_module.last_model()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--agent", action="append", dest="agents",
                        help="agent directory name; repeatable. Default: three judgement agents.")
    args = parser.parse_args()
    names = args.agents or DEFAULT_AGENTS

    print("The same percept, three models, one harness")
    print()
    print("  Only the capability tier changes between columns. The prompt, the sensor")
    print("  schemas, the actuator list and both validation gates are identical.")
    print()

    agreed = disagreed = 0
    models_seen: dict[str, str] = {}

    for name in names:
        agent_dir = AGENTS_ROOT / name
        if not (agent_dir / "agent.yaml").is_file():
            print(f"  no such agent: {name}")
            continue
        import yaml

        config = yaml.safe_load((agent_dir / "agent.yaml").read_text(encoding="utf-8"))["agent"]
        cases = cases_for(agent_dir, config)

        print(f"  {name}  ({config['architecture']}, declared tier {config['behavior']['tier']})")
        print(f"    {'case':<30} {'small':<26} {'mid':<26} {'frontier':<26} same?")

        for case in cases:
            row = []
            for tier in TIERS:
                action, model = run_one(agent_dir, case["input"], tier)
                if model:
                    models_seen[tier] = model
                row.append(action)
            same = len(set(row)) == 1
            agreed += same
            disagreed += not same
            print(f"    {case['id'][:29]:<30} {row[0]:<26} {row[1]:<26} {row[2]:<26} "
                  f"{'yes' if same else 'NO'}")
        print()

    total = agreed + disagreed
    print("  Models behind the columns:")
    for tier in TIERS:
        print(f"    {tier:<9} {models_seen.get(tier, 'not reached')}")
    print()
    if total:
        print(f"  {agreed} of {total} percepts produced the same action from all three.")
        print()
        if agreed:
            print("  On those rows the architecture decided the outcome. The percept, the")
            print("  actuator list and the prompt left one defensible answer, and the")
            print("  cheapest model found it. That is the claim this repository makes about")
            print("  where an LLM belongs: inside a structure that has already narrowed the")
            print("  question, not in place of one.")
        if disagreed:
            print()
            print(f"  The other {disagreed} are where capability bought something. Those are the")
            print("  decisions worth paying a frontier tier for, and the point of measuring")
            print("  is that it is a shorter list than it feels like.")
    print()
    print("  Not a benchmark. One run, one date, three model versions, and a handful of")
    print("  percepts chosen because they are the cases these agents are asserted to")
    print("  handle rather than because they are hard.")


if __name__ == "__main__":
    main()
