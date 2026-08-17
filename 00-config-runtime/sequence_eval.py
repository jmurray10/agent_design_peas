"""Test whether history changes an agent's answer, which `evaluate()` cannot.

    python 00-config-runtime/sequence_eval.py
    python 00-config-runtime/sequence_eval.py agents/triage-tuner

`evaluate()` judges each case on its percept. That is the right default -- a suite where
case four only passes because case three ran first tells you very little about case four
-- but it is structurally unable to test the claim that makes a model-based or learning
agent worth building: that what happened earlier changes what the agent does now.

This harness tests that claim the only way it can be tested, by running the same percept
twice with different histories in front of it:

    control    the percept alone, on a fresh agent
    primed     the same percept, after a preamble the agent is expected to learn from

A pass is the two answers differing in the declared direction. A test whose control and
primed answers are the same has found either an agent that is not using its state or a
preamble that carries nothing, and the report says which by printing both.

The distinction that matters, and the reason this file is separate rather than a flag on
`evaluate()`: a single-percept suite measures whether each action is reachable and
defensible on its own, which is what the HTTP contract promises a caller. A sequence suite
measures whether the agent has a memory that does anything. Both are worth having and they
answer different questions, so conflating them would leave neither answered.

A sequence file is `eval/sequences.json` beside the agent's `test_cases.json`:

    [{"id": "...", "why": "...",
      "preamble": [ <percept>, ... ],
      "percept": <percept>,
      "control_action": "<what it does with no history>",
      "primed_action":  "<what it should do after the preamble>"}]

Agents without one are reported as having no sequence claim to test, which is honest for
a stateless agent and a gap for a stateful one. The report says which of the two it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from runtime import ActionRejected, ConfigDrivenAgent, PerceptRejected  # noqa: E402


def _act(agent: ConfigDrivenAgent, percept: dict) -> str:
    """One turn, with a refusal reported rather than raised.

    A refusal is a legitimate answer here -- it is what the deterministic layer does with
    a percept it will not accept -- and a sequence test that crashed on one would be
    measuring the harness rather than the agent.
    """
    try:
        return agent.run(percept)["action"]
    except (PerceptRejected, ActionRejected) as refusal:
        return f"refused ({refusal.__class__.__name__})"


def run_sequences(agent_dir: Path) -> list[dict]:
    """Run every sequence declared for this agent. Empty list if it declares none."""
    agent_dir = Path(agent_dir)
    path = agent_dir / "eval" / "sequences.json"
    if not path.is_file():
        return []

    sequences = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for sequence in sequences:
        # A fresh agent per arm, so the control genuinely has no history. Reusing one and
        # clearing its state would leave the transcript's replay position advanced, and
        # the control would answer with the primed run's recording.
        control_agent = ConfigDrivenAgent(agent_dir)
        control = _act(control_agent, sequence["percept"])

        primed_agent = ConfigDrivenAgent(agent_dir)
        for percept in sequence.get("preamble", []):
            _act(primed_agent, percept)
        primed = _act(primed_agent, sequence["percept"])

        expected_control = sequence.get("control_action")
        expected_primed = sequence.get("primed_action")
        results.append({
            "id": sequence["id"],
            "why": sequence.get("why", ""),
            "control": control,
            "primed": primed,
            "expected_control": expected_control,
            "expected_primed": expected_primed,
            "control_ok": expected_control is None or control == expected_control,
            "primed_ok": expected_primed is None or primed == expected_primed,
            "history_mattered": control != primed,
            "state_after": dict(primed_agent.state or {}),
        })
    return results


def report(agent_dir: Path, results: list[dict]) -> bool:
    """Print one agent's sequence results. Returns True if every declared test passed."""
    agent = ConfigDrivenAgent(agent_dir)
    name = agent.config["name"]
    stateful = agent.state_schema is not None

    if not results:
        if stateful:
            print(f"  {name:<17} declares state and no sequence test. Its memory is "
                  f"untested.")
        else:
            print(f"  {name:<17} stateless, so there is no sequence claim to test.")
        # A stateless agent with no sequence file is complete, not failing.
        return not stateful

    ok = True
    print(f"  {name}")
    for row in results:
        passed = row["control_ok"] and row["primed_ok"] and row["history_mattered"]
        ok &= passed
        print(f"    {'ok  ' if passed else 'FAIL'} {row['id']}")
        print(f"         control {row['control']}"
              + (f"  (expected {row['expected_control']})" if not row["control_ok"] else ""))
        print(f"         primed  {row['primed']}"
              + (f"  (expected {row['expected_primed']})" if not row["primed_ok"] else ""))
        if not row["history_mattered"]:
            # The failure worth naming precisely: the agent answered identically with and
            # without the history, so nothing it carried reached the decision.
            print(f"         the two answers are the same, so the preamble changed nothing")
        if row["why"]:
            print(f"         {row['why']}")
        if row["state_after"]:
            print(f"         state carried: {', '.join(sorted(row['state_after']))}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("agent_dir", nargs="?",
                        help="one agent; default is every agent under agents/")
    args = parser.parse_args()

    if args.agent_dir:
        directories = [Path(args.agent_dir)]
    else:
        directories = sorted(d for d in (HERE / "agents").iterdir()
                             if (d / "agent.yaml").is_file())

    print("Sequence evaluation: does history change the answer?")
    print("Each test runs the same percept twice -- once cold, once after a preamble.")
    print()

    all_ok = True
    for directory in directories:
        all_ok &= report(directory, run_sequences(directory))
    print()
    print("Every stateful agent has a sequence test and every test passed."
          if all_ok else
          "Above: a stateful agent with no sequence test, or a test whose two arms agreed.")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
