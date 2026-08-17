"""Two agents, one runtime, no agent-specific code.

    python 00-config-runtime/demo.py

Everything below drives `ConfigDrivenAgent` from runtime.py. The only difference between
the two runs is the directory path handed to the constructor. A simple reflex agent and a
model-based one, with different sensors, different actuators, different prompts, different
schemas and different capability tiers, are the same class pointed at different folders.

The last section checks that claim mechanically: it reads runtime.py as text and looks
for every name that belongs to a specific agent. A reader who does not trust the prose
can read that check instead.

Requires pyyaml. No API key -- with no provider configured, shared/llm.py returns canned
responses and prints a banner on the first call.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from runtime import (  # noqa: E402
    VALIDATOR_NAME,
    ActionRejected,
    ConfigDrivenAgent,
    PerceptRejected,
    validate,
)

AGENTS = HERE / "agents"

# One percept sequence per agent. These are the percepts from 01-reflex-agents/simple and
# 01-reflex-agents/model-based, in the shape their sensor schemas declare.
# Deliberately malformed. Missing the required key, an unknown key, and a value outside
# the declared enum. Fed to each agent to show that validation happens before the model
# is called, not after it answers.
# A percept no agent can accept: an undeclared key, which every sensor schema in this
# directory rejects because they all set additionalProperties false. Generic on purpose,
# so the refusal check works for an agent this file has never heard of.
MALFORMED = {"not_a_declared_field": "this is not a percept for any agent here"}


def percepts_from_eval(agent: ConfigDrivenAgent, limit: int = 3) -> list[dict]:
    """The percepts this agent's own evaluation suite already declares.

    Hardcoding a percept list here would be agent-specific knowledge in the demo, and
    adding an agent would mean editing this file -- which is the thing the directory
    claims you never have to do. The eval cases are percepts the agent is expected to
    handle, written next to the agent, so they are the honest source.
    """
    cases = json.loads((Path(agent.config_path).parent /
                        agent.config["performance"]["eval"]).read_text(encoding="utf-8"))
    cases = cases.get("cases", cases) if isinstance(cases, dict) else cases
    return [case["input"] for case in cases[:limit] if "input" in case]


def run_agent(agent_dir: Path, percepts: list[dict] | None = None) -> ConfigDrivenAgent:
    agent = ConfigDrivenAgent(agent_dir)
    if percepts is None:
        percepts = percepts_from_eval(agent)

    print(f"=== {agent.name} " + "=" * (58 - len(agent.name)))
    print(f"driven by:    {agent.config_path.relative_to(REPO).as_posix()}")
    print(f"architecture: {agent.config['architecture']}")
    print(f"tier:         {agent.tier}")
    print(f"prompts:      {', '.join(sorted(agent.prompts))}")
    print(f"schemas:      {', '.join(sorted(agent.schemas))}"
          + ("  (+ state)" if agent.state_schema is not None else ""))
    print()

    for percept in percepts:
        print(f"see:  {percept}")
        try:
            result = agent.run(percept)
        except PerceptRejected as refusal:
            print("  [refused] no sensor schema accepted this percept, no model called")
            for sensor, errors in refusal.errors.items():
                print(f"            {sensor}: {errors[0]}")
            print()
            continue
        except ActionRejected as refusal:
            print(f"  [refused] {refusal}")
            for error in refusal.errors:
                print(f"            {error}")
            print()
            continue
        print(f"  sensor:      {result['sensor']}")
        print(f"  task prompt: {result['task_prompt']}")
        print(f"  do:          {result['action']}  {result['args']}")
        if agent.state is not None:
            print(f"  state:       {agent.state}")
        print()

    print("performance, as observed by this run:")
    for line in agent.performance.report():
        print(f"  {line}")
    print()
    return agent


def run_eval(agent: ConfigDrivenAgent) -> None:
    results = agent.evaluate()
    passed = sum(1 for row in results if row["passed"])
    print(f"eval, {agent.config['performance']['eval']} "
          f"-- {passed} of {len(results)} cases matched:")
    for row in results:
        mark = "pass" if row["passed"] else "FAIL"
        print(f"  [{mark}] {row['id']}: expected {row['expected']!r}, "
              f"observed {row['observed']!r}")
        if not row["passed"]:
            print(f"         {row['note']}")
    print()


def check_no_agent_specific_code() -> None:
    """Look for anything in runtime.py that names a specific agent.

    Every token is pulled out of the agent directories at check time, so adding an agent
    widens the check automatically. If runtime.py ever grows a special case for one agent,
    the name of that agent has to appear in it, and this fails.

    The search is over runtime.py's STRING LITERALS, parsed with `ast`, not over its text.
    Agent-specific code reads a value out of a config or compares against a name, and both
    of those are string literals; a method called `report` is not agent-specific knowledge
    just because an agent has a sensor called `report`. Substring-searching the raw text
    said otherwise the moment a sixth agent arrived, and a check that constrains what an
    agent author may name a field is enforcing the coupling it exists to forbid.

    Comments and docstrings are excluded for the same reason -- prose about an agent is
    not a branch on one. A comment naming an agent is still worth avoiding, and there is
    no way for this check to tell one from an explanation without reading it.
    """
    tree = ast.parse((HERE / "runtime.py").read_text(encoding="utf-8"))
    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    literals = {
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node not in docstrings
    }

    tokens: set[str] = set()
    # Same guard the discovery loop uses. Without it an empty directory left behind by a
    # removed agent -- git rm takes the files and leaves the directories -- makes this
    # check die on a missing agent.yaml instead of reporting on the agents that exist.
    for agent_dir in sorted(d for d in AGENTS.iterdir() if (d / "agent.yaml").is_file()):
        config = yaml.safe_load((agent_dir / "agent.yaml").read_text(encoding="utf-8"))["agent"]
        tokens.add(agent_dir.name)
        tokens.add(config["name"])
        for actuator in config["actuators"]:
            tokens.add(actuator["name"])
            tokens.add(actuator.get("output_schema", ""))
        for sensor in config["sensors"]:
            tokens.add(sensor["name"])
            tokens.add(sensor.get("input_schema", ""))
        for path in config["prompts"].values():
            tokens.add(path)
        if "state" in config:
            tokens.add(config["state"]["schema"])
    tokens.discard("")

    found = sorted(token for token in tokens
                   if any(token.lower() in literal for literal in literals))
    print(f"no-agent-specific-code check: {len(tokens)} agent-specific names "
          f"(actuators, sensors, prompt files, schema files, agent names) "
          f"searched against runtime.py's string literals")
    if found:
        print(f"  FAIL: runtime.py mentions {found}")
        raise SystemExit(1)
    print("  none of them appear. The runtime does not know which agent it is running.")


def main() -> None:
    directories = sorted(d for d in AGENTS.iterdir() if (d / "agent.yaml").is_file())
    print(f"Config-driven runtime. {len(directories)} agent directories, "
          f"one ConfigDrivenAgent class.")
    print(f"schema validator: {VALIDATOR_NAME}")
    print()

    for agent_dir in directories:
        agent = run_agent(agent_dir)

        print("The same agent, handed a percept that is not a percept:")
        print(f"see:  {MALFORMED}")
        try:
            agent.run(MALFORMED)
            print("  [bug] the malformed percept was accepted")
            raise SystemExit(1)
        except PerceptRejected as refusal:
            print("  [refused] no sensor schema accepted this percept, no model called")
            for sensor, errors in sorted(refusal.errors.items())[:2]:
                print(f"            {sensor}: {errors[0]}")
        print()

        if agent.state_schema is not None:
            # The cost of the declared fallback, stated rather than assumed. After a
            # hand-merged percept the agent's beliefs no longer satisfy the schema its own
            # config declares.
            drift = validate(agent.state, agent.state_schema)
            print("does the final state still satisfy the declared state schema?")
            if drift:
                print(f"  no -- {len(drift)} violation(s), because the fallback fired:")
                for error in drift[:3]:
                    print(f"    {error}")
                print("  the run survived a bad state update and paid for it in state quality")
            else:
                print("  yes")
            print()

        run_eval(agent)

    check_no_agent_specific_code()


if __name__ == "__main__":
    main()
