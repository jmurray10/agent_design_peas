"""The runtime that the config-driven pattern says you have to write yourself.

The source page for Article 1 is explicit that this pattern is "not a framework you can
install": the YAML defines a shape, and any team adopting it builds the runtime that
reads that shape. This file is that runtime, for the agents in this repository.

`ConfigDrivenAgent` is the class from the source page, completed. It loads `agent.yaml`,
loads prompts and schemas from the files the config points at, and runs one turn of the
oscillation:

    DETERMINISTIC  validate the raw percept against the sensor schemas
    LLM            assemble system prompt + routed task prompt + percept, ask for an action
    DETERMINISTIC  parse the answer, check the actuator exists, validate against its schema
    DETERMINISTIC  merge any state update, or fall back to the config's declared fallback
    DETERMINISTIC  dispatch, record

The constraint that makes it worth reading: there is no agent-specific code below this
line. No name checks, no special cases, nothing that knows what any particular agent
does. Every difference between two agents lives in their directories. `demo.py` checks
that claim mechanically rather than asking you to take it on faith.

Requires pyyaml. That is the only third-party dependency anywhere in this repository,
and it lives here because reading YAML is the whole job.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Run from the repo root as `python 00-config-runtime/demo.py`. Python puts the script's
# own directory on sys.path, never the root, so the root has to be added by hand.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

import shared.llm as llm_module
from shared.llm import llm_call
from shared.model_json import loads as model_loads  # noqa: E402

# Schema validation is the deterministic half of the pattern, so the validator must never
# be the reason a reader cannot run this. jsonschema if it happens to be installed, a
# hand-rolled required-key/type/enum check if not. Which one ran is printed at startup:
# the two accept and reject the same inputs for the schemas in this directory, but they
# word their complaints differently, and a reader comparing output to the README should
# be able to tell which they got.
try:
    import jsonschema

    VALIDATOR_NAME = "jsonschema (installed)"
except ImportError:  # pragma: no cover - depends on the reader's machine, not on input
    jsonschema = None
    VALIDATOR_NAME = "built-in key/type check (jsonschema not installed)"


class PerceptRejected(ValueError):
    """Raised when raw input matches none of the declared sensor schemas."""

    def __init__(self, errors: dict[str, list[str]]):
        self.errors = errors
        super().__init__("percept matched no declared sensor schema")


class ActionRejected(ValueError):
    """Raised when a model answer is not a legal action for this agent."""

    def __init__(self, reason: str, errors: list[str] | None = None):
        self.errors = errors or []
        super().__init__(reason)


# -- validation ----------------------------------------------------------------------


def validate(instance: Any, schema: dict, label: str = "root") -> list[str]:
    """Return a list of schema violations. Empty list means valid.

    No model is involved and no exception escapes: the caller decides what a violation
    means. That is the whole point of validating in code rather than asking the model to
    notice that its own answer was wrong.
    """
    if jsonschema is not None:
        errors = jsonschema.Draft202012Validator(schema).iter_errors(instance)
        return sorted(
            f"{'.'.join(str(p) for p in e.path) or 'root'}: {e.message}" for e in errors
        )
    return sorted(_builtin_validate(instance, schema, label))


def _builtin_validate(instance: Any, schema: dict, label: str = "root") -> list[str]:
    """The no-dependency path: declared type, enum, required keys, property types.

    Deliberately partial. It covers exactly the keywords the schemas in `agents/` use,
    because a half-implemented validator that silently ignores a keyword is worse than
    one that never saw it.
    """
    expected = schema.get("type")
    if expected is not None and not _type_matches(instance, expected):
        return [f"{label}: expected {expected}, got {_json_type_name(instance)}"]

    errors: list[str] = []
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{label}: {instance!r} is not one of {schema['enum']}")

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{label}: required key {key!r} is missing")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{label}: {key!r} is not a declared property")
        for key, value in instance.items():
            if key in properties:
                child = key if label == "root" else f"{label}.{key}"
                errors += _builtin_validate(value, properties[key], child)
    return errors


_JSON_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, one) for one in expected)
    # JSON Schema treats true as a boolean and never as a number. Python disagrees --
    # bool subclasses int -- so the check has to be made before isinstance runs.
    if isinstance(value, bool) and expected in ("number", "integer"):
        return False
    if expected == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, _JSON_TYPES[expected])


def _json_type_name(value: Any) -> str:
    for name, python_type in _JSON_TYPES.items():
        if name == "integer":
            continue
        if isinstance(value, python_type) and not (
            isinstance(value, bool) and name == "number"
        ):
            return "integer" if isinstance(value, int) and not isinstance(value, bool) else name
    return type(value).__name__


# -- the pieces the source page names but does not define -----------------------------


def load_tools(actuators: list[dict]) -> dict[str, dict]:
    """Index the declared actuators by name.

    This runtime dispatches actions; it does not execute them. A production version
    resolves each name to a callable here, which is exactly where the agent stops being
    portable, so the demonstration stops just short of it.
    """
    return {actuator["name"]: actuator for actuator in actuators}


class PerformanceTracker:
    """Records what actually happened. Does not score it.

    The P in PEAS is a declaration of what good looks like, not a measurement of it.
    `metrics` from the config is carried around and reported verbatim as a declaration,
    because a runtime that printed a number next to "customer satisfaction" would be
    inventing one.
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self.records: list[dict] = []

    def record(self, result: dict) -> None:
        self.records.append(result)

    def report(self) -> list[str]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record["action"]] = counts.get(record["action"], 0) + 1
        lines = [f"actions dispatched: {len(self.records)}"]
        for name, count in sorted(counts.items()):
            lines.append(f"  {name}: {count}")
        for metric in self.spec.get("metrics", []):
            lines.append(f"  declared metric, not measured here: {metric}")
        return lines


# -- the runtime ----------------------------------------------------------------------


class ConfigDrivenAgent:
    def __init__(self, agent_dir: str | Path):
        self.base = Path(agent_dir)
        self.config_path = self.base / "agent.yaml"
        # The config files in this repo wrap everything in a top-level `agent:` key, as
        # the source page's YAML does. The source's runtime reads the inner mapping, so
        # unwrap it here rather than repeating "agent" at every lookup.
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))["agent"]
        self.name = self.config["name"]

        # Load prompts from files
        self.prompts: dict[str, str] = {}
        for name, path in self.config.get("prompts", {}).items():
            self.prompts[name] = (self.base / path).read_text(encoding="utf-8")

        # Load schemas from files
        self.schemas: dict[str, dict] = {}
        for actuator in self.config.get("actuators", []):
            if "output_schema" in actuator:
                schema_path = self.base / actuator["output_schema"]
                self.schemas[actuator["name"]] = json.loads(
                    schema_path.read_text(encoding="utf-8")
                )

        # Load input validation schemas
        for sensor in self.config.get("sensors", []):
            if "input_schema" in sensor:
                schema_path = self.base / sensor["input_schema"]
                self.schemas[f"input_{sensor['name']}"] = json.loads(
                    schema_path.read_text(encoding="utf-8")
                )

        # Load state schema if model-based
        # Appended to this agent's transcript filename. See decide().
        self.transcript_suffix: str = ""
        self.state_schema: dict | None = None
        self.state: dict | None = None
        if "state" in self.config:
            state_path = self.base / self.config["state"]["schema"]
            self.state_schema = json.loads(state_path.read_text(encoding="utf-8"))
            self.state = {}

        self.tools = load_tools(self.config["actuators"])
        self.performance = PerformanceTracker(self.config["performance"])

        # Capability tier for every model call this agent makes, declared in agent.yaml
        # next to a comment justifying it. It belongs to the agent, not to the runtime:
        # a one-label-from-four decision and a multi-turn state object are not the same
        # job, and the runtime has no way to tell them apart without reading the config.
        self.tier = self.config.get("behavior", {}).get("tier", "default")

        # Selects which canned response comes back in mock mode, so a multi-turn run
        # reads like a run instead of the same string repeated. Every real backend
        # ignores mock_key, so this counter has no effect once a provider is configured.
        self.step = 0
        # Which task prompt the config routed to on the last turn. Recorded so the demo
        # can print it. Nothing in the runtime branches on it.
        self.routed_prompt = ""

    # -- one turn ---------------------------------------------------------------------

    def run(self, input_data: dict) -> dict:
        # DETERMINISTIC: validate input against sensor schema
        percept, sensor = self.validate_input(input_data)

        # LLM: decide action (system prompt + task prompt + percept)
        action = self.decide(percept)

        # DETERMINISTIC: validate action against output schema
        validated_action = self.validate_output(action)

        # DETERMINISTIC: fold any proposed state update in, under a declared fallback
        self.update_state(percept, validated_action)

        # DETERMINISTIC: counters the config declares, incremented by arithmetic
        self.count_percept(sensor)

        # DETERMINISTIC: execute
        result = self.act(validated_action)
        result["sensor"] = sensor
        # Whose answer this is. On a replay it names the model the recording came from,
        # which is what a reader watching an offline run actually wants to know.
        model = llm_module.last_model()
        if model:
            result["model"] = model
        result["task_prompt"] = self.routed_prompt or "(none routed)"

        # DETERMINISTIC: measure
        self.performance.record(result)
        return result

    def decide(self, percept: dict) -> str:
        # Assemble context from loaded prompts
        system = self.prompts["system"]
        task_prompt = self.select_task_prompt(percept)

        state_line = ""
        if self.state_schema is not None:
            state_line = f"Internal state: {json.dumps(self.state)}\n"

        prompt = f"""{system}

{task_prompt}

Available actions: {[t["name"] for t in self.config["actuators"]]}
Current observation: {json.dumps(percept)}
{state_line}Output schemas by action: {json.dumps(self._actuator_schemas())}
Strategy: {self.config["behavior"]["decision_strategy"]}

Pick the next action. Return valid JSON matching the output schema."""

        self.step += 1
        # Record under this agent's own name. Without it every agent in this directory
        # shares one transcript, because they all reach llm_call through this file, and
        # re-recording one of them throws away the rest.
        # transcript_suffix lets a caller keep its recordings in their own file. Entries
        # are keyed by prompt content and hold one model each, so a study that runs the
        # same prompt at several tiers would otherwise have each recording overwrite the
        # last and leave the agent replaying a tier it did not ask for. Empty by default,
        # which is every ordinary run.
        source = f"agent__{self.config['name'].replace('-', '_')}{self.transcript_suffix}"
        with llm_module.transcript_source(source):
            return llm_call(prompt, mock_key=self._mock_key(), tier=self.tier)

    def select_task_prompt(self, percept: dict) -> str:
        """Pick the right task prompt based on the current situation.

        The source page routes on substrings hardcoded in the runtime ("pdf", "review")
        onto prompt names hardcoded in the runtime ("extraction", "escalation"). Both are
        agent-specific, which is the one thing this file is not allowed to contain, so
        the routing table moved into agent.yaml under `behavior.task_prompts`. An agent
        that declares no routing gets no task prompt, and the system prompt carries the
        whole instruction.
        """
        haystack = json.dumps(percept).lower()
        for rule in self.config.get("behavior", {}).get("task_prompts", []):
            if rule["when"].lower() in haystack:
                self.routed_prompt = rule["use"]
                return self.prompts.get(rule["use"], "")
        self.routed_prompt = ""
        return ""

    # -- the deterministic boundaries -------------------------------------------------

    def validate_input(self, input_data: dict) -> tuple[dict, str]:
        """DETERMINISTIC: validate raw input against sensor schemas.

        The sensors in a PEAS config are alternatives, not a conjunction: an agent with
        three sensors receives one percept at a time, from one of them. So the rule is
        that a percept must satisfy at least one declared sensor schema, and the name of
        the sensor it satisfied comes back with it. A percept that satisfies none is
        refused here, before a single token is spent.
        """
        declared = [
            (sensor["name"], self.schemas[f"input_{sensor['name']}"])
            for sensor in self.config.get("sensors", [])
            if "input_schema" in sensor
        ]
        if not declared:
            return input_data, "(no input schema declared)"

        failures: dict[str, list[str]] = {}
        for name, schema in declared:
            errors = validate(input_data, schema)
            if not errors:
                return input_data, name
            failures[name] = errors
        raise PerceptRejected(failures)

    def validate_output(self, action: str) -> dict:
        """DETERMINISTIC: validate LLM action output against actuator schema.

        `model_loads` rather than `json.loads`, because a model asked for JSON routinely
        returns JSON inside a markdown fence. A bare parse rejects that, and this runtime
        rejected it for a long time: against a live model every eval case in one of the
        agent directories failed with "answer was not JSON" while the model had in fact
        answered correctly and wrapped it. The fence is a formatting habit, not a
        malformed answer, and unwrapping it before the parse is deterministic work that
        belongs on this side of the boundary. Genuinely broken output still raises, so the
        refusal path below stays reachable.

        (An earlier version of this comment named the agent directory it happened to, and
        `demo.py`'s no-agent-specific-code check failed the build over it. Correctly: a
        runtime that knows an agent's name, even in prose, is one edit away from behaving
        differently for it.)
        """
        try:
            parsed = model_loads(action)
        except json.JSONDecodeError as broken:
            raise ActionRejected(f"answer was not JSON: {broken}") from broken
        if not isinstance(parsed, dict):
            raise ActionRejected(f"answer was a JSON {_json_type_name(parsed)}, not an object")

        tool_name = parsed.get("action", "")
        # The prompt names the legal actions, but a prompt is a request. This is the
        # guarantee: an actuator this agent does not declare cannot be dispatched, no
        # matter how confidently the model named it.
        if tool_name not in self.tools:
            raise ActionRejected(f"{tool_name!r} is not an actuator declared in agent.yaml")

        if tool_name in self.schemas:
            errors = validate(parsed, self.schemas[tool_name])
            if errors:
                raise ActionRejected(
                    f"{tool_name!r} output failed its schema", errors
                )
        return parsed

    def update_state(self, percept: dict, validated_action: dict) -> None:
        """DETERMINISTIC: fold a proposed state update in, or take the declared fallback.

        Only agents that declare a state schema have state at all, which is the whole
        difference between a simple reflex agent and a model-based one -- expressed as
        the presence of six lines in a YAML file rather than as a different class.
        """
        if self.state_schema is None:
            return
        proposed = validated_action.get("state_update")
        if proposed is None:
            return

        # A config may declare which state fields the model is allowed to write. Anything
        # outside that list is dropped before it reaches the state, whatever the model
        # said about it.
        #
        # This exists because triage-tuner asked its model for `outcomes_seen` -- a count
        # of the outcomes it had been handed -- in an agent whose own config says the
        # reward is "computed by arithmetic over what happened" and that a critic which is
        # a model call is an agent grading its own homework. Live, the count was right in
        # 7 of 20 runs. Nothing asserted it: the eval suite compares actions, the sequence
        # harness compares actions, and the state schema accepts any integer, so a wrong
        # count was indistinguishable from a right one.
        #
        # Generic, not agent-specific: the runtime reads a list out of the config and
        # obeys it. An agent that declares nothing keeps the old behaviour.
        writable = self.config.get("state", {}).get("model_writable")
        if writable is not None:
            dropped = sorted(set(proposed) - set(writable))
            proposed = {k: v for k, v in proposed.items() if k in writable}
            if dropped:
                print(f"  [state] dropped {', '.join(dropped)}: not model-writable "
                      f"for this agent")

        errors = validate(proposed, self.state_schema)
        if not errors:
            self.state.update(proposed)
            return

        fallback = self.config["state"].get("fallback")
        if fallback != "merge-percept-data":
            raise ActionRejected(
                f"state update failed its schema and agent.yaml declares no usable "
                f"fallback (got {fallback!r})",
                errors,
            )
        # The fallback the config asked for: keep the run alive on hand-written logic
        # rather than raising. It degrades the state -- a raw percept field lands where
        # a normalized one belonged -- and that is the trade the config signed up for.
        print(f"  [fallback] state update failed its schema: {errors[0]}")
        self.state.update(percept)
        print(f"             -> merged the percept in by hand, "
              f"state keys now: {', '.join(self.state)}")

    def count_percept(self, sensor: str) -> None:
        """DETERMINISTIC: increment any counters this config declares for this sensor.

        A count is arithmetic, and this method exists because one agent here was asking a
        model to do it. triage-tuner's prompt requested `outcomes_seen`, a running total
        of the outcomes it had been handed, in an agent whose own config says the reward
        is computed by arithmetic over what happened. Measured live over twenty runs the
        model got that total right seven times. Nothing in the repository could tell a
        wrong count from a right one: the eval suite compares actions, the sequence
        harness compares actions, and the state schema accepts any integer.

        Generic in the same way `model_writable` is. The runtime reads a {field: sensor}
        mapping out of the config and counts; an agent that declares none gets none, and
        no agent's name appears here.
        """
        if self.state is None:
            return
        counted = (self.config.get("state") or {}).get("counted") or {}
        for field, counted_sensor in counted.items():
            if counted_sensor == sensor:
                self.state[field] = self.state.get(field, 0) + 1

    def act(self, validated_action: dict) -> dict:
        """DETERMINISTIC: dispatch the validated action to its actuator."""
        name = validated_action["action"]
        return {
            "agent": self.name,
            "action": name,
            "actuator_type": self.tools[name].get("type", "unspecified"),
            "args": {
                key: value
                for key, value in validated_action.items()
                if key not in ("action", "state_update")
            },
        }

    # -- evaluation -------------------------------------------------------------------

    def evaluate(self) -> list[dict]:
        """Run the eval cases the config points at.

        Loaded here and nowhere else. The source page's own table says eval cases are
        read at evaluation time and never during a production run, so they are not
        touched in `__init__`.

        In mock mode this compares canned responses against expectations, so it measures
        whether the pipeline carries an answer end to end -- not whether a model is any
        good. Point it at a real provider and the same code measures the model.
        """
        raw = json.loads(
            (self.base / self.config["performance"]["eval"]).read_text(encoding="utf-8")
        )
        # A bare list of cases, or an object with them under "cases". Both shapes appear
        # in the wild and neither is more correct; refusing one would make the eval file
        # format a thing an agent author has to look up rather than guess right.
        cases = raw["cases"] if isinstance(raw, dict) else raw

        results = []
        for case in cases:
            # Per case, not once per suite. A stateful agent that carried state from one
            # case into the next would be scored on the order its cases happen to sit in
            # the file, and moving two of them would change the result without anything
            # about the agent changing. Found live: aml-alert's last case closed an alert
            # it should have asked about, because an earlier case had already established
            # an innocent explanation for a different customer. Alone it answers correctly
            # five times out of five.
            #
            # This is what makes the two suites separable. Here a case is judged on its
            # percept, which is what the HTTP contract promises a caller. Whether history
            # changes an answer is the sequence harness's question, and it builds the
            # history deliberately rather than inheriting it by accident.
            if self.state_schema is not None:
                self.state = {}
            try:
                observed = self.run(case["input"])["action"]
            except (PerceptRejected, ActionRejected) as refusal:
                observed = f"refused ({refusal.__class__.__name__})"
            results.append(
                {
                    "id": case["id"],
                    "expected": case["expect_action"],
                    "observed": observed,
                    "passed": observed == case["expect_action"],
                    "note": case.get("note", ""),
                }
            )
        return results

    # -- internals --------------------------------------------------------------------

    def _actuator_schemas(self) -> dict[str, dict]:
        return {
            name: schema
            for name, schema in self.schemas.items()
            if name in self.tools
        }

    def _mock_key(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")
        return f"runtime_{slug}_{self.step}"
