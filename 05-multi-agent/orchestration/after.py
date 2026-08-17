"""LLM multi-agent orchestration with deterministic contracts between the agents.

Source: reference/05-multi-agent-systems-before-after.md, "Cooperative: Orchestrated
Pipelines / AFTER: LLM Multi-Agent System".

The source page shows AgentConfig, WorkerAgent and Orchestrator, and its config block
promises `validation_between_agents: true`. The code on the page never implements it.
This file does, because that line is the whole argument: the patterns -- workflow,
parallel, routing -- are ordinary control flow that predates LLMs by decades. What makes
a chain of models into a system is that the output of agent N is checked against the
input schema of agent N+1 by code, before agent N+1 is called.

Four runs, in order:

    workflow    sequential, contract checked at every hand-off
    parallel    three independent reads of the same document
    routing     classify, then dispatch to one handler
    failure     the extractor emits plausible-looking garbage and the pipeline halts

Everything runs offline. With no provider configured, `llm_call` returns canned
responses and the full control flow still executes -- prompts built, JSON parsed,
schemas checked, the halt taken.
"""

import asyncio
import json
import sys
from pathlib import Path
from textwrap import indent

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.llm import llm_call
from shared.model_json import loads as model_loads

# The task this file exists for is deterministic validation between agents, so the
# validator itself must never be the reason a reader cannot run the file. jsonschema if
# it happens to be installed, a hand-rolled required-key and type check if not. Which one
# ran is printed at startup, because the two produce differently worded messages and a
# reader comparing output against the README should be able to tell which they got.
try:
    import jsonschema

    VALIDATOR_NAME = "jsonschema (installed)"
except ImportError:  # pragma: no cover - depends on the reader's machine, not on input
    jsonschema = None
    VALIDATOR_NAME = "built-in key/type check (jsonschema not installed)"


# -- the contracts -------------------------------------------------------------------
#
# One input schema per agent. Read as a pipeline, each schema is simultaneously the
# thing the agent accepts and the thing its predecessor is obliged to emit. That is the
# whole of `validation_between_agents: true` from config.yaml. In a production version
# these live in agents/<name>/schemas/input.json, exactly as config.yaml says; they are
# inline here so the file stays readable in a screenshot.
#
# Note there is no `additionalProperties: false`. A contract is a floor, not a cage --
# the extractor is free to return extra fields, and does.

SCHEMAS: dict[str, dict] = {
    "extractor": {
        "type": "object",
        "required": ["document"],
        "properties": {"document": {"type": "string"}},
    },
    "validator": {
        "type": "object",
        "required": ["name", "amount", "confidence"],
        "properties": {
            "name": {"type": "string"},
            "amount": {"type": "number"},
            "confidence": {"type": "number"},
        },
    },
    "formatter": {
        "type": "object",
        "required": ["status", "name", "amount"],
        "properties": {
            "status": {"type": "string"},
            "name": {"type": "string"},
            "amount": {"type": "number"},
            "errors": {"type": "array"},
        },
    },
}

_JSON_TYPE_NAMES = {
    dict: "object",
    list: "array",
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    type(None): "null",
}


def validate_against_schema(instance: dict, schema: dict) -> list[str]:
    """Return a list of contract violations. Empty list means the hand-off is legal.

    No model is involved and no exception escapes: the caller decides what a violation
    means. That is the point of doing this in code rather than asking agent N+1 to
    notice that its input is wrong.
    """
    if jsonschema is not None:
        errors = jsonschema.Draft202012Validator(schema).iter_errors(instance)
        return sorted(
            f"{'.'.join(str(p) for p in e.path) or 'root'}: {e.message}" for e in errors
        )
    return sorted(_builtin_validate(instance, schema))


def _builtin_validate(instance: object, schema: dict, label: str = "root") -> list[str]:
    """The no-dependency path: declared type, required keys, and property types."""
    expected = schema.get("type")
    if expected is not None and not _type_matches(instance, expected):
        return [f"{label}: expected {expected}, got {_json_type_name(instance)}"]

    errors: list[str] = []
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{key}: required property is missing")
        for key, subschema in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(_builtin_validate(instance[key], subschema, key))
    return errors


def _type_matches(instance: object, expected: str) -> bool:
    # bool is a subclass of int in Python, so the obvious isinstance check would let
    # True through as a number. JSON does not agree, and neither does jsonschema.
    if isinstance(instance, bool):
        return expected == "boolean"
    if expected == "number":
        return isinstance(instance, (int, float))
    if expected == "integer":
        return isinstance(instance, int)
    return isinstance(instance, _EXPECTED_PYTHON_TYPES.get(expected, object))


_EXPECTED_PYTHON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _json_type_name(instance: object) -> str:
    return _JSON_TYPE_NAMES.get(type(instance), type(instance).__name__)


# -- the agents ----------------------------------------------------------------------

class AgentConfig:
    def __init__(self, config: dict):
        self.name = config['name']
        self.role = config['role']
        self.performance = config['performance']
        self.actions = config['actuators']
        self.system_prompt = config['behavior']['system_prompt']
        # Two fields the source page does not carry. `tier` is the capability level this
        # agent's job actually needs; `mock_key` selects the canned response that stands
        # in for a model when no provider is configured. Both belong in the config for
        # the same reason the prompt does -- changing which model does a job should not
        # be a code change.
        self.tier = config.get('tier', 'mid')
        self.mock_key = config.get('mock_key', 'default')
        # The contract this agent's output has to satisfy, which is the NEXT agent's input
        # schema. The source page's config gives every actuator an `output_schema` and its
        # code never reads one; this is that field, wired up.
        self.output_schema = config.get('output_schema')
        # A fixed response to return instead of calling the model. The failure run below
        # needs the extractor to emit a specific bad record every time, and mock_key can
        # only arrange that offline -- every real backend ignores it, so with a key set the
        # extractor behaved itself and the demonstration of the gate halting silently
        # stopped happening. Injecting the payload makes the run identical in both modes.
        self.forced_output = config.get('forced_output')


class WorkerAgent:
    """Single agent with its own PEAS spec and performance measure."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.performance_log = []

    async def process(self, input_data: dict) -> dict:
        if self.config.forced_output is not None:
            print(f"  note  {self.config.name} output injected by the demo, no model call")
            result = dict(self.config.forced_output)
            self.performance_log.append(self._evaluate(result))
            return result

        prompt = f"""{self.config.system_prompt}
Input: {json.dumps(input_data)}
Available actions: {self.config.actions}
{self._contract_clause()}Process the input. Return JSON."""

        # Tier is per agent and justified at each config below -- frontier for turning an
        # unseen document layout into a fixed record, mid for the validator's bounded
        # judgement, small for the formatter's templating. Dispatched to a worker thread
        # so that `_parallel`'s asyncio.gather genuinely overlaps three calls instead of
        # serialising them behind a blocking client (see README, "What changed").
        response = await asyncio.to_thread(
            llm_call, prompt, mock_key=self.config.mock_key, tier=self.config.tier
        )
        try:
            result = model_loads(response)
        except json.JSONDecodeError:
            result = {'raw_output': response, 'parse_error': True}

        score = self._evaluate(result)
        self.performance_log.append(score)
        return result

    def _evaluate(self, result: dict) -> float:
        score = 0.0
        if not result.get('parse_error'):
            score += 0.5
        if self._confidence(result) > 0.8:
            score += 0.5
        return score

    def _contract_clause(self) -> str:
        """Tell the model the shape the next agent will hold it to.

        Without this the pipeline asked for "JSON" and then rejected the answer for not
        having the fields it never mentioned. claude-opus-5 returned a perfectly sensible
        record with the fields nested one level down, the gate reported `'name' is a
        required property`, and the workflow halted on its first hand-off -- a contract
        failure manufactured by not stating the contract.

        Naming the required fields is not a substitute for the gate, and the gate below is
        unchanged. It is the difference between a contract and a trap: the model is now
        wrong when it misses a field, rather than unlucky.
        """
        schema = self.config.output_schema
        if not schema:
            return ""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        fields = ", ".join(
            f"{name} ({properties.get(name, {}).get('type', 'any')})" for name in required
        )
        return f"Return a JSON object with these keys at the top level: {fields}.\n" if fields else ""

    @staticmethod
    def _confidence(result: dict) -> float:
        """Read `confidence` out of a model's JSON without trusting its type.

        The source page compares `result.get('confidence', 0) > 0.8` directly. That holds
        for as long as the model returns a number, and claude-opus-5 returned the string
        "0.93", which raises TypeError on the comparison and takes the whole pipeline down.

        Where it happened is the interesting part. This runs inside the agent's own
        performance measure, which fires before the contract gate between agents -- and the
        gate is exactly the thing that would have rejected a string where a number belongs.
        The deterministic layer was correct and one call too late to help. Anything reading
        a model's output has to be defensive on its own account, including the code whose
        job is to score that output.

        A confidence that will not parse is treated as no confidence rather than as an
        error, because the alternative is an agent that cannot report on itself.
        """
        try:
            return float(result.get('confidence', 0))
        except (TypeError, ValueError):
            return 0.0


class Orchestrator:
    """
    Coordinates agents. Patterns (mapped to architectures):
    - workflow: sequential pipeline (goal-based, fixed plan)
    - routing: classify then dispatch (reflex-based)
    - parallel: concurrent workers (separate performance measures)
    """

    def __init__(self, agents: list, pattern: str = 'workflow'):
        self.agents = agents
        self.pattern = pattern

    async def run(self, input_data: dict) -> dict:
        if self.pattern == 'workflow':
            return await self._workflow(input_data)
        elif self.pattern == 'parallel':
            return await self._parallel(input_data)
        elif self.pattern == 'routing':
            return await self._routing(input_data)
        # The source page falls off the end here and returns None. A typo in a config
        # file should not look like a pipeline that ran and produced nothing.
        raise ValueError(f"unknown orchestration pattern: {self.pattern!r}")

    async def _workflow(self, data: dict) -> dict:
        current = data
        results = []
        producer = 'pipeline input'
        for position, agent in enumerate(self.agents):
            # The contract gate. Whatever produced `current` has to satisfy this agent's
            # input schema before the call fires. Two consequences worth noticing: no
            # tokens are spent on data that cannot be processed, and the error names the
            # producer rather than the victim. A missing schema is a KeyError on purpose
            # -- an agent with no declared contract is a configuration error, not a
            # default-to-permissive case.
            errors = validate_against_schema(current, SCHEMAS[agent.config.name])
            if errors:
                blocked = [a.config.name for a in self.agents[position:]]
                print(f"  gate  {producer} -> {agent.config.name}   CONTRACT VIOLATION")
                for message in errors:
                    print(f"          {message}")
                return {
                    'error': f'{producer} produced output that {agent.config.name} cannot accept',
                    'failed_at': producer,
                    'schema_errors': errors,
                    'blocked': blocked,
                    'partial': results,
                }
            print(f"  gate  {producer} -> {agent.config.name}   contract ok")
            print(f"  call  {agent.config.name} (tier={agent.config.tier})")

            result = await agent.process(current)
            if result.get('parse_error'):
                return {'error': f'{agent.config.name} failed', 'partial': results}
            results.append({'agent': agent.config.name, 'result': result})
            current = result
            producer = agent.config.name
        return {'status': 'complete', 'results': results, 'final': current}

    async def _parallel(self, data: dict) -> dict:
        # No gate here, and that is not an oversight: a fan-out has no agent N+1. Every
        # worker reads the same input, so the only contract to enforce is the pipeline's
        # own entry schema, and merging the three results is the caller's problem.
        tasks = [a.process(data) for a in self.agents]
        results = await asyncio.gather(*tasks)
        return {'status': 'complete', 'results': [
            {'agent': a.config.name, 'result': r}
            for a, r in zip(self.agents, results)
        ]}

    async def _routing(self, data: dict) -> dict:
        names = [a.config.name for a in self.agents]
        prompt = f"""Classify to one handler: {names}
Input: {json.dumps(data)}
Return just the handler name."""
        # small: picking one label out of a three-item list of handler names. There is
        # nothing here a larger model would get more right, and the deterministic
        # agent_map lookup below already contains any answer that is not on the list.
        category = llm_call(prompt, mock_key="orchestration_router", tier="small").strip()
        agent_map = {a.config.name: a for a in self.agents}
        target = agent_map.get(category, self.agents[0])
        result = await target.process(data)
        return {'routed_to': target.config.name, 'result': result}


# -- build from configs ---------------------------------------------------------------
#
# Three configs, one per agent. In the shape config.yaml describes, each of these is a
# directory: prompts in files, schemas in files, eval cases beside them. New agent = new
# directory. New pipeline = new orchestration config pointing at existing directories.

EXTRACTOR_CONFIG = {
    'name': 'extractor', 'role': 'data extraction',
    'performance': {'metrics': ['accuracy', 'completeness']},
    'actuators': ['extract_fields', 'flag_unclear'],
    'behavior': {'system_prompt': 'Extract structured data from the document. Return JSON.'},
    # frontier: this is the component that has to absorb document layouts nobody wrote a
    # parser for -- ambiguous natural language in, a fixed record out, which is the case
    # the cheaper tiers get subtly wrong. Subtly wrong matters here: the schema gate
    # catches a malformed record and halts, which is correct behaviour and useless to the
    # user. This is the single most expensive call in the pipeline and the one worth it.
    'tier': 'frontier',
    'mock_key': 'orchestration_extractor',
}

VALIDATOR_CONFIG = {
    'name': 'validator', 'role': 'data validation',
    'performance': {'metrics': ['error_detection', 'false_positive_rate']},
    'actuators': ['approve', 'reject', 'flag'],
    'behavior': {'system_prompt': 'Validate extracted data for completeness and correctness. Return JSON.'},
    # mid: bounded judgement over a record that has already passed a schema check. The
    # hard checks -- required fields, types -- are the gate's job and were done before
    # this agent was called. What is left is plausibility ("is a $2,450 invoice from this
    # payer normal") plus a confidence number, which is exactly the mid case.
    'tier': 'mid',
    'mock_key': 'orchestration_validator',
}

FORMATTER_CONFIG = {
    'name': 'formatter', 'role': 'output formatting',
    'performance': {'metrics': ['format_compliance']},
    'actuators': ['format_report', 'format_json'],
    'behavior': {'system_prompt': 'Format validated data into the requested output. Return JSON.'},
    # small: every field it emits was validated upstream, so this is templating, not
    # reasoning. Paying frontier prices to turn three known values into a sentence is the
    # most common way a multi-agent system ends up costing 15x for no gain.
    'tier': 'small',
    'mock_key': 'orchestration_formatter',
}


# What the extractor emits in the failure run. Valid JSON, plausible to a human, and
# wrong in two ways the next agent's schema is explicit about: the amount is the string
# printed on the invoice rather than a number, and confidence is missing entirely.
BAD_EXTRACTION = {
    "name": "Acme Manufacturing LLC",
    "amount": "$2,450.00",
    "note": "amount transcribed exactly as printed on the invoice",
}


def agents_for(extractor_mock_key: str = 'orchestration_extractor',
               extractor_forced_output: dict | None = None) -> list:
    """Fresh agents per run so each run's performance_log is its own.

    Both extractor arguments are test seams, not architecture. `mock_key` picks a canned
    response and only bites offline; `forced_output` skips the call entirely and therefore
    works with a real key too, which is what the failure run needs.
    """
    # Each agent's output contract is literally the next agent's input schema -- the same
    # dict the gate checks against, handed to the producer instead of only to the judge.
    # Read down the pipeline, SCHEMAS is both at once, which is the point of the config.
    return [
        WorkerAgent(AgentConfig({**EXTRACTOR_CONFIG, 'mock_key': extractor_mock_key,
                                 'forced_output': extractor_forced_output,
                                 'output_schema': SCHEMAS['validator']})),
        WorkerAgent(AgentConfig({**VALIDATOR_CONFIG, 'output_schema': SCHEMAS['formatter']})),
        WorkerAgent(AgentConfig(FORMATTER_CONFIG)),
    ]


# The document before.py could not parse. Nothing about it is unusual; it is simply not
# the format the hand-coded parser assumed.
DOCUMENT = "INVOICE\nBill to: Acme Manufacturing LLC\nTotal amount: $2,450.00\nTerms: Net 30"


def print_performance(agents: list) -> None:
    for agent in agents:
        if agent.performance_log:
            print(f"    {agent.config.name:<10} {agent.performance_log}")


async def main() -> None:
    payload = {'document': DOCUMENT}
    print(f"schema validator: {VALIDATOR_NAME}")
    print()
    print("Input document -- the one before.py could not parse:")
    print(indent(DOCUMENT, "    "))
    print()

    print("=== pattern: workflow -- sequential, contract checked at every hand-off ===")
    agents = agents_for()
    result = await Orchestrator(agents, pattern='workflow').run(payload)
    if 'status' not in result:
        # This run is supposed to succeed. If it did not, say why in one line rather
        # than letting the next print raise a KeyError over the top of the real reason.
        raise SystemExit(f"  workflow did not complete: {result.get('error')}")
    print(f"  status: {result['status']}")
    print(f"  final:  {json.dumps(result['final'])}")
    print("  performance measure per agent (each scores its own output):")
    print_performance(agents)
    print()

    print("=== pattern: parallel -- three independent reads of the same document ===")
    agents = agents_for()
    result = await Orchestrator(agents, pattern='parallel').run(payload)
    print(f"  status: {result['status']}")
    for entry in result['results']:
        print(f"  {entry['agent']:<10} {json.dumps(entry['result'])}")
    print("  no gate ran: a fan-out has no agent N+1 to hold to a contract")
    print()

    print("=== pattern: routing -- classify, then dispatch to one handler ===")
    agents = agents_for()
    result = await Orchestrator(agents, pattern='routing').run(payload)
    print(f"  routed_to: {result['routed_to']}")
    print(f"  result:    {json.dumps(result['result'])}")
    print("  the router's answer is contained by agent_map.get(category, agents[0]) --")
    print("  a handler name the model invents can never dispatch to something that is")
    print("  not in the agent list")
    print()

    print("=== failure run: the extractor emits output the validator cannot accept ===")
    print("  same three agents, same document, one canned extractor response swapped")
    agents = agents_for(extractor_forced_output=BAD_EXTRACTION)
    result = await Orchestrator(agents, pattern='workflow').run(payload)
    print(f"  halted at:    {result['failed_at']}")
    print(f"  message:      {result['error']}")
    print(f"  never called: {', '.join(result['blocked'])}")
    print("  what the extractor returned:")
    print(f"    {json.dumps(result['partial'][0]['result'])}")
    print("  its own performance measure scored that:")
    print_performance(agents)
    print("  -- a signal, not a stop. The schema check is the stop.")
    print()
    print("  The extractor's output parsed as valid JSON and looked entirely reasonable.")
    print("  It returned the amount as it appears on the invoice, '$2,450.00', instead of")
    print("  as a number, and omitted confidence. Nothing downstream had to be written to")
    print("  cope with that, and the validator was never asked to.")


if __name__ == "__main__":
    asyncio.run(main())
