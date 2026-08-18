"""Generate an OpenAPI 3.1 document from a PEAS config.

Nothing here is hand-written per agent. The spec is derived from `agent.yaml` and the
schema files it already references, which is the point: a PEAS config is an interface
description that happens to be written in YAML, and an API description is the same
information written in JSON. Sensors are the request contract. Actuators are the response
contract. Both already carry their JSON Schema because the runtime validates against them.

So the docs cannot drift from the agent. There is no second place to update, and adding an
agent adds an API without anyone writing an API.

The mapping, and it is worth reading once because it is the argument rather than plumbing:

    sensors      -> the request body. Sensors are alternatives -- the runtime accepts a
                    percept if ANY sensor schema matches -- so the body is a `oneOf` over
                    the sensor schemas, exactly as `validate_input` treats them.
    actuators    -> the enumeration of actions a response may name, and the schema each
                    one's arguments must satisfy.
    performance  -> description prose. The P in PEAS is what the endpoint is for.
    environment  -> description prose.
    state        -> whether the agent carries state between calls, which a caller has to
                    know to use it correctly.

The two deterministic gates in the oscillation pattern become the two error responses:
a percept no sensor schema accepts is 422, and an action the actuator schema refuses is
502. That distinction is not cosmetic. 422 says the caller sent something wrong; 502 says
the model did.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OPENAPI_VERSION = "3.1.0"


def build_spec(agent_dir: str | Path, version: str = "1.0.0") -> dict[str, Any]:
    """Return an OpenAPI document describing the agent in `agent_dir`."""
    import yaml

    base = Path(agent_dir)
    config = yaml.safe_load((base / "agent.yaml").read_text(encoding="utf-8"))["agent"]

    sensors = config.get("sensors", [])
    actuators = config.get("actuators", [])

    schemas: dict[str, Any] = {}
    percept_refs = []
    for sensor in sensors:
        if "input_schema" not in sensor:
            # A sensor with no schema is accepted as-is by the runtime, so the API cannot
            # promise a shape for it either. Saying that plainly beats inventing one.
            continue
        name = f"Percept_{sensor['name']}"
        schemas[name] = _load_schema(base / sensor["input_schema"], sensor)
        percept_refs.append({"$ref": f"#/components/schemas/{name}"})

    for actuator in actuators:
        if "output_schema" not in actuator:
            continue
        name = f"Action_{actuator['name']}"
        if name not in schemas:
            schemas[name] = _load_schema(base / actuator["output_schema"], actuator)

    schemas["AgentResult"] = _result_schema(actuators)
    schemas["Refusal"] = _refusal_schema()

    examples = _examples_from_eval(base, config)
    unschemaed = [s["name"] for s in sensors if "input_schema" not in s]
    request_schema: dict[str, Any]
    if percept_refs:
        request_schema = {"oneOf": percept_refs} if len(percept_refs) > 1 else percept_refs[0]
    else:
        request_schema = {"type": "object"}

    return {
        "openapi": OPENAPI_VERSION,
        "info": {
            "title": f"{config['name']} ({config['architecture']})",
            "version": version,
            "description": _description(config, unschemaed, base),
        },
        "servers": [{"url": "/", "description": "this agent, as served by serve.py"}],
        "tags": [
            {"name": "agent", "description":
                "The agent loop. One percept in, one action out, with deterministic "
                "validation on both sides of the model call."},
            {"name": "introspection", "description":
                "What this agent is and whether it is answering. Neither costs a model call."},
        ],
        "paths": {
            "/act": _act_path(request_schema, examples, actuators),
            "/agent": _agent_path(),
            "/health": _health_path(),
            "/openapi.json": _spec_path(),
            "/docs": _docs_path(),
            "/": _root_path(),
        },
        "components": {"schemas": schemas},
    }


def _eval_summary(config: dict, base: Path | None) -> str:
    """One sentence of fact about this agent's evaluation suite.

    Counted from the files rather than stated, so an agent that loses coverage says so
    here instead of continuing to advertise it.
    """
    actuators = [a["name"] for a in config.get("actuators", [])]
    if base is None:
        return f"{len(actuators)} actuators declared."

    try:
        raw = json.loads((base / config["performance"]["eval"]).read_text(encoding="utf-8"))
    except (OSError, KeyError, json.JSONDecodeError):
        return f"{len(actuators)} actuators declared; no evaluation suite was readable."

    cases = raw["cases"] if isinstance(raw, dict) else raw
    asserted = {c.get("expect_action") for c in cases}
    covered = [a for a in actuators if a in asserted]
    uncovered = [a for a in actuators if a not in asserted]

    summary = (f"{len(cases)} evaluation cases, asserting {len(covered)} of "
               f"{len(actuators)} declared actuators.")
    if uncovered:
        summary += (" Declared but never asserted by a case: "
                    + ", ".join(uncovered)
                    + ". An actuator no case reaches is a capability nobody has checked.")
    else:
        summary += (" Every declared actuator is reached by at least one case, so none of "
                    "them is a capability that has never been exercised.")

    sequences = base / "eval" / "sequences.json"
    if sequences.is_file():
        try:
            count = len(json.loads(sequences.read_text(encoding="utf-8")))
            noun = "sequence test runs" if count == 1 else f"{count} sequence tests run"
            summary += (f" A further {noun} the same percept twice, once cold and once "
                        "after a preamble, because a single-percept suite cannot test "
                        "whether history changes the answer.")
        except (OSError, json.JSONDecodeError):
            pass
    return summary


def _load_schema(path: Path, owner: dict) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    # $schema is a JSON Schema keyword and not an OpenAPI one. Harmless, but it makes
    # some validators complain, and the file it came from is the authority anyway.
    schema.pop("$schema", None)
    schema.setdefault("title", owner["name"])
    return schema


def _result_schema(actuators: list[dict]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["agent", "action", "actuator_type"],
        "properties": {
            "agent": {"type": "string"},
            "action": {
                "type": "string",
                "enum": [a["name"] for a in actuators],
                "description": "Always one of the declared actuators. The runtime refuses "
                               "anything else before it reaches an actuator.",
            },
            "actuator_type": {"type": "string"},
            "args": {"type": "object"},
            "sensor": {"type": "string", "description": "Which sensor schema matched the percept."},
            "model": {
                "type": "string",
                "description": (
                    "The model behind this answer. On a live call it is the model the "
                    "declared tier resolved to. With no backend configured it is the "
                    "model the replayed recording came from, so an offline run still "
                    "tells you whose output you are reading rather than leaving you to "
                    "assume."),
            },
            "task_prompt": {"type": "string"},
        },
    }


def _refusal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["error", "reason"],
        "properties": {
            "error": {"type": "string"},
            "reason": {"type": "string"},
            "detail": {
                "type": "object",
                "description": "Per-sensor or per-actuator validation errors, when there are any.",
            },
        },
    }


# The environment line is the textbook's vocabulary, and it is worth keeping: it is the
# same five axes every agent in this repository is classified on. It is also meaningless
# to most people reading a Swagger page, so each term gets a plain reading beside it.
# Kept as a lookup rather than prose so a term nobody translated shows up as missing
# instead of being quietly dropped.
ENVIRONMENT_IN_PLAIN_WORDS = {
    "fully-observable": "it can see everything it needs to decide",
    "partially-observable": "it never sees the whole picture",
    "deterministic": "the same situation always behaves the same way",
    "stochastic": "the same situation can turn out differently",
    "episodic": "each case stands on its own",
    "sequential": "what it does now changes what comes next",
    "static": "nothing moves while it is deciding",
    "dynamic": "the situation can change while it is deciding",
    "discrete": "a fixed set of choices",
    "continuous": "a value on a scale, not a choice from a list",
}


def _environment_in_plain_words(environment: str) -> str:
    """Translate the environment classification, or say which term has no reading yet."""
    readings = []
    for term in (t.strip() for t in environment.split(",")):
        if not term:
            continue
        readings.append(ENVIRONMENT_IN_PLAIN_WORDS.get(term, f"{term} (no plain reading written yet)"))
    return "; ".join(readings)


def _description(config: dict, unschemaed: list[str], base: Path | None = None) -> str:
    performance = config.get("performance", {})
    metrics = ", ".join(performance.get("metrics", [])) or "not declared"
    lines = []

    # What the agent is for, in the domain's own terms, before any of the mechanics. A
    # reader who lands on a Swagger page should not have to infer the problem from an
    # actuator list. This comes out of the config like everything else, so it cannot
    # describe an agent the runtime is not running.
    use_case = config.get("use_case")
    stateful = "state" in config
    if use_case:
        lines += [
            f"**{use_case.get('domain', '')}**",
            "",
            use_case.get("decision", ""),
            "",
            "WHY THIS IS WORTH LOOKING AT",
            "",
            use_case.get("why_it_matters", ""),
            "",
            "TRY IT, IN THREE STEPS",
            "",
            "  1. Open POST /act below and press Try it out.",
            "  2. Choose an entry from the Examples dropdown. Those are this agent's own "
            "evaluation cases -- real inputs it is asserted to handle -- so you do not "
            "have to invent a request body.",
            "  3. Press Execute. A 200 comes back carrying the action this agent chose, the "
            "arguments it chose for that action, and which model answered.",
            "",
            "You need no API key and no network for any of that. With nothing configured "
            "the agent replays a real model's recorded answer to that exact request.",
            "",
            "WHAT YOU JUST SAW",
            "",
            "A model chose that action -- but it chose from a list it cannot add to, "
            "after the request was checked against a shape it cannot change, and the "
            "answer was checked again before you saw it. Send something this agent does "
            "not handle and you get a 422 before any model is called at all.",
            "",
            "That is the whole idea. The model supplies judgement in one specific place. "
            "Ordinary code decides what it is allowed to be asked, what it is allowed to "
            "answer, and what happens next. Neither half is doing the other's job, and "
            "you can check that here rather than take anyone's word for it.",
            "",
            "THREE WORDS THIS PAGE KEEPS USING",
            "",
            "  percept    one input. A single alert, message, claim or ticket -- whatever "
            "this agent reads. The request body below is exactly one percept.",
            "  actuator   one action it is allowed to take. The list is fixed in "
            "configuration. The model chooses from it and cannot add to it, which is the "
            "whole reason this agent is safe to point at anything.",
            "  tier       how capable a model it asks for: small, mid or frontier. That is "
            "a line of configuration, not a code change.",
            "",
        ]
        if stateful:
            lines += [
                "This agent remembers. Send the examples in order under one X-Session-Id "
                "header and each answer is shaped by the ones before it -- that is the "
                "point of a stateful agent. Send them with no header and each request "
                "stands alone, which is why the examples work in any order.",
                "",
            ]
        lines += [

            "WHAT IT DELIBERATELY CANNOT DO",
            "",
            use_case.get("boundary", ""),
            "",
            "---",
            "",
        ]

    lines += [
        f"Architecture: {config['architecture']}.",
        f"Environment: {config.get('environment', {}).get('type', 'not declared')}.",
        "  In plain words: "
        + _environment_in_plain_words(config.get("environment", {}).get("type", "")) + ".",
        f"Performance measure: {metrics}.",
        f"Capability tier requested for the model call: {config.get('behavior', {}).get('tier', 'default')}.",
        "",
        "State: "
        + ("carried between calls; this endpoint is not stateless."
           if "state" in config else "none; every call is independent."),
        "",
        "This document is generated from agent.yaml at start-up. It is not maintained by "
        "hand and cannot describe an agent the runtime is not actually running.",
        "",
        "HOW THIS IS BUILT, AND WHY THAT IS THE POINT",
        "",
        "This agent is a directory of YAML, prompts and JSON Schema. There is no code "
        "specific to it anywhere: one ConfigDrivenAgent class serves every agent in this "
        "container, and a check parses that class for every name belonging to a "
        "specific agent and finds none of them.",
        "",
        "The classical architecture is doing the work and the model occupies one slot "
        "inside it. Read the request and response schemas as the two halves of that:",
        "",
        "  sensors    -> the request body. A percept is accepted only if some declared "
        "sensor schema matches it, and that check runs before any model call.",
        "  the model  -> chooses one action and its arguments. It decides nothing about "
        "what actions exist.",
        "  actuators  -> the response contract. The action must be one of the declared "
        "names and its arguments must satisfy that actuator's schema, or the answer is "
        "refused after the model produced it.",
        "",
        "That is the oscillation the whole repository is about: deterministic, model, "
        "deterministic. The two deterministic halves are what make the model's "
        "contribution safe to accept, and they are why the boundary above is a property "
        "of this API rather than a promise in a prompt.",
        "",
        "WHAT IS ACTUALLY CHECKED",
        "",
        _eval_summary(config, base),
        "",
        "Those cases are not a benchmark and they are not a large suite. They are the "
        "cases this agent is asserted to handle, they are the request examples below, and "
        "you can run them against a live model yourself:",
        "",
        "    python 00-config-runtime/demo.py",
        "",
        "Where a model disagreed with a case, the disagreement was read rather than "
        "tuned away. Several expectations turned out to be wrong and were corrected "
        "because the agent argued better than the case did; each is a separate commit "
        "with the reason attached. That distinction is the only thing separating an "
        "evaluation suite from a set of assertions that always pass.",
        "",
        "WHAT EACH RESPONSE CODE MEANS",
        "",
        "Read them as a diagnosis rather than a failure:",
        "  200  the model answered and the actuator contract accepted it.",
        "  422  no sensor schema accepted your percept. Your request is wrong, and no "
        "model was called, so nothing was spent.",
        "  502  the model answered and the answer failed the actuator contract. Your "
        "request was fine; the answer was not.",
        "",
        "Three more this endpoint returns, outside that pattern:",
        "  400  the body was not a JSON object.",
        "  405  the route exists and does not take that method; see the Allow header.",
        "  503  the percept is valid, no backend is configured, and no recording exists "
        "for the prompt it makes. Nothing is invented here, so there is nothing to "
        "return. Send one of the examples above, or set a key.",
        "",
        "WHICH MODEL IS ANSWERING",
        "",
        "GET /health reports the backend. With no credentials configured the agent "
        "replays recorded real model responses from shared/transcripts, so this API works "
        "with no key and no network -- and the answers are recordings of a real model, not "
        "invented ones. Set ANTHROPIC_API_KEY to put it on a live model; nothing else "
        "changes. A prompt that has no recording raises rather than inventing a reply.",
    ]
    if unschemaed:
        lines += [
            "",
            "Sensors without a declared input schema, accepted without validation: "
            + ", ".join(unschemaed) + ".",
        ]
    return "\n".join(lines)


def _examples_from_eval(base: Path, config: dict) -> dict[str, Any]:
    """Request examples, taken from the agent's own evaluation cases.

    Not written by hand. `eval/test_cases.json` already holds percepts this agent is
    expected to handle, with the action each one should produce, because the evaluation
    suite runs them. Reusing them as OpenAPI examples means the examples are known-good
    inputs rather than something plausible someone typed into a docstring, and they cannot
    drift from the agent -- change the eval case and the documented example changes with
    it.

    Without this, "Try it out" in Swagger UI hands a reader an empty box and a schema, and
    asks them to invent a percept. That is the difference between a spec and a usable one.
    """
    eval_path = config.get("performance", {}).get("eval")
    if not eval_path:
        return {}
    try:
        raw = json.loads((base / eval_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    cases = raw.get("cases", raw) if isinstance(raw, dict) else raw
    examples: dict[str, Any] = {}
    for case in cases if isinstance(cases, list) else []:
        if not isinstance(case, dict) or "input" not in case:
            continue
        summary = case.get("id", f"case {len(examples) + 1}")
        expected = case.get("expect_action")
        note = case.get("note", "")
        description = " ".join(part for part in (
            f"Expected action: {expected}." if expected else "",
            note,
            "From this agent's evaluation suite.",
        ) if part)
        examples[summary] = {"summary": summary, "description": description,
                             "value": case["input"]}
    return examples


def _act_path(request_schema: dict, examples: dict, actuators: list[dict]) -> dict[str, Any]:
    return {
        "post": {
            "summary": "Send one percept, get one action",
            "description": (
                "The agent loop for a single percept: validate the input against the "
                "sensor schemas, call the model, validate the action against the actuator "
                "schema, return it. Deterministic on both sides of the model call."
            ),
            "operationId": "act",
            "tags": ["agent"],
            "parameters": [{
                "name": "X-Session-Id",
                "in": "header",
                "required": False,
                "schema": {"type": "string"},
                "description": (
                    "Names a conversation. Without it every request runs on its own agent "
                    "with empty state, so two callers never share one conversation and the "
                    "examples below can be sent in any order. With it, a stateful agent "
                    "carries what it learned into the next request under the same id -- "
                    "which is what makes the difference between asking 'where is my order' "
                    "and answering it."
                ),
                "example": "my-conversation-1",
            }],
            "requestBody": {
                "required": True,
                "content": {"application/json": dict(
                    {"schema": request_schema},
                    **({"examples": examples} if examples else {}),
                )},
            },
            "responses": {
                "200": {
                    "description": "An action the actuator schema accepted.",
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/AgentResult"}}},
                },
                "422": {
                    "description": (
                        "No sensor schema accepted the percept. The caller sent something "
                        "this agent does not perceive. No model was called and nothing was spent."
                    ),
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Refusal"}}},
                },
                "502": {
                    "description": (
                        "The model answered and the answer failed the actuator contract -- "
                        "an action not on the list, or arguments the schema refused. The "
                        "request was fine; the upstream answer was not."
                    ),
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Refusal"}}},
                },
                # The three below are not part of the oscillation pattern, which is why they
                # sit apart from 422 and 502. They are still responses this endpoint really
                # returns, and a spec that omits them generates a client that cannot handle
                # them.
                "400": {
                    "description": (
                        "The body was not a JSON object. Nothing was validated and no model "
                        "was called."
                    ),
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Refusal"}}},
                },
                "405": {
                    "description": (
                        "This route exists and does not accept that method. The Allow header "
                        "names the one it does."
                    ),
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Refusal"}}},
                },
                "503": {
                    "description": (
                        "The percept is valid, this deployment has no model backend "
                        "configured, and no recording exists for the prompt it produces. "
                        "Nothing here invents a response, so there is no answer to return. "
                        "Set a backend key, or send one of the recorded examples above."
                    ),
                    "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Refusal"}}},
                },
            },
        }
    }


def _agent_path() -> dict[str, Any]:
    return {
        "get": {
            "summary": "The PEAS spec this agent is running",
            "description": "The parsed agent.yaml. What the API is generated from.",
            "operationId": "agent",
            "tags": ["introspection"],
            "responses": {"200": {"description": "The agent's PEAS configuration.",
                                  "content": {"application/json": {"schema": {"type": "object"}}}}},
        }
    }


def _spec_path() -> dict[str, Any]:
    """This document, describing itself.

    Self-reference looks like a curiosity and is not. A generated client is built from a
    spec fetched at some URL, and if that URL is not in the document the client cannot
    refresh itself against the thing it was generated from.
    """
    return {
        "get": {
            "summary": "This document",
            "description": (
                "The OpenAPI 3.1 spec for this agent, generated from its agent.yaml at "
                "startup rather than written by hand or kept in sync by discipline."
            ),
            "operationId": "spec",
            "tags": ["introspection"],
            "responses": {"200": {"description": "This document.",
                                  "content": {"application/json": {"schema": {"type": "object"}}}}},
        }
    }


def _docs_path() -> dict[str, Any]:
    return {
        "get": {
            "summary": "Swagger UI for this agent",
            "description": (
                "A browser page rendering /openapi.json. The UI is fetched from a CDN, so "
                "this route needs network access and /openapi.json does not -- an "
                "air-gapped run still gets the spec."
            ),
            "operationId": "docs",
            "tags": ["introspection"],
            "responses": {"200": {"description": "An HTML page.",
                                  "content": {"text/html": {"schema": {"type": "string"}}}}},
        }
    }


def _root_path() -> dict[str, Any]:
    return {
        "get": {
            "summary": "Where everything else is",
            "description": "This agent's name, architecture, and the routes it answers on.",
            "operationId": "root",
            "tags": ["introspection"],
            "responses": {"200": {"description": "Links to the routes above.",
                                  "content": {"application/json": {"schema": {"type": "object"}}}}},
        }
    }


def _health_path() -> dict[str, Any]:
    return {
        "get": {
            "summary": "Liveness, and which backend is configured",
            "operationId": "health",
            "tags": ["introspection"],
            "responses": {"200": {"description": "Serving.",
                                  "content": {"application/json": {"schema": {
                                      "type": "object",
                                      "properties": {
                                          "status": {"type": "string"},
                                          "agent": {"type": "string"},
                                          "provider": {"type": "string"},
                                      }}}}}},
        }
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "00-config-runtime/agents/uptime-triage"
    print(json.dumps(build_spec(target), indent=2))
