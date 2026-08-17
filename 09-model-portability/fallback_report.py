"""Which deterministic layer caught what, case by case.

`compare_models.py` prints one number per backend. This is the version underneath it:
for every case in the suite, whether a guard fired, which one, what the agent itself
said when it fired, and what the run did next.

Three things get counted, and all three already existed before this file did. Nothing
here adds a failure mode, simulates one, or decides on its own that something went
wrong -- the agent in `01-reflex-agents/model-based/after.py` prints a line when a guard
fires, Task 13's harness captures those lines, and this script sorts them.

    action_not_allowed          the model named an action that is not an actuator
    update_state_json_parse     the model's state update would not parse as JSON
    predict_effect_json_parse   the model's effect prediction would not parse as JSON

The fourth layer people expect, output-schema validation, is not on this agent's path.
It exists in this repository, in `00-config-runtime/runtime.py` and
`05-multi-agent/orchestration/after.py`, and it is reported below as absent rather than
as zero -- a layer that is not installed and a layer that never fired are different
findings, and only one of them is reassuring.

The section this exists for is the last one. Guards catch malformed output. They do not
catch a well-formed, permitted, wrong answer, and the suite contains those too.

    python 09-model-portability/fallback_report.py
    python 09-model-portability/fallback_report.py --all
    python 09-model-portability/fallback_report.py --provider ollama
    python 09-model-portability/fallback_report.py --case c04 c16 c17 --json report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_REL = "08-production-patterns/evaluation/run_eval.py"
AGENT_REL = "01-reflex-agents/model-based/after.py"


def _load_module(name: str, path: Path):
    """Import a module by file path. The directory names here are not identifiers."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Importing run_eval.py defines the harness without running the suite: its own demo is
# under `if __name__ == "__main__"`. Loading it also loads the agent under test.
ev = _load_module("run_eval", ROOT / EVAL_REL)

# What each layer is, in the words of the code it guards. `where` is a real location in
# a file this task did not write and did not modify.
LAYERS: dict[str, dict[str, str]] = {
    "action_not_allowed": {
        "where": f"{AGENT_REL}: agent_function",
        "guards": "the action name the model returned",
        "test": "membership in available_actions",
        "then": "substitutes the no_op sentinel; the action is never executed",
        "per": "action choices",
    },
    "update_state_json_parse": {
        "where": f"{AGENT_REL}: llm_update_state",
        "guards": "the model's updated state object",
        "test": "json.loads, under try/except",
        "then": "merges the percept into the previous state by hand and keeps going",
        "per": "state updates",
    },
    "predict_effect_json_parse": {
        "where": f"{AGENT_REL}: llm_predict_effect",
        "guards": "the model's prediction of the state after the action",
        "test": "json.loads, under try/except",
        "then": "leaves state unchanged, so the effect of the action is not recorded",
        "per": "effect predictions",
    },
}

# Present in the repository, absent from this agent. Reported, not silently omitted.
ABSENT = {
    "output_schema_validation": {
        "where": "00-config-runtime/runtime.py: validate_output, and "
                 "05-multi-agent/orchestration/after.py",
        "guards": "the shape of a structured action or handoff, not just its name",
        "why_absent": "this agent's actuators take no arguments, so there is no payload "
                      "to validate a schema against",
    },
}


def collect(cases: list[dict]) -> list:
    """Run each case and return its record. One call per case, so a failure is local."""
    records = []
    for case in cases:
        # run_case is Task 13's per-case entry point. Everything below reads the record
        # it returns; nothing re-derives a verdict here.
        records.append(ev.run_case(case))
    return records


def layer_counts(records: list) -> dict[str, int]:
    counts = {layer: 0 for layer in LAYERS}
    for record in records:
        for fallback in record.fallbacks:
            counts[fallback["layer"]] = counts.get(fallback["layer"], 0) + 1
    return counts


def _lines_by_step(transcript: list[str]) -> dict[int, list[str]]:
    """Regroup the harness's flat transcript, which is prefixed `step N: `, by step."""
    grouped: dict[int, list[str]] = {}
    for line in transcript:
        prefix, _, rest = line.partition(": ")
        try:
            step = int(prefix.split()[1])
        except (IndexError, ValueError):
            continue
        grouped.setdefault(step, []).append(rest)
    return grouped


def classify(record) -> str:
    """Which of the four quadrants this case landed in."""
    caught = bool(record.fallbacks)
    if record.success:
        return "passed_with_guard" if caught else "passed_clean"
    return "failed_with_guard" if caught else "failed_uncaught"


# -- output ---------------------------------------------------------------------------

RULE = "=" * 78


def print_layers() -> None:
    print(RULE)
    print("THE LAYERS")
    print(RULE)
    print("Every one of these is ordinary Python sitting between a model call and the")
    print("rest of the run. None of them were added for this report.")
    print()
    for name, layer in LAYERS.items():
        print(f"  {name}")
        print(f"    lives in   {layer['where']}")
        print(f"    guards     {layer['guards']}")
        print(f"    test       {layer['test']}")
        print(f"    on fire    {layer['then']}")
    for name, layer in ABSENT.items():
        print(f"  {name}   NOT ON THIS AGENT'S PATH")
        print(f"    lives in   {layer['where']}")
        print(f"    guards     {layer['guards']}")
        print(f"    absent     {layer['why_absent']}")
    print()
    print("  The absent layer is listed rather than counted as zero. A guard that is not")
    print("  installed and a guard that never fired look identical in a total and mean")
    print("  opposite things.")


def print_counts(records: list, metrics: dict) -> None:
    counts = layer_counts(records)
    steps = sum(len(record.actions) for record in records)
    calls = metrics["llm_calls_total"]

    print()
    print(RULE)
    print("WHAT EACH LAYER CAUGHT")
    print(RULE)
    # Each step is one state update, one action choice and one effect prediction, so the
    # denominator for every layer is the number of steps, not the number of cases.
    print(f"{len(records)} case(s), {steps} step(s), {calls} model call(s).")
    print("Each step is one state update, one action choice, one effect prediction, so")
    print(f"every layer below had {steps} chance(s) to fire.")
    print()
    print(f"  {'layer':<28}{'fired':>7}{'of':>7}{'rate':>9}")
    for name, layer in LAYERS.items():
        rate = counts[name] / steps if steps else 0.0
        print(f"  {name:<28}{counts[name]:>7}{steps:>7}{rate:>9.1%}   "
              f"per {layer['per']}")
    for name in ABSENT:
        print(f"  {name:<28}{'n/a':>7}{'n/a':>7}{'n/a':>9}   not on this agent's path")
    total = sum(counts.values())
    rate = total / calls if calls else 0.0
    print(f"  {'all layers':<28}{total:>7}{calls:>7}{rate:>9.1%}   per model call")

    if total != metrics["fallbacks_total"]:
        print(f"  NOTE: the harness totalled {metrics['fallbacks_total']} fallback(s) and "
              f"this report attributed {total}. They should match.")


def print_cases(records: list, show_all: bool) -> None:
    print()
    print(RULE)
    print("CASE BY CASE")
    print(RULE)
    print("Quoted lines are the agent's own stdout, captured by the harness. That is")
    print("what makes these counts triggers rather than assertions about triggers.")
    if not show_all:
        print("Only cases where a guard fired or the case failed. --all shows every case.")
    print()

    shown = 0
    for record in records:
        interesting = record.fallbacks or not record.success
        if not (interesting or show_all):
            continue
        shown += 1
        verdict = "PASS" if record.success else "FAIL"
        print(f"  {record.case_id}  {verdict}  {record.quality}/{record.difficulty}  "
              f"{record.title[:52]}")
        actions = " > ".join(ev.SHORT.get(action, action) for action in record.actions)
        print(f"        actions: {actions or '(none)'}")

        said = _lines_by_step(record.transcript)
        caught_steps = sorted({fallback["step"] for fallback in record.fallbacks})
        for step in caught_steps:
            fired = [f["layer"] for f in record.fallbacks if f["step"] == step]
            print(f"        step {step}: caught by {', '.join(fired)}")
            for layer in fired:
                print(f"          -> {LAYERS[layer]['then']}")
            for line in said.get(step, []):
                print(f"          agent said: {line}")
        # A printed line at a step where no layer was attributed would mean the agent
        # narrated something this report does not know how to file. Show it rather than
        # drop it.
        for step in sorted(set(said) - set(caught_steps)):
            for line in said[step]:
                print(f"        step {step}: the agent also printed: {line}")
        for failure in record.failures:
            print(f"        unmet expectation: {failure}")
        if not record.fallbacks and not record.success:
            print("        no layer fired: the output was well formed, on the actuator")
            print("        list, and wrong")
            print("        caught instead by the expected-outcome check in the eval suite")
        print()
    if shown == 0:
        print("  Nothing fired and nothing failed.")
        print()


def print_quadrants(records: list) -> None:
    buckets: dict[str, list[str]] = {
        "passed_clean": [], "passed_with_guard": [],
        "failed_with_guard": [], "failed_uncaught": [],
    }
    for record in records:
        buckets[classify(record)].append(record.case_id)

    print(RULE)
    print("WHAT THE LAYERS ARE FOR, AND WHAT THEY ARE NOT FOR")
    print(RULE)
    print(f"  {'':<10}{'guard fired':>14}{'no guard fired':>18}")
    print(f"  {'passed':<10}{len(buckets['passed_with_guard']):>14}"
          f"{len(buckets['passed_clean']):>18}")
    print(f"  {'failed':<10}{len(buckets['failed_with_guard']):>14}"
          f"{len(buckets['failed_uncaught']):>18}")
    print()
    if buckets["passed_with_guard"]:
        print(f"  passed with a guard fired: {', '.join(buckets['passed_with_guard'])}")
        print("    A model call came back unusable and the case still met its expected")
        print("    outcome. That is the architecture working: the deterministic layer")
        print("    absorbed the bad output instead of the run stopping on it.")
    if buckets["failed_with_guard"]:
        print(f"  failed with a guard fired: {', '.join(buckets['failed_with_guard'])}")
        print("    The guard fired and the case still missed its expected outcome. The")
        print("    fallback kept the run alive; it did not make the answer right.")
    if buckets["failed_uncaught"]:
        print(f"  failed with no guard fired: {', '.join(buckets['failed_uncaught'])}")
        print("    This is the row to read twice. Every layer in this agent checks")
        print("    whether a response is well formed and permitted. None of them checks")
        print("    whether it is correct, because nothing in the code knows what correct")
        print("    is. These cases were caught by the expected-outcome comparison in the")
        print("    eval suite, which is a test written in advance by a person -- not by")
        print("    anything the agent could run on itself in production.")
    print()
    print("  A fallback rate is therefore a measure of how often a model returns")
    print("  unusable output, not of how often it is wrong. Those are different numbers")
    print("  and only one of them has a guard behind it.")


def print_caveats(records: list, metrics: dict, provider: str) -> None:
    print(RULE)
    print("READING THIS REPORT")
    print(RULE)
    print(f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')} local time, "
          f"python {platform.python_version()}, {platform.system()} {platform.machine()}, "
          f"provider={provider}.")
    if metrics["all_scripted"]:
        print("Replayed, not measured today. Every response in this run came from")
        print("shared/transcripts/, so the output is a replay of a real run rather than a fresh")
        print("one. The trigger paths are real and the code that caught them is the agent's")
        print("own. Which responses arrived was decided by a model on the recorded date and")
        print("not by hand, so the rates describe that model then rather than one now.")
        print("Point the shim at a backend and every rate here becomes an observation")
        print("of that model, on that day, at that version.")
    else:
        print("Measured. At least one response in this run did not match a canned")
        print("string, so a model produced it. These rates are one observation of one")
        print("model on one run of this suite. Model versions move under a fixed name.")
        print("Run it again, and run it yourself, before quoting any of it.")
    missing = sorted({key for record in records for key in record.missing_mock_keys})
    if missing:
        print()
        print("calls fell back to the default canned string, so the cases involved did")
        print("not run the trajectory they were written for:")
        for key in missing:
            print(f"  {key}")


# -- entry point ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", metavar="NAME",
                        choices=["replay", "ollama", "anthropic", "openai_compatible"],
                        help="run against this backend (default: the shim's own search "
                             "order, which lands on mock with no setup)")
    parser.add_argument("--case", nargs="+", metavar="ID",
                        help="run only these case ids")
    parser.add_argument("--all", action="store_true",
                        help="show every case, not only the ones a guard touched")
    parser.add_argument("--json", metavar="PATH",
                        help="write the per-case attribution to a JSON file")
    args = parser.parse_args()

    # Set before the first model call: the shim resolves its provider once and caches it.
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    cases = ev.load_cases()
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise SystemExit(f"unknown case id(s): {', '.join(sorted(unknown))}")

    print(f"Fallback report: {len(cases)} case(s) from {EVAL_REL}")
    print(f"Agent under test: {AGENT_REL}, unmodified")
    print()

    records = collect(cases)
    metrics = ev.aggregate(records)
    # The shim's own resolver, so this line cannot disagree with what actually ran.
    provider = shim_provider()

    print()
    print_layers()
    print_counts(records, metrics)
    print_cases(records, args.all)
    print_quadrants(records)
    print_caveats(records, metrics, provider)

    print()
    print("Backend-by-backend version of the same numbers: python "
          "09-model-portability/compare_models.py")

    if args.json:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "provider": provider,
            "scripted": metrics["all_scripted"],
            "suite": EVAL_REL,
            "agent": AGENT_REL,
            "layers": {name: layer_counts(records)[name] for name in LAYERS},
            "layers_absent": {name: layer["where"] for name, layer in ABSENT.items()},
            "steps": sum(len(record.actions) for record in records),
            "llm_calls": metrics["llm_calls_total"],
            "cases": [
                {
                    "case_id": record.case_id,
                    "success": record.success,
                    "quadrant": classify(record),
                    "fallbacks": record.fallbacks,
                    "failures": record.failures,
                    "transcript": record.transcript,
                }
                for record in records
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


def shim_provider() -> str:
    """Which backend the shim actually selected for this process."""
    from shared import llm

    return llm._select_provider()


if __name__ == "__main__":
    main()
