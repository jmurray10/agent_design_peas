"""Run the twenty test cases against the support agent and report the six metrics.

The agent under test is the one from `01-reflex-agents/model-based/after.py`, loaded by
path and left completely unmodified. Its six actuators are the actions every case is
scored against.

What this harness adds around it:

    a wrapper on llm_call     counts calls and estimated tokens, times each call, and
                              routes mock_key so every case gets its own canned
                              trajectory instead of all twenty replaying case one
    stdout capture            the agent already prints its own fallback and validation
                              lines; those printed lines are the evidence this harness
                              counts, so a fallback is only recorded when one fired
    a deterministic checker   `expected` in test_cases.json, evaluated in Python

Nothing here decides whether an action was right by asking a model. That is judge.py,
it is opt-in behind --judge, and its score never enters the success rate.

Per-case execution is `run_case`, which returns a `CaseRecord` rather than printing.
Task 17 (cross-model) and Task 20 (drift) both re-run this suite through that function.

Runs with no API key:

    python 08-production-patterns/evaluation/run_eval.py
    python 08-production-patterns/evaluation/run_eval.py --judge
    python 08-production-patterns/evaluation/run_eval.py --case c01 --verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import statistics
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path

# run_eval.py is launched from the repo root, which puts this file's directory on
# sys.path but not the root. parents[2] is the root; the directory itself is added
# explicitly so `import judge` works however this module was loaded.
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import shared.llm as llm_module
from shared.llm import llm_call as _real_llm_call  # noqa: E402

import judge  # noqa: E402

CASES_PATH = HERE / "test_cases.json"
AGENT_PATH = ROOT / "01-reflex-agents" / "model-based" / "after.py"

# Short names for the table. The full action names are what the agent returns and what
# test_cases.json is written in; these are only for the width of a terminal.
SHORT = {
    "check_order_status": "check",
    "issue_refund": "refund",
    "escalate_to_manager": "escalate",
    "reply_to_customer": "reply",
    "request_more_info": "ask",
    "close_ticket": "close",
    "no_op": "REFUSED",
}

# The three deterministic layers inside the agent, keyed by the line each one prints.
FALLBACK_LAYERS = {
    "[fallback] llm_update_state": "update_state_json_parse",
    "[fallback] llm_predict_effect": "predict_effect_json_parse",
    "[validation]": "action_not_allowed",
}


def _load_module(name: str, path: Path):
    """Import a module by file path. Directory names here are not valid identifiers."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The demo in after.py sits under `if __name__ == "__main__"`, so loading it this way
# defines the class without running the three-percept ticket.
support = _load_module("support_after", AGENT_PATH)


def estimate_tokens(text: str) -> int:
    """Four characters per token.

    An estimator, not a tokenizer. Counting real tokens means a dependency and this repo
    installs nothing. Every prompt and every response in every case goes through this one
    function, so the numbers are comparable with each other and approximate against a
    provider's bill.
    """
    return (len(text) + 3) // 4


@dataclass
class CaseRecord:
    """Everything one case produced. Downstream tasks read this, not the printed table."""

    case_id: str
    title: str = ""
    quality: str = ""
    difficulty: str = ""
    actions: list[str] = field(default_factory=list)
    final_action: str = ""
    validation: str = "passed"          # "passed" | "refused"
    invalid_actions: int = 0
    fallbacks: list[dict] = field(default_factory=list)   # {"step": int, "layer": str}
    llm_calls: int = 0
    tiers: list[str] = field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    tokens: int = 0
    latency_s: float = 0.0              # end to end, this case
    model_latency_s: float = 0.0        # the part of it spent inside llm_call
    escalated: bool = False
    success: bool = False
    failures: list[str] = field(default_factory=list)
    scripted: bool = True               # every response matched a canned string
    missing_mock_keys: list[str] = field(default_factory=list)
    judge: dict | None = None
    final_state: dict = field(default_factory=dict)
    transcript: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_suite(path: Path | str = CASES_PATH) -> dict:
    """Load test_cases.json whole, including the agent block and the notes."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cases(path: Path | str = CASES_PATH) -> list[dict]:
    return load_suite(path)["cases"]


# The role and the actuator list come from the suite file rather than being repeated
# here, so the cases and the agent they are scored against cannot drift apart.
SUITE = load_suite()


def _make_wrapper(case_id: str, record: CaseRecord, tier_override: str | None):
    """Wrap llm_call for one case: route mock keys, count tokens, time the call.

    The agent asks for mock_key "reflex_model_state_2". This rewrites it to
    "evaluation_c07_state_2" so each case replays its own canned trajectory. Every real
    backend ignores mock_key entirely, so in real mode this rewriting changes nothing
    about what the model is asked or what it answers.
    """

    def wrapped(prompt: str, mock_key: str = "default", tier: str = "default") -> str:
        key = mock_key
        if mock_key.startswith("reflex_model_"):
            key = mock_key.replace("reflex_model_", f"evaluation_{case_id}_", 1)
        use_tier = tier_override or tier

        start = time.perf_counter()
        raw = _real_llm_call(prompt, mock_key=key, tier=use_tier)
        record.model_latency_s += time.perf_counter() - start

        record.llm_calls += 1
        record.tiers.append(use_tier)
        record.prompt_tokens += estimate_tokens(prompt)
        record.response_tokens += estimate_tokens(raw)

        # Whether a model produced this response or a recording replayed it is a property
        # of the shim's mode, not of the text, so it is read from there. There is no third
        # possibility to detect: an unrecorded prompt raises rather than returning a
        # stand-in, which is what makes the distinction two-valued.
        record.scripted = llm_module._select_provider() == "replay"
        return raw

    return wrapped


def run_case(case: dict, tier_override: str | None = None,
             judge_enabled: bool = False) -> CaseRecord:
    """Run one test case end to end and return what it did.

    `case` is one entry from test_cases.json. `tier_override` forces every model call in
    the run to one capability tier, which is how Task 20 perturbs a run without editing
    the agent. `judge_enabled` adds an LLM-as-judge score for cases carrying
    judge_criteria; the score is reported separately and never affects `success`.
    """
    record = CaseRecord(
        case_id=case["id"],
        title=case.get("title", ""),
        quality=case.get("quality", ""),
        difficulty=case.get("difficulty", ""),
    )

    agent = support.LLMModelBasedReflexAgent(
        role=SUITE["agent"]["role"],
        available_actions=list(SUITE["agent"]["available_actions"]),
    )

    original_llm_call = support.llm_call
    support.llm_call = _make_wrapper(case["id"], record, tier_override)
    start = time.perf_counter()
    try:
        for step, percept_data in enumerate(case["percepts"], start=1):
            buffer = io.StringIO()
            # The agent narrates its own fallbacks to stdout. Capturing that keeps the
            # table readable and, more usefully, makes the agent's own words the source
            # of the fallback counts rather than a second guess at them here.
            with redirect_stdout(buffer):
                action = agent.agent_function(support.Percept(percept_data))
            _absorb_output(buffer.getvalue(), step, record)

            record.actions.append(action)
            record.tool_calls += 1
            if action == "no_op":
                record.validation = "refused"
                record.invalid_actions += 1
                record.tool_errors += 1
    finally:
        support.llm_call = original_llm_call
        record.latency_s = time.perf_counter() - start

    record.tokens = record.prompt_tokens + record.response_tokens
    record.final_action = record.actions[-1] if record.actions else ""
    record.escalated = "escalate_to_manager" in record.actions
    record.final_state = agent.state
    record.failures = check_expected(case.get("expected", {}), record)
    record.success = not record.failures

    if judge_enabled and case.get("judge_criteria"):
        record.judge = judge.evaluate_response(
            render_agent_output(case, record),
            case["judge_criteria"],
            mock_key=f"evaluation_judge_{case['id']}",
        )

    return record


def _absorb_output(captured: str, step: int, record: CaseRecord) -> None:
    """Record the agent's printed lines, and let the shim's mode banner through."""
    for line in captured.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Printed once per process by shared/llm.py. It belongs on the reader's screen,
        # not buried in a captured transcript.
        if stripped.startswith("[replay]") or stripped.startswith("[live]"):
            print(stripped)
            continue
        record.transcript.append(f"step {step}: {stripped}")
        for marker, layer in FALLBACK_LAYERS.items():
            if stripped.startswith(marker):
                record.fallbacks.append({"step": step, "layer": layer})


def check_expected(expected: dict, record: CaseRecord) -> list[str]:
    """Deterministic verification. Returns one string per unmet expectation.

    Only the keys a case actually specifies are checked, so a case that constrains one
    thing is not silently held to five.
    """
    failures = []
    actions = record.actions

    for required in expected.get("must_include_actions", []):
        if required not in actions:
            failures.append(f"never took required action {required}")
    for forbidden in expected.get("must_not_include_actions", []):
        if forbidden in actions:
            failures.append(f"took forbidden action {forbidden}")

    if "final_action" in expected and record.final_action != expected["final_action"]:
        failures.append(
            f"final action was {record.final_action or 'none'}, "
            f"expected {expected['final_action']}"
        )
    if "max_actions" in expected and len(actions) > expected["max_actions"]:
        failures.append(f"took {len(actions)} actions, limit was {expected['max_actions']}")
    if "escalates" in expected and record.escalated != bool(expected["escalates"]):
        failures.append(
            "escalated when it should not have" if record.escalated
            else "did not escalate when it should have"
        )
    if "validation" in expected and record.validation != expected["validation"]:
        failures.append(
            f"validation outcome was {record.validation}, expected {expected['validation']}"
        )
    return failures


def render_agent_output(case: dict, record: CaseRecord) -> str:
    """Flatten one run into the text the judge reads.

    The support agent emits actions, not prose, so what gets judged is the handling: what
    it saw, what it did, and the state it ended in.
    """
    lines = []
    for step, (percept, action) in enumerate(zip(case["percepts"], record.actions), start=1):
        lines.append(f"percept {step}: {json.dumps(percept)}")
        lines.append(f"action {step}: {action}")
    lines.append(f"final internal state: {json.dumps(record.final_state)}")
    return "\n".join(lines)


def run_suite(cases: list[dict] | None = None, tier_override: str | None = None,
              judge_enabled: bool = False) -> list[CaseRecord]:
    """Run every case and return the records in suite order."""
    return [run_case(case, tier_override, judge_enabled)
            for case in (cases if cases is not None else load_cases())]


def aggregate(records: list[CaseRecord]) -> dict:
    """The six metrics from the source page, plus what they were computed from."""
    n = len(records)
    if n == 0:
        return {}

    successes = sum(1 for r in records if r.success)
    tool_calls = sum(r.tool_calls for r in records)
    tool_errors = sum(r.tool_errors for r in records)
    tokens = [r.tokens for r in records]
    latencies = [r.latency_s for r in records]
    escalations = sum(1 for r in records if r.escalated)

    by_layer: dict[str, int] = {}
    for record in records:
        for fallback in record.fallbacks:
            by_layer[fallback["layer"]] = by_layer.get(fallback["layer"], 0) + 1

    judged = [r.judge for r in records if r.judge is not None]
    judge_mean, judge_scored, judge_refused = judge.mean_score(judged) if judged else (None, 0, 0)

    return {
        "cases": n,
        "successes": successes,
        "task_success_rate": successes / n,
        "tokens_total": sum(tokens),
        "tokens_per_task": sum(tokens) / n,
        "tool_calls_total": tool_calls,
        "tool_calls_per_task": tool_calls / n,
        "tool_errors": tool_errors,
        "tool_error_rate": (tool_errors / tool_calls) if tool_calls else 0.0,
        "latency_total_s": sum(latencies),
        "latency_mean_s": sum(latencies) / n,
        "latency_p50_s": statistics.median(latencies),
        "escalations": escalations,
        "escalation_rate": escalations / n,
        "llm_calls_total": sum(r.llm_calls for r in records),
        "fallbacks_total": sum(len(r.fallbacks) for r in records),
        "fallbacks_by_layer": by_layer,
        "all_scripted": all(r.scripted for r in records),
        "judge_mean_score": judge_mean,
        "judge_scored": judge_scored,
        "judge_refused": judge_refused,
    }


# -- output -------------------------------------------------------------------------

def _short_actions(record: CaseRecord) -> str:
    return " > ".join(SHORT.get(a, a) for a in record.actions) or "(none)"


def print_table(records: list[CaseRecord]) -> None:
    header = (f"{'id':<5}{'quality':<10}{'difficulty':<13}{'actions':<28}"
              f"{'tokens':>8}{'ms':>8}{'fb':>4}  result")
    print(header)
    print("-" * len(header))
    for record in records:
        fallbacks = str(len(record.fallbacks)) if record.fallbacks else "-"
        print(f"{record.case_id:<5}{record.quality:<10}{record.difficulty:<13}"
              f"{_short_actions(record):<28}{record.tokens:>8}"
              f"{record.latency_s * 1000:>8.1f}{fallbacks:>4}  "
              f"{'PASS' if record.success else 'FAIL'}")
        for failure in record.failures:
            print(f"{'':<5}{'':<10}-> {failure}")
        for fallback in record.fallbacks:
            print(f"{'':<5}{'':<10}-> fallback fired at step {fallback['step']}: "
                  f"{fallback['layer']}")
        if record.judge:
            # Printed beside the verdict, never folded into it. The judge is an opinion
            # about handling quality; PASS and FAIL above are assertions.
            score = record.judge["score"]
            rendered = f"{score}/5" if score is not None else f"none ({record.judge['error']})"
            print(f"{'':<5}{'':<10}-> judge (subjective, not part of the verdict): {rendered}")
    print("-" * len(header))
    print("actions: check=check_order_status  refund=issue_refund  "
          "escalate=escalate_to_manager")
    print("         reply=reply_to_customer  ask=request_more_info  close=close_ticket")
    print("         REFUSED=the model named an action that is not on the actuator list")


def print_metrics(metrics: dict) -> None:
    def row(label: str, value: str, maps_to: str) -> None:
        print(f"  {label:<22}{value:<39}{maps_to}")

    print()
    print("THE SIX METRICS")
    row("", "", "maps to (PEAS)")
    row("task success rate",
        f"{metrics['successes']} / {metrics['cases']} = {metrics['task_success_rate']:.1%}",
        "performance measure")
    row("tokens per task",
        f"{metrics['tokens_per_task']:.0f} mean, {metrics['tokens_total']} total (estimated)",
        "cost and scalability")
    row("tool calls per task",
        f"{metrics['tool_calls_per_task']:.2f} mean, {metrics['tool_calls_total']} total",
        "actuator efficiency")
    row("tool error rate",
        f"{metrics['tool_errors']} / {metrics['tool_calls_total']} = "
        f"{metrics['tool_error_rate']:.1%}",
        "actuator reliability")
    row("latency (end to end)",
        f"{metrics['latency_mean_s'] * 1000:.1f} ms mean, "
        f"{metrics['latency_p50_s'] * 1000:.1f} ms p50",
        "performance measure")
    row("escalation rate",
        f"{metrics['escalations']} / {metrics['cases']} = {metrics['escalation_rate']:.1%}",
        "confidence calibration")

    print()
    print(f"  model calls            {metrics['llm_calls_total']}")
    print(f"  fallbacks fired        {metrics['fallbacks_total']}")
    for layer, count in sorted(metrics["fallbacks_by_layer"].items()):
        print(f"    {layer:<28}{count}")
    if metrics.get("judge_scored"):
        mean = metrics["judge_mean_score"]
        print(f"  judge score (subjective, excluded from success): "
              f"{mean:.2f}/5 over {metrics['judge_scored']} case(s), "
              f"{metrics['judge_refused']} unusable")


def print_caveats(metrics: dict, records: list[CaseRecord]) -> None:
    print()
    print("=" * 78)
    if metrics["all_scripted"]:
        print("REPLAYED, NOT MEASURED TODAY")
        print(
            "Every action in the table was replayed from shared/transcripts/ -- what a real\n"
            "model returned to that exact prompt on the date stored beside it. Nobody wrote\n"
            "these actions. The rate measures that model on that day: not this harness\n"
            "against an authored run, and not any model's behaviour today.\n"
            "\n"
            "The failures are the model's too. No case was arranged to fail and no response\n"
            "was arranged to be unparseable -- the counters above report what happened, and\n"
            "a counter reading zero means that layer had nothing to catch on this\n"
            "recording. Set ANTHROPIC_API_KEY, or point the shim at a local model, and\n"
            "every number here becomes an observation of that model instead."
        )
    else:
        print("MEASURED")
        print(
            "Real mode. A model chose every action in the table, so the success rate,\n"
            "the fallback counts and the escalation rate are all observations of this\n"
            "run. They will vary between runs and between models. One run is not a\n"
            "benchmark."
        )
    print("=" * 78)

    print()
    print("Reading the numbers:")
    print("  Tokens are estimated at four characters each, prompts and responses both.")
    print("  No tokenizer is installed. Treat the totals as approximate and the")
    print("  comparisons between cases, which share the estimator, as sound.")
    if metrics["all_scripted"]:
        print("  Latency on a replay is a dictionary lookup and some Python. It measures")
        print("  this harness on this machine and says nothing whatsoever about any")
        print("  provider. The column is printed anyway because a metric that only")
        print("  appears once a key is set is a metric nobody has ever tested.")
        print("  The first case in any run also absorbs the shim's one-time provider")
        print("  selection, which is why its millisecond column is an outlier and why the")
        print("  p50 is the more useful of the two latency figures.")
    else:
        print("  Latency is wall clock per case against a live provider, on this machine,")
        print("  on this network, at this moment. It is not a property of the model.")
    print("  A tool error here is an action the deterministic check refused because it")
    print("  was not on the actuator list. Parse fallbacks are counted separately.")

    missing = sorted({key for record in records for key in record.missing_mock_keys})
    if missing:
        print()
        print("calls fell back to the default canned string, so the cases below are not")
        print("running the trajectory they were written for:")
        for key in missing:
            print(f"  {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", nargs="+", metavar="ID",
                        help="run only these case ids, e.g. --case c01 c16")
    parser.add_argument("--judge", action="store_true",
                        help="also run the LLM-as-judge on cases with judge_criteria")
    parser.add_argument("--tier", choices=["small", "mid", "frontier"],
                        help="force every model call in the run to one capability tier")
    parser.add_argument("--verbose", action="store_true",
                        help="print each agent's own output per case")
    parser.add_argument("--json", metavar="PATH",
                        help="write the per-case records and metrics to a JSON file")
    args = parser.parse_args()

    cases = load_cases()
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            raise SystemExit(f"unknown case id(s): {', '.join(sorted(unknown))}")

    print(f"Evaluation suite: {len(cases)} case(s) against "
          f"{SUITE['agent']['class']} from {SUITE['agent']['source']}")
    if args.tier:
        print(f"Tier override: every model call forced to tier={args.tier}")
    print()

    records = run_suite(cases, tier_override=args.tier, judge_enabled=args.judge)

    if args.verbose:
        for record in records:
            print(f"\n--- {record.case_id} {record.title}")
            for line in record.transcript:
                print(f"  {line}")
            if record.judge:
                print(f"  judge: {record.judge}")
        print()

    print_table(records)
    metrics = aggregate(records)
    print_metrics(metrics)
    print_caveats(metrics, records)

    if args.json:
        payload = {
            "suite": SUITE["suite"],
            "agent": SUITE["agent"],
            "tier_override": args.tier,
            "metrics": metrics,
            "records": [record.to_dict() for record in records],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
