"""Run the twenty-case evaluation suite against every backend configured on this
machine and print one table.

The suite is Task 13's, unchanged: `08-production-patterns/evaluation/run_eval.py`
against the support agent in `01-reflex-agents/model-based/after.py`. Nothing here
edits either one. Swapping the model is a change to `LLM_PROVIDER` and
`shared/providers.yaml`, which is the claim this directory exists to make checkable.

With no setup at all there is exactly one row, the replayed one, and it is labelled as
replayed rather than measured. Every backend this machine is not configured for still
gets a row, marked `not run`, with the reason and what to do about it. A backend that
was configured but failed mid-run gets a row saying so and no numbers -- a missing
number is honest, an invented one is not.

Each backend runs in its own process. `shared/llm.py` resolves its provider once and
caches it, so a fresh process is the only way to ask for a second one without reaching
into the shim's internals; it also means every row was produced by a command a reader
can type by hand and check.

    python 09-model-portability/compare_models.py
    python 09-model-portability/compare_models.py --provider mock ollama
    python 09-model-portability/compare_models.py --case c01 c04 c16 c17
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The shim's own port probe, deliberately. If this file asked "is Ollama up" a
# different way it could advertise a backend the shim would then decline to select,
# and the table would be describing a machine that does not exist.
import shared.llm as shim  # noqa: E402

EVAL_REL = "08-production-patterns/evaluation/run_eval.py"
EVAL_SCRIPT = ROOT / EVAL_REL
PROVIDERS_YAML = ROOT / "shared" / "providers.yaml"

# Order of the rows. Mock first because it is the row everyone gets.
PROVIDER_ORDER = ["replay", "ollama", "anthropic", "gemini", "openai_compatible"]


@dataclass
class Backend:
    """One possible row: whether it can run here, and why not if it cannot."""

    provider: str
    available: bool
    detail: str
    enable: str = ""


@dataclass
class RunResult:
    """What one subprocess produced. `payload` is run_eval.py's --json output."""

    provider: str
    ok: bool = False
    payload: dict | None = None
    error: str = ""
    wall_s: float = 0.0
    banner: str = ""
    command: str = ""
    notes: list[str] = field(default_factory=list)


# -- which backends exist on this machine ---------------------------------------------

def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def detect_backends() -> list[Backend]:
    """Ask the same questions `shared/llm.py` asks, and record the answers.

    Detection is deliberately conservative: a backend is only called available when
    everything it needs is present. Anything else is a `not run` row with a reason,
    because a row that silently disappears reads as a backend that has no problems.
    """
    backends: list[Backend] = []

    backends.append(Backend(
        "replay", True,
        "always available; recorded real responses from shared/transcripts/, no key, no network",
    ))

    # A real backend also needs the tier map, which is the shim's only non-stdlib import.
    yaml_missing = "pyyaml is not installed, so shared/providers.yaml cannot be read"

    if shim._ollama_is_up():
        if _installed("yaml"):
            backends.append(Backend(
                "ollama", True,
                f"something is listening on {shim.OLLAMA_HOST}",
            ))
        else:
            backends.append(Backend("ollama", False, yaml_missing, "pip install pyyaml"))
    else:
        backends.append(Backend(
            "ollama", False,
            f"nothing is listening on {shim.OLLAMA_HOST}",
            "start ollama and pull a model named under `ollama:` in shared/providers.yaml",
        ))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        backends.append(Backend(
            "anthropic", False, "ANTHROPIC_API_KEY is not set",
            "export ANTHROPIC_API_KEY=... and pip install anthropic",
        ))
    elif not _installed("anthropic"):
        backends.append(Backend(
            "anthropic", False,
            "ANTHROPIC_API_KEY is set but the anthropic package is not installed",
            "pip install anthropic",
        ))
    elif not _installed("yaml"):
        backends.append(Backend("anthropic", False, yaml_missing, "pip install pyyaml"))
    else:
        backends.append(Backend(
            "anthropic", True,
            "ANTHROPIC_API_KEY is set and the anthropic package is importable",
        ))

    # No SDK check here, unlike Anthropic: shared/llm.py reaches Google over its REST
    # endpoint with urllib, so a key and the tier map are the whole requirement.
    if not os.environ.get("GEMINI_API_KEY"):
        backends.append(Backend(
            "gemini", False, "GEMINI_API_KEY is not set",
            "export GEMINI_API_KEY=... (aistudio.google.com/apikey)",
        ))
    elif not _installed("yaml"):
        backends.append(Backend("gemini", False, yaml_missing, "pip install pyyaml"))
    else:
        backends.append(Backend("gemini", True, "GEMINI_API_KEY is set"))

    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
    if not base_url:
        backends.append(Backend(
            "openai_compatible", False, "OPENAI_COMPATIBLE_BASE_URL is not set",
            "export OPENAI_COMPATIBLE_BASE_URL=... (vLLM, Together, Groq, OpenRouter, "
            "LM Studio) and check the model names under `openai_compatible:` in "
            "shared/providers.yaml",
        ))
    elif not _installed("yaml"):
        backends.append(Backend("openai_compatible", False, yaml_missing, "pip install pyyaml"))
    else:
        backends.append(Backend(
            "openai_compatible", True,
            f"OPENAI_COMPATIBLE_BASE_URL is set to {base_url}",
        ))

    return backends


# -- running one backend --------------------------------------------------------------

def run_backend(provider: str, case_ids: list[str] | None = None,
                timeout: float = 1800.0) -> RunResult:
    """Run the whole eval suite against one backend, in its own process.

    `LLM_PROVIDER` is set explicitly rather than left to the shim's search order. On a
    machine with Ollama running, an unset variable would make the row labelled `mock`
    a live Ollama run under somebody else's name.
    """
    env = dict(os.environ)
    env["LLM_PROVIDER"] = provider
    shown = (f"LLM_PROVIDER={provider} python {EVAL_REL} --json {provider}.json"
             + (f" --case {' '.join(case_ids)}" if case_ids else ""))

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"{provider}.json"
        command = [sys.executable, str(EVAL_SCRIPT), "--json", str(out_path)]
        if case_ids:
            command += ["--case", *case_ids]

        start = time.perf_counter()
        try:
            # encoding is pinned: a live model can return anything, and a console
            # code page that cannot represent it would kill the comparison rather
            # than the row.
            proc = subprocess.run(
                command, cwd=str(ROOT), env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RunResult(provider, False, None,
                             f"no result after {timeout:.0f}s, killed",
                             time.perf_counter() - start, "", shown)
        wall = time.perf_counter() - start
        banner = _banner_line(proc.stdout)

        if proc.returncode != 0:
            return RunResult(provider, False, None,
                             _tail(proc.stderr) or _tail(proc.stdout) or
                             f"exit code {proc.returncode}", wall, banner, shown)
        if not out_path.exists():
            return RunResult(provider, False, None,
                             "the run exited cleanly but wrote no JSON", wall, banner, shown)
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    return RunResult(provider, True, payload, "", wall, banner, shown)


def _banner_line(stdout: str) -> str:
    """The shim's one-time mode banner, which names the provider and model it chose."""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("[real mode]") or stripped.startswith("[mock mode]"):
            return stripped
    return ""


def _tail(text: str, lines: int = 1) -> str:
    kept = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return " | ".join(kept[-lines:])


# -- turning a run into a row ---------------------------------------------------------

def models_used(provider: str, tiers: list[str]) -> dict[str, str]:
    """Resolve the tiers this run actually asked for to model names.

    The mapping comes from shared/providers.yaml, the same file the shim resolves
    through, rather than from anything typed in here. Mock mode has no model, and
    reading the file needs pyyaml, so both cases return nothing and the caller says
    so instead of guessing a name.
    """
    if provider == "replay" or not tiers:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    mapping = yaml.safe_load(PROVIDERS_YAML.read_text(encoding="utf-8")).get(provider, {})
    resolved = {}
    for tier in tiers:
        resolved[tier] = mapping.get(tier) or mapping.get("default", "(unmapped)")
    return resolved


def summarize(result: RunResult) -> dict:
    """The one row's worth of numbers, all of them read out of the run's own record.

    Fallback rate is fallbacks divided by model calls, because in this agent every
    model call is followed by exactly one deterministic guard: the state parse, the
    allowed-action membership test, the effect parse. So the rate reads as the share
    of model outputs that a deterministic layer had to catch.
    """
    payload = result.payload or {}
    metrics = payload.get("metrics", {})
    records = payload.get("records", [])
    layers = metrics.get("fallbacks_by_layer", {})

    parse_failures = (layers.get("update_state_json_parse", 0)
                      + layers.get("predict_effect_json_parse", 0))
    invalid_actions = layers.get("action_not_allowed", 0)
    calls = metrics.get("llm_calls_total", 0)

    tiers: list[str] = []
    for record in records:
        for tier in record.get("tiers", []):
            if tier not in tiers:
                tiers.append(tier)

    notes = list(result.notes)
    # The layer count comes from the agent's printed [validation] lines; tool_errors
    # counts the no_op actions those refusals produced. They are two readings of the
    # same event and they should agree. Say so out loud when they do not.
    if invalid_actions != metrics.get("tool_errors", invalid_actions):
        notes.append(f"{result.provider}: the allowed-action layer fired "
                     f"{invalid_actions} time(s) but {metrics['tool_errors']} action(s) "
                     f"came back refused -- those two counts should match")

    return {
        "provider": result.provider,
        "models": models_used(result.provider, tiers),
        "tiers": tiers,
        "cases": metrics.get("cases", 0),
        "successes": metrics.get("successes", 0),
        "success_rate": metrics.get("task_success_rate", 0.0),
        "fallbacks": metrics.get("fallbacks_total", 0),
        "llm_calls": calls,
        "fallback_rate": (metrics.get("fallbacks_total", 0) / calls) if calls else 0.0,
        "fallbacks_by_layer": layers,
        "invalid_actions": invalid_actions,
        "parse_failures": parse_failures,
        "latency_p50_ms": metrics.get("latency_p50_s", 0.0) * 1000,
        "tokens_per_task": metrics.get("tokens_per_task", 0.0),
        "tokens_total": metrics.get("tokens_total", 0),
        "scripted": metrics.get("all_scripted", False),
        "wall_s": result.wall_s,
        "banner": result.banner,
        "command": result.command,
        "notes": notes,
    }


# -- output ---------------------------------------------------------------------------

HEADER = (f"{'provider':<18}{'model':<24}{'success':>13}{'fb rate':>8}"
          f"{'invalid':>9}{'parse':>7}{'p50 ms':>9}{'tok/case':>10}")

# What a cell holds when a backend produced no number. Never a zero: a backend that did
# not run has no fallback rate, and printing 0.0% for one is the whole failure mode this
# script is supposed to avoid.
BLANK = (f"{'--':>13}{'--':>8}{'--':>9}{'--':>7}{'--':>9}{'--':>10}")


def _model_cell(row: dict) -> str:
    if row["provider"] == "replay":
        # Not one model. A replayed row serves whatever model each recording names, and
        # the transcripts hold three of them, so a single cell here would be a fiction.
        return "(recorded; see shared/transcripts/)"
    names = list(dict.fromkeys(row["models"].values()))
    if not names:
        return "(see providers.yaml)"
    cell = ", ".join(names)
    if len(cell) <= 23:
        return cell
    # Truncating a list of two model names to one would name the wrong model. Say how
    # many there were and let the detail block below print them in full.
    if len(names) > 1:
        return f"{len(names)} models, listed below"
    return cell[:20] + "..."


def print_table(rows: list[dict], skipped: list[Backend], failed: list[RunResult]) -> None:
    print(HEADER)
    print("-" * len(HEADER))
    order = {name: index for index, name in enumerate(PROVIDER_ORDER)}
    entries: list[tuple[int, str]] = []

    for row in rows:
        success = f"{row['successes']}/{row['cases']} = {row['success_rate']:.0%}"
        entries.append((order.get(row["provider"], 99),
                        f"{row['provider']:<18}{_model_cell(row):<24}{success:>13}"
                        f"{row['fallback_rate']:>8.1%}{row['invalid_actions']:>9}"
                        f"{row['parse_failures']:>7}{row['latency_p50_ms']:>9.1f}"
                        f"{row['tokens_per_task']:>10.0f}"))
    for result in failed:
        entries.append((order.get(result.provider, 99),
                        f"{result.provider:<18}{'run failed':<24}{BLANK}"))
    for backend in skipped:
        entries.append((order.get(backend.provider, 99),
                        f"{backend.provider:<18}{'not run':<24}{BLANK}"))

    for _, line in sorted(entries, key=lambda pair: pair[0]):
        print(line)
    print("-" * len(HEADER))
    print("success   cases whose actions met the expectations stated in test_cases.json")
    print("fb rate   fallbacks fired / model calls. Every model call in this agent has")
    print("          exactly one deterministic guard behind it, so this is the share of")
    print("          model outputs a deterministic layer had to catch")
    print("invalid   actions refused because they were not on the actuator list")
    print("parse     model responses that would not parse as JSON, both call sites")
    print("p50 ms    median wall clock per case, end to end, on this machine")
    print("tok/case  mean estimated tokens per case, four characters per token")


def print_run_detail(rows: list[dict], failed: list[RunResult]) -> None:
    print()
    print("HOW EACH ROW WAS PRODUCED")
    for row in sorted(rows, key=lambda r: PROVIDER_ORDER.index(r["provider"])):
        print(f"  {row['provider']}")
        print(f"    {row['command']}")
        if row["banner"]:
            print(f"    shim said: {row['banner']}")
        if row["models"]:
            tiers = "  ".join(f"{tier}={model}" for tier, model in row["models"].items())
            print(f"    tiers this run asked for, resolved through shared/providers.yaml:")
            print(f"      {tiers}")
        elif row["provider"] != "replay":
            print("    model names unavailable: pyyaml is not installed here, so this "
                  "script could not read shared/providers.yaml")
        kind = ("replayed, not measured today" if row["scripted"]
                else "measured: a model was called for this row")
        print(f"    {row['fallbacks']} fallback(s) over {row['llm_calls']} model call(s); "
              f"{row['wall_s']:.1f}s wall clock; {kind}")
        for layer, count in sorted(row["fallbacks_by_layer"].items()):
            print(f"      {layer:<28}{count}")
        for note in row["notes"]:
            print(f"    NOTE: {note}")

    for result in failed:
        print(f"  {result.provider}")
        print(f"    {result.command}")
        if result.banner:
            print(f"    shim said: {result.banner}")
        print(f"    FAILED after {result.wall_s:.1f}s: {result.error}")
        print("    No numbers are reported for this backend. A row that did not finish")
        print("    has nothing to say about the model behind it. Run the command above")
        print("    directly to see the whole error.")


def print_skipped(skipped: list[Backend]) -> None:
    if not skipped:
        return
    print()
    print("BACKENDS NOT RUN")
    for backend in sorted(skipped, key=lambda b: PROVIDER_ORDER.index(b.provider)):
        print(f"  {backend.provider:<20}{backend.detail}")
        if backend.enable:
            print(f"  {'':<20}to add this row: {backend.enable}")


def print_caveats(rows: list[dict], backends: list[Backend],
                  failed: list[RunResult]) -> None:
    measured = [row for row in rows if not row["scripted"]]
    replayed = [row for row in rows if row["scripted"]]

    print()
    print("=" * len(HEADER))
    print("WHAT THIS TABLE IS")
    print(f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')} local time, "
          f"python {platform.python_version()}, {platform.system()} {platform.machine()}.")
    print(f"{len(rows)} of {len(backends)} backend(s) produced numbers.")
    if replayed:
        print()
        print("Replayed, not measured today: "
              + ", ".join(row["provider"] for row in replayed) + ".")
        print("Every action in those rows came out of shared/transcripts/ -- what a named")
        print("model actually returned to that exact prompt on a named date, not a string")
        print("written to make a point. But no model was called just now, so the latency is")
        print("this harness reading files, and the success rate is that recorded run's")
        print("rather than a fresh one. They are here so the table exists with no setup,")
        print("and so the comparison path is exercised by every reader rather than only by")
        print("the ones holding an API key.")
    if measured:
        print()
        print("Measured: " + ", ".join(row["provider"] for row in measured) + ".")
        print("A model chose every action in those rows. That makes them observations of")
        print("one model, at one version, on one day, on this machine and this network,")
        print("over one run of twenty cases. Model versions change under a fixed name and")
        print("results move with them. One run is not a benchmark, and the numbers here")
        print("are not published figures -- they are whatever happened when this ran.")
    if not measured and failed:
        print()
        print(f"No measured row: {len(failed)} configured backend(s) failed mid-run and")
        print("are reported above with the error rather than with numbers. Fix the")
        print("backend, or run the command printed under it by hand, and try again.")
    elif not measured:
        print()
        print("Nothing here compares two models yet: no real backend was configured. The")
        print("table has one replayed row and the rest say why they did not run. Configure")
        print("a backend from the list above and run this again to get a comparison.")
    print()
    print("Fallback counts are not simulated. They are counted from the lines the agent")
    print("itself prints when a guard fires, in code written long before this script.")
    print("Token counts are estimated at four characters per token; no tokenizer is")
    print("installed. Latency is wall clock here, including this machine and, in a real")
    print("row, the network between it and the provider.")
    print("=" * len(HEADER))


# -- entry point ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", nargs="+", metavar="NAME", choices=PROVIDER_ORDER,
                        help="only consider these backends (default: all five)")
    parser.add_argument("--case", nargs="+", metavar="ID",
                        help="run only these case ids, which keeps a real run cheap")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="seconds to wait for one backend before giving up "
                             "(default 1800)")
    parser.add_argument("--json", metavar="PATH",
                        help="write the rows and the backend detection to a JSON file")
    args = parser.parse_args()

    backends = detect_backends()
    if args.provider:
        wanted = set(args.provider)
        backends = [backend for backend in backends if backend.provider in wanted]

    print(f"Cross-model comparison: {EVAL_REL}")
    print("Same suite, same agent, same expectations. Only the backend changes.")
    print()
    for backend in sorted(backends, key=lambda b: PROVIDER_ORDER.index(b.provider)):
        mark = "will run " if backend.available else "not run  "
        print(f"  {mark} {backend.provider:<20}{backend.detail}")
    print()

    rows: list[dict] = []
    failed: list[RunResult] = []
    for backend in sorted(backends, key=lambda b: PROVIDER_ORDER.index(b.provider)):
        if not backend.available:
            continue
        print(f"running {backend.provider} ...", flush=True)
        result = run_backend(backend.provider, args.case, args.timeout)
        if result.ok and (result.payload or {}).get("metrics", {}).get("cases"):
            rows.append(summarize(result))
        else:
            # A run that finished but scored nothing is not a row of zeroes.
            if result.ok:
                result.ok = False
                result.error = "the run finished but scored no cases"
            failed.append(result)
    print()

    skipped = [backend for backend in backends if not backend.available]
    print_table(rows, skipped, failed)
    print_run_detail(rows, failed)
    print_skipped(skipped)
    print_caveats(rows, backends, failed)

    print()
    print("Each row above is one of these, run for you:")
    print(f"  LLM_PROVIDER=<backend> python {EVAL_REL} --json out.json")
    print(f"  PowerShell: $env:LLM_PROVIDER='ollama'; python {EVAL_REL}")
    print("Which layer caught what, case by case: python "
          "09-model-portability/fallback_report.py")

    if args.json:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "suite": EVAL_REL,
            "cases_filter": args.case,
            "backends": [
                {"provider": b.provider, "available": b.available, "detail": b.detail,
                 "enable": b.enable}
                for b in backends
            ],
            "rows": rows,
            "failed": [{"provider": r.provider, "error": r.error, "command": r.command}
                       for r in failed],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
