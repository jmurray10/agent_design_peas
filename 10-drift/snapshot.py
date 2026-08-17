"""Record what the agent does today, so tomorrow's run has something to be diffed against.

    python 10-drift/snapshot.py --label baseline
    python 10-drift/snapshot.py --label weak-model --tier small

Runs the twenty cases from 08-production-patterns/evaluation and writes
`10-drift/baselines/<label>.json`: per case the action sequence, whether a fallback fired
and which one, tokens, latency and the validation outcome -- plus aggregate distributions.

The distributions are the reason this file exists. Pass and fail move late. An agent that
has started escalating twice as often is still passing most of its cases, and the only
place that shows up early is the shape of the action distribution.

Runs with no API key once the condition has been recorded: every model call is either live
or replayed from `shared/transcripts/`, and a prompt with no recording raises rather than
being answered by a stand-in. Read the provenance block it prints before quoting anything
from it -- a replay is a recording of one run on one date, not a fresh measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drift_harness as dh  # noqa: E402


def print_case_table(snapshot: dict) -> None:
    header = (f"{'id':<5}{'actions':<44}{'valid':<9}"
              f"{'tokens':>8}{'ms':>8}{'fb':>4}  result")
    print(header)
    print("-" * len(header))
    for case_id, entry in snapshot["cases"].items():
        actions = " > ".join(dh.run_eval.SHORT.get(a, a) for a in entry["actions"])
        fallbacks = str(len(entry["fallbacks"])) if entry["fallbacks"] else "-"
        print(f"{case_id:<5}{actions or '(none)':<44}{entry['validation']:<9}"
              f"{entry['tokens']:>8}{entry['latency_ms']:>8.1f}{fallbacks:>4}  "
              f"{'PASS' if entry['success'] else 'FAIL'}")
    print("-" * len(header))


def print_distributions(snapshot: dict) -> None:
    agg = snapshot["aggregate"]
    print()
    print("DISTRIBUTIONS  (the part that moves first)")
    print(f"  cases                     {agg['cases']}")
    print(f"  actions taken             {agg['actions_total']}")
    print("  action distribution")
    for action, count in agg["action_frequency"].items():
        share = agg["action_share"][action]
        print(f"    {action:<24}{count:>4}  {share:>6.1%}")
    print(f"  escalation rate           {agg['escalations']}/{agg['cases']} = "
          f"{agg['escalation_rate']:.1%}")
    print(f"  fallback rate             {agg['cases_with_fallback']}/{agg['cases']} = "
          f"{agg['fallback_rate']:.1%}")
    print(f"  task success rate         {agg['successes']}/{agg['cases']} = "
          f"{agg['task_success_rate']:.1%}")

    print()
    print("STRUCTURAL COUNTS  (deterministic checks; never added to the above)")
    structural = agg["structural"]
    print(f"  invalid action names      {structural['invalid_action_names']}"
          f"        (agent: allowed-action check)")
    print(f"  JSON parse failures       {structural['json_parse_failures']}"
          f"        (agent: its own fallback lines)")
    print(f"    of those, recoverable   {structural['recoverable_parse_failures']}"
          f"        (harness: tolerant parser accepts them)")
    print(f"  schema violations         {structural['schema_violations']}"
          f"        (harness: required fields "
          f"{', '.join(dh.REQUIRED_STATE_FIELDS)})")

    print()
    print("COST  (neither structural nor behavioral -- kept apart from both)")
    cost = agg["cost"]
    print(f"  model calls               {cost['llm_calls_total']}")
    print(f"  estimated tokens          {cost['tokens_total']} total, "
          f"{cost['tokens_per_case']:.0f} per case")
    print(f"  latency                   {cost['latency_mean_ms']:.1f} ms mean, "
          f"{cost['latency_p50_ms']:.1f} ms p50")
    if cost["injected_latency_total_ms"]:
        print(f"    of which injected       {cost['injected_latency_total_ms']:.1f} ms "
              f"by --degraded-tools, on purpose, not measured from any tool")


def add_perturbation_flags(parser: argparse.ArgumentParser) -> None:
    """The five perturbations. Shared with replay.py so both accept the same words."""
    parser.add_argument("--tier", choices=["small", "mid", "frontier"],
                        help="same provider, weaker or stronger model")
    parser.add_argument("--provider", help="different backend entirely, e.g. ollama")
    parser.add_argument("--prompt", metavar="PATH",
                        help=f"system prompt to use instead of "
                             f"{dh.DEFAULT_PROMPT.relative_to(dh.ROOT).as_posix()}")
    parser.add_argument("--noisy-input", action="store_true",
                        help="typos, truncation and format shifts on the percepts")
    parser.add_argument("--degraded-tools", action="store_true",
                        help="tool results with missing fields, injected latency and "
                             "intermittent errors")


def config_from_args(args: argparse.Namespace) -> dh.RunConfig:
    prompt = Path(args.prompt).resolve() if args.prompt else dh.DEFAULT_PROMPT
    if not prompt.exists():
        raise SystemExit(f"No such system prompt: {prompt}")
    return dh.RunConfig(
        system_prompt=prompt,
        tier=args.tier,
        provider=args.provider,
        noisy_input=args.noisy_input,
        degraded_tools=args.degraded_tools,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="baseline",
                        help="name of the snapshot file to write (default: baseline)")
    add_perturbation_flags(parser)
    args = parser.parse_args()

    config = config_from_args(args)
    print(f"Snapshot {args.label!r}: {config.describe()}")
    print()

    snapshot = dh.run_labeled(config, args.label)

    dh.print_provenance(snapshot)
    print()
    print_case_table(snapshot)
    print_distributions(snapshot)

    path = dh.write_snapshot(snapshot)
    print()
    print(f"Wrote {path.relative_to(dh.ROOT).as_posix()}")
    print("Diff a perturbed run against it with:")
    print(f"  python 10-drift/replay.py --baseline {args.label} "
          f"--prompt 10-drift/prompts/system_v2.md")


if __name__ == "__main__":
    main()
