"""Run the same twenty cases under one perturbation and diff against a named baseline.

    python 10-drift/replay.py --baseline baseline --prompt 10-drift/prompts/system_v2.md
    python 10-drift/replay.py --baseline baseline --tier small
    python 10-drift/replay.py --baseline baseline --provider ollama
    python 10-drift/replay.py --baseline baseline --noisy-input
    python 10-drift/replay.py --baseline baseline --degraded-tools

The output has two sections and it will never have one.

    STRUCTURAL DRIFT    the output stopped conforming. Deterministic code caught it,
                        counted it, and named the layer that fired.

    BEHAVIORAL DRIFT    the output conformed and the agent changed. Nothing caught it,
                        because there is nothing in a schema check that could.

Adding them together would produce a number that goes up when a model starts emitting
broken JSON and stays flat when it starts escalating every ticket, and a reader would
have no way to tell which had happened. The first one wakes somebody up at 3am. The
second one is the one that runs for three weeks.

Runs with no API key once the condition has been recorded. Every perturbation changes the
real input to a real model, and transcript entries are keyed by the prompt, so a perturbed
run needs its own recording and will say so if it does not have one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drift_harness as dh  # noqa: E402
from snapshot import add_perturbation_flags, config_from_args  # noqa: E402

STRUCTURAL_ROWS = [
    ("invalid action names", "invalid_action_names",
     "agent refused it: not on the actuator list"),
    ("JSON parse failures", "json_parse_failures",
     "agent fell back: its own bare json.loads raised"),
    ("schema violations", "schema_violations",
     "harness: parsed clean, required fields missing"),
]


def _delta(before: float, after: float) -> str:
    """A signed change, or a dash. Zero is printed as a dash so movement is findable."""
    diff = after - before
    if abs(diff) < 1e-9:
        return "  --"
    return f"{diff:+g}"


def print_structural(base: dict, new: dict) -> bool:
    """The section the deterministic layer can speak to. Returns True if anything moved."""
    before, after = base["aggregate"]["structural"], new["aggregate"]["structural"]
    moved = False

    print("STRUCTURAL DRIFT     (caught by the deterministic layer)")
    for label, key, provenance in STRUCTURAL_ROWS:
        b, a = before[key], after[key]
        moved = moved or b != a
        print(f"  {label:<26}{b:>4} -> {a:<4}  {_delta(b, a):>5}   {provenance}")
        if key == "json_parse_failures":
            rb, ra = before["recoverable_parse_failures"], after["recoverable_parse_failures"]
            print(f"    {'of those, recoverable':<24}{rb:>4} -> {ra:<4}  {_delta(rb, ra):>5}   "
                  f"a tolerant parser accepts them; the agent threw them away")

    refusals_b = base["aggregate"]["refusals"]
    refusals_a = new["aggregate"]["refusals"]
    print(f"  {'cases ending in refusal':<26}{refusals_b:>4} -> {refusals_a:<4}  "
          f"{_delta(refusals_b, refusals_a):>5}   no_op, the sentinel for a blocked action")

    print()
    if moved:
        print("  Everything above was visible without an eval suite. A parse error, an")
        print("  off-list action name and a missing required field all announce")
        print("  themselves at the point they happen.")
    else:
        print("  Nothing moved. Every response in the perturbed run was exactly as well")
        print("  formed as the baseline's: same parse failures, same refusals, same")
        print("  fields present. The deterministic layer has nothing to report.")
    return moved


def print_behavioral(base: dict, new: dict, structural_moved: bool) -> bool:
    """The section only a performance measure can speak to.

    `structural_moved` is passed in for one reason: `no_op` is in the action distribution,
    and `no_op` is a refused action. When structural drift is also present, part of this
    section is downstream of the section above, and the closing line has to say so rather
    than claim credit for detecting something the allowed-action check already caught.
    """
    b, a = base["aggregate"], new["aggregate"]
    moved = False

    print("BEHAVIORAL DRIFT     (caught only by the eval suite)")

    eb, ea = b["escalation_rate"], a["escalation_rate"]
    moved = moved or abs(eb - ea) > 1e-9
    print(f"  {'escalation rate':<26}{eb:>6.1%} -> {ea:<6.1%}"
          f"{(ea - eb) * 100:+6.1f} pts")

    sb, sa = b["task_success_rate"], a["task_success_rate"]
    moved = moved or abs(sb - sa) > 1e-9
    print(f"  {'task success rate':<26}{sb:>6.2f} -> {sa:<6.2f}{sa - sb:+6.2f}")

    fb, fa = b["fallback_rate"], a["fallback_rate"]
    print(f"  {'fallback rate':<26}{fb:>6.1%} -> {fa:<6.1%}"
          f"{(fa - fb) * 100:+6.1f} pts")

    print("  action distribution")
    actions = sorted(set(b["action_frequency"]) | set(a["action_frequency"]))
    for action in actions:
        cb = b["action_frequency"].get(action, 0)
        ca = a["action_frequency"].get(action, 0)
        pb = b["action_share"].get(action, 0.0)
        pa = a["action_share"].get(action, 0.0)
        moved = moved or cb != ca
        flag = "  <--" if cb != ca else ""
        print(f"    {action:<24}{cb:>3} ({pb:>5.1%}) -> {ca:>3} ({pa:>5.1%}){flag}")

    changed = [case_id for case_id, entry in new["cases"].items()
               if base["cases"].get(case_id, {}).get("actions") != entry["actions"]]
    moved = moved or bool(changed)
    print(f"  {'cases whose sequence moved':<28}{len(changed)} of {a['cases']}"
          + (f"   {', '.join(changed)}" if changed else ""))

    flipped = [(case_id, base["cases"][case_id]["success"], entry["success"])
               for case_id, entry in new["cases"].items()
               if case_id in base["cases"]
               and base["cases"][case_id]["success"] != entry["success"]]
    if flipped:
        print("  verdict flips")
        for case_id, was, now in flipped:
            print(f"    {case_id:<24}{'PASS' if was else 'FAIL'} -> "
                  f"{'PASS' if now else 'FAIL'}")

    refusals_moved = (b["action_frequency"].get("no_op", 0)
                      != a["action_frequency"].get("no_op", 0))
    print()
    if not moved:
        print("  Nothing moved. Same actions, same distribution, same verdicts.")
    elif refusals_moved:
        print("  Part of this is downstream of the section above: every no_op in the")
        print("  distribution is an action the allowed-action check refused, and that")
        print("  much did get reported. The rest of it did not. Nothing announced the")
        print("  escalation rate moving or a verdict flipping on a well-formed response.")
    elif structural_moved:
        print("  No refused actions were added, so nothing in the distribution above is")
        print("  an artifact of the section before it. A parse failure upstream can still")
        print("  change what the agent decides next -- that consequence is behavioral")
        print("  even where its cause was structural, and only this section sees it.")
    else:
        print("  Not one line above was reported by the agent, by a schema, or by a")
        print("  fallback. It took twenty cases with stated expected outcomes to see it.")
    return moved


def print_cost(base: dict, new: dict) -> None:
    """Kept out of both sections on purpose: cost is not correctness in either direction."""
    b, a = base["aggregate"]["cost"], new["aggregate"]["cost"]
    print("COST                 (neither structural nor behavioral)")
    print(f"  {'estimated tokens/case':<26}{b['tokens_per_case']:>7.0f} -> "
          f"{a['tokens_per_case']:<7.0f}{a['tokens_per_case'] - b['tokens_per_case']:+.0f}")
    print(f"  {'model calls':<26}{b['llm_calls_total']:>7} -> "
          f"{a['llm_calls_total']:<7}{_delta(b['llm_calls_total'], a['llm_calls_total']):>5}")
    print(f"  {'latency mean ms':<26}{b['latency_mean_ms']:>7.1f} -> "
          f"{a['latency_mean_ms']:<7.1f}"
          f"{a['latency_mean_ms'] - b['latency_mean_ms']:+.1f}")
    if a["injected_latency_total_ms"] or b["injected_latency_total_ms"]:
        print(f"    {'of which injected, total':<24}{b['injected_latency_total_ms']:>7.1f} -> "
              f"{a['injected_latency_total_ms']:<7.1f}  put there on purpose by "
              f"--degraded-tools,")
        print(f"    {'':<24}{'':>7}    {'':<7}  not measured from any real tool")
    if new["mode"] == "replay":
        print()
        print("  Token counts hold on a replay -- the estimator runs over the text")
        print("  actually sent, so a changed system prompt or a truncated percept really")
        print("  does change the count. Latency does not: replaying a recording is a file")
        print("  lookup and says nothing whatsoever about any provider's response time.")


def print_verdict(structural_moved: bool, behavioral_moved: bool) -> None:
    print(dh.BAR)
    if behavioral_moved and not structural_moved:
        print("BEHAVIORAL DRIFT WITH ZERO STRUCTURAL DRIFT")
        print(
            "This is the case the harness exists for. Every response parsed. Every\n"
            "action was on the actuator list. Every state object carried its required\n"
            "fields. No fallback fired that did not fire in the baseline, no validation\n"
            "refused anything new, and nothing anywhere logged a problem -- and the\n"
            "agent is doing measurably different work.\n\n"
            "A production monitor watching error rates would have shown a flat line\n"
            "through this. The only instrument that moved was the eval suite, because\n"
            "the eval suite is the only thing here holding an opinion about what the\n"
            "right answer was."
        )
    elif structural_moved and behavioral_moved:
        print("BOTH KINDS OF DRIFT")
        print(
            "The structural half is already handled: it was caught, counted, and the\n"
            "layer that caught it is named above. The behavioral half was not caught by\n"
            "anything. Do not let the first section's numbers stand in for the second's\n"
            "-- a fixed parser would clear the top block and leave the bottom one\n"
            "exactly where it is."
        )
    elif structural_moved:
        print("STRUCTURAL DRIFT ONLY")
        print(
            "The output stopped conforming and the deterministic layer said so. Behavior\n"
            "held: same distribution, same verdicts. This is the failure mode that is\n"
            "already solved, and the one that gets all the attention."
        )
    else:
        print("NO DRIFT DETECTED")
        print(
            "Identical on both axes: same actions, same distribution, same verdicts.\n"
            "Check the perturbation line in the provenance block above before reading\n"
            "this as a result -- an unperturbed run against its own baseline is supposed\n"
            "to come out empty, and says nothing about the model's stability."
        )
    print(dh.BAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default="baseline",
                        help="label of the snapshot to diff against (default: baseline)")
    parser.add_argument("--save", metavar="LABEL",
                        help="also write the perturbed run as a snapshot under this label")
    add_perturbation_flags(parser)
    args = parser.parse_args()

    base = dh.read_snapshot(args.baseline)
    config = config_from_args(args)

    print(f"Baseline {args.baseline!r}: {base['config']['system_prompt']}"
          f"{', tier=' + base['config']['tier'] if base['config']['tier'] else ''}"
          f"  ({base['created_utc']}, {base['mode']} mode)")
    print(f"Replay:              {config.describe()}")
    print()

    new = dh.run_labeled(config, args.save or f"replay-{'-'.join(config.slugs()) or 'none'}")

    dh.print_provenance(new, base)
    print()
    structural_moved = print_structural(base, new)
    print()
    print_behavioral(base, new, structural_moved)
    print()
    behavioral_moved = _behavioral_moved(base, new)
    print_cost(base, new)
    print()
    print_verdict(structural_moved, behavioral_moved)

    if args.save:
        path = dh.write_snapshot(new)
        print(f"\nWrote {path.relative_to(dh.ROOT).as_posix()}")


def _behavioral_moved(base: dict, new: dict) -> bool:
    """Recomputed rather than returned from the printer, so the verdict cannot drift
    from what was printed by someone editing one of them."""
    b, a = base["aggregate"], new["aggregate"]
    if abs(b["escalation_rate"] - a["escalation_rate"]) > 1e-9:
        return True
    if abs(b["task_success_rate"] - a["task_success_rate"]) > 1e-9:
        return True
    if b["action_frequency"] != a["action_frequency"]:
        return True
    return any(base["cases"].get(case_id, {}).get("actions") != entry["actions"]
               for case_id, entry in new["cases"].items())


if __name__ == "__main__":
    main()
