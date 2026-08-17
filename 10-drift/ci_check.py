"""Fail the build when the twenty cases stop behaving the way the committed baseline says.

    python 10-drift/ci_check.py

This is the whole of what `.github/workflows/drift.yml` runs on every push. It takes a
fresh snapshot, diffs the aggregate against `baselines/baseline.json`, and exits nonzero
on a breach. It needs no secrets, because with no backend configured every model call is
replayed from `shared/transcripts/`.

Three groups of checks, kept apart for the same reason `replay.py` keeps its two sections
apart -- a single combined number would go up for a broken parser and sit still for an
agent that has started escalating everything:

    CONFIGURATION   the run must be the same experiment the baseline recorded. A changed
                    system prompt shows up here as a changed sha1, whether or not anything
                    downstream of it moved.

    STRUCTURAL      output that stopped conforming. Any increase fails. These are the
                    failures the deterministic layer already catches at runtime; CI
                    catching them again is cheap and says nothing new.

    BEHAVIORAL      output that conforms while the agent changes. Nothing in the codebase
                    reports these at runtime. The thresholds below are the entire guard,
                    which is why they are three named constants with reasons attached
                    rather than a tolerance buried in a comparison.

Cost is printed and never fails anything. Latency is not compared at all: it is a property
of the machine that ran the job, and thresholding on it would produce a red build every
time a runner was busy.

WHERE THE RESPONSES COME FROM. Every model call goes through `shared/llm.py`: live when a
backend is configured, replayed from `shared/transcripts/` when none is. A replay is what
a real model returned to that exact prompt on a recorded date, which is what lets this job
run offline without pretending offline is the same as fresh.

Transcript entries are keyed by the prompt, so editing a system prompt in place no longer
slips past. The edited prompt misses its recording and the run raises at the first model
call, before the configuration check gets as far as noticing the changed sha1. Two
instruments see that edit now; under canned responses exactly one could.

Exit codes:  0 every check passed
             1 at least one check failed
             2 the check could not be run at all (wrong backend, mismatched baseline)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drift_harness as dh  # noqa: E402
from snapshot import add_perturbation_flags, config_from_args  # noqa: E402

# Twenty cases, so one ticket changing its mind about escalating moves the rate by five
# points. Ten points is two cases: tight enough that a system prompt telling the agent to
# escalate whenever it is unsure cannot pass unnoticed, loose enough that one borderline
# ticket decided differently by a live provider does not redden the build every morning.
ESCALATION_RATE_BAND = 0.10

# Only a fall fails. A rise is not a regression, and a check that reddens on improvement
# teaches people to ignore it. A rise still gets printed, because a baseline the agent has
# outgrown is a stale baseline.
SUCCESS_RATE_DROP = 0.10

# Thirty-six actions across the suite, so ten share points is about three and a half
# actions. This is the row that catches the perturbations the headline rates miss: a
# distribution can slide visibly -- one action displacing another -- while the escalation
# rate and the success rate both sit still.
ACTION_SHARE_BAND = 0.10

STRUCTURAL_CHECKS = [
    ("invalid action names", "invalid_action_names"),
    ("JSON parse failures", "json_parse_failures"),
    ("schema violations", "schema_violations"),
]

CONFIG_FIELDS = [
    ("system_prompt", "system prompt"),
    ("system_prompt_sha1", "system prompt sha1"),
    ("tier", "tier"),
    ("provider", "provider"),
    ("noisy_input", "noisy input"),
    ("degraded_tools", "degraded tools"),
]


@dataclass
class Row:
    """One line of the report. `status` is "ok", "FAIL", or blank for a plain observation."""

    status: str
    label: str
    detail: str

    @property
    def is_check(self) -> bool:
        return self.status in ("ok", "FAIL")

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


def _fmt(key: str, value: object) -> str:
    """Full value, except the sha1, where the first twelve characters are the readable part
    and the other twenty-eight only make the line hard to scan."""
    text = str(value)
    return text[:12] + "..." if key.endswith("sha1") and len(text) > 15 else text


def config_rows(base: dict, new: dict) -> list[Row]:
    """The run has to be the same experiment before its numbers mean anything.

    A configuration change is not drift -- somebody typed it. CI cannot tell an intended
    change from an accidental one, so it fails and asks, which is the correct behaviour for
    a machine that does not know why the file changed.
    """
    rows = []
    for key, label in CONFIG_FIELDS:
        want, got = base["config"].get(key), new["config"].get(key)
        if want == got:
            rows.append(Row("ok", label, _fmt(key, want)))
        else:
            rows.append(Row("FAIL", label, f"{_fmt(key, want)} -> {_fmt(key, got)}"))
    return rows


def structural_rows(base: dict, new: dict) -> list[Row]:
    """Any increase fails. These already announce themselves at runtime; a rise here means
    a fallback that used to fire rarely is now load-bearing."""
    before, after = base["aggregate"]["structural"], new["aggregate"]["structural"]
    rows = []
    for label, key in STRUCTURAL_CHECKS:
        b, a = before[key], after[key]
        status = "FAIL" if a > b else "ok"
        rows.append(Row(status, label,
                        f"{b:>3} -> {a:<3} {a - b:+d}   limit: no increase"))
    rb, ra = before["recoverable_parse_failures"], after["recoverable_parse_failures"]
    # Not a check: a subset of the row above, and failing it too would count the same
    # response twice. Printed because a rise means the deterministic layer is throwing away
    # answers a tolerant parser would have accepted.
    rows.append(Row("", "  of those, recoverable",
                    f"{rb:>3} -> {ra:<3} {ra - rb:+d}   reported, not a check"))
    return rows


def behavioral_rows(base: dict, new: dict) -> list[Row]:
    """The thresholds. Nothing else in this repository looks at any of these at runtime."""
    b, a = base["aggregate"], new["aggregate"]
    rows = []

    eb, ea = b["escalation_rate"], a["escalation_rate"]
    rows.append(Row("FAIL" if abs(ea - eb) > ESCALATION_RATE_BAND + 1e-9 else "ok",
                    "escalation rate",
                    f"{eb:>6.1%} -> {ea:<6.1%} {(ea - eb) * 100:+5.1f} pts   "
                    f"limit: +/-{ESCALATION_RATE_BAND * 100:.0f} pts"))

    sb, sa = b["task_success_rate"], a["task_success_rate"]
    rows.append(Row("FAIL" if sb - sa > SUCCESS_RATE_DROP + 1e-9 else "ok",
                    "task success rate",
                    f"{sb:>6.2f} -> {sa:<6.2f} {sa - sb:+5.2f}       "
                    f"limit: -{SUCCESS_RATE_DROP:.2f} (a rise never fails)"))
    if sa > sb + 1e-9:
        rows.append(Row("", "  success rate rose",
                        "not a failure, but the baseline no longer describes this agent"))

    for action in sorted(set(b["action_frequency"]) | set(a["action_frequency"])):
        cb = b["action_frequency"].get(action, 0)
        ca = a["action_frequency"].get(action, 0)
        pb = b["action_share"].get(action, 0.0)
        pa = a["action_share"].get(action, 0.0)
        rows.append(Row("FAIL" if abs(pa - pb) > ACTION_SHARE_BAND + 1e-9 else "ok",
                        f"share {action}",
                        f"{cb:>3} ({pb:>5.1%}) -> {ca:>3} ({pa:>5.1%}) "
                        f"{(pa - pb) * 100:+5.1f} pts"))
    return rows


def print_cost(base: dict, new: dict) -> None:
    b, a = base["aggregate"]["cost"], new["aggregate"]["cost"]
    print("COST                 (printed, never a failure -- cost is not correctness)")
    print(f"        {'estimated tokens/case':<28}{b['tokens_per_case']:>7.0f} -> "
          f"{a['tokens_per_case']:<7.0f}{a['tokens_per_case'] - b['tokens_per_case']:+.0f}")
    print(f"        {'model calls':<28}{b['llm_calls_total']:>7} -> "
          f"{a['llm_calls_total']:<7}{a['llm_calls_total'] - b['llm_calls_total']:+d}")
    print(f"        {'latency':<28}not compared: it is a property of the machine that "
          f"ran the job")


def print_section(title: str, note: str, rows: list[Row]) -> None:
    print(f"{title:<21}({note})")
    for row in rows:
        print(f"  {row.status:<6}{row.label:<26}{row.detail}")


def print_verdict(failures: list[Row], checks: int, replaying: bool) -> None:
    print(dh.BAR)
    if not failures:
        print(f"PASS   {checks} checks, 0 failures")
        print(
            "The distribution the baseline recorded is the distribution this run\n"
            "produced. On a replay that stays true until something in the repository\n"
            "changes, which is what this job is: a regression test on the agent, its\n"
            "prompt, its recordings and its eval suite. It is not monitoring -- nothing\n"
            "here watches a production model, and a replay is one recorded run on one\n"
            "date, not today's answer."
        )
        print(dh.BAR)
        return

    print(f"FAIL   {checks} checks, {len(failures)} "
          f"failure{'' if len(failures) == 1 else 's'}")
    for row in failures:
        print(f"         {row.label:<26}{row.detail}")
    print()
    config_only = all(row.label in dict(CONFIG_FIELDS).values() for row in failures)
    if config_only and replaying:
        print(
            "Only the configuration moved, and the distribution below it did not. That\n"
            "combination is worth a second look on a replay: this run answered from\n"
            "recordings, so the distribution can only be the one that was recorded under\n"
            "the configuration named in the baseline. What the configuration change means\n"
            "for behaviour is not in this output. Record the new condition and diff it:\n"
            "  ANTHROPIC_API_KEY=... LLM_RECORD=1 python 10-drift/replay.py "
            "--baseline baseline \\\n"
            "      --prompt <the prompt you changed to>\n"
        )
    print(
        "Either the change was intended, in which case the committed baseline is now out\n"
        "of date, or it was not, in which case nothing else was going to tell you. CI\n"
        "cannot tell those two apart and should not guess. Retake it deliberately:\n"
        "  python 10-drift/snapshot.py --label baseline\n"
        "and commit the diff, so the change is a line in the history with a name on it."
    )
    print(dh.BAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default="baseline",
                        help="label of the committed snapshot to check against")
    parser.add_argument("--save", metavar="LABEL",
                        help="also write this run to baselines/<LABEL>.json")
    parser.add_argument("--allow-real", action="store_true",
                        help="permit a live backend; the thresholds were set against a "
                             "deterministic replay and a live run will move them")
    add_perturbation_flags(parser)
    args = parser.parse_args()

    try:
        base = dh.read_snapshot(args.baseline)
    except SystemExit as missing:
        # Re-raised as 2 rather than 1: no baseline is "the check did not run", which is a
        # different thing from "the check ran and the agent moved".
        print(missing, file=sys.stderr)
        raise SystemExit(2) from None

    try:
        config = config_from_args(args)
    except SystemExit as unusable:   # a --prompt path that is not there
        print(unusable, file=sys.stderr)
        raise SystemExit(2) from None

    replaying = dh.backend_is_replay()

    if not replaying and not args.allow_real:
        # Refusing rather than warning: a CI job that silently starts billing an API is a
        # worse outcome than a red build, and twenty cases against a live provider is not
        # what anybody pushing a commit asked for.
        print(
            "This check replays recordings on purpose: free, offline and deterministic.\n"
            "A live backend is selected right now. Unset LLM_PROVIDER and any provider\n"
            "key so the run replays shared/transcripts/, or pass --allow-real if you\n"
            "meant to spend the calls.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"Baseline {args.baseline!r}  ({base['created_utc']}, {base['mode']} mode, "
          f"{base['config']['system_prompt']})")
    print(f"This run             {config.describe()}")
    print()

    new = dh.run_labeled(config, args.save or "ci")
    dh.print_provenance(new, base)
    print()

    if base["mode"] != new["mode"]:
        print(
            f"Baseline was taken in {base['mode']} mode and this run is {new['mode']} "
            f"mode.\nThose are not comparable: a replay returns exactly what was recorded "
            f"every time, and a\nlive provider does not. Retake the baseline in this mode "
            f"before checking against it.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    sections = [
        ("CONFIGURATION", "must match the baseline, or the diff is not a diff",
         config_rows(base, new)),
        ("STRUCTURAL", "any increase fails; the deterministic layer sees these too",
         structural_rows(base, new)),
        ("BEHAVIORAL", "nothing else sees these; the thresholds are the whole guard",
         behavioral_rows(base, new)),
    ]
    rows = [row for _, _, section_rows in sections for row in section_rows]
    for title, note, section_rows in sections:
        print_section(title, note, section_rows)
        print()
    print_cost(base, new)
    print()

    if args.save:
        path = dh.write_snapshot(new)
        print(f"Wrote {path.relative_to(dh.ROOT).as_posix()}")
        print()

    failures = [row for row in rows if row.failed]
    print_verdict(failures, sum(1 for row in rows if row.is_check), replaying)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
