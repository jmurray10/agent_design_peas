"""The same floor, one layer up: when is it worth fanning agent calls out?

`benchmark_floor.py` measures the crossover for arithmetic on this CPU. This script
measures the same shape for agent calls, where the unit of work is an LLM round trip
and the fixed cost is orchestration rather than process startup.

Threads, not processes. The work here is I/O-bound by construction -- an agent call is
almost entirely time spent waiting on a network round trip and a remote forward pass.
The GIL is released across that wait, so threads give real concurrency and cost about
three orders of magnitude less to start than processes. Reaching for
`multiprocessing` here would buy nothing and pay for a great deal.

Runs offline. `shared/llm.py` in mock mode returns instantly, which would make every
row take zero time and prove nothing, so this script injects the latency itself with a
`sleep` around each call. The delay lives here, not in the shim -- the shim is shared
and must stay honest about what it is.

Read the output's "simulated vs measured" block before quoting any number from it. The
per-call latency is an assumption you can change from the command line. What survives
changing it is the structure: a fixed orchestration cost amortized over a per-task
saving always has a floor, and below that floor the sequential loop is the faster
program.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.llm import llm_call
from shared.model_json import loads as model_loads

# A fan-out is only ever as wide as your rate limit and your context budget allow. The
# source page makes the same point about GPU block size: the sweet spot is matched to
# throughput capacity, not to the maximum the hardware will accept. This cap is the
# deterministic guard on whatever the planner proposes.
MAX_CONCURRENCY = 8

# Stand-in for one agent round trip: network out, queue, forward pass, network back.
# Deliberately small so the script finishes quickly. Real calls are far slower; pass
# --latency to use a number you have actually measured against your own endpoint.
DEFAULT_LATENCY_S = 0.05

# An orchestrator's planning call is not the same size as a worker call -- longer
# prompt, longer output, usually a more capable tier. This multiplier is an assumption,
# printed in the output, and it is what puts a floor under the concurrent path.
PLAN_COST_MULTIPLIER = 2.5

TASK_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)

LABELS: tuple[str, ...] = ("billing", "outage", "feature-request", "account-access", "other")

TICKETS: tuple[str, ...] = (
    "Charged twice for the March invoice, need one reversed.",
    "Dashboard has been returning 503 for about twenty minutes.",
    "Can you add CSV export to the reporting tab?",
    "Locked out after the SSO migration, reset link never arrives.",
    "Is there a student discount?",
)


def build_route_prompt(ticket: str) -> str:
    """The worker prompt. Built for real on every call, mock mode or not."""
    return (
        "Route this support ticket to exactly one queue.\n"
        f"Queues: {', '.join(LABELS)}\n"
        "Reply with the queue name and nothing else.\n\n"
        f"Ticket: {ticket}"
    )


def build_plan_prompt(task_count: int) -> str:
    """The orchestrator prompt: how wide should this batch fan out?"""
    return (
        "You are planning a batch of independent support-ticket routing calls.\n"
        f"Batch size: {task_count}\n"
        "The calls share no state and can run in any order.\n"
        'Reply with JSON only: {"concurrency": <int>, "rationale": "<one sentence>"}'
    )


def route_ticket(ticket: str, latency: float) -> str:
    """One agent call: build prompt, wait, call, normalize, validate.

    Returns a queue name from LABELS, or "unrouted" if the model answered with
    something that is not one. The validation is deterministic and stays that way --
    the model picks the label, the code decides whether the label is real.
    """
    prompt = build_route_prompt(ticket)

    # Injected here rather than in shared/llm.py: mock mode answers in microseconds,
    # which would collapse every row of this benchmark to zero and measure nothing.
    time.sleep(latency)

    # tier=small: pick one label from a fixed five-item list. No reasoning, no
    # synthesis, no free-form output -- the entire job is closed-set classification,
    # which is where the cheapest tier is not a compromise but the correct choice.
    raw = llm_call(prompt, mock_key="parallel_worker_route", tier="small")

    label = raw.strip().lower()
    return label if label in LABELS else "unrouted"


def plan_concurrency(task_count: int, latency: float, mock_key: str) -> tuple[int, str]:
    """Ask for a fan-out width, then refuse to trust it.

    Returns `(concurrency, note)` where note records what the deterministic layer had
    to do to the model's answer. `mock_key` is a parameter so the caller can show both
    the healthy path and the malformed-response path without touching the shim.
    """
    prompt = build_plan_prompt(task_count)

    # The planning call is charged the same simulated latency as a worker call, scaled
    # by PLAN_COST_MULTIPLIER. This is the fixed cost that creates the floor.
    time.sleep(latency * PLAN_COST_MULTIPLIER)

    # tier=mid: structured JSON with a bounded integer, consumed by code that clamps
    # it and can fall back without the model. Not small -- small tiers are unreliable
    # at emitting parseable JSON under a schema. Not frontier -- there is no ambiguity
    # to resolve here, just a number inside a range, and paying frontier prices for a
    # value the code is going to clamp anyway is how orchestration budgets disappear.
    raw = llm_call(prompt, mock_key=mock_key, tier="mid")

    try:
        proposed = int(model_loads(raw)["concurrency"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # The model answered in prose, or omitted the field, or gave a non-integer.
        # Fall back to a width the code can defend on its own.
        fallback = min(MAX_CONCURRENCY, task_count)
        return fallback, f"unparseable response, fell back to {fallback}"

    concurrency = max(1, min(proposed, MAX_CONCURRENCY, task_count))
    if concurrency != proposed:
        return concurrency, f"model proposed {proposed}, clamped to {concurrency}"
    return concurrency, f"model proposed {proposed}, accepted"


def run_sequential(tickets: list[str], latency: float) -> list[str]:
    """One call after another. No planner, no pool, no coordination."""
    return [route_ticket(ticket, latency) for ticket in tickets]


def run_concurrent(tickets: list[str], latency: float) -> tuple[list[str], int, str]:
    """Plan the fan-out, then execute it. The planning call is part of the cost."""
    concurrency, note = plan_concurrency(len(tickets), latency, "parallel_fanout_plan")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda t: route_ticket(t, latency), tickets))
    return results, concurrency, note


def tickets_for(count: int) -> list[str]:
    """Cycle the sample tickets up to `count`. Content does not affect timing."""
    return [TICKETS[i % len(TICKETS)] for i in range(count)]


def timed(fn):
    """Run `fn`, return `(result, seconds)`."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def main(latency: float) -> None:
    print("Agent parallelization floor -- sequential vs concurrent agent calls")
    print()

    # Warm-up before any timing: this resolves the provider, prints the mode banner
    # once, and keeps both out of the measured region. It also means the banner is
    # never emitted from inside a worker thread.
    route_ticket(TICKETS[0], 0.0)

    print()
    print(f"  platform            {sys.platform}")
    print(f"  logical CPUs        {os.cpu_count()}")
    print(f"  concurrency model   threads (agent calls are I/O-bound by construction)")
    print(f"  concurrency cap     {MAX_CONCURRENCY}")
    print(f"  simulated latency   {latency * 1000:.0f} ms per worker call")
    print(f"  planning call cost  {PLAN_COST_MULTIPLIER}x a worker call "
          f"({latency * PLAN_COST_MULTIPLIER * 1000:.0f} ms)")
    print()

    print("Control-flow check on the planner, before timing anything:")
    healthy, healthy_note = plan_concurrency(16, 0.0, "parallel_fanout_plan")
    print(f"  well-formed JSON    -> concurrency={healthy}  ({healthy_note})")
    degraded, degraded_note = plan_concurrency(16, 0.0, "parallel_fanout_plan_degraded")
    print(f"  prose, not JSON     -> concurrency={degraded}  ({degraded_note})")
    print("  Both paths return a usable width. The model never sets the fan-out on its")
    print("  own -- MAX_CONCURRENCY and the batch size bound it either way.")
    print()

    header = (f"{'tasks':>7}  {'sequential':>12}  {'concurrent':>12}  {'speedup':>9}  "
              f"{'width':>5}  verdict")
    print(header)
    print("-" * len(header))

    crossover_from: int | None = None
    crossover_to: int | None = None
    last_losing_count: int | None = None
    every_count_lost = True
    total_disagreements = 0
    total_labels = 0

    for count in TASK_COUNTS:
        tickets = tickets_for(count)

        sequential_labels, sequential_s = timed(lambda: run_sequential(tickets, latency))
        (concurrent_labels, width, _note), concurrent_s = timed(
            lambda: run_concurrent(tickets, latency)
        )

        # A speed comparison between two functions that disagree is worth less, so the
        # disagreement is counted and reported rather than assumed away.
        #
        # This was an assert. It held on every offline run and could not do otherwise:
        # both paths replay one recording per prompt, so both read the same answer by
        # construction. Live it fires, because the two paths are two separate calls and a
        # model is free to answer the same ticket differently twice -- which is a fact
        # about models, not a defect in either path, and crashing on it tells the reader
        # the parallel path is broken when it is not.
        disagreements = sum(
            1 for seq, con in zip(sequential_labels, concurrent_labels) if seq != con
        )
        total_disagreements += disagreements
        total_labels += len(sequential_labels)

        speedup = sequential_s / concurrent_s
        won = speedup > 1.0
        if won:
            every_count_lost = False
            if crossover_to is None and last_losing_count is not None:
                crossover_from, crossover_to = last_losing_count, count
        else:
            last_losing_count = count

        print(f"{count:>7}  {sequential_s * 1000:>9.1f} ms  {concurrent_s * 1000:>9.1f} ms  "
              f"{speedup:>8.2f}x  {width:>5}  "
              f"{'concurrent wins' if won else 'concurrent LOSES'}")

    print()
    if total_disagreements:
        print(f"The two paths disagreed on {total_disagreements} of {total_labels} routing")
        print("decisions. Both ran the same validation over the same tickets, so the")
        print("disagreement is the model answering the same question differently on two")
        print("separate calls, not one path routing worse than the other. Read the speedup")
        print("column as a comparison of two paths doing equivalent work, not identical work.")
    else:
        print(f"The two paths agreed on all {total_labels} routing decisions, so the speedup")
        print("column compares two paths that did the same work.")
    print()
    if crossover_to is not None:
        print(f"Crossover at these settings: between {crossover_from} and "
              f"{crossover_to} tasks.")
        print("Below it, the planning call and the fan-out cost more than the sequential")
        print("loop they were meant to replace.")
    elif every_count_lost:
        print(f"Concurrent lost at every batch size tested, up to {TASK_COUNTS[-1]} tasks. "
              f"At this latency the fixed orchestration cost never amortizes.")
    else:
        print("Concurrent won at every batch size tested, including a single task. "
              "Lower the planning cost multiplier or the latency and the floor reappears.")

    print()
    print("Simulated vs measured, because the difference matters:")
    print(f"  simulated  the {latency * 1000:.0f} ms per-call latency. It is an assumption")
    print("             set by --latency, not a measurement of any provider's API.")
    print(f"  simulated  the {PLAN_COST_MULTIPLIER}x planning-call cost, on the assumption")
    print("             that a fan-out prompt is longer than a routing prompt.")
    print("  measured   every millisecond in the table: real wall clock, real thread")
    print("             creation, real queue handoff, real result collection, on this")
    print("             machine, in this run.")
    print()
    print("  The floor is set by the RATIO of fixed orchestration cost to per-task cost,")
    print("  not by the latency itself. Scale both and the crossover barely moves -- try")
    print("  --latency 0.5 and compare. Drop it far enough (--latency 0.001) and real")
    print("  thread overhead starts to dominate, which pushes the floor UP, not down.")
    print("  Measure your own ratio rather than borrowing this one.")
    print()
    print("  No GPU was involved in anything this script ran. See README.md for the")
    print("  published GPU figures, which are not these.")


def parse_latency(argv: list[str]) -> float:
    """Read --latency from argv. Kept trivial on purpose -- argparse for one flag is
    more machinery than a demonstration script should carry."""
    if "--latency" in argv:
        return float(argv[argv.index("--latency") + 1])
    return DEFAULT_LATENCY_S


if __name__ == "__main__":
    main(parse_latency(sys.argv[1:]))
