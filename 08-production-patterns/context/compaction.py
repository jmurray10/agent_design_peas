"""Compaction: summarize what is old, keep what is recent.

The source page shows `compact_context` with a deterministic budget check wrapped
around one LLM call. This file is that function, plus the two things the page left
out: a token estimator that actually counts something, and a driver that grows a
history until compaction fires so a reader can watch it happen.

Run it from the repository root:

    python 08-production-patterns/context/compaction.py

The architecture here is unchanged from the classical one. A ring buffer with a
fixed retention policy has the same shape: check occupancy, keep the tail, throw
away the head. The only component the LLM replaces is "throw away the head" --
instead of discarding the old entries it condenses them, so the agent keeps a
lossy trace of what it already did instead of no trace at all.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.llm import llm_call  # noqa: E402


# -- the token estimator ------------------------------------------------------------
#
# Read this before you read anything else in this file, because every number the
# script prints comes out of it.

CHARS_PER_TOKEN = 4.0


def estimate_tokens_in_text(text: str) -> int:
    """Approximate how many tokens `text` would cost. This is an estimate.

    It divides the character count by four, the usual rule of thumb for English
    prose. It is not a tokenizer. It does not know about byte-pair merges, and it
    will be wrong -- usually low -- on code, JSON, URLs, non-English text, and long
    identifiers, all of which pack fewer characters into a token than prose does.

    A real count comes from the model's own tokenizer or from the API's token
    counting endpoint. Both need a dependency or a network call, and this repo
    promises neither, so the honest move is to approximate and say so rather than
    to print a number that looks authoritative and is not.

    Every token figure this script prints is an estimate produced here.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_tokens(history: list[dict[str, str]]) -> int:
    """Approximate the token cost of a whole history.

    Serializing first is deliberate: role labels, braces, and quoting are real
    characters that a real request pays for, so counting them is closer to the
    truth than summing the content fields alone. It still ignores the few tokens
    of per-message framing that the API itself adds.
    """
    return estimate_tokens_in_text(json.dumps(history))


# -- compaction ---------------------------------------------------------------------

RECENT_WINDOW = 5

# Which canned summary to serve on each successive compaction. Mock-mode plumbing
# only -- real backends ignore mock_key entirely. The second one is deliberately
# degenerate so the validation path below is exercised on a real run rather than
# merely existing.
_MOCK_KEYS = (
    "context_compaction_summary_1",
    "context_compaction_summary_2",
    "context_compaction_summary_3",
)
_compactions = 0


def compact_context(history: list[dict[str, str]], max_tokens: int = 50000) -> list[dict[str, str]]:
    """Compact `history` if it is over budget, otherwise return it untouched.

    Deterministic check: are we over budget?
    LLM call: summarize older entries.
    Keep recent entries intact.
    """
    global _compactions

    if estimate_tokens(history) < max_tokens:
        return history  # no compaction needed

    if len(history) <= RECENT_WINDOW:
        # Not in the source page, and needed. With fewer entries than the recent
        # window, `older` is empty: the function would spend a call summarizing
        # nothing and then append that summary, growing the context it was asked to
        # shrink. Being over budget with nothing old to drop means the individual
        # entries are too large, which is a tool-design problem, not a compaction one.
        return history

    recent = history[-RECENT_WINDOW:]   # always keep last 5 entries
    older = history[:-RECENT_WINDOW]

    _compactions += 1
    summary = llm_call(
        f"Summarize these results concisely:\n{older}",
        mock_key=_MOCK_KEYS[min(_compactions - 1, len(_MOCK_KEYS) - 1)],
        # mid: condensing many tool results into a paragraph that a later turn can
        # still act on is judgment, not lookup -- it has to decide which of a dozen
        # findings still matter. But the output is one bounded paragraph with a
        # deterministic fallback right behind it, so frontier capability buys
        # nothing here that the budget check does not already guarantee.
        tier="mid",
    )

    summary = summary.strip()
    # A model that returns a bare label, an empty completion, or something longer
    # than the text it was asked to replace has not summarized anything. Compaction
    # is supposed to shrink the context, so the caller checks that it did.
    if summary.lower().startswith("summary:"):
        summary = summary[len("summary:"):].strip()
    if not summary or estimate_tokens_in_text(summary) >= estimate_tokens(older):
        print("          [fallback] summary came back empty or no shorter than what "
              "it replaced;\n          using a deterministic manifest instead")
        summary = _manifest(older)

    return [{"role": "summary", "content": summary}] + recent


def _manifest(older: list[dict[str, str]]) -> str:
    """Deterministic replacement for a summary that failed validation.

    Retrying the model is the tempting move and the wrong one: the context is
    already over budget, so the failure mode of a retry loop is an agent that
    stalls at exactly the moment it has the most work queued. A first line per
    dropped entry is worse than a summary and much better than nothing.
    """
    lines = [entry["content"].splitlines()[0][:70] for entry in older]
    return f"unsummarized ({len(older)} entries dropped): " + " | ".join(lines)


# -- the demo -----------------------------------------------------------------------
#
# A research agent accumulating tool results. The entries below are invented but
# sized like real ones; nothing here was captured from a live run.

SEARCH_RESULTS = (
    """SEARCH carrier_appetite "commercial auto, fleet under 25 units"
Result 1 of 3: Meridian Mutual publishes appetite for fleets of 5-40 power units in
NY, NJ, CT and PA, minimum three years in business, no more than two at-fault losses
in the trailing 36 months. Radius restricted to 300 miles. Requires MVRs for all
listed drivers at bind. Declines: sand and gravel haulers, long-haul refrigerated,
any risk with a DOT out-of-service rate above the national average.""",
    """SEARCH loss_runs policy=AUTO-2024-001 years=3
Retrieved 3 loss run documents totalling 11 pages. 2023: two claims, both physical
damage, incurred 14,200 total, both closed. 2024: one claim, bodily injury, reserved
at 85,000, open, litigation flag set 2024-11-02. 2025: no claims to date. Loss ratio
on the trailing three years computes to 0.61 against written premium of 163,400.
Note: the 2024 open reserve was strengthened twice, most recently 2025-03-14.""",
    """SEARCH regulatory_bulletins state=NJ line=commercial_auto after=2025-01-01
Bulletin 25-04 requires carriers to file revised UM/UIM offer forms by 2026-01-01.
Bulletin 25-11 clarifies that telematics-derived surcharges must be disclosed at
quote, not at bind, and must be reversible on request within 30 days. Bulletin 25-19
extends the mandatory grace period for non-payment cancellations from 10 to 15 days
for policies with an annual premium above 25,000.""",
    """SEARCH internal_kb "VIN validation failure handling"
Three articles matched. KB-3312: the extraction service returns a null VIN when the
scanned field fails checksum validation, which is common on faxed ACORD 127 forms
where the character 0 and the letter O collide. KB-3318: never infer a VIN from make
and model; route to human review. KB-3350: a null VIN blocks rating but not quoting,
so the quote can proceed with a flagged vehicle if the underwriter accepts.""",
    """SEARCH filed_rates carrier=Meridian state=NY effective=2026
Base rate per power unit 2,140 for radius under 50 miles, 2,890 for radius 50-200,
3,410 for radius 200-300. Fleet credit schedule: 5 percent at 10 units, 9 percent at
20 units, 12 percent at 30 units. Telematics credit up to 15 percent on a documented
program. Surcharges: 22 percent for any driver under 23, 40 percent for a driver with
a major violation in the trailing 36 months.""",
    """SEARCH claim_notes policy=HOME-2024-003
Four notes on file. 2025-02-11: adjuster records that the loss location address does
not match the address on the declarations page; unit number differs. 2025-02-19:
insured states the unit number on the application was a typo. 2025-03-02: underwriting
requests a corrected ACORD 80 before endorsement. 2025-03-30: still outstanding, task
reassigned. No coverage decision has been recorded.""",
    """SEARCH competitor_filings line=commercial_auto state=PA after=2025-06-01
Two filings matched. Keystone Casualty filed a 6.4 percent overall rate increase
effective 2026-02-01, concentrated in the 200-300 mile radius bands. Allegheny
Indemnity filed a new telematics program with a maximum 20 percent credit and a
5 percent participation-only credit in the first term. Neither filing changes the
minimum-years-in-business eligibility rule.""",
    """SEARCH reinsurance_treaty_terms year=2026 line=commercial_auto
Quota share retention increased to 500,000 per occurrence from 350,000. Aggregate
stop loss attaches at a 78 percent loss ratio. The treaty excludes any risk written
with a radius above 300 miles, which makes the radius question on this submission a
treaty question and not only an appetite question. Cession statements are due within
45 days of quarter end.""",
)


def build_entry(step: int) -> dict[str, str]:
    """One tool result, numbered so the reader can follow which entries survive."""
    body = SEARCH_RESULTS[step % len(SEARCH_RESULTS)]
    return {"role": "tool_result", "content": f"[result {step:02d}] {body}"}


def main() -> None:
    # The source signature defaults to 50000 tokens. Reaching that here would need
    # roughly 200KB of invented search results, so the demo passes a smaller budget
    # explicitly. The mechanism is identical at either number.
    budget = 1500
    cycles = 12

    print("Compaction demo -- a research agent accumulating tool results")
    print(f"Token budget: {budget} (source default is 50000; lowered so compaction "
          f"fires within {cycles} cycles)")
    print(f"Recent window: last {RECENT_WINDOW} entries are never summarized")
    print("All token counts below are ESTIMATES from a character-count "
          "approximation, not a tokenizer.")
    print()

    history: list[dict[str, str]] = []
    step = 0

    for cycle in range(1, cycles + 1):
        # Two new tool results per cycle, the way a research agent actually grows.
        for _ in range(2):
            step += 1
            history.append(build_entry(step))

        before_tokens = estimate_tokens(history)
        before_len = len(history)
        recent_before = [entry["content"] for entry in history[-RECENT_WINDOW:]]

        # The "before" line goes out before the call so that anything compaction
        # itself prints -- the fallback notice, for one -- lands under the cycle it
        # belongs to instead of trailing the one before it.
        print(f"cycle {cycle:2d}  before: {before_len:2d} entries, "
              f"est. {before_tokens:5d} tokens")

        history = compact_context(history, max_tokens=budget)

        after_tokens = estimate_tokens(history)
        after_len = len(history)
        fired = after_len != before_len

        print(f"          after:  {after_len:2d} entries, est. {after_tokens:5d} "
              f"tokens   {'COMPACTED' if fired else 'under budget, untouched'}")

        if fired:
            recent_after = [entry["content"] for entry in history[-RECENT_WINDOW:]]
            preserved = recent_after == recent_before
            print(f"          recent window preserved verbatim: {preserved}")
            print(f"          summary entry: {history[0]['content'][:88]}...")

    print()
    print(f"Compactions fired: {_compactions}")
    print(f"Final history: {len(history)} entries, "
          f"est. {estimate_tokens(history)} tokens, budget {budget}")
    print("Without compaction the same run would have ended at "
          f"est. {estimate_tokens([build_entry(i + 1) for i in range(step)])} tokens.")
    print()
    print("What is deterministic here: the budget check, the size of the recent")
    print("window, the decision to compact, and the check that the returned summary")
    print("is actually shorter than what it replaced. Only the summary text itself")
    print("comes from the model.")


if __name__ == "__main__":
    main()
