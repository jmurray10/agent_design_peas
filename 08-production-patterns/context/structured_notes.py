"""Structured note-taking: keep a state document, not a transcript.

The source page shows the `<agent_state>` XML document and says it is "the
model-based reflex agent's internal state made explicit and token-efficient." It
does not show the code that maintains it. This file does, and then measures the
claim: the same five-step run is carried twice, once as a state document and once
as raw accumulated history, and both are counted at every step.

Run it from the repository root:

    python 08-production-patterns/context/structured_notes.py

The comparison is a real measurement of these two representations of this run.
It is not a benchmark of anything else. The ratio you see depends entirely on how
verbose the tool results are, and the tool results here are invented -- realistic
in shape, but written by hand for this demo. Change them and the ratio changes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.llm import llm_call  # noqa: E402


# -- the token estimator ------------------------------------------------------------
#
# Deliberately identical to the one in compaction.py, and deliberately duplicated:
# each script in this directory has to run on its own from the repository root, and
# a shared helper module is an abstraction the source page does not have.

CHARS_PER_TOKEN = 4.0


def estimate_tokens_in_text(text: str) -> int:
    """Approximate how many tokens `text` would cost. This is an estimate.

    Character count divided by four, the usual rule of thumb for English prose.
    Not a tokenizer: no byte-pair merges, and it runs low on code, JSON, and
    identifiers. Every token number this script prints comes from here, and both
    sides of the comparison are measured the same way, which is what makes the
    comparison meaningful even though the absolute numbers are approximate.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# -- the state document -------------------------------------------------------------

MAX_ENTRY_CHARS = 120

# The four sections the source page's document has, in the order it has them.
LIST_SECTIONS = ("completed", "pending", "key_findings")
ITEM_TAG = {"completed": "step", "pending": "action", "key_findings": "finding"}


def initial_state() -> dict:
    """The document at step zero: a task and a plan, nothing done yet."""
    return {
        "current_task": "Processing ACORD form batch",
        "completed": [],
        "pending": [
            "Load batch manifest",
            "Extract fields from all forms",
            "Validate extracted records",
            "Generate completion report",
        ],
        "key_findings": [],
    }


def render_state(state: dict) -> str:
    """Render the state dict as the `<agent_state>` document from the source page.

    Rendering deterministically instead of asking the model to emit XML is the
    whole trick. The document can then never be malformed, never lose a section,
    and never drift in shape between turns, no matter what the model returns.
    """
    lines = ["<agent_state>", f"  <current_task>{escape(state['current_task'])}</current_task>"]
    for section in LIST_SECTIONS:
        lines.append(f"  <{section}>")
        for item in state[section]:
            tag = ITEM_TAG[section]
            lines.append(f"    <{tag}>{escape(item)}</{tag}>")
        lines.append(f"  </{section}>")
    lines.append("</agent_state>")
    return "\n".join(lines)


def update_state(state: dict, tool_result: str, mock_key: str) -> dict:
    """Fold one tool result into the state document.

    The LLM proposes a patch. Deterministic code validates it, clamps it, and
    applies it. If the patch does not parse or does not match the schema, a
    deterministic fallback keeps the run going with a worse note rather than
    stopping or, worse, silently dropping the step.
    """
    prompt = f"""You are maintaining an agent state document. Here it is now:

{render_state(state)}

A step just completed. Here is its raw output:

{tool_result}

Return ONLY a JSON object with these optional keys:
  "current_task":  string, the task the agent is on now
  "completed":     list of short strings to append to <completed>
  "pending_done":  list of exact <pending> strings that are now finished
  "pending_add":   list of short strings to append to <pending>
  "key_findings":  list of short strings to append to <key_findings>
Each string must be under {MAX_ENTRY_CHARS} characters and must be a note a later
turn can act on, not a copy of the raw output."""

    raw = llm_call(
        prompt,
        mock_key=mock_key,
        # mid: this is structured JSON generation with a deterministic fallback --
        # the exact case the tier guidance names. The judgment involved is real
        # (which line of a 900-character tool dump is worth carrying forward for
        # the rest of the task) but it is bounded by a fixed schema, a character
        # cap, and a validator, so a frontier model would be paying for headroom
        # the schema already forecloses. A small model is the wrong direction: it
        # would return valid JSON containing useless notes, which the validator
        # cannot detect.
        tier="mid",
    )

    patch = _parse_patch(raw)
    if patch is None:
        print("    [fallback] patch did not parse as the expected JSON object; "
              "recording a raw one-line note instead")
        patch = {"completed": ["unsummarized: " + tool_result.splitlines()[0]]}

    return _apply_patch(state, patch)


def _parse_patch(raw: str) -> dict | None:
    """Return a validated patch, or None if the model did not produce one.

    Returning None rather than raising is the point: an agent that dies because a
    note-taking call came back malformed has traded a small loss for a total one.
    """
    text = raw.strip()

    # Fenced code blocks are the single most common wrapper around JSON that a
    # model returns, so strip them before giving up on parsing.
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1])

    try:
        patch = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(patch, dict):
        return None

    allowed = {"current_task", "completed", "pending_done", "pending_add", "key_findings"}
    if not set(patch) <= allowed:
        return None
    if "current_task" in patch and not isinstance(patch["current_task"], str):
        return None
    for key in allowed - {"current_task"}:
        if key in patch:
            value = patch[key]
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                return None
    return patch


def _apply_patch(state: dict, patch: dict) -> dict:
    """Apply a validated patch. The character cap is enforced here, not requested.

    The prompt asks for short entries. Asking is not enforcing, and the reason to
    keep a state document at all is that it stays bounded, so the bound is applied
    on this side of the call.
    """
    updated = {
        "current_task": state["current_task"],
        "completed": list(state["completed"]),
        "pending": list(state["pending"]),
        "key_findings": list(state["key_findings"]),
    }

    if "current_task" in patch:
        updated["current_task"] = _clamp(patch["current_task"])
    for item in patch.get("completed", []):
        updated["completed"].append(_clamp(item))
    for item in patch.get("key_findings", []):
        updated["key_findings"].append(_clamp(item))
    for item in patch.get("pending_done", []):
        if item in updated["pending"]:
            updated["pending"].remove(item)
    for item in patch.get("pending_add", []):
        updated["pending"].append(_clamp(item))
    return updated


def _clamp(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_ENTRY_CHARS else text[: MAX_ENTRY_CHARS - 3] + "..."


# -- the simulated run --------------------------------------------------------------
#
# Five steps of an ACORD batch job. The tool outputs are invented for this demo and
# written at the verbosity real extraction tools actually emit.

STEPS: tuple[tuple[str, str], ...] = (
    (
        "load_batch(batch_id='ACORD-2026-07')",
        """LOAD_BATCH ok batch=ACORD-2026-07 forms=15 source=sftp://intake/2026-07/
AUTO-2024-001 ACORD-127 pages=4 scanned=2026-07-02 ocr_conf=0.91
AUTO-2024-002 ACORD-127 pages=4 scanned=2026-07-02 ocr_conf=0.97
AUTO-2024-004 ACORD-127 pages=5 scanned=2026-07-02 ocr_conf=0.95
AUTO-2024-007 ACORD-127 pages=4 scanned=2026-07-03 ocr_conf=0.93
AUTO-2024-011 ACORD-127 pages=6 scanned=2026-07-03 ocr_conf=0.89
HOME-2024-003 ACORD-80  pages=3 scanned=2026-07-03 ocr_conf=0.88
HOME-2024-005 ACORD-80  pages=3 scanned=2026-07-04 ocr_conf=0.96
HOME-2024-008 ACORD-80  pages=3 scanned=2026-07-04 ocr_conf=0.94
HOME-2024-012 ACORD-80  pages=4 scanned=2026-07-04 ocr_conf=0.92
WORK-2024-002 ACORD-130 pages=7 scanned=2026-07-05 ocr_conf=0.90
WORK-2024-006 ACORD-130 pages=7 scanned=2026-07-05 ocr_conf=0.95
WORK-2024-009 ACORD-130 pages=6 scanned=2026-07-05 ocr_conf=0.93
GENL-2024-001 ACORD-125 pages=5 scanned=2026-07-06 ocr_conf=0.97
GENL-2024-004 ACORD-125 pages=5 scanned=2026-07-06 ocr_conf=0.94
GENL-2024-010 ACORD-125 pages=6 scanned=2026-07-06 ocr_conf=0.91
manifest checksum sha256:7f19c2... queue position 1 of 1""",
    ),
    (
        "extract_fields(rows=1-6)",
        """EXTRACT_FIELDS batch=ACORD-2026-07 rows=1-6 status=ok elapsed=41.2s
AUTO-2024-001 named_insured='Kellerman Freight LLC' fein=27-1188402 units=14
  garaging_zip=07030 radius=250 drivers=17 eff=2026-08-01 exp=2027-08-01
AUTO-2024-002 named_insured='Bay Ridge Courier Inc' fein=45-2210983 units=6
  garaging_zip=11209 radius=60 drivers=8 eff=2026-08-15 exp=2027-08-15
AUTO-2024-004 named_insured='Ostrander Bros Hauling' fein=81-4417720 units=22
  garaging_zip=18018 radius=290 drivers=26 eff=2026-09-01 exp=2027-09-01
AUTO-2024-007 named_insured='Pike Valley Distribution' fein=33-9902114 units=9
  garaging_zip=06010 radius=140 drivers=11 eff=2026-08-01 exp=2027-08-01
AUTO-2024-011 named_insured='Corbin Logistics Group' fein=52-7731098 units=31
  garaging_zip=07728 radius=300 drivers=38 eff=2026-10-01 exp=2027-10-01
HOME-2024-003 named_insured='Alvarez, Marisol' dwelling_type=condo units=1
  loc_zip=10312 coverage_a=410000 eff=2026-08-01 exp=2027-08-01
exceptions=0 fields_extracted=214 fields_null=0""",
    ),
    (
        "extract_fields(rows=7-12)",
        """EXTRACT_FIELDS batch=ACORD-2026-07 rows=7-12 status=ok_with_exceptions elapsed=52.8s
HOME-2024-005 named_insured='Petrosyan, Armen' dwelling_type=single units=1
  loc_zip=07446 coverage_a=685000 eff=2026-08-10 exp=2027-08-10
HOME-2024-008 named_insured='Whitfield, Dana' dwelling_type=single units=1
  loc_zip=11375 coverage_a=520000 eff=2026-09-01 exp=2027-09-01
HOME-2024-012 named_insured='Nakamura, Kenji' dwelling_type=townhome units=1
  loc_zip=06880 coverage_a=760000 eff=2026-08-20 exp=2027-08-20
WORK-2024-002 named_insured='Redhill Millwork Co' class_codes=[2802,5645] payroll=1840000
WORK-2024-006 named_insured='Sandoval Electric Inc' class_codes=[5190] payroll=2260000
WORK-2024-009 named_insured='Trellis Facilities Mgmt' class_codes=[9014,9015] payroll=980000
EXCEPTION AUTO-2024-001 field=vin[3] value=null reason=checksum_failed ocr_conf=0.61
  raw_scan='1FTfW1E5xKFA1O237' note character O at position 15 may be zero
  KB-3318 applies: do not infer VIN from make and model, route to human review
exceptions=1 fields_extracted=239 fields_null=1""",
    ),
    (
        "extract_fields(rows=13-15)",
        """EXTRACT_FIELDS batch=ACORD-2026-07 rows=13-15 status=ok_with_exceptions elapsed=33.4s
GENL-2024-001 named_insured='Harborline Property Mgmt' class_code=61212 sales=4100000
GENL-2024-004 named_insured='Quill & Bramble Retail' class_code=18435 sales=2350000
GENL-2024-010 named_insured='Ashcroft Event Services' class_code=46622 sales=1120000
EXCEPTION HOME-2024-003 field=loc_address value_mismatch
  declarations='118 Rivington St Unit 4B, Staten Island NY 10312'
  application='118 Rivington St Unit 4D, Staten Island NY 10312'
  claim note 2025-02-11 records the same discrepancy, still unresolved
exceptions=1 fields_extracted=97 fields_null=0""",
    ),
    (
        "validate_batch()",
        """VALIDATE_BATCH batch=ACORD-2026-07 records=15 schema=acord_v2026.1 elapsed=8.9s
PASS 12 records validated with no findings
FAIL AUTO-2024-001 rule=vin_required severity=blocking
  rating cannot proceed without a VIN for unit 3 of 14; quoting may proceed flagged
FAIL HOME-2024-003 rule=address_consistency severity=blocking
  declarations and application disagree on unit number; corrected ACORD 80 required
WARN AUTO-2024-011 rule=radius_ceiling severity=advisory
  radius 300 is at the treaty exclusion boundary; reinsurance review recommended
routing: 3 records to human_review queue, 12 records to rating queue
validation report written to s3://batch-reports/ACORD-2026-07/validation.json""",
    ),
)

MOCK_KEYS = (
    "context_notes_step_1",
    "context_notes_step_2",
    "context_notes_step_3",
    "context_notes_step_4",
    "context_notes_step_5",
)


def main() -> None:
    print("Structured note-taking vs raw history accumulation")
    print("Same five-step run, carried two ways, counted at every step.")
    print("Token counts are ESTIMATES from a character-count approximation, not a "
          "tokenizer.")
    print()

    state = initial_state()

    # The baseline: everything the agent saw, kept verbatim. Both representations
    # start from the same task line so the comparison is like for like.
    raw_history: list[str] = [f"TASK: {state['current_task']}"]

    rows: list[tuple[int, int, int]] = []

    for index, ((action, tool_result), mock_key) in enumerate(zip(STEPS, MOCK_KEYS), start=1):
        raw_history.append(f"ACTION: {action}")
        raw_history.append(f"RESULT: {tool_result}")

        # Header first, so anything update_state prints -- the fallback notice, for
        # one -- lands under the step it belongs to instead of trailing the previous.
        print(f"--- step {index}: {action} ---")

        state = update_state(state, tool_result, mock_key)

        document = render_state(state)
        structured_tokens = estimate_tokens_in_text(document)
        raw_tokens = estimate_tokens_in_text("\n".join(raw_history))
        rows.append((index, structured_tokens, raw_tokens))

        print(document)
        print(f"    state document: est. {structured_tokens} tokens")
        print(f"    raw history:    est. {raw_tokens} tokens")
        print()

    print("Measured on this run, with the estimator above:")
    print()
    print("  step   state doc   raw history   raw / state")
    for index, structured_tokens, raw_tokens in rows:
        ratio = raw_tokens / structured_tokens
        print(f"  {index:4d}   {structured_tokens:9d}   {raw_tokens:11d}   {ratio:10.1f}x")
    print()

    final_structured, final_raw = rows[-1][1], rows[-1][2]
    print(f"After 5 steps: {final_structured} est. tokens as a state document, "
          f"{final_raw} est. tokens as raw history.")
    print("That is a measurement of these two representations of this run, not a "
          "published figure and not a benchmark.")
    print("The ratio is a function of how verbose the tool results are. These tool")
    print("results are invented for the demo. Make them terser and the gap shrinks.")
    print()
    print("What the state document loses: everything not written down. The raw")
    print("history still contains the OCR confidences, the FEINs, and the exact")
    print("scan string that failed the VIN checksum. The document does not.")


if __name__ == "__main__":
    main()
