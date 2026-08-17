"""LLM-as-judge: a score for the part of an answer no assertion can reach.

The function is the one on the source page. What surrounds it is the same deterministic
frame every other model call in this repository gets: parse, validate, refuse.

Two rules hold this file together, and both exist because a judge that is trusted too
far is worse than no judge at all.

    1. The judge never produces a number the harness could have computed. Where a
       deterministic check exists -- was the refund issued, was the ticket closed, was
       the action on the actuator list -- run_eval.py runs that check and the judge is
       not asked. The judge is for "was the customer actually told anything", which no
       assertion in this repository can decide.

    2. A judge that cannot be parsed, or that returns a score outside 1-5, scores
       nothing. It does not score 3. An invented middle number is indistinguishable
       from a real one in the aggregate, and that is the whole failure.

See README.md for why this is not the critic from 04-learning/q-learning/after.py.

Runs with no API key:

    python 08-production-patterns/evaluation/judge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# judge.py is run from the repo root ("python 08-production-patterns/evaluation/judge.py"),
# which puts this file's directory on sys.path but not the root. parents[2] is the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.llm import llm_call  # noqa: E402

SCORE_MIN, SCORE_MAX = 1, 5


def evaluate_response(agent_output: str, expected_criteria: str,
                      mock_key: str = "evaluation_judge_demo_clear") -> dict:
    """Score one agent output against one criterion. Returns a dict, always.

    Keys: score (int or None), reasoning (str), recovered (bool), error (str or None).
    A None score means the judge did not produce a usable one. Callers must treat that
    as an absent measurement rather than a bad one -- see `mean_score` below, which
    drops them from the average and reports how many it dropped.

    `mock_key` only selects which canned response comes back with no API key; every
    real backend ignores it, exactly as `llm_call` does.
    """
    prompt = f"""Rate this agent output on a scale of 1-5.
Output: {agent_output}
Criteria: {expected_criteria}
Return JSON: {{"score": int, "reasoning": str}}"""

    # tier=mid: bounded numeric judgment with a fixed output schema and a deterministic
    # guard underneath it -- the mid tier's description almost exactly. small returns a
    # bare number or prose about half the time, which the guard below would reject, and
    # a rejected score is a lost measurement. frontier is not worth paying for a
    # five-point scale against a one-sentence criterion, and this call runs once per
    # case per eval run, which is the volume that makes tier choice a real cost.
    raw = llm_call(prompt, mock_key=mock_key, tier="mid")
    return _parse_score(raw)


def _parse_score(raw: str) -> dict:
    """Turn whatever came back into a score, or into a refusal with a reason.

    The source page does `json.loads(llm_call(prompt))` and lets the exception out. In an
    eval harness that means one chatty judge response ends a twenty-case run partway
    through, which is a strange way to find out that the model likes preambles.
    """
    result = {"score": None, "reasoning": "", "recovered": False, "error": None,
              "raw": raw.strip()}

    payload, recovered = _extract_json_object(raw)
    if payload is None:
        result["error"] = "no JSON object in the judge response"
        return result
    result["recovered"] = recovered

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        result["error"] = f"judge response did not parse: {exc.msg}"
        return result

    if not isinstance(parsed, dict):
        result["error"] = "judge returned JSON that is not an object"
        return result

    score = parsed.get("score")
    result["reasoning"] = str(parsed.get("reasoning", "")).strip()

    # bool is a subclass of int, so True would otherwise sail through as a score of 1.
    if isinstance(score, bool) or not isinstance(score, int):
        result["error"] = f"score is not an integer: {score!r}"
        return result
    if not SCORE_MIN <= score <= SCORE_MAX:
        result["error"] = f"score {score} is outside {SCORE_MIN}-{SCORE_MAX}"
        return result

    result["score"] = score
    return result


def _extract_json_object(text: str) -> tuple[str | None, bool]:
    """Return the outermost braced span of `text` and whether it had to be dug out."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped, False
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, False
    return text[start:end + 1], True


def mean_score(results: list[dict]) -> tuple[float | None, int, int]:
    """Average the usable scores. Returns (mean or None, scored count, refused count)."""
    scores = [r["score"] for r in results if r["score"] is not None]
    refused = len(results) - len(scores)
    if not scores:
        return None, 0, refused
    return sum(scores) / len(scores), len(scores), refused


# -- demonstration ------------------------------------------------------------------

# Four replies and one criterion each. The last three exist to be handled badly by the
# judge, because a judge is only worth having if you know what it does when it fails.
FIXTURES = [
    {
        "label": "specific reply, clear commitment",
        "mock_key": "evaluation_judge_demo_clear",
        "criteria": "Does the reply tell the customer what happened and what happens next?",
        "output": (
            "Order 7734 was held at our Leeds depot after a mis-scan on 12 March. It is "
            "back on the network and due with you on Friday 21 March. If it has not "
            "arrived by Monday, reply here and I will refund the delivery charge without "
            "you having to ask again."
        ),
    },
    {
        "label": "accurate but empty reply",
        "mock_key": "evaluation_judge_demo_terse",
        "criteria": "Does the reply tell the customer what happened and what happens next?",
        "output": "Your order is currently in transit. Thank you for your patience.",
    },
    {
        "label": "judge wraps its JSON in prose",
        "mock_key": "evaluation_judge_demo_prose",
        "criteria": "Is the refund amount, method and timing all stated?",
        "output": (
            "I have refunded you for the missing filter. It will be back with you "
            "shortly."
        ),
    },
    {
        "label": "judge declines to score",
        "mock_key": "evaluation_judge_demo_refuses",
        "criteria": "Was the policy applied correctly for this customer's tier?",
        "output": "I am afraid a 13 month old motor is outside the warranty period.",
    },
    {
        "label": "judge returns a score off the scale",
        "mock_key": "evaluation_judge_demo_out_of_range",
        "criteria": "Does the reply tell the customer what happens next?",
        "output": "Escalated to a manager, who will contact you within one working day.",
    },
]


if __name__ == "__main__":
    print("LLM-as-judge, five agent replies.\n")

    results = []
    for fixture in FIXTURES:
        result = evaluate_response(
            fixture["output"], fixture["criteria"], mock_key=fixture["mock_key"]
        )
        results.append(result)

        print(f"--- {fixture['label']}")
        print(f"    criterion: {fixture['criteria']}")
        if result["score"] is None:
            print(f"    SCORE:     none -- {result['error']}")
            print(f"    judge said: {result['raw'][:78]!r}")
            print("    This case contributes no score. It is not a 3.")
        else:
            note = "  (JSON recovered from prose)" if result["recovered"] else ""
            print(f"    SCORE:     {result['score']}/5{note}")
            print(f"    because:   {result['reasoning'][:150]}")
        print()

    mean, scored, refused = mean_score(results)
    print(f"Mean judge score: {mean:.2f} over {scored} of {len(results)} replies "
          f"({refused} produced no usable score).")
    print()
    print("What that number is not:")
    print("  It is not accuracy, and nothing in this repository learns from it. The judge")
    print("  reads an output and forms an opinion with the same faculty that wrote it. It")
    print("  is a cheap second reader for qualities no assertion can check, and it belongs")
    print("  next to the deterministic metrics in run_eval.py, never in place of them.")
    print("  The critic in 04-learning/q-learning/after.py is a different job entirely:")
    print("  that one is arithmetic over observed outcomes and must never be a model.")
