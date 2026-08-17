"""`json.loads` for text that came out of a language model.

Same contract as `json.loads`: returns the parsed value, raises
`json.JSONDecodeError` if it cannot. Drop it in wherever an example parses a model
response and the existing `try/except json.JSONDecodeError` fallback keeps working
unchanged.

The difference is what counts as parseable. Asked for "valid JSON only", real models
routinely answer with valid JSON wrapped in something else:

    ```json
    {"action": "escalate"}
    ```

    Here is the updated state:
    {"action": "escalate"}

Both are the model doing its job. Neither survives a bare `json.loads`, and treating
them as parse failures is not robustness -- it is the deterministic layer discarding a
correct answer and then congratulating itself for degrading gracefully. Measured against
claude-sonnet-5, the naive parse failed on six of six calls in
`01-reflex-agents/model-based/after.py`, and the agent ran the whole demo on its fallback
path without the model contributing anything.

This is the deterministic half of the oscillation pattern doing real work: normalizing a
nondeterministic output format into a structured one, before validation gets a turn. It
is not a way of making the model deterministic, and it does not repair genuinely broken
JSON. Truncated output, invented syntax, and prose with no JSON in it all still raise,
which is what keeps every fallback in this repository reachable and honest.
"""

from __future__ import annotations

import json
import re
from typing import Any

# A fenced block, with or without a language tag. Non-greedy so the first complete block
# wins rather than everything up to the last fence in the response.
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def loads(text: str) -> Any:
    """Parse JSON out of a model response, or raise `json.JSONDecodeError`."""
    for candidate in _candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Nothing worked. Raise against the original text so the error message points at what
    # the model actually said rather than at some intermediate slice of it.
    raise json.JSONDecodeError("no parseable JSON in model response", text or "", 0)


def _candidates(text: str) -> list[str]:
    """Substrings of `text` worth attempting, cheapest and most likely first."""
    stripped = (text or "").strip()
    if not stripped:
        return []

    candidates = [stripped]

    # Fenced blocks, in the order they appear.
    candidates.extend(match.group(1).strip() for match in _FENCE.finditer(stripped))

    # Outermost brace- or bracket-delimited span. Catches "Here is the state: {...}" and
    # any trailing commentary after the value. Scanning from the first opener to the last
    # matching closer is deliberately blunt: a model that emits two separate objects gets
    # a parse failure rather than a silent pick of one of them.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if 0 <= start < end:
            candidates.append(stripped[start:end + 1])

    return candidates
