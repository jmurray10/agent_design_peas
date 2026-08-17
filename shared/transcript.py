"""Recorded model responses, so a reader with no API key still sees a real one.

This replaces the canned mock responses this repository used to ship. The difference is
not cosmetic. A canned string is something a person wrote to make a demo come out a
particular way, and every headline number in a demo built that way is authored rather
than measured. Several of this repository's were, and running the same code against a
live model moved them a long way -- a suite that reported 85 percent success reported 45,
and a fallback rate of 74 percent turned out to be a parsing bug rather than a model
limitation.

A transcript entry is what a named model actually returned, on a named date, to the exact
prompt stored beside it. Replaying one is not a simulation of an agent. It is a recording
of one.

    ANTHROPIC_API_KEY=... LLM_RECORD=1 python <script>     record while running live
    python <script>                                        replay the recording

Entries are keyed by the SHA-256 of the prompt, which means a prompt change misses the
recording and fails loudly instead of replaying an answer to a question nobody asked any
more. That is deliberate: the silent version of that mistake is what
`10-drift/` exists to talk about.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

TRANSCRIPT_DIR = Path(__file__).resolve().parent / "transcripts"

_lock = threading.Lock()
_cache: dict[str, dict] = {}
_position: dict[str, int] = {}
# Sources this process has already started a fresh recording for.
_fresh: set[str] = set()


def prompt_key(prompt: str) -> str:
    """Stable identity for a prompt. Whitespace-normalized so trivial edits do not miss."""
    return hashlib.sha256(" ".join(prompt.split()).encode("utf-8")).hexdigest()[:24]


_SAFE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."


def _path_for(source: str) -> Path:
    # `python -c` reports its filename as "<string>", which no filesystem will accept.
    safe = "".join(c if c in _SAFE else "_" for c in source) or "unattributed"
    return TRANSCRIPT_DIR / f"{safe}.json"


def _load(source: str) -> dict:
    if source not in _cache:
        path = _path_for(source)
        if path.exists():
            _cache[source] = json.loads(path.read_text(encoding="utf-8"))
        else:
            _cache[source] = {"entries": {}}
    return _cache[source]


def replay(source: str, prompt: str, tier: str | None = None) -> str:
    """Return what a real model said to this prompt, or explain why we cannot.

    `tier` is checked against the recording. Prompts are keyed by content, and a tier is
    not part of the prompt, so without this check changing a tier replays the answer a
    *different model* gave -- silently, under the new tier's name. That is the failure
    this repository exists to argue about, sitting in the machinery meant to prevent it.
    A tier comparison run offline would have compared one model against itself and
    reported whatever the recording happened to hold.
    """
    entries = _load(source).get("entries", {})
    key = prompt_key(prompt)
    if key not in entries:
        raise LookupError(
            f"No recorded response for this prompt in shared/transcripts/{source}.json.\n"
            f"  prompt digest: {key}\n"
            f"  This happens when a prompt changed, or when the example is new. Record it:\n"
            f"    ANTHROPIC_API_KEY=... LLM_RECORD=1 python <the script you just ran>\n"
            f"  Nothing is faked in its place on purpose -- a stand-in answer here is how a\n"
            f"  demo keeps printing plausible output after it stopped meaning anything."
        )

    entry = entries[key]
    recorded_tier = entry.get("tier")
    if tier is not None and recorded_tier is not None and tier != recorded_tier:
        raise LookupError(
            f"Recorded at tier {recorded_tier!r}, replayed at tier {tier!r}, in "
            f"shared/transcripts/{source}.json.\n"
            f"  prompt digest: {key}  recorded model: {entry.get('model', 'unknown')}\n"
            f"  A tier selects a different model, so this recording is another model's\n"
            f"  answer to this prompt. Returning it under the new tier would make a tier\n"
            f"  comparison compare one model against itself. Re-record:\n"
            f"    ANTHROPIC_API_KEY=... LLM_RECORD=1 python <the script you just ran>"
        )

    responses = entry["responses"]
    # A prompt can legitimately recur within one run and deserve a different answer each
    # time, so responses are a list replayed in the order they were recorded. Past the end
    # the last one repeats, which is the closest honest thing to what a model would do.
    slot = f"{source}:{key}"
    index = _position.get(slot, 0)
    _position[slot] = index + 1
    return responses[min(index, len(responses) - 1)]


def recorded_model(source: str, prompt: str) -> str | None:
    """Which model produced the recording for this prompt, if there is one.

    Replay hands back a response and nothing about where it came from. A reader
    watching a replayed answer should be able to see whose answer it is, and the
    transcript already stores it -- this reads it without changing replay()'s
    contract.
    """
    store = _load(source)
    entry = store.get("entries", {}).get(prompt_key(prompt))
    return entry.get("model") if entry else None


def record(source: str, prompt: str, response: str, model: str, recorded_at: str,
           tier: str | None = None) -> None:
    """Add a real response to the transcript for `source`.

    Prompts already captured are skipped before this is reached, so a repeated recording
    pass fills gaps rather than appending a second answer beside the first. That matters:
    responses are replayed in recorded order, so a doubled entry would hand the first
    occurrence of a prompt one session's answer and the second occurrence another's,
    interleaving two runs into a replay that never happened.

    See `is_fresh_run` for the one way to remove anything.
    """
    with _lock:
        if source not in _fresh and is_fresh_run():
            _fresh.add(source)
            _cache[source] = {"entries": {}}
        data = _load(source)
        entries = data.setdefault("entries", {})
        key = prompt_key(prompt)
        entry = entries.setdefault(
            key, {"prompt": prompt, "model": model, "tier": tier,
                  "recorded_at": recorded_at, "responses": []}
        )
        entry["responses"].append(response)
        entry["model"] = model
        entry["tier"] = tier
        entry["recorded_at"] = recorded_at
        data["source"] = source

        TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        _path_for(source).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def reset_positions() -> None:
    """Start replay from the beginning again. Used when one process runs a suite twice."""
    _position.clear()


def describe(source: str) -> str:
    data = _load(source)
    entries = data.get("entries", {})
    if not entries:
        return f"{source}: no recording"
    models = sorted({e.get("model", "?") for e in entries.values()})
    dates = sorted({e.get("recorded_at", "?")[:10] for e in entries.values()})
    return f"{source}: {len(entries)} prompts, {', '.join(models)}, recorded {', '.join(dates)}"


def is_recording() -> bool:
    return os.environ.get("LLM_RECORD", "").strip() not in ("", "0", "false", "False")


def is_fresh_run() -> bool:
    """LLM_RECORD=fresh: discard this source's transcript and record it from scratch.

    Recording FILLS IN what is missing by default and never deletes. That default was
    chosen the hard way. Resetting on the first write of a process gives the tidier
    property -- one transcript is exactly one run -- and it cost 615 recorded prompts when
    an interrupted job restarted at a later step and truncated a file that had taken half
    an hour of live calls to build. Tidiness is not worth a destructive default on data
    that costs money to produce.

    Entries carry their own `recorded_at`, so a transcript stitched from several sessions
    says so entry by entry. Use `fresh` when a transcript should stop containing answers
    to prompts a script no longer asks; stale entries are otherwise inert, since nothing
    ever looks them up.
    """
    return os.environ.get("LLM_RECORD", "").strip().lower() == "fresh"


def already_recorded(source: str, prompt: str, tier: str | None = None) -> bool:
    """Has this exact prompt already been captured for this source, at this tier?

    A prompt captured at another tier does not count: it is a different model's answer,
    and reusing it during a recording pass would bake that confusion into the file.
    """
    entry = _load(source).get("entries", {}).get(prompt_key(prompt))
    if entry is None:
        return False
    recorded_tier = entry.get("tier")
    return recorded_tier is None or tier is None or recorded_tier == tier
