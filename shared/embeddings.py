"""One entry point for embeddings, on the same terms as `shared/llm.py`.

`llm_call` sends a prompt and gets text back. An embedding sends text and gets
coordinates back, which is a different contract, so it gets its own function rather than
a second parameter on the one function that shim deliberately exposes.

Everything else is identical, including the part that matters:

    backend configured   -> live call
    nothing configured   -> replay a RECORDED real response from shared/transcripts/
    no recording exists  -> raise, loudly, with instructions for recording one

A vector is stored as its JSON text, so the transcript store needs no special case: a
recording is still what a named model returned, on a named date, to the exact input
stored beside it. Change the input and the digest misses, which is the point.

Model names live in `shared/providers.yaml` and nowhere else, the same rule the LLM shim
follows. Calling code asks for a tier.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from shared import transcript  # noqa: E402

HF_ENDPOINT = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
TIMEOUT_SECONDS = 60

_banner_shown = False


def embed(text: str, tier: str = "default", source: str = "embeddings") -> list[float]:
    """Return the embedding of `text`, live or replayed. Never invented.

    `source` names the transcript file, so one example's recordings do not collide with
    another's -- the same reason `llm_call` attributes by calling example.
    """
    provider = _select_provider()

    if provider == "replay":
        _banner(
            "[replay] No embedding backend configured. Replaying recorded vectors from "
            "shared/transcripts/. These are real model outputs, not invented ones."
        )
        return _decode(transcript.replay(source, text, tier))

    if (transcript.is_recording() and not transcript.is_fresh_run()
            and transcript.already_recorded(source, text, tier)):
        _banner("[live] embeddings recording; inputs already captured are reused")
        return _decode(transcript.replay(source, text, tier))

    model = _model_for(tier)
    _banner(f"[live] embeddings tier={tier} model={model}"
            + ("  RECORDING" if transcript.is_recording() else ""))

    vector = _call_hf(model, text)
    if transcript.is_recording():
        transcript.record(source, text, json.dumps(vector), model, _now(), tier)
    return vector


def _select_provider() -> str:
    if os.environ.get("EMBEDDINGS_PROVIDER") == "replay":
        return "replay"
    if _token():
        return "huggingface"
    return "replay"


def _token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _model_for(tier: str) -> str:
    import yaml

    table = yaml.safe_load((_HERE / "providers.yaml").read_text(encoding="utf-8"))
    models = table["huggingface_embeddings"]
    return models.get(tier) or models["default"]


def _call_hf(model: str, text: str) -> list[float]:
    request = urllib.request.Request(
        HF_ENDPOINT.format(model=model),
        data=json.dumps({"inputs": text}).encode("utf-8"),
        headers={"Authorization": f"Bearer {_token()}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(
            f"Embedding call failed ({err.code}) for {model}.\n  {detail}\n"
            "  A 401 means the token cannot reach the inference API; a 404 usually means "
            "the model is not served by that provider."
        ) from err

    # feature-extraction returns either a vector or per-token vectors depending on the
    # model. Mean-pool the second shape so callers always get one vector per input, which
    # is what a sentence embedding means.
    if payload and isinstance(payload[0], list):
        width = len(payload[0])
        return [sum(row[i] for row in payload) / len(payload) for i in range(width)]
    return list(payload)


def _decode(raw: str) -> list[float]:
    return json.loads(raw)


def _now() -> str:
    from datetime import date

    return date.today().isoformat()


def _banner(message: str) -> None:
    global _banner_shown
    if not _banner_shown:
        print(message)
        _banner_shown = True
