"""The one LLM entry point for this repository.

Every `after.py` imports `llm_call` from here. Nobody writes their own client.

    llm_call(prompt, mock_key="default", tier="default") -> str

`tier` is a capability hint -- "small", "mid", "frontier" -- not a model name. Callers
request a capability level and never name a vendor. `providers.yaml` maps tiers to
actual models. Both extra arguments have defaults, so the bare `llm_call(prompt)` that
the source pages show is still exactly what those pages show.

Five backends behind that one signature:

    replay             recorded real responses from shared/transcripts/, the default
    anthropic          ANTHROPIC_API_KEY
    gemini             GEMINI_API_KEY
    ollama             http://localhost:11434, local open-weight models, no key
    openai_compatible  OPENAI_COMPATIBLE_BASE_URL, covers vLLM/Together/Groq/
                       OpenRouter and the Hugging Face router

Selection order: an explicit LLM_PROVIDER wins; otherwise Ollama if the local endpoint
answers, then Anthropic, then Gemini, then an OpenAI-compatible base URL, then replay.
Ollama outranks the paid APIs on purpose -- if a local model is already running, the
free one should be the default. Set LLM_PROVIDER to override the whole order, which is
what running one example against three vendors in turn actually needs.

Replay is not a mock and not a stub that raises. It returns what a named model actually
returned to this exact prompt on a named date, so the calling code's parsing, validation,
and fallback paths all execute against real model output. It imports nothing outside the
standard library, which is what makes the zero-setup promise true.
"""

from __future__ import annotations

import contextvars
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

from shared import transcript


def _widen_stdout() -> None:
    """Make stdout able to print whatever a model hands back.

    A Windows console defaults to a legacy code page -- cp1252 on the machine this was
    written on. Print a response containing an arrow, an em dash, or a non-Latin script
    and the process dies on UnicodeEncodeError, several seconds and several API calls
    into a demo. Mock mode never hits it, because the canned responses in mocks.py are
    ASCII, so it is invisible until the first real run.

    Done here because this module is imported by exactly the files that print model
    output, and nowhere else. `errors="replace"` means this can only ever widen what
    prints successfully -- it has no failure mode of its own.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            pass  # already utf-8, or replaced by something that cannot be reconfigured


_widen_stdout()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Generous on purpose. At 1024 this repo looked like it worked and did not: asked for a
# formal goal as JSON, claude-opus-5 wrote past the cap and the response came back cut off
# mid-object. Every downstream json.loads then failed, every deterministic fallback fired
# exactly as designed, and every example still printed a plausible answer -- produced
# entirely without the model. The validation layer reported malformed output, which was
# true, and said nothing about the cause, because a truncated response is indistinguishable
# from a badly formatted one once you are only looking at the text.
MAX_TOKENS = 4096

_PROVIDERS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "providers.yaml")

# A caller that knows its own identity better than the call stack does can say so. The
# config runtime is the case that forced this: six agents all run through runtime.py, so
# every one of them attributed to the same transcript file, and re-recording one agent
# discarded the other five. An agent is the unit that gets recorded, not the file that
# happens to execute it.
_source_override: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "peas_transcript_source", default=None)

# What answered the most recent call, so a caller can report it without llm_call having
# to return two things. The conventions keep that signature at one string in and one
# string out on purpose, and widening it to carry provenance would change every call site
# away from what the source pages show.
_last_model: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "peas_last_model", default=None)


def last_model() -> str | None:
    """The model behind the most recent llm_call on this context.

    On a live call it is the model that was asked. On a replay it is the model the
    recording names, which is the more useful answer: it says whose output you are
    watching rather than whose you would have got.
    """
    return _last_model.get()

_banner_printed = False
_provider: str | None = None
_example_name: str | None = None
_tier_map: dict[str, dict[str, str]] | None = None


def llm_call(prompt: str, mock_key: str = "default", tier: str = "default") -> str:
    """Send `prompt` to a model at capability level `tier`, or replay a recorded answer.

    With a backend configured this is a live call. With none configured it replays what a
    real model returned to this exact prompt, from `shared/transcripts/`. There is no
    third mode: nothing here invents a response.

    `mock_key` is retained so that the call sites and the source pages still read the way
    the articles show them. It no longer selects anything -- prompts are matched by their
    own content, which is what makes a changed prompt miss the recording instead of
    silently replaying an answer to a question it no longer asks.
    """
    provider = _select_provider()
    source = _calling_example()

    if provider == "replay":
        _banner(
            "[replay] No backend configured. Replaying recorded responses from "
            "shared/transcripts/. These are real model outputs, not invented ones -- "
            "see shared/README.md."
        )
        _last_model.set(transcript.recorded_model(source, prompt))
        return transcript.replay(source, prompt, tier)

    # Recording fills in what is missing, so an interrupted pass resumes instead of paying
    # again for everything it already captured.
    #
    # LLM_RECORD=fresh has to bypass this, not just reach the reset inside record(): the
    # reuse below returns before record() is ever called, so a fresh pass would keep
    # replaying the entries it was asked to replace and report itself as a recording while
    # capturing nothing.
    if (transcript.is_recording() and not transcript.is_fresh_run()
            and transcript.already_recorded(source, prompt, tier)):
        _banner(f"[live] provider={provider} recording; prompts already captured are "
                f"reused rather than re-paid for")
        _last_model.set(transcript.recorded_model(source, prompt))
        return transcript.replay(source, prompt, tier)

    model = _model_for(provider, tier)
    _last_model.set(model)
    _banner(f"[live] provider={provider} tier={tier} model={model}"
            + ("  RECORDING" if transcript.is_recording() else ""))

    if provider == "anthropic":
        response = _call_anthropic(model, prompt)
    elif provider == "gemini":
        response = _call_gemini(model, prompt)
    elif provider == "ollama":
        response = _call_ollama(model, prompt)
    elif provider == "openai_compatible":
        response = _call_openai_compatible(model, prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    if transcript.is_recording():
        transcript.record(source, prompt, response, model, _now(), tier)
    return response


def transcript_source(name: str):
    """Name the transcript this call belongs to, for the duration of a `with` block.

    A contextvar rather than a module global, so two agents served concurrently in one
    process -- which is exactly what `serve.py` does -- cannot record into each other's
    files.
    """
    import contextlib

    @contextlib.contextmanager
    def _scope():
        token = _source_override.set(name)
        try:
            yield
        finally:
            _source_override.reset(token)

    return _scope()


def _now() -> str:
    # Imported here rather than at module scope: nothing else in this file needs a clock,
    # and a recording is the only thing that has to be dated.
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _calling_example() -> str:
    """Name the example that made this call, for the transcript filename.

    Derived from the call stack rather than from an argument, so an example cannot record
    into one file and replay from another, and so adding an example needs no registration
    step anywhere.

    The stack is not always enough. `05-multi-agent/orchestration/after.py` dispatches
    through `asyncio.to_thread`, and a worker thread's stack starts at the thread
    bootstrap with no repository frame anywhere in it -- which silently filed most of that
    example's calls under "unattributed" and recorded one prompt where there should have
    been a dozen. So the first frame that does resolve is remembered for the process, and
    stands in whenever a later call is made from somewhere the stack cannot see. One
    script is one process, so that attribution is the right one.
    """
    override = _source_override.get()
    if override:
        return override

    global _example_name
    import inspect

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shared_dir = os.path.join(repo, "shared")
    for frame in inspect.stack():
        path = os.path.abspath(frame.filename)
        if path.startswith(shared_dir) or not path.startswith(repo):
            continue
        relative = os.path.relpath(path, repo)
        name = os.path.splitext(relative)[0].replace(os.sep, "__").replace("-", "_")
        if _example_name is None:
            _example_name = name
        return name

    # No repository frame in this stack. The script that was actually run is the right
    # answer and is always available, so a threaded call files alongside its own example
    # rather than into a shared "unattributed" bucket.
    entry = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    if entry.startswith(repo) and os.path.isfile(entry):
        return os.path.splitext(os.path.relpath(entry, repo))[0].replace(os.sep, "__").replace("-", "_")
    return _example_name or "unattributed"


# -- provider selection -------------------------------------------------------------

def _select_provider() -> str:
    """Resolve the backend once per process and remember it."""
    global _provider
    if _provider is not None:
        return _provider

    explicit = os.environ.get("LLM_PROVIDER")
    if explicit:
        _provider = explicit
    elif _ollama_is_up():
        _provider = "ollama"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        _provider = "anthropic"
    elif os.environ.get("GEMINI_API_KEY"):
        _provider = "gemini"
    elif os.environ.get("OPENAI_COMPATIBLE_BASE_URL"):
        _provider = "openai_compatible"
    else:
        _provider = "replay"
    return _provider


def _ollama_is_up() -> bool:
    """Is something listening on the Ollama port?

    A bare TCP connect rather than an HTTP request: urllib consults proxy settings
    before opening a socket, which costs over a second on some machines. That second
    would be paid by every reader running in mock mode, who is exactly the person the
    zero-setup promise is for.

    The timeout is deliberately tiny. A machine actually running Ollama connects on
    loopback in about a millisecond, so the budget only has to cover "is it there",
    not "is it healthy". A machine that is not running Ollama should not notice this
    happened -- and some of them do not refuse the connection, they drop it, which is
    why a generous timeout would be a tax on the common case rather than a courtesy.
    Set LLM_PROVIDER to skip the check entirely.
    """
    parsed = urllib.parse.urlparse(OLLAMA_HOST)
    host, port = parsed.hostname or "localhost", parsed.port or 11434
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False

    for family, socktype, proto, _canonname, address in addresses:
        try:
            with socket.socket(family, socktype, proto) as probe:
                probe.settimeout(0.05)
                if probe.connect_ex(address) == 0:
                    return True
        except OSError:
            continue
    return False


def _model_for(provider: str, tier: str) -> str:
    """Map a capability tier to a concrete model name via providers.yaml."""
    global _tier_map
    if _tier_map is None:
        # Imported lazily so that mock mode never needs pyyaml, and so a reader with a
        # bare Python install can still run every example.
        import yaml

        with open(_PROVIDERS_YAML, encoding="utf-8") as handle:
            _tier_map = dict(yaml.safe_load(handle))

    tiers = _tier_map.get(provider)
    if not tiers:
        raise ValueError(f"No tier mapping for provider {provider!r} in providers.yaml")
    return tiers.get(tier) or tiers["default"]


# -- backends -----------------------------------------------------------------------

def _warn_if_truncated(reason: str | None, model: str) -> None:
    """Say so, loudly, when a response was cut off at the token cap.

    Truncated output is not malformed output, but it is indistinguishable from it by the
    time a caller runs json.loads. Every fallback in this repository would then fire,
    correctly, for the wrong reason, and the run would look fine. Whether the deterministic
    layer should have to guess at causes is the argument in 10-drift/; not making it guess
    costs one line here.
    """
    if reason in ("max_tokens", "length"):
        print(
            f"[warning] {model} hit the {MAX_TOKENS}-token cap and its response was cut "
            f"off. Anything parsing it will see malformed output. Raise MAX_TOKENS in "
            f"shared/llm.py.",
            file=sys.stderr,
        )


def _call_anthropic(model: str, prompt: str) -> str:
    # Imported inside the branch so replay mode has no third-party dependency at all.
    import anthropic

    # The SDK retries with exponential backoff; the default of 2 attempts is tuned for an
    # interactive call. A recording pass is hundreds of sequential calls, where a single
    # 529 from a busy API throws away everything already paid for -- which is exactly how
    # the first attempt at the critic experiment died, forty minutes in. Retrying more is
    # the cheap side of that trade.
    response = anthropic.Anthropic(max_retries=8).messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    _warn_if_truncated(response.stop_reason, model)

    # Same guard as _call_gemini, and it belongs here more than it belongs there: every
    # recording in this repository was made through this backend.
    #
    # claude-sonnet-5 returns a `thinking` block on every call now. Those tokens count
    # against max_tokens and are stripped here, so a prompt whose visible answer is 236
    # characters can spend the entire budget reasoning and arrive with no text at all.
    # Measured on 03-utility-based/value-iteration/real_world.py: 3 runs in 14 came back
    # empty or with half a JSON object. Returning "" sent that downstream, where every
    # deterministic fallback fired exactly as designed and blamed the model's formatting
    # for what is a budget problem -- the failure _warn_if_truncated's own docstring
    # describes, which it warns about and then permits.
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        kinds = sorted({block.type for block in response.content})
        raise RuntimeError(
            f"{model} returned no answer text (stop_reason="
            f"{response.stop_reason!r}, blocks={kinds}). The budget went on reasoning. "
            f"Raise MAX_TOKENS in shared/llm.py, or shorten the prompt."
        )
    return text


def _call_gemini(model: str, prompt: str) -> str:
    """Google's REST endpoint directly, with no SDK.

    Two details that are not optional. The key goes in a header rather than the query
    string Google's own examples use, because urllib raises HTTPError carrying the full
    URL -- a 400 from a query-string call prints the key into the traceback, and from
    there into whatever captured the run.

    And the Gemini 3 models reason before answering, returning those tokens as parts
    flagged `thought`. They are not the answer. Concatenating them yields a string that
    begins with the model talking to itself, which every JSON parse downstream then
    fails on for a reason that looks nothing like the cause.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS},
    }
    body = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        payload,
        {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    # A blocked prompt comes back 200 with no candidates at all, so the obvious
    # body["candidates"][0] raises IndexError and blames the shim. This repository sends
    # prompt-injection strings on purpose in 08-production-patterns/permissions/, which
    # is exactly the input a safety filter is built to stop.
    candidates = body.get("candidates") or []
    if not candidates:
        reason = (body.get("promptFeedback") or {}).get("blockReason", "unknown")
        raise RuntimeError(
            f"{model} returned no candidates; the prompt was blocked "
            f"(blockReason={reason!r}). This is the provider refusing the input, not a "
            f"parse failure."
        )
    candidate = candidates[0]
    # Gemini spells it MAX_TOKENS where Anthropic says max_tokens and Ollama says length.
    _warn_if_truncated((candidate.get("finishReason") or "").lower(), model)

    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    if not text:
        # A thinking model that spends the entire budget reasoning returns thought parts
        # and no answer. Silently returning "" sends an empty string into a parser and
        # blames the model's formatting for what is a budget problem.
        raise RuntimeError(
            f"{model} returned no answer text (finishReason="
            f"{candidate.get('finishReason')!r}, {len(parts)} part(s), all reasoning). "
            f"Raise MAX_TOKENS in shared/llm.py."
        )
    return text


def _call_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": MAX_TOKENS},
    }
    body = _post_json(f"{OLLAMA_HOST}/api/generate", payload)
    # Ollama reports "length" here when num_predict stopped it.
    _warn_if_truncated(body.get("done_reason"), model)
    return body["response"]


def _call_openai_compatible(model: str, prompt: str) -> str:
    base = os.environ["OPENAI_COMPATIBLE_BASE_URL"].rstrip("/")
    key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = _post_json(f"{base}/chat/completions", payload, headers)
    choice = body["choices"][0]
    _warn_if_truncated(choice.get("finish_reason"), model)
    return choice["message"]["content"]


def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    """POST JSON and parse JSON back, using only the standard library.

    An HTTPError is re-raised carrying the provider's own explanation. urllib's default
    message is the status line and nothing else -- "HTTP Error 500: Internal Server Error"
    for a body that says exactly which field was wrong. The body is the whole diagnostic,
    and it is on the exception object the entire time.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read().decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 -- a body that cannot be read is not the story
            pass
        # The URL is deliberately not repeated here. Google's own examples put the API
        # key in the query string, and urllib puts the full URL in the default message.
        raise urllib.error.HTTPError(
            error.url, error.code, f"{error.reason}: {detail[:600]}" if detail
            else str(error.reason), error.headers, None
        ) from None


# -- output -------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print the mode banner once per process, not once per call."""
    global _banner_printed
    if not _banner_printed:
        print(text)
        _banner_printed = True
