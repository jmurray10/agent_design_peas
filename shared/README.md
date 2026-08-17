# shared

One LLM entry point for the whole repository. Every `after.py` imports it. Nobody writes
their own client.

    from shared.llm import llm_call

    llm_call(prompt: str, mock_key: str = "default", tier: str = "default") -> str

Both extra arguments have defaults, so the bare `llm_call(prompt)` shown on every source
page is still exactly the call the source pages show.

## Two modes, and neither one invents anything

The mode is chosen by environment, not by an argument:

    a backend configured  ->  live API call
    nothing configured    ->  replay a recorded real response

A one-line banner prints once per process saying which. Once, not per call.

There is no third mode. If nothing is configured and no recording exists for a prompt,
`llm_call` raises and tells you how to record one. It does not substitute a stand-in,
because a stand-in is how a demo keeps printing plausible output after it has stopped
meaning anything.

## Transcripts

`transcripts/` holds what real models actually returned. One file per example, keyed by
the SHA-256 of the prompt, each entry carrying the prompt, the model, the date, and the
response.

    ANTHROPIC_API_KEY=... LLM_RECORD=1 python 01-reflex-agents/simple/after.py

records while running live. Running the same file with no key replays it.

This replaced hand-written mock responses, and the reason is the whole argument of this
repository. A canned string is not a model, so an example built on one demonstrates
everything except the claim it is making. That was not hypothetical here. On canned
responses the evaluation suite reported 85 percent success; against a live model it
reported 45. A fallback rate of 74 percent turned out to be a JSON parsing bug rather
than a model limitation. A search example never reached its own subject, because the model
returned the board under a key the code did not read. The source page's own scoring code
raised TypeError when a model returned a confidence of "0.93" as a string. All of them
passed offline.

Keying on prompt content means a changed prompt misses its recording and fails loudly
rather than replaying an answer to a question it no longer asks. That is deliberate. The
silent version of that mistake is the subject of `10-drift/`.

A replay is a recording of an agent, not a simulation of one. It is still not the same as
running your own: model versions move, and the only way to know what a model does today
is to point a key at it.

## Tiers

A tier is a capability hint, not a model name:

    small      pick one label from a short list
    mid        structured generation with a fallback, bounded numeric judgment
    frontier   ambiguous language into a formal spec, synthesis across examples

Calling code asks for a capability level and never names a vendor. `providers.yaml` maps
tiers to concrete models and is the only file in the repository where vendor model names
appear. Changing which model an agent runs on is a change to that file, not to any
agent's code.

On replay the tier is whatever was recorded, because the recording is of a specific model.
Changing a tier and re-running offline replays the old model's answer, so a tier comparison
is only meaningful live.

## Backends

    replay             recorded real responses, zero setup, the default
    anthropic          ANTHROPIC_API_KEY
    gemini             GEMINI_API_KEY, over Google's REST endpoint, no SDK
    ollama             http://localhost:11434, local open-weight models, no key, no cost
    openai_compatible  OPENAI_COMPATIBLE_BASE_URL, covers vLLM, Together, Groq, OpenRouter,
                       and the Hugging Face router at https://router.huggingface.co/v1

Selection order: an explicit `LLM_PROVIDER` wins; otherwise Ollama if the local endpoint
answers, then Anthropic if a key exists, then Gemini, then an OpenAI-compatible base URL
if one is set, then replay.

Two backends need a word about what they return. The Gemini 3 models reason before
answering and hand those tokens back as parts flagged `thought`; the shim drops them and
keeps the answer, because concatenating the two produces a string that fails every JSON
parse downstream for a reason that looks nothing like the cause. And the Hugging Face
router serves only some of what the Hub hosts -- a model name it does not serve comes
back as an HTTP 400, not a fallback, so the names under `openai_compatible:` in
providers.yaml are ones checked against that router rather than picked from the Hub.

Ollama outranks a paid API deliberately. If a local model is already running, the free
one should be the default. Set `LLM_PROVIDER=anthropic` to override, which also skips
the port check.

The port check is a bare TCP connect with a 50ms budget per address, not an HTTP
request. A machine running Ollama answers on loopback in about a millisecond. A machine
that is not running it should not be able to tell this happened.

## model_json.py

`json.loads` for text a model produced. Same contract — returns the parsed value, raises
`json.JSONDecodeError` — so it drops into an existing `try/except` without changing the
shape of the fallback around it.

The difference is what counts as parseable. Asked for "valid JSON only", real models
routinely return valid JSON inside a ```json fence, or after a sentence introducing it.
Both are the model doing its job, neither survives a bare `json.loads`, and calling that
a parse failure is the deterministic layer discarding a correct answer and then taking
credit for degrading gracefully.

It does not repair broken JSON. Truncated output, invented syntax, and prose containing
no JSON all still raise, which is what keeps every fallback in this repository reachable.

## What the first live run found

Everything in this repository ran green in mock mode before any of it ran against a real
model. Three defects survived that, and all three were invisible offline:

- **Responses were being truncated.** `MAX_TOKENS` was 1024. Asked for a formal goal as
  JSON, `claude-opus-5` wrote past the cap and the response arrived cut off mid-object.
  Every downstream parse failed, every fallback fired exactly as designed, and every
  example still printed a plausible answer produced entirely without the model. The
  validation layer reported malformed output, which was true and useless. `llm_call` now
  warns on `stop_reason: max_tokens` — a truncated response is not a malformed one, and
  the two are indistinguishable by the time a caller sees only text.
- **Fenced JSON was being thrown away**, which is what `model_json.py` above exists for.
- **Non-ASCII output crashed the process on Windows.** A console at cp1252 cannot encode
  an arrow or an em dash. The old canned responses never hit it because they were ASCII.
  `llm_call` widens stdout on import.

Four more surfaced later, in files that had also passed offline: a `TypeError` when a
model returned `confidence` as `"0.93"`; a contract gate rejecting fields the prompt never
asked for; a "deliberate failure" run that only failed when no key was set; and an
evaluation prompt that made a model write 4096-token essays in answer to "return just a
number".

Seven defects, none visible offline. That record is why this repository stopped shipping
canned responses.

## Failure behavior

Live API errors propagate. They are not caught and quietly converted into a replay. A run
that silently degraded from a live model to a recording would produce output that looks
like a result and is not one, which is the specific failure this repository argues against
everywhere else.
