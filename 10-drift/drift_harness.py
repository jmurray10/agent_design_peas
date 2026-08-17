"""Shared machinery for the drift harness. `snapshot.py` and `replay.py` both import it.

Two failures wear the same word and behave nothing alike, so this module keeps them in
separate counters from the first line to the last:

    structural drift    the output stops conforming -- unparseable JSON, an action name
                        that is not on the actuator list, a state object missing required
                        fields. Deterministic code catches all of it and says so.

    behavioral drift    the output conforms perfectly and the agent is worse. Escalation
                        rate moves, the action distribution shifts, the success rate
                        falls. Every response passes every check. Nothing logs.

Nothing here ever adds the two together. There is no single "drift score" and there is
not going to be one, because the number would be dominated by whichever category happened
to be countable and would hide the other.

The agent under test and the twenty cases are not reimplemented. `run_case` from
08-production-patterns/evaluation/run_eval.py runs each case; this module wraps the one
function that module calls to reach a model, which is the only seam a perturbation needs.

WHERE THE RESPONSES COME FROM. Every model call goes through `shared/llm.py`. With a
backend configured it is a live call. With nothing configured it is replayed from
`shared/transcripts/`, which holds what a real model actually returned to that exact
prompt on a recorded date. Nothing is invented in either mode: a prompt with no recording
raises rather than substituting a stand-in.

Every perturbation this module offers changes the real input to a real model -- a
different system prompt, noisier percepts, degraded tool results, a weaker tier, another
backend. Nothing here simulates a model changing its mind. Because transcript entries are
keyed by the SHA-256 of the prompt, a perturbation that rewrites the prompt misses the
baseline recording and fails loudly, so baseline and perturbation are recorded separately.
That is the honest shape of the measurement: comparing two conditions means running both.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from shared import llm as shim  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402

EVAL_DIR = ROOT / "08-production-patterns" / "evaluation"
BASELINE_DIR = HERE / "baselines"
PROMPT_DIR = HERE / "prompts"
DEFAULT_PROMPT = PROMPT_DIR / "system_v1.md"

# Every state and effect response in the twenty-case baseline carries these three fields.
# That is what makes them checkable: a response missing one of them is a shape change the
# suite can name, not a matter of taste. The agent itself does not check this -- it only
# checks that the text parses -- so this counter belongs to the harness, and the report
# says which of the three structural checks lives where.
REQUIRED_STATE_FIELDS = ("ticket_id", "issue_type", "conversation")

# Percept keys that stand in for a tool result rather than something a customer typed.
# `--degraded-tools` damages these and leaves the customer's own words alone, because a
# degraded tool and a confused customer are different failures.
TOOL_KEY_SUFFIXES = ("_lookup", "_record", "_records", "_history", "_notice", "_context")
TOOL_KEYS = {"account"}

# Injected per degraded tool result, inside the measured window. Not a claim about any
# real tool: an amount chosen to be visible against a dictionary lookup and labelled as
# injected everywhere it is reported.
TOOL_LATENCY_S = 0.010


def _display_path(path: Path) -> str:
    """Repo-relative if it is in the repo, absolute if it is not.

    `--prompt` is allowed to point anywhere, including outside the tree, which is what a
    reader trying their own wording will do.
    """
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_module(name: str, path: Path):
    """Import a module by file path. `08-production-patterns` is not an identifier."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_eval = _load_module("drift_run_eval", EVAL_DIR / "run_eval.py")

ACTUATORS = list(run_eval.SUITE["agent"]["available_actions"])


# -- configuration under test ---------------------------------------------------------

@dataclass
class RunConfig:
    """Everything a perturbation can change about a run.

    A snapshot records this block, so two snapshots can be compared knowing exactly which
    variable moved. One field, one perturbation flag.
    """

    system_prompt: Path = DEFAULT_PROMPT
    tier: str | None = None
    provider: str | None = None
    noisy_input: bool = False
    degraded_tools: bool = False

    def slugs(self) -> list[str]:
        """Short names for the perturbations this configuration applies, in apply order.

        Empty for an unperturbed run. Used to name a snapshot file and to say in one line
        which variable moved; the perturbations themselves reach the model as real changes
        to the prompt, the tier or the backend.
        """
        names = []
        if self.system_prompt.resolve() != DEFAULT_PROMPT.resolve():
            names.append(f"prompt_{self.system_prompt.stem}")
        if self.tier:
            names.append(f"tier_{self.tier}")
        if self.provider:
            names.append(f"provider_{self.provider}")
        if self.noisy_input:
            names.append("noisy_input")
        if self.degraded_tools:
            names.append("degraded_tools")
        return names

    def describe(self) -> str:
        parts = [f"system prompt {_display_path(self.system_prompt)}"]
        if self.tier:
            parts.append(f"tier={self.tier}")
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.noisy_input:
            parts.append("noisy input")
        if self.degraded_tools:
            parts.append("degraded tools")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        text = self.system_prompt.read_text(encoding="utf-8")
        return {
            "system_prompt": _display_path(self.system_prompt),
            "system_prompt_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
            "tier": self.tier,
            "provider": self.provider,
            "noisy_input": self.noisy_input,
            "degraded_tools": self.degraded_tools,
        }


def backend_is_replay() -> bool:
    """Will this run be replayed from `shared/transcripts/` rather than reach a live model?

    The name is a holdover from when the offline path returned canned strings; what it
    reports now is "no live backend is selected, so every call is a replay of a recorded
    real response".

    Mirrors the selection order in the shim rather than asking it, because the answer is
    needed before `--provider` is allowed to touch the environment, and the shim caches
    its choice on the first call. `_ollama_is_up` is reused instead of reimplemented so
    the two cannot end up disagreeing about what a live Ollama is.
    """
    # Ask the shim rather than re-deriving it. This used to reimplement the selection
    # rules and test for a provider called "mock", which stopped existing when canned
    # responses were replaced by recorded ones -- so a keyless run reported itself as
    # live, the baseline and the replay disagreed about what they were, and ci_check
    # refused to compare them. One source of truth removes the whole class of that bug.
    return shim._select_provider() == "replay"


# -- percept perturbations ------------------------------------------------------------

def _is_tool_key(key: str) -> bool:
    return key in TOOL_KEYS or key.endswith(TOOL_KEY_SUFFIXES)


def _typo(text: str, rng: random.Random) -> str:
    """Transpose one adjacent pair. The commonest real typo and the least destructive."""
    if len(text) < 4:
        return text
    i = rng.randrange(len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def add_input_noise(case: dict, rng: random.Random) -> dict:
    """Typos, truncation and format shifts on what the customer said.

    Seeded per case, so a noisy run is reproducible: the same case always gets the same
    damage. Tool results are left alone -- `--degraded-tools` is the other flag.
    """
    noisy = copy.deepcopy(case)
    for percept in noisy["percepts"]:
        for key in list(percept):
            value = percept[key]
            if _is_tool_key(key) or not isinstance(value, str):
                continue
            text = value
            for _ in range(max(1, len(text) // 60)):
                text = _typo(text, rng)
            if len(text) > 120:
                # Hard truncation, no ellipsis. A form that posts a truncated field does
                # not announce that it did.
                text = text[: int(len(text) * 0.8)]
            if rng.random() < 0.5:
                text = text.lower().rstrip(".!?")
            percept[key] = text
        # Format shift: the same content arriving under a different field name, which is
        # what a changed upstream form actually looks like.
        if "message" in percept and rng.random() < 0.5:
            percept["msg"] = percept.pop("message")
    return noisy


def degrade_tools(case: dict, rng: random.Random) -> tuple[dict, set[int]]:
    """Missing fields, intermittent errors and empty values in tool results.

    Returns the damaged case and the 1-based step numbers that carry a damaged result, so
    the injected latency lands on the steps where a slow tool would have been felt.
    """
    damaged = copy.deepcopy(case)
    hit: set[int] = set()
    seen = 0
    for step, percept in enumerate(damaged["percepts"], start=1):
        for key in list(percept):
            if not _is_tool_key(key) or not isinstance(percept[key], str):
                continue
            hit.add(step)
            mode = seen % 3
            seen += 1
            value = percept[key]
            if mode == 0:
                # Missing fields: everything after the first separator is gone.
                for sep in (";", ",", " - "):
                    if sep in value:
                        percept[key] = value.split(sep)[0]
                        break
                else:
                    percept[key] = value[: len(value) // 2]
            elif mode == 1:
                percept[key] = f"<tool error: {key} timed out after 30s, no data>"
            else:
                head = value.split(":")[0]
                percept[key] = f"{head}: null"
    return damaged, hit


# -- the probe ------------------------------------------------------------------------

@dataclass
class CaseObservation:
    """What the harness saw at the model boundary for one case.

    Separate from `CaseRecord`, which is what the agent did. The two are cross-checked
    rather than merged: parse failures are counted from the agent's own printed fallback
    lines, and these counters only supply what the agent never looks at.
    """

    case_id: str = ""
    calls: int = 0
    strict_parse_failures: int = 0        # plain json.loads said no -- the agent's view
    recoverable_parse_failures: int = 0   # ...but a tolerant parser said yes
    schema_violations: int = 0            # parsed clean, wrong shape
    off_list_actions: int = 0             # the model named something not on the list
    # Always zero. Nothing is ever substituted for a response now; the counter is kept so
    # that snapshots taken before recordings replaced canned strings still parse.
    substitutions: int = 0
    injected_latency_s: float = 0.0
    # run_eval counts the prompt it was handed, which is the prompt before this harness
    # prepends a system prompt to it. Counted here and added on, so swapping the system
    # prompt shows up in the cost column instead of being invisible.
    prefix_tokens: int = 0


class Probe:
    """Stands in for `run_eval._real_llm_call` for the duration of a run.

    Three jobs, in order: prepend the system prompt under test, sleep where a degraded
    tool would have made the caller wait, and classify whatever comes back. It chooses
    nothing about the response itself -- that comes from a live backend or from the
    recording of one. It never touches `shared/`.
    """

    def __init__(self, config: RunConfig, replaying: bool,
                 degraded_steps: dict[str, set[int]] | None = None):
        self.config = config
        self.replaying = replaying
        self.slugs = config.slugs() if replaying else []
        self.system_prompt = config.system_prompt.read_text(encoding="utf-8").strip()
        self.degraded_steps = degraded_steps or {}
        self.prefix = f"{self.system_prompt}\n\n"
        self.prefix_tokens = run_eval.estimate_tokens(self.prefix)
        self.observation = CaseObservation()

    def begin(self, case_id: str) -> None:
        self.observation = CaseObservation(case_id=case_id)

    def finish(self) -> CaseObservation:
        return self.observation

    def __call__(self, prompt: str, mock_key: str = "default",
                 tier: str = "default") -> str:
        obs = self.observation
        obs.calls += 1

        # A real change to the text sent, live or replayed alike. It is why a --prompt
        # run's token counts move -- and, because a transcript entry is keyed by the
        # prompt, why a --prompt run needs its own recording rather than the baseline's.
        full_prompt = self.prefix + prompt
        obs.prefix_tokens += self.prefix_tokens

        case_id, kind, step = _parse_key(mock_key)
        if self.config.degraded_tools and kind == "state" \
                and step in self.degraded_steps.get(case_id, set()):
            # The lookup result arrives as a percept, and the first model call of that
            # step is the earliest point inside the measured window. Sleeping here puts
            # an injected delay where a slow tool's delay would have shown up.
            time.sleep(TOOL_LATENCY_S)
            obs.injected_latency_s += TOOL_LATENCY_S

        key = self._resolve_key(mock_key)
        if key != mock_key:
            obs.substitutions += 1

        raw = shim.llm_call(full_prompt, mock_key=key, tier=tier)
        self._classify(kind, raw)
        return raw

    def _resolve_key(self, mock_key: str) -> str:
        """Pass the caller's key through untouched. Nothing selects a response here.

        Kept as a named seam so the one place a perturbation could have reached in and
        chosen an answer is visible, and visibly does not.
        """
        # This used to swap one canned response for another, which made drift something
        # the harness authored rather than something it observed. There are no canned
        # responses now. Every perturbation this harness offers -- a different prompt,
        # noisier percepts, degraded tool results, a weaker tier -- changes the real input
        # to a real model, so the drift is the model's and the harness only measures it.
        return mock_key

    def _classify(self, kind: str, raw: str) -> None:
        obs = self.observation
        if kind == "action":
            if raw.strip() not in ACTUATORS:
                obs.off_list_actions += 1
            return
        if kind not in ("state", "effect"):
            return

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # The agent uses a bare json.loads, so this is the count that matches what its
            # fallback actually did. The tolerant parser is asked second, and only to say
            # whether the deterministic layer threw away a correct answer.
            obs.strict_parse_failures += 1
            try:
                model_loads(raw)
            except json.JSONDecodeError:
                return
            obs.recoverable_parse_failures += 1
            return

        if not isinstance(parsed, dict) or not parsed:
            obs.schema_violations += 1
            return
        if any(field_name not in parsed for field_name in REQUIRED_STATE_FIELDS):
            obs.schema_violations += 1


def _parse_key(mock_key: str) -> tuple[str, str, int]:
    """`evaluation_c04_state_3` -> ('c04', 'state', 3). ('', '', 0) if it is not one."""
    parts = mock_key.split("_")
    if len(parts) != 4 or parts[0] != "evaluation":
        return "", "", 0
    try:
        return parts[1], parts[2], int(parts[3])
    except ValueError:
        return "", "", 0


# -- running a labelled suite ---------------------------------------------------------

def run_labeled(config: RunConfig, label: str) -> dict:
    """Run all twenty cases under `config` and return the snapshot dict.

    The snapshot is the unit of comparison. It carries per-case detail and aggregate
    distributions, and the distributions are the part that matters: behavioral drift
    shows up as a shifted distribution long before any single case starts failing.
    """
    replaying = backend_is_replay()
    if not replaying and config.provider:
        # Only when a live backend was already selected, and only after the decision
        # above: setting this on a replay would point the shim at a backend that is not
        # running. Which model answered is a property of the recording, so --provider and
        # --tier only mean anything live.
        os.environ["LLM_PROVIDER"] = config.provider

    cases = run_eval.load_cases()
    prepared: list[dict] = []
    degraded_steps: dict[str, set[int]] = {}
    for case in cases:
        prepared_case = case
        if config.noisy_input:
            prepared_case = add_input_noise(
                prepared_case, random.Random(f"noisy:{case['id']}"))
        if config.degraded_tools:
            prepared_case, hit = degrade_tools(
                prepared_case, random.Random(f"degraded:{case['id']}"))
            degraded_steps[case["id"]] = hit
        prepared.append(prepared_case)

    probe = Probe(config, replaying, degraded_steps)
    original = run_eval._real_llm_call
    run_eval._real_llm_call = probe

    entries: dict[str, dict] = {}
    records = []
    observations = []
    try:
        for case in prepared:
            probe.begin(case["id"])
            record = run_eval.run_case(case, tier_override=config.tier)
            observation = probe.finish()
            records.append(record)
            observations.append(observation)
            entries[case["id"]] = _case_entry(record, observation)
    finally:
        run_eval._real_llm_call = original

    return {
        "label": label,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "replay" if replaying else "live",
        "mock_variation": {
            "slugs": probe.slugs,
            "substitutions": sum(o.substitutions for o in observations),
        },
        "config": config.to_dict(),
        "suite": {
            "name": run_eval.SUITE["suite"],
            "path": EVAL_DIR.relative_to(ROOT).as_posix() + "/test_cases.json",
            "cases": len(prepared),
        },
        "agent": run_eval.SUITE["agent"],
        "cases": entries,
        "aggregate": aggregate(entries),
    }


def _case_entry(record, observation: CaseObservation) -> dict:
    """One case, flattened. Provenance of every structural count is fixed here.

    `json_parse_failures` and `invalid_action_names` come from the lines the agent itself
    printed when its own checks fired. `schema_violations` and the recoverable subset come
    from the probe, because the agent has no such check and pretending otherwise would
    credit the deterministic layer with work it does not do.
    """
    parse_fallbacks = [f for f in record.fallbacks if f["layer"].endswith("json_parse")]
    return {
        "title": record.title,
        "actions": list(record.actions),
        "final_action": record.final_action,
        "validation": record.validation,
        "escalated": record.escalated,
        "success": record.success,
        "failures": list(record.failures),
        "structural": {
            "invalid_action_names": record.invalid_actions,
            "json_parse_failures": len(parse_fallbacks),
            "recoverable_parse_failures": observation.recoverable_parse_failures,
            "schema_violations": observation.schema_violations,
        },
        "fallbacks": [dict(f) for f in record.fallbacks],
        "llm_calls": record.llm_calls,
        "tokens": record.tokens + observation.prefix_tokens,
        "latency_ms": round(record.latency_s * 1000, 2),
        "injected_latency_ms": round(observation.injected_latency_s * 1000, 2),
        "mock_substitutions": observation.substitutions,
        "probe_parse_failures": observation.strict_parse_failures,
        "probe_off_list_actions": observation.off_list_actions,
    }


def aggregate(entries: dict[str, dict]) -> dict:
    """Distributions first, pass/fail second, structural counts in their own block."""
    n = len(entries) or 1
    values = list(entries.values())

    frequency: dict[str, int] = {}
    for entry in values:
        for action in entry["actions"]:
            frequency[action] = frequency.get(action, 0) + 1
    total_actions = sum(frequency.values()) or 1

    latencies = [e["latency_ms"] for e in values]
    structural_keys = ("invalid_action_names", "json_parse_failures",
                       "recoverable_parse_failures", "schema_violations")

    return {
        "cases": len(entries),
        "actions_total": sum(frequency.values()),
        "action_frequency": dict(sorted(frequency.items())),
        "action_share": {a: c / total_actions for a, c in sorted(frequency.items())},
        "escalations": sum(1 for e in values if e["escalated"]),
        "escalation_rate": sum(1 for e in values if e["escalated"]) / n,
        "successes": sum(1 for e in values if e["success"]),
        "task_success_rate": sum(1 for e in values if e["success"]) / n,
        "cases_with_fallback": sum(1 for e in values if e["fallbacks"]),
        "fallback_rate": sum(1 for e in values if e["fallbacks"]) / n,
        "refusals": sum(1 for e in values if e["validation"] == "refused"),
        "structural": {
            key: sum(e["structural"][key] for e in values) for key in structural_keys
        },
        "cost": {
            "llm_calls_total": sum(e["llm_calls"] for e in values),
            "tokens_total": sum(e["tokens"] for e in values),
            "tokens_per_case": sum(e["tokens"] for e in values) / n,
            "latency_mean_ms": sum(latencies) / n,
            "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
            "injected_latency_total_ms": sum(e["injected_latency_ms"] for e in values),
        },
    }


# -- snapshot files -------------------------------------------------------------------

def snapshot_path(label: str) -> Path:
    return BASELINE_DIR / f"{label}.json"


def write_snapshot(snapshot: dict) -> Path:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(snapshot["label"])
    path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return path


def read_snapshot(label: str) -> dict:
    path = snapshot_path(label)
    if not path.exists():
        raise SystemExit(
            f"No baseline named {label!r} at {path}.\n"
            f"Take one first:  python 10-drift/snapshot.py --label {label}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# -- honesty ---------------------------------------------------------------------------

BAR = "=" * 78


def print_provenance(snapshot: dict, other: dict | None = None) -> None:
    """Say what produced these numbers before printing any of them.

    A drift table is exactly the kind of output that gets screenshotted without its
    caption, so whether these responses came from a live model or from a recording of one
    belongs above the deltas rather than in a footnote below them.
    """
    runs = [snapshot] + ([other] if other else [])
    print(BAR)
    if all(r["mode"] == "replay" for r in runs):
        slugs = sorted({s for r in runs for s in r["mock_variation"]["slugs"]})
        print("REPLAYED FROM RECORDED RUNS")
        print(
            "No backend is configured, so every response below was replayed from\n"
            "shared/transcripts/ -- what a real model returned to that exact prompt on\n"
            "the date stored beside it. A real model chose these actions. Nothing was\n"
            "written by hand to make a point, and a prompt with no recording raises\n"
            "instead of being answered by a stand-in."
        )
        if slugs:
            print(f"\n  perturbation(s) applied    {', '.join(slugs)}")
            print("  Each condition is its own recording. Entries are keyed by the")
            print("  prompt, so a perturbed run replays answers given to the perturbed")
            print("  prompt and cannot borrow the baseline's.")
        else:
            print("\n  No perturbation was requested: this is the baseline condition.")
        print(
            "\nWhat a replay is not: a fresh measurement. It is one run on one date, and\n"
            "model versions move underneath a tier name. Twenty cases is a sample and\n"
            "re-recording will move it. Set ANTHROPIC_API_KEY, or run a local Ollama, to\n"
            "ask a model what it does today."
        )
    else:
        print("MEASURED LIVE")
        print(
            "At least one run reached a live provider, so its actions, fallbacks and\n"
            "distributions are observations of that model on this suite at this moment.\n"
            "Every perturbation reaches the model as a changed prompt, a changed tier, a\n"
            "changed backend or changed input. One run is not a benchmark; a distribution\n"
            "from twenty cases is a sample, and re-running will move it."
        )
    print(BAR)
