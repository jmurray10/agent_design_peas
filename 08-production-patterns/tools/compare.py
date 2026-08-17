"""Run one identical agent task against both tool sets and count what it cost.

The task, the fixture data, the agent loop, the serializer, and the token estimator are
the same for both columns. The only variable is tool design.

Read the banner the script prints above the table before quoting any number out of it.
In mock mode the tool-call sequence in each column is a canned script from
`shared/transcripts/`, so the call counts are a replay of one recorded run; the token counts
are still measured, because the tool definitions and the tool responses being counted are
the real ones. With a real key, a model chooses the calls and both numbers are
observations. The script says which of the two it just did.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# parents[2] is the repo root: compare.py -> tools -> 08-production-patterns -> repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# And this file's own directory, so `import tools_bad` works no matter how the script was
# launched (a plain `python .../compare.py` gets this for free; runpy and imports do not).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import shared.llm as llm_module  # noqa: E402
from shared.llm import llm_call  # noqa: E402

import tools_bad  # noqa: E402
import tools_good  # noqa: E402

# One task, both columns. Phrased the way a person would phrase it, and deliberately
# needing all three capabilities plus a window longer than either usage tool allows in
# one call -- the error path is part of the job, not an accident.
TASK = (
    "Meridian Freight comes up for renewal next month. Find the account owner and their "
    "email, find the renewal paperwork in the drive, and summarize how much the account "
    "has used the product over the last 120 days."
)

MAX_STEPS = 10
PREVIEW_CHARS = 96


@dataclass
class Run:
    """Everything compare.py counted for one tool set."""

    label: str
    definition_tokens: int = 0
    response_tokens: int = 0
    prompt_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    failed_tool_calls: int = 0
    recovered_parses: int = 0
    answer: str = ""
    scripted: bool = True
    trace: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Four characters per token.

    This is an estimator, not a tokenizer -- counting real tokens would mean a
    dependency, and this repo runs with nothing installed. Both columns go through the
    same estimator on the same kind of text, so the comparison between them survives the
    approximation even though any single absolute number is only in the neighbourhood.
    """
    return (len(text) + 3) // 4


def render_catalog(tools: list[dict]) -> str:
    """Render tool definitions the way they are sent to the model: every turn, in full."""
    blocks = []
    for tool in tools:
        params = ", ".join(f"{name}: {spec}" for name, spec in tool["parameters"].items())
        blocks.append(f"{tool['name']}({params})\n{tool['description']}")
    return "\n\n".join(blocks)


def build_prompt(catalog: str, transcript: list[str]) -> str:
    """Assemble the turn. Tool definitions and history are both re-sent every call, which
    is why a verbose tool response is not paid for once but once per remaining turn."""
    return (
        "You are an account agent. Use the tools below to complete the task.\n\n"
        f"TOOLS\n{catalog}\n\n"
        f"TASK\n{TASK}\n\n"
        f"HISTORY\n{chr(10).join(transcript) if transcript else '(nothing yet)'}\n\n"
        'Reply with one JSON object and nothing else. To call a tool: '
        '{"tool": "<name>", "args": {...}}. When you can answer the task: '
        '{"answer": "<answer>"}.'
    )


def parse_decision(raw: str) -> tuple[dict | None, bool]:
    """Parse the model's turn into a decision.

    Returns (decision, recovered). `recovered` is True when the strict parse failed and
    the JSON had to be dug out of surrounding prose or a code fence -- a real and common
    model behavior, so the loop handles it instead of crashing. Returns (None, False)
    when there is no JSON object at all, which ends the run rather than guessing.
    """
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        return None, False
    try:
        return json.loads(match.group(0)), True
    except json.JSONDecodeError:
        return None, False


def run_agent(tool_module, label: str, mock_prefix: str) -> Run:
    """Drive the agent loop over one tool set and record what it consumed."""
    run = Run(label=label)
    catalog = render_catalog(tool_module.TOOLS)
    run.definition_tokens = estimate_tokens(catalog)
    transcript: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(catalog, transcript)
        run.prompt_tokens += estimate_tokens(prompt)

        mock_key = f"{mock_prefix}_step{step}"
        # tier=mid: the model picks one tool from a catalog and emits a JSON object of
        # arguments. Structured generation with a deterministic parse and a recovery path
        # behind it -- a small model drops required arguments often enough to matter here,
        # and nothing in this loop needs frontier-level reasoning about ambiguity.
        raw = llm_call(prompt, mock_key=mock_key, tier="mid")
        run.model_calls += 1

        # Whether a model chose this turn or a recording replayed it is a property of the
        # shim's mode, not of the text. The honesty banner below is keyed on it, and it is
        # observed rather than asserted.
        if llm_module._select_provider() != "replay":
            run.scripted = False

        decision, recovered = parse_decision(raw)
        if recovered:
            run.recovered_parses += 1
        # Held back rather than printed now, so the note lands under the step it belongs
        # to instead of trailing the previous one.
        note = (
            "\n         [parse] reply was not bare JSON; recovered the object from prose"
            if recovered
            else ""
        )

        if decision is None:
            run.trace.append(f"  {step}. unparseable response, stopping")
            break

        if "answer" in decision:
            run.answer = decision["answer"]
            run.trace.append(f"  {step}. final answer{note}")
            break

        name = decision.get("tool", "")
        args = decision.get("args", {})
        response, ok = tool_module.call(name, args)

        run.tool_calls += 1
        run.response_tokens += estimate_tokens(response)
        if not ok:
            run.failed_tool_calls += 1

        transcript.append(f"called {name}({json.dumps(args)}) -> {response}")
        marker = "ok   " if ok else "ERROR"
        rendered_args = ", ".join(f"{key}={value!r}" for key, value in args.items())
        run.trace.append(
            f"  {step}. {marker} {name}({rendered_args}){note}"
            f"\n         {estimate_tokens(response):>5} tokens  {_preview(response)}"
        )

    return run


def _preview(text: str) -> str:
    flat = text.replace("\n", " ")
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + " ..."


def _row(caption: str, left: object, right: object) -> str:
    return f"{caption:<34}{str(left):>18}{str(right):>18}"


def main() -> None:
    bad = run_agent(tools_bad, "unhelpful tools", "tools_bad")
    good = run_agent(tools_good, "principled tools", "tools_good")

    for run in (bad, good):
        print(f"\n--- {run.label} " + "-" * (56 - len(run.label)))
        for line in run.trace:
            print(line)
        answer = run.answer or "(the run ended without one)"
        print(textwrap.fill(answer, width=74, initial_indent="  answer: ", subsequent_indent="          "))

    print("\n" + "=" * 70)
    if not (bad.answer and good.answer):
        # Almost always means the canned trajectory for one column is missing from
        # shared/transcripts/. Say so rather than letting a half-run fill in a table.
        print("INCOMPLETE RUN -- at least one column stopped before answering.")
        print("The counts below describe a partial run and should not be compared.")
        print("=" * 70)
        print()

    if bad.scripted and good.scripted:
        print("REPLAYED, NOT MEASURED TODAY")
        print(
            "Every tool call in both columns was replayed from shared/transcripts/ -- what a\n"
            "real model chose when this was recorded, rather than a script written to make\n"
            "a point. The token counts are mechanical either way: the size of the real tool\n"
            "definitions and the real tool responses, counted by one estimator on both\n"
            "sides. Set ANTHROPIC_API_KEY to let a model choose afresh."
        )
    else:
        print("MEASURED")
        print(
            "Real mode. A model chose every tool call in both columns. Call counts and\n"
            "token counts are both observations of this run, and both will vary between\n"
            "runs and between models."
        )
    print("=" * 70)

    print()
    print(_row("", bad.label, good.label))
    print("-" * 70)
    print(_row("tool definitions (tokens)", bad.definition_tokens, good.definition_tokens))
    print(_row("tool responses (tokens)", bad.response_tokens, good.response_tokens))
    print(_row("prompt tokens sent, all turns", bad.prompt_tokens, good.prompt_tokens))
    print(_row("model calls", bad.model_calls, good.model_calls))
    print(_row("tool calls", bad.tool_calls, good.tool_calls))
    print(_row("tool calls returning an error", bad.failed_tool_calls, good.failed_tool_calls))
    print(_row("JSON parses needing recovery", bad.recovered_parses, good.recovered_parses))
    print("-" * 70)

    ratio = bad.prompt_tokens / good.prompt_tokens if good.prompt_tokens else 0.0
    print(
        f"On this run, the unhelpful set sent {ratio:.1f}x the prompt tokens of the "
        f"principled set."
    )
    print(
        "That ratio is a property of these fixtures and this trajectory. It is not a\n"
        "published figure and it is not a claim about your data."
    )
    print(
        "\nNote the first row: prompt-engineered descriptions make the principled tool\n"
        "definitions the more expensive of the two. They are re-sent every turn and they\n"
        "still come out ahead, because the responses are where the tokens actually go."
    )
    print("Tokens are estimated at 4 characters each. No tokenizer is installed.")


if __name__ == "__main__":
    main()
