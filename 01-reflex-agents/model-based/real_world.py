"""The same model-based reflex agent, on a loan file instead of a two-cell floor.

`before.py` and `after.py` are the partially observable vacuum world, because that is the
example the source page shows. The agent sees one cell at a time and has to remember the
other, which demonstrates exactly what internal state is for.

A loan application is the same shape and somebody's mortgage depends on it. Documents
arrive one at a time over days: payslips, then a bank statement, then an ID that turns out
to be expired. Nobody ever sees the whole file in one percept, and the only thing that
knows whether the application is complete is what the agent remembered.

`ModelBasedReflexAgent` is imported from `before.py` rather than reimplemented. The architecture is unchanged: percept in, state update, rule match, effect
predicted, action out.

What the model replaces is `update_state`. The vacuum's version reads a location and a
status out of a dict, because a two-cell world has two facts in it. A loan file arrives as
"scanned the payslips, two of the three, and the bank statement is the joint account not
the personal one" -- and turning that into a change to what is on file is the reading a
rule table cannot do.

What it does not replace is the rule table over the state. Whether an application with
verified income, an expired ID and no address proof can proceed is a policy question with
a written answer, and it stays written.

Run it:

    python 01-reflex-agents/model-based/real_world.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

# before.py runs its vacuum demonstration at module level, exactly as the source page
# prints it. The listing is left alone rather than wrapped in an entry-point guard --
# readers arrive having seen that file -- so the output is swallowed here instead.
import contextlib  # noqa: E402
import io  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from before import ModelBasedReflexAgent  # noqa: E402

from shared.llm import llm_call  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402

# What a complete file needs. Policy, not judgement, and it is written down.
REQUIRED = ["income_verified", "identity_verified", "address_verified", "affordability_run"]

# What arrives, in the order it arrives. Each is one percept: a note from whoever handled
# the post that morning.
INBOX = [
    "Applicant sent three payslips, all consecutive, employer matches the application.",
    "Bank statement came in but it's the joint account, not the personal one we asked for.",
    "Passport scan received. Expiry date is last March, so it's out of date.",
    "Driving licence arrived, in date, address matches the application form.",
    "Ran affordability on the verified income figure. Passes with room.",
]


def read_note(note: str, on_file: dict) -> dict:
    """The model call, standing in for update_state. Prose in, a state change out."""
    prompt = f"""A loan application is being assembled. Update what is on file.

Already on file: {on_file}

New note from the processing team:
"{note}"

Return only the fields this note changes, as JSON. Do not repeat unchanged fields and do
not invent a field that is not listed here.

Fields:
  income_verified      true once income is evidenced and consistent with the application
  identity_verified    true only for identity evidence that is currently valid
  address_verified     true once address is evidenced
  affordability_run    true once an affordability assessment has been completed
  note                 one short sentence on anything a human should know

An expired document does not verify anything. Evidence for the wrong account or the wrong
person does not verify anything either.

Return JSON only, for example: {{"income_verified": true, "note": "three payslips, consistent"}}"""
    # tier=mid: reading a processing note and deciding which requirement it actually
    # satisfies is ordinary comprehension with a trap in it -- an expired passport looks
    # like identity evidence and is not. Every field it returns is filtered below.
    raw = llm_call(prompt, mock_key="loan_note", tier="mid")
    return model_loads(raw)


def rules_over_state(state: dict) -> str:
    """The policy, unchanged by any of this. A rule table over what is on file."""
    outstanding = [r for r in REQUIRED if not state.get(r)]
    if not outstanding:
        return "send_to_underwriting"
    if len(outstanding) == 1:
        return f"request_final_item:{outstanding[0]}"
    return f"await_documents:{len(outstanding)}_outstanding"


def main() -> None:
    print("The same model-based reflex agent, on a loan file")
    print()
    print("  ModelBasedReflexAgent is imported from before.py, which runs a two-cell")
    print("  vacuum world with it. Only what state means changed.")
    print()

    agent = ModelBasedReflexAgent()
    # The state this agent carries is the loan file, not a floor. Same attribute, same
    # class, and the runtime never learns which it is holding.
    agent.state = {}

    print("Documents arrive one at a time. Nobody sees the whole file at once.")
    print()

    for note in INBOX:
        change = read_note(note, dict(agent.state))

        # DETERMINISTIC: only declared fields land, and only as booleans. A model that
        # decided to set "approved": true would find nothing to write it to.
        applied = {}
        for field in REQUIRED:
            if field in change:
                applied[field] = bool(change[field])
        agent.state.update(applied)

        action = rules_over_state(agent.state)
        print(f"  note: {note}")
        print(f"    model read it as: {applied or 'nothing verified by this'}"
              + (f"   [{change['note']}]" if change.get("note") else ""))
        print(f"    on file now:      {sorted(k for k, v in agent.state.items() if v) or 'nothing'}")
        print(f"    do:               {action}")
        print()

    print("  The passport is the case worth watching. It is identity evidence, it is the")
    print("  document the file is waiting for, and it is expired -- so it verifies")
    print("  nothing, and the agent keeps waiting. A rule table keyed on 'passport"
          " received'")
    print("  marks that requirement satisfied and sends an incomplete file to")
    print("  underwriting.")
    print()
    print("  What changed: update_state. What did not: the policy over the state, which")
    print("  is a rule table anyone can read and argue with, and the filter that decides")
    print("  which fields a model is allowed to write at all.")


if __name__ == "__main__":
    main()
