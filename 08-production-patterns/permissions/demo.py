"""The support agent asking for refunds, with an authorization layer under it.

Run from the repository root:

    python 08-production-patterns/permissions/demo.py

Part one is the agent from 01-reflex-agents/model-based/after.py, unchanged and loaded
from that file, working a ticket until it decides to issue a refund. Its architecture
validates that `issue_refund` is a real action and then stops asking questions. It never
asks how much, whose money, how many times today, or who said yes.

Part two puts the same decision through permissions.py at four magnitudes in one session:

    18.50   under the autonomous limit          executes
    240.00  over it                             escalates, does not execute
    240.00  again, with a manager's approval     executes
    9.99    under the limit                     executes, spends the last of the budget
    12.00   under the limit, budget gone        blocked

The last one is the interesting row. Nothing about the amount was wrong.

Part three covers the other two tiers and the two fail-closed paths: an actor whose role
may not request refunds at all, and an audit-required actuator with nowhere to write.

Everything the layer decides is printed with the rule that decided it, and the whole run
is reprinted as an audit table at the end.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from permissions import (  # noqa: E402
    Actor,
    ActionRequest,
    Approval,
    AuditLog,
    PermissionLayer,
    load_actuator_spec,
)
from shared.llm import llm_call  # noqa: E402

# Copied from the __main__ block of 01-reflex-agents/model-based/after.py, which is where
# that file defines them. They live under `if __name__ == "__main__"`, so they cannot be
# imported -- but they must match, or this layer would be authorizing a different agent's
# actions. The names are also the actuator names in actuators.yaml.
AVAILABLE_ACTIONS = [
    "reply_to_customer", "escalate_to_manager",
    "check_order_status", "issue_refund",
    "request_more_info", "close_ticket",
]

SESSION = "shift-2026-08-03-a"
BOT = Actor(identity="support-bot-7", role="support_agent")
TRIAGE = Actor(identity="triage-bot-2", role="triage_bot")


def _load(name: str, path: Path):
    """Load a module from a path. Directory names here have hyphens, so imports do not."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# -- the actuators, such as they are ------------------------------------------------

_refund_ids = itertools.count(1001)


def issue_refund_actuator(request: ActionRequest) -> str:
    """Stand-in for the payments API. Only ever called on an ALLOW."""
    return f"RF-{next(_refund_ids)} posted to payments for {request.amount:,.2f} on {request.reference}"


def reply_actuator(request: ActionRequest) -> str:
    return f"reply sent on {request.reference}"


def close_actuator(request: ActionRequest) -> str:
    return f"{request.reference} closed"


def no_side_effect_actuator(request: ActionRequest) -> str:
    """Actions that change nothing outside the conversation."""
    return f"{request.actuator} performed on {request.reference}"


# Which code runs for which action. The lookup is by the actuator name on the request,
# which is the name the allowed-action check just validated.
#
# This used to be a function passed in alongside the request, and the two could disagree.
# Live, the model answered a late-parcel ticket with `check_order_status` -- a good answer,
# and not the refund the caller had assumed. The permission layer correctly authorised the
# lookup, and the demo then ran the refund actuator anyway, on a request carrying no
# amount. Dispatching a validated action to an actuator chosen somewhere else is the exact
# failure this directory argues against, and it was in the file making the argument.
ACTUATORS = {
    "issue_refund": issue_refund_actuator,
    "reply_to_customer": reply_actuator,
    "close_ticket": close_actuator,
    "escalate_to_manager": no_side_effect_actuator,
    "check_order_status": no_side_effect_actuator,
    "request_more_info": no_side_effect_actuator,
}


# -- one attempt --------------------------------------------------------------------

def propose(mock_key: str, ticket: str, situation: str) -> ActionRequest:
    """Ask the model to fill in the refund actuator's output schema for one ticket."""
    prompt = f"""You are a customer support agent working ticket {ticket}.
Situation: {situation}
Available actions: {AVAILABLE_ACTIONS}
Return JSON only, with keys "action", "amount" (USD, omit if the action has no amount)
and "reason"."""
    # tier=mid: structured JSON against a fixed three-key schema plus one bounded numeric
    # judgment -- what is this refund worth. Small models get the number right and the
    # syntax wrong often enough that the recovery path would become the normal path, and
    # nothing above this needs a frontier model, because the number is not trusted: the
    # permission layer bounds it either way.
    response = llm_call(prompt, mock_key=mock_key, tier="mid")
    request = ActionRequest.from_model_output(response, reference=ticket)

    amount = "" if request.amount is None else f"  {request.amount:,.2f} USD"
    print(f"  model proposed: {request.actuator}{amount}")
    if request.parse_note:
        print(f"  [parse]      {request.parse_note}")
    return request


def validate(request: ActionRequest) -> bool:
    """The check every architecture page in this repo already has, and all it proves."""
    if request.actuator not in AVAILABLE_ACTIONS:
        print(f"  [validation] {request.actuator!r} is not an allowed action -> no_op")
        return False
    print(f"  [validation] {request.actuator} is in available_actions -- passes")
    return True


def attempt(layer, request, actor, approvals=()) -> None:
    """Validate, authorize, and only then run. Prints every step.

    The actuator is looked up from the action the request actually carries, so the code
    that runs can never be for a different action than the one just authorised.
    """
    if not validate(request):
        return
    executor = ACTUATORS[request.actuator]
    decision = layer.execute(request, actor, executor, approvals)
    print(decision.render())
    if decision.executed:
        print(f"  [actuator]   {decision.result}")
        print(f"               budget: {layer.usage(request.actuator)}")
    else:
        print("  [actuator]   not run")
    print()


def rule(title: str) -> None:
    print(f"--- {title} " + "-" * max(0, 74 - len(title)))


# -- the run ------------------------------------------------------------------------

if __name__ == "__main__":
    spec = load_actuator_spec()
    audit = AuditLog()
    layer = PermissionLayer(spec, session=SESSION, audit_log=audit)

    # An action the agent can name and the spec does not cover would be denied at runtime
    # as an unknown actuator, which is the right answer at the wrong time. Check at start.
    uncovered = [name for name in AVAILABLE_ACTIONS if name not in layer.permissions]
    if uncovered:
        raise SystemExit(f"actuators.yaml has no permissions block for: {', '.join(uncovered)}")

    print("Actuator permissions: the support agent, one session.")
    print(f"Spec loaded from {spec['_loaded_from']}, covering all "
          f"{len(AVAILABLE_ACTIONS)} of the agent's actions.")
    print(f"A session is {spec['session_scope']}; refunds are capped at "
          f"{layer.permissions['issue_refund'].rate_limit.describe()}.\n")

    rule("part 1: the agent decides, the way it already does")
    support = _load("support_after", REPO / "01-reflex-agents" / "model-based" / "after.py")
    agent = support.LLMModelBasedReflexAgent(
        role="customer support agent", available_actions=AVAILABLE_ACTIONS
    )
    for percept in [
        support.Percept({'message': 'My order hasnt arrived and its been 2 weeks'}),
        support.Percept({'order_lookup': 'Order #4521 - shipped 12 days ago, stuck in transit'}),
        support.Percept({'message': 'Third time this has happened. I want a refund.'}),
    ]:
        print(f"  see: {list(percept.data.values())[0]}")
        print(f"  do:  {agent.agent_function(percept)}")
    print()
    print("  Three actions, three membership tests, three passes. That is the whole")
    print("  authorization story in the architecture as published: a check against six")
    print("  strings. How much, whose money, how often, approved by whom -- none of those")
    print("  questions have anywhere to live in it.")
    print()
    print("  Worth noticing what the agent did not name: issue_refund, on a ticket that")
    print("  asks for one outright. Whether it reaches for that action is a property of")
    print("  the model and the day, and it is exactly why part 2 stops asking the model")
    print("  and tests the layer's thresholds directly.\n")

    rule("part 2: the same action, four magnitudes, one session")

    print("ticket 4521 -- shipping charge on a parcel stuck in transit")
    # What the model asks for is printed, and then set aside. Part 2 is a test of the
    # layer's thresholds, and a threshold test needs the amounts chosen by the test rather
    # than by the model: against a real key the model answers this ticket with
    # `check_order_status`, which is a better first move than a refund and would leave
    # every row below unexercised. Part 1 above and injection_demo.py are where the model
    # drives; here it is the variable being held still.
    proposed = propose("permissions_refund_4521", "T-4521",
                       "Parcel 12 days late, customer wants the shipping charge back.")
    if proposed.actuator != "issue_refund" or proposed.amount is None:
        print("  (not a refund proposal, so part 2 uses its own amounts from here on)")
    attempt(layer, ActionRequest(actuator="issue_refund", amount=18.50,
                                 reference="T-4521"), BOT)

    print("ticket 4622 -- damaged item, customer wants the order refunded")
    propose("permissions_refund_4622", "T-4622",
            "Item arrived damaged, customer wants the full order refunded.")
    over_limit = ActionRequest(actuator="issue_refund", amount=240.00, reference="T-4622")
    attempt(layer, over_limit, BOT)

    print("ticket 4622 -- resubmitted after a manager approved it out of band")
    # The approval is constructed here, by runtime code, from something a human did in an
    # approval system. It is not a field in a model response, and there is no code path in
    # permissions.py that could build one from a model response.
    approval = Approval(
        actuator="issue_refund",
        approver="d.whitfield",
        approver_role="support_manager",
        token="APR-7781",
        max_amount=500.00,
        kind="human_approval",
        reference="T-4622",
    )
    attempt(layer, over_limit, BOT, approvals=[approval])

    print("ticket 4703 -- duplicate charge on a small item")
    propose("permissions_refund_4703", "T-4703", "Customer was charged twice for one item.")
    attempt(layer, ActionRequest(actuator="issue_refund", amount=9.99,
                                 reference="T-4703"), BOT)

    print("ticket 4810 -- promo code was not applied at checkout")
    propose("permissions_refund_4810", "T-4810",
            "Promo code failed at checkout, customer wants the difference.")
    attempt(layer, ActionRequest(actuator="issue_refund", amount=12.00,
                                 reference="T-4810"), BOT)

    print("  12.00 was the smallest refund of the session and the only one refused on")
    print("  count. Magnitude and frequency are separate bounds and both have to hold.\n")

    rule("part 3: the other tiers, and two ways to fail closed")

    print("an autonomous actuator needs no ceremony")
    attempt(layer, propose("permissions_reply_4521", "T-4521", "Tell the customer the refund is on its way."),
            BOT)

    print("a reversible actuator at requires_confirmation takes a customer confirmation")
    close = propose("permissions_close_4703", "T-4703", "The duplicate charge is refunded, nothing else is open.")
    attempt(layer, close, BOT)

    print("  the customer confirms in the next message, and the runtime resubmits")
    attempt(layer, close, BOT, approvals=[Approval(
        actuator="close_ticket",
        approver="customer:AC-77120",
        approver_role="customer",
        token="CNF-3390",
        kind="confirmation",
        reference="T-4703",
    )])

    print("the same 18.50 refund, asked for by a read-only triage bot")
    attempt(layer, ActionRequest(actuator="issue_refund", amount=18.50, reference="T-4521"),
            TRIAGE)

    print("an audit-required actuator on a layer with no audit sink")
    unlogged = PermissionLayer(spec, session="shift-2026-08-03-a-unlogged", audit_log=None)
    unlogged_decision = unlogged.check(
        ActionRequest(actuator="issue_refund", amount=18.50, reference="T-4521"), BOT
    )
    print(unlogged_decision.render())
    print("  [actuator]   not run")
    print("               that decision is in the layer's decision list and in no audit")
    print("               log, because there is no audit log. Which is why it is a DENY.\n")

    rule("every decision this session, with the rule that produced it")
    print(audit.render_table())
    print()
    print(f"{len(audit.decisions())} decisions, {sum(1 for d in audit.decisions() if d.executed)} executed.")
    print("Not one of them was decided by a model. The layer is a few hundred lines of")
    print("Python reading a spec file, which is the only reason its answers are the same")
    print("on the tenth run as on the first.")
