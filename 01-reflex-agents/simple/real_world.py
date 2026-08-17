"""The same simple reflex agent, on the GL coding table an AP team already maintains.

`before.py` and `after.py` are the vacuum world, because that is the example the source
page shows and readers arrive having seen it. It demonstrates the mechanism honestly and
answers no question about why anyone would run a reflex agent in production.

This is that answer. Same architecture, same class -- `SimpleReflexAgent` is imported from
`before.py` rather than reimplemented -- on accounts payable coding.

Every invoice that arrives has to be assigned a general ledger account before it can be
paid. Most finance teams do this with a vendor table: this supplier is always travel, that
one is always software. The table is fast, total over the vendors it knows, auditable and
free, and it is exactly right for the invoices that come from suppliers you pay every
month. Nobody should replace it with a model.

The tail is the problem, and it costs in a way the table cannot see. A vendor nobody has
coded before falls through `rule_match` and the invoice goes to a human queue. At volume
that queue is the cost centre: each one is a few minutes of an accountant reading a line
item to work out whether "professional services" means legal, consulting or contract
engineering. Miscode it and the department budget is wrong, the accrual is wrong, and
somebody finds out at quarter end.

Run it:

    python 01-reflex-agents/simple/real_world.py

The first section is the table alone. The second is the same agent with the model asked
about the fall-through only -- the oscillation this repository is about. The table keeps
every vendor it already knows, and the model reads the line item for the ones nobody has
coded yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

# before.py runs its vacuum demonstration at module level, exactly as the source page
# prints it, so importing the class also prints four lines about a floor. The listing is
# left alone rather than wrapped in an entry-point guard: readers arrive having seen that
# file in a screenshot, and indenting its body to import it cleanly would change the one
# thing rule 5 asks to keep recognisable. Swallowing the output here is the smaller edit.
import contextlib  # noqa: E402
import io  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    from before import Percept, SimpleReflexAgent  # noqa: E402

from shared.llm import llm_call  # noqa: E402
from shared.model_json import loads as model_loads  # noqa: E402

# The vendor table an AP team maintains by hand. Keyed the way the source page's rules are
# keyed: a substring the interpreted percept has to contain. Four vendors here; a real one
# has hundreds and answers most of the volume.
VENDOR_RULES = {
    "AWS": "6820_cloud_infrastructure",
    "WeWork": "7100_rent_and_facilities",
    "Slack": "6810_software_subscriptions",
    "Hertz": "7420_travel_ground",
}

# The chart of accounts this company uses. The model may choose one of these and nothing
# else. That is the deterministic half: a wrong answer miscodes an invoice, which an
# accountant catches in review, rather than inventing an account that does not exist and
# failing the posting run at month end.
CHART_OF_ACCOUNTS = {
    "6810_software_subscriptions": "Software and SaaS subscriptions",
    "6820_cloud_infrastructure": "Cloud hosting and infrastructure",
    "7100_rent_and_facilities": "Rent, utilities, facilities",
    "7200_professional_legal": "Legal counsel and related fees",
    "7210_professional_consulting": "Management and strategy consulting",
    "7220_contract_engineering": "Contracted engineering and development",
    "7420_travel_ground": "Ground transport and vehicle hire",
    "9000_uncoded": "Held for manual coding",
}

KNOWN = [
    {"vendor": "AWS", "amount": 48210.55, "line": "EC2, S3 and data transfer, October"},
    {"vendor": "WeWork", "amount": 12400.00, "line": "Office licence fee, November, 40 desks"},
    {"vendor": "Slack", "amount": 3180.00, "line": "Business+ annual, 212 seats"},
]

# First-time vendors: the invoices that reach a human today. All three could be read as
# "professional services" from the vendor name, and all three belong in different
# accounts, which is why the line item is the evidence and a name table cannot do it.
FIRST_TIME = [
    {"vendor": "Aldergate Partners LLP", "amount": 28750.00,
     "line": "Advisory re: share purchase agreement, disclosure schedules, completion mechanics"},
    {"vendor": "Northwind Delivery Co", "amount": 9640.00,
     "line": "Sprint capacity, two backend engineers, September, statement of work 4"},
    {"vendor": "Calder & Vance", "amount": 61000.00,
     "line": "Operating model review, phase 1 diagnostic and target state design"},
]


def code_invoice(invoice: dict) -> tuple[str, str]:
    """Only for a vendor the table has no rule for. Returns (account, reason)."""
    accounts = "\n".join(f"  {code}  {name}" for code, name in CHART_OF_ACCOUNTS.items())
    prompt = f"""Assign a general ledger account to this invoice. The vendor is not in our
coding table, so there is no history to copy.

Vendor: {invoice['vendor']}
Amount: {invoice['amount']:,.2f}
Line item: {invoice['line']}

Chart of accounts:
{accounts}

Code on what the work actually was, not on how the vendor styles itself. A firm called
"Partners LLP" may be doing legal work or consulting, and the line item is the evidence.
If the line item does not distinguish them, use 9000_uncoded rather than guessing: a held
invoice costs a few minutes, and a miscoded one costs a wrong department budget and a
correcting entry after somebody notices at quarter end.

Return JSON only: {{"account": "<code from the list>", "reason": "<one sentence>"}}"""
    # tier=small: choosing one of eight accounts from a line item is a bounded
    # classification, and it runs on every uncoded invoice -- which for AP is the whole
    # point, because a frontier model per invoice costs more than the review it replaces.
    raw = llm_call(prompt, mock_key="gl_coding", tier="small")
    answer = model_loads(raw)
    account = answer.get("account", "")
    # DETERMINISTIC: the model does not get to invent an account. An unknown code holds
    # the invoice rather than posting it somewhere that does not exist.
    if account not in CHART_OF_ACCOUNTS:
        return "9000_uncoded", f"model returned {account!r}, which is not in the chart"
    return account, answer.get("reason", "")


def main() -> None:
    agent = SimpleReflexAgent(VENDOR_RULES)

    print("BEFORE: the vendor table alone")
    print()
    print("  The coding table an AP team maintains. It answers most of the volume and")
    print("  costs nothing to run.")
    print()
    for invoice in KNOWN:
        account = agent.agent_function(Percept(data=invoice))
        print(f"  {invoice['vendor']:<24} {invoice['amount']:>11,.2f}  -> {account}")

    print()
    for invoice in FIRST_TIME:
        account = agent.agent_function(Percept(data=invoice))
        print(f"  {invoice['vendor']:<24} {invoice['amount']:>11,.2f}  -> {account}"
              f"   <- no rule, goes to a human")

    held = sum(i["amount"] for i in FIRST_TIME)
    print()
    print(f"  Three first-time vendors, {held:,.2f} held for manual coding. The table is")
    print("  right about everything it knows and silent about everything else, and that")
    print("  silence is indistinguishable from an invoice that genuinely needs review.")
    print()

    print("AFTER: the same agent, model asked only where the table fell through")
    print()
    for invoice in KNOWN:
        account = agent.agent_function(Percept(data=invoice))
        print(f"  {invoice['vendor']:<24} -> {account}   (table, no model call)")
    print()
    for invoice in FIRST_TIME:
        account, reason = code_invoice(invoice)
        print(f"  {invoice['vendor']:<24} -> {account}   [model]")
        print(f"    {reason}")
    print()
    print("  All three line items read as professional services from the vendor name, and")
    print("  all three belong in different accounts. That distinction lives in the")
    print("  description of the work, which is why a name table cannot make it and why an")
    print("  accountant was making it by hand.")
    print()
    print("  What changed: one branch. rule_match still answers every vendor it knows, at")
    print("  the same speed and cost as before. What did not: SimpleReflexAgent is")
    print("  imported from before.py rather than reimplemented, and an account outside the")
    print("  chart holds the invoice instead of posting it.")


if __name__ == "__main__":
    main()
