"""Three capabilities, each one violating the five tool-design principles.

The capabilities are: find a person, find a document, fetch usage data for an account.
`tools_good.py` implements exactly the same three against the same fixture data. Nothing
here is a strawman -- every anti-pattern below is a shape real internal APIs actually
have, because these functions were written by wrapping a database schema instead of
wrapping a workflow.

What is wrong, principle by principle:

    1. Choose the right tools     list_contacts and get_files return everything. The
                                  model has to read the whole table to find one row.
    2. Namespace your tools       list_contacts, get_files, data. Bolt a second CRM or a
                                  second file store onto this agent and the names collide.
    3. Return meaningful context  UUIDs, MIME types, epoch timestamps, abbreviated keys.
                                  The account is a foreign key, not a name, so the model
                                  cannot connect a contact to the account in its task.
    4. Token efficiency           No pagination, no filtering, no aggregation, no
                                  truncation indicator, and "Error: Invalid input" as the
                                  entire diagnostic for two unrelated failure modes.
    5. Prompt-engineer            One-line descriptions that restate the function name.
       descriptions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The fixture module sits next to this file. Python only puts the *script's* directory on
# sys.path, so an importer living elsewhere (compare.py does not, but a notebook might)
# would otherwise fail on the next line.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from records import ACCOUNT_IDS, CONTACTS, FILES, daily_usage, epoch  # noqa: E402

# Principle 5, violated: the description says nothing the name did not already say.
# Principle 2, violated: no service or resource prefix anywhere.
TOOLS: list[dict] = [
    {
        "name": "list_contacts",
        "description": "Lists contacts.",
        "parameters": {},
    },
    {
        "name": "get_files",
        "description": "Gets files.",
        "parameters": {},
    },
    {
        "name": "data",
        "description": "Returns usage data.",
        "parameters": {"customer": "string", "days": "int"},
    },
]


def list_contacts() -> list[dict]:
    """Return every contact in the CRM.

    Principle 1's named anti-pattern. There is no query parameter, so an agent looking
    for one person pays for all of them and then does a brute-force scan token by token.
    """
    return [
        {
            "id": contact["uuid"],
            "cn": contact["name"],
            "eml": contact["email"],
            "tel": contact["phone"],
            "ttl": contact["title"],
            # A foreign key where a name belongs. The model is holding an id it has no
            # way to resolve, which is exactly how hallucinated account matches happen.
            "org": _account_id(contact["account"]),
            "rec_type": "application/vnd.crm.contact+json",
            "mod_ts": epoch(contact["updated"]),
        }
        for contact in CONTACTS
    ]


def get_files() -> list[dict]:
    """Return every file in the drive.

    Same anti-pattern, plus the response the source page prints as its example of a bad
    one: an opaque id and a MIME type, with no human-readable file type.
    """
    return [
        {
            "id": file["uuid"],
            "name": file["name"],
            "mime_type": file["mime"],
            "sz": file["size_bytes"],
            "mod_ts": epoch(file["modified"]),
            "own": _account_id(file["account"]),
            "fld": file["folder"],
        }
        for file in FILES
    ]


def data(customer: str, days: int) -> list[dict] | str:
    """Return daily usage rows for a customer.

    Two independent validation failures collapse into one string. The model cannot tell
    whether it passed a bad identifier or a bad window, so its only recovery strategy is
    to vary one parameter at a time and call again -- which is what the mock trajectory
    in compare.py has it do, three times, before it gets a row back.
    """
    if customer not in ACCOUNT_IDS or not 1 <= days <= 90:
        return "Error: Invalid input"

    # No aggregation and no cap: `days` daily rows, each one repeating the account name,
    # the source system, and the unit, because that is what the underlying table stores.
    return [
        {
            "d": row["date"],
            "c": row["api_calls"],
            "s": row["active_seats"],
            "st": row["storage_gb"],
            "e": row["errors"],
            "acct": customer,
            "src": "usage_v2",
            "unit": "count",
        }
        for row in daily_usage(customer, days)
    ]


def call(name: str, args: dict) -> tuple[str, bool]:
    """Dispatch a tool call and serialize the result.

    Returns (serialized response, ok). `tools_good.py` exposes an identical function with
    an identical signature and the identical serializer, so compare.py measures the two
    responses on exactly the same terms.
    """
    handlers = {"list_contacts": list_contacts, "get_files": get_files, "data": data}
    handler = handlers.get(name)
    if handler is None:
        return json.dumps("Error: Invalid input"), False

    try:
        result = handler(**args)
    except TypeError:
        # A wrong or misspelled argument name. In character for this tool set: the model
        # gets the same five words it gets for every other failure and learns nothing.
        return json.dumps("Error: Invalid input"), False

    ok = not (isinstance(result, str) and result.startswith("Error"))
    return json.dumps(result, separators=(",", ":")), ok


def _account_id(account: str) -> str:
    return ACCOUNT_IDS.get(account, "00000000-0000-0000-0000")


if __name__ == "__main__":
    print("tools_bad.py -- three capabilities, five principles violated\n")

    contacts = list_contacts()
    print(f"list_contacts() -> {len(contacts)} records, no query parameter")
    print(f"  first record: {json.dumps(contacts[0], separators=(',', ':'))}\n")

    files = get_files()
    print(f"get_files() -> {len(files)} records, no query parameter")
    print(f"  first record: {json.dumps(files[0], separators=(',', ':'))}\n")

    print('data(customer="3e91c07b-5a44-4d18-b0f7", days=120) ->')
    print(f"  {data('3e91c07b-5a44-4d18-b0f7', 120)}")
    print('data(customer="Meridian Freight", days=120) ->')
    print(f"  {data('Meridian Freight', 120)}")
    print("  (same message, different cause -- the id was fine that time, the window was not)\n")

    rows = data("Meridian Freight", 90)
    print(f'data(customer="Meridian Freight", days=90) -> {len(rows)} daily rows')
    print(f"  first row: {json.dumps(rows[0], separators=(',', ':'))}")
