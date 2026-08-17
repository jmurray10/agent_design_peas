"""The same three capabilities, following the five tool-design principles.

Find a person, find a document, fetch usage data for an account -- the same three jobs
`tools_bad.py` does, against the same fixture data in `records.py`. Only the tool design
differs.

What each principle bought:

    1. Choose the right tools     search_contacts / search_files take a query and return
                                  matches, so the model skips to the answer instead of
                                  scanning a table. The usage capability also exposes a
                                  bulk-export path, because its error message promises one.
    2. Namespace your tools       crm_contacts_search, gdrive_files_search,
                                  analytics_usage_report. Service, then resource, then verb.
    3. Return meaningful context  Names, human-readable file types, ISO dates. Identifiers
                                  appear only in response_format="detailed", and even there
                                  they are the short handles the source page's good example
                                  uses -- {"id": 1, ...} -- not the underlying uuids.
    4. Token efficiency           page / page_size, total_results and truncated on every
                                  result set, monthly rollups instead of raw daily rows,
                                  and error messages that state the constraint, the value
                                  received, and the way forward.
    5. Prompt-engineer            Descriptions written like onboarding notes: query syntax,
       descriptions               what the tool is for, what it is not for, common mistakes.

Note on tool count: the good set lists four tools for three capabilities. Usage data has a
windowed report and a bulk-export escape hatch, because the source page's example of a good
error message points the agent at exactly such an escape hatch, and an error message that
names a tool which does not exist is worse than no message. The extra entry costs the good
column tokens in the tool-definition row of compare.py. It does not save it any.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# See the same comment in tools_bad.py: the fixture module is a sibling file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from records import ACCOUNTS, CONTACTS, FILES, daily_usage, monthly_usage  # noqa: E402

MAX_USAGE_DAYS = 90
DEFAULT_PAGE_SIZE = 5

TOOLS: list[dict] = [
    {
        "name": "crm_contacts_search",
        "description": (
            "Search CRM contacts by person name, account name, job title, or email.\n"
            "Matching is case-insensitive substring matching across all four fields, so "
            "'meridian' finds every contact at Meridian Freight.\n"
            "Best practice: search by account name first, then read the 'role' field to "
            "pick the right person. Roles are: account owner, executive sponsor, "
            "technical, billing, internal renewals owner.\n"
            "Returns 5 results per page. Use page=2 for the next 5; check 'total_results' "
            "before paging, it is exact.\n"
            "Use response_format='concise' when you only need to read the answer, and "
            "'detailed' when you need a contact_id for a follow-up call.\n"
            "Avoid: broad one-word queries such as 'director', which match across every "
            "account. Prefer the account name."
        ),
        "parameters": {
            "query": "string, required",
            "page": "int, default 1",
            "page_size": "int, default 5, max 25",
            "response_format": "enum: concise | detailed, default concise",
        },
    },
    {
        "name": "gdrive_files_search",
        "description": (
            "Search Google Drive file names for a keyword or phrase.\n"
            "Searches file names only, not file contents. Files are named "
            "'<Account> - <Document>', so 'Meridian Freight renewal' narrows to one "
            "account's renewal paperwork; multi-word queries match files containing all "
            "of the words in any order.\n"
            "Internal documents that belong to no customer are filed under the account "
            "'internal'.\n"
            "Returns 5 results per page with 'total_results' and 'truncated' so you know "
            "what you did not see.\n"
            "Use response_format='detailed' only if you need file_id, folder, or size."
        ),
        "parameters": {
            "query": "string, required",
            "page": "int, default 1",
            "page_size": "int, default 5, max 25",
            "response_format": "enum: concise | detailed, default concise",
        },
    },
    {
        "name": "analytics_usage_report",
        "description": (
            "Usage summary for one account over a trailing window of up to 90 days.\n"
            "'account' must be an exact account name, not a contact name and not an id. "
            "Resolve it with crm_contacts_search first if you are unsure.\n"
            "concise returns totals, averages, peak day, and a trend label -- enough to "
            "answer 'how are they doing'. detailed adds daily rows and is large; ask for "
            "it only when you need a specific date.\n"
            "For a window longer than 90 days use analytics_download_full_history."
        ),
        "parameters": {
            "account": "string, required, exact account name",
            "days": "int, default 30, range 1-90",
            "response_format": "enum: concise | detailed, default concise",
        },
    },
    {
        "name": "analytics_download_full_history",
        "description": (
            "Complete usage history for one account, rolled up by calendar month.\n"
            "Use this instead of several analytics_usage_report calls when the window you "
            "need is longer than 90 days. Returns one row per month, so it stays small "
            "no matter how long the account has existed.\n"
            "Month-level only. If you need a specific date, use analytics_usage_report "
            "with response_format='detailed'."
        ),
        "parameters": {
            "account": "string, required, exact account name",
            "response_format": "enum: concise | detailed, default concise",
        },
    },
]


def crm_contacts_search(
    query: str,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    response_format: str = "concise",
) -> dict:
    """Principle 1: a query parameter turns a table scan into a lookup."""
    terms = query.lower().split()
    matches = [
        contact
        for contact in CONTACTS
        if all(
            term in " ".join(
                [contact["name"], contact["account"], contact["title"], contact["email"]]
            ).lower()
            for term in terms
        )
    ]

    def shape(contact: dict) -> dict:
        concise = {
            "name": contact["name"],
            "title": contact["title"],
            "account": contact["account"],
            "role": contact["role"],
            "email": contact["email"],
        }
        if response_format == "concise":
            return concise
        # Principle 3: an identifier the model can carry around, not the raw uuid.
        return {
            **concise,
            "contact_id": contact["id"],
            "phone": contact["phone"],
            "last_updated": contact["updated"],
        }

    return _page(matches, page, page_size, shape, query)


def gdrive_files_search(
    query: str,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    response_format: str = "concise",
) -> dict:
    """Principle 3: 'Excel Spreadsheet', not 'application/vnd.ms-excel'."""
    terms = query.lower().split()
    matches = [
        file
        for file in FILES
        if all(term in f"{file['name']} {file['account']} {file['kind']}".lower() for term in terms)
    ]
    matches.sort(key=lambda file: file["modified"], reverse=True)

    def shape(file: dict) -> dict:
        concise = {
            "name": file["name"],
            "file_type": file["kind"],
            "account": file["account"],
            "modified": file["modified"],
        }
        if response_format == "concise":
            return concise
        return {
            **concise,
            "file_id": file["id"],
            "folder": file["folder"],
            "size": _human_size(file["size_bytes"]),
        }

    return _page(matches, page, page_size, shape, query)


def analytics_usage_report(
    account: str, days: int = 30, response_format: str = "concise"
) -> dict:
    """Principle 4: the error message is the interesting part of this function.

    Both failure modes say which parameter is wrong, what was received, and what to do
    instead. The `days` message is the one printed on the source page.
    """
    if account not in ACCOUNTS:
        suggestions = ", ".join(f"'{name}'" for name in ACCOUNTS)
        return {
            "error": (
                f"No account named '{account}'. The 'account' parameter takes an exact "
                f"account name, not a contact name or an id. Known accounts: {suggestions}. "
                f"Use crm_contacts_search to resolve a person or an id to an account name."
            )
        }

    if not 1 <= days <= MAX_USAGE_DAYS:
        return {
            "error": (
                f"The 'days' parameter must be between 1 and {MAX_USAGE_DAYS}. You "
                f"provided {days}. To get {days} days of data, break into two "
                f"{days // 2}-day queries or use analytics_download_full_history()."
            )
        }

    rows = daily_usage(account, days)
    total_calls = sum(row["api_calls"] for row in rows)
    total_errors = sum(row["errors"] for row in rows)
    peak = max(rows, key=lambda row: row["api_calls"])
    first_half = sum(row["api_calls"] for row in rows[: len(rows) // 2])
    second_half = total_calls - first_half

    report = {
        "account": account,
        "window": f"{rows[0]['date']} to {rows[-1]['date']} ({days} days)",
        "total_api_calls": total_calls,
        "avg_daily_api_calls": round(total_calls / days),
        "peak_day": {"date": peak["date"], "api_calls": peak["api_calls"]},
        "active_seats_latest": rows[-1]["active_seats"],
        "error_rate_pct": round(100 * total_errors / total_calls, 3),
        "trend": "rising" if second_half > first_half else "flat or falling",
    }

    if response_format == "concise":
        return report

    # Principle 4 again: even the detailed format caps itself and says that it did.
    shown = rows[-30:]
    report["daily"] = shown
    report["truncated"] = len(shown) < len(rows)
    report["total_results"] = len(rows)
    report["showing"] = len(shown)
    return report


def analytics_download_full_history(account: str, response_format: str = "concise") -> dict:
    """The escape hatch analytics_usage_report's error message promises."""
    if account not in ACCOUNTS:
        suggestions = ", ".join(f"'{name}'" for name in ACCOUNTS)
        return {
            "error": (
                f"No account named '{account}'. Known accounts: {suggestions}. "
                f"Use crm_contacts_search to resolve a person or an id to an account name."
            )
        }

    months = monthly_usage(account)
    history = {
        "account": account,
        "granularity": "calendar month",
        "months_returned": len(months),
        "months": months,
    }
    if response_format == "detailed":
        history["total_api_calls"] = sum(month["total_api_calls"] for month in months)
        history["total_errors"] = sum(month["total_errors"] for month in months)
    return history


def call(name: str, args: dict) -> tuple[str, bool]:
    """Dispatch a tool call and serialize the result.

    Same signature and same serializer as `tools_bad.call`, so compare.py is not quietly
    giving one side prettier JSON than the other.
    """
    handlers = {
        "crm_contacts_search": crm_contacts_search,
        "gdrive_files_search": gdrive_files_search,
        "analytics_usage_report": analytics_usage_report,
        "analytics_download_full_history": analytics_download_full_history,
    }
    handler = handlers.get(name)
    if handler is None:
        known = ", ".join(sorted(handlers))
        result: dict = {"error": f"No tool named '{name}'. Available tools: {known}."}
    else:
        try:
            result = handler(**args)
        except TypeError as problem:
            # Principle 4 applied to the dispatcher itself: name the tool, name the
            # parameters it does accept, and repeat back what arrived, so the retry is
            # informed rather than a guess.
            accepted = next(tool for tool in TOOLS if tool["name"] == name)["parameters"]
            result = {
                "error": (
                    f"Bad arguments for {name}: {problem}. Accepted parameters: "
                    f"{json.dumps(accepted)}. You sent: {json.dumps(sorted(args))}."
                )
            }

    ok = "error" not in result
    return json.dumps(result, separators=(",", ":")), ok


def _page(matches: list[dict], page: int, page_size: int, shape, query: str) -> dict:
    """Shared paging envelope: what you got, out of how many, and whether there is more."""
    page = max(1, page)
    page_size = max(1, min(page_size, 25))
    start = (page - 1) * page_size
    window = matches[start : start + page_size]
    return {
        "query": query,
        "page": page,
        "page_size": page_size,
        "total_results": len(matches),
        "showing": len(window),
        "truncated": start + len(window) < len(matches),
        "results": [shape(match) for match in window],
    }


def _human_size(size_bytes: int) -> str:
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    return f"{size_bytes / 1024:.0f} KB"


if __name__ == "__main__":
    print("tools_good.py -- three capabilities, five principles applied\n")

    contacts = crm_contacts_search("Meridian Freight")
    print('crm_contacts_search(query="Meridian Freight") ->')
    print(f"  {json.dumps(contacts, separators=(',', ':'))}\n")

    files = gdrive_files_search("Meridian Freight renewal", page_size=3)
    print('gdrive_files_search(query="Meridian Freight renewal", page_size=3) ->')
    print(f"  {json.dumps(files, separators=(',', ':'))}\n")

    print('analytics_usage_report(account="3e91c07b-5a44-4d18-b0f7", days=30) ->')
    print(f"  {analytics_usage_report('3e91c07b-5a44-4d18-b0f7', 30)['error']}\n")

    print('analytics_usage_report(account="Meridian Freight", days=120) ->')
    print(f"  {analytics_usage_report('Meridian Freight', 120)['error']}\n")

    print('analytics_download_full_history(account="Meridian Freight") ->')
    print(f"  {json.dumps(analytics_download_full_history('Meridian Freight'), separators=(',', ':'))}")
