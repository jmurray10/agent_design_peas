"""Fixture data shared by both tool sets.

Both `tools_bad.py` and `tools_good.py` read from here on purpose. If each file carried
its own copy of the data, `compare.py` would be measuring two different datasets and the
comparison would prove nothing. One dataset, two tool designs, is the whole point.

Nothing here is real. The account names, people, and files are invented; the usage series
is generated from a fixed formula against a fixed anchor date so that every run of
`compare.py` prints the same numbers on every machine.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# Fixed rather than date.today() so the printed output in the README stays true tomorrow.
TODAY = date(2026, 3, 31)

ACCOUNTS = ["Meridian Freight", "Halcyon Labs", "Northwind Traders", "Blue Harbor Foods"]

ACCOUNT_IDS = {
    "Meridian Freight": "b41c8f70-2d95-4e63-a118",
    "Halcyon Labs": "c92e4a11-77bd-4c05-8f3a",
    "Northwind Traders": "d17b6c88-31fa-4a92-b5e0",
    "Blue Harbor Foods": "e5a0d234-9b17-4f76-8c41",
}

# Canonical contact records. Each tool set reshapes these into its own response format;
# neither invents a field the other cannot see.
CONTACTS: list[dict] = [
    {
        "uuid": "3e91c07b-5a44-4d18-b0f7",
        "name": "Dana Whitfield",
        "title": "Operations Director",
        "email": "d.whitfield@meridianfreight.example",
        "phone": "+1-503-555-0142",
        "account": "Meridian Freight",
        "role": "account owner",
        "updated": "2026-02-14",
    },
    {
        "uuid": "9c2f18ad-4e60-41b3-9a75",
        "name": "Priya Raghunathan",
        "title": "VP Logistics",
        "email": "p.raghunathan@meridianfreight.example",
        "phone": "+1-503-555-0188",
        "account": "Meridian Freight",
        "role": "executive sponsor",
        "updated": "2026-01-09",
    },
    {
        "uuid": "51db73e9-8a2c-4f40-97be",
        "name": "Tomas Berg",
        "title": "Billing Contact",
        "email": "t.berg@meridianfreight.example",
        "phone": "+1-503-555-0119",
        "account": "Meridian Freight",
        "role": "billing",
        "updated": "2025-11-27",
    },
    {
        "uuid": "6f4a2b10-c37d-4e88-81aa",
        "name": "Alice Nakamura",
        "title": "Head of Platform",
        "email": "a.nakamura@halcyonlabs.example",
        "phone": "+1-415-555-0170",
        "account": "Halcyon Labs",
        "role": "account owner",
        "updated": "2026-03-02",
    },
    {
        "uuid": "27c9e6f3-1b48-4d59-a032",
        "name": "Marcus Oyelaran",
        "title": "Staff Engineer",
        "email": "m.oyelaran@halcyonlabs.example",
        "phone": "+1-415-555-0133",
        "account": "Halcyon Labs",
        "role": "technical",
        "updated": "2026-02-21",
    },
    {
        "uuid": "84be1d27-9f03-4a6c-b7d1",
        "name": "Rosa Delgado",
        "title": "Procurement Manager",
        "email": "r.delgado@northwindtraders.example",
        "phone": "+1-312-555-0155",
        "account": "Northwind Traders",
        "role": "account owner",
        "updated": "2026-03-11",
    },
    {
        "uuid": "b0f37c45-6e29-4b81-9d3c",
        "name": "Ken Iwasaki",
        "title": "Warehouse Systems Lead",
        "email": "k.iwasaki@northwindtraders.example",
        "phone": "+1-312-555-0126",
        "account": "Northwind Traders",
        "role": "technical",
        "updated": "2026-01-30",
    },
    {
        "uuid": "af62905d-73c1-4e07-8b44",
        "name": "Yusuf Demir",
        "title": "CFO",
        "email": "y.demir@northwindtraders.example",
        "phone": "+1-312-555-0198",
        "account": "Northwind Traders",
        "role": "billing",
        "updated": "2025-12-18",
    },
    {
        "uuid": "1d8c4e60-2af9-4713-95b8",
        "name": "Georgia Pham",
        "title": "Director of Supply",
        "email": "g.pham@blueharborfoods.example",
        "phone": "+1-206-555-0164",
        "account": "Blue Harbor Foods",
        "role": "account owner",
        "updated": "2026-03-20",
    },
    {
        "uuid": "7a03fb18-5d42-4c96-8e2f",
        "name": "Ibrahim Toure",
        "title": "Cold Chain Manager",
        "email": "i.toure@blueharborfoods.example",
        "phone": "+1-206-555-0102",
        "account": "Blue Harbor Foods",
        "role": "technical",
        "updated": "2026-02-05",
    },
    {
        "uuid": "c48e2d95-0b76-4f31-a9c7",
        "name": "Helen Vasquez",
        "title": "Accounts Payable",
        "email": "h.vasquez@blueharborfoods.example",
        "phone": "+1-206-555-0177",
        "account": "Blue Harbor Foods",
        "role": "billing",
        "updated": "2025-10-14",
    },
    {
        "uuid": "5b7f31ca-e284-4d0a-96f5",
        "name": "Owen Castellanos",
        "title": "Renewals Manager",
        "email": "o.castellanos@internal.example",
        "phone": "+1-503-555-0100",
        "account": "Meridian Freight",
        "role": "internal renewals owner",
        "updated": "2026-03-24",
    },
]

# Canonical file records. `kind` is the human-readable type, `mime` the technical one --
# the two tool sets differ in which of the pair they hand back to the model.
FILES: list[dict] = [
    {
        # This uuid and mime type are the exact pair the source page uses as its example
        # of an unhelpful tool response, and the one this example exists to show.
        "uuid": "a7f3d9e2-8c4b-4f1a-9d2e",
        "name": "Meridian Freight - Renewal Agreement 2026",
        "kind": "Excel Spreadsheet",
        "mime": "application/vnd.ms-excel",
        "size_bytes": 184320,
        "modified": "2026-03-02",
        "account": "Meridian Freight",
        "folder": "Renewals/2026",
    },
    {
        "uuid": "f28d0b91-6c37-4a15-83be",
        "name": "Meridian Freight - Master Services Agreement (signed)",
        "kind": "PDF Document",
        "mime": "application/pdf",
        "size_bytes": 921600,
        "modified": "2023-04-18",
        "account": "Meridian Freight",
        "folder": "Contracts/Signed",
    },
    {
        "uuid": "0c5a76e4-b813-4d29-97fa",
        "name": "Meridian Freight - Renewal Pricing Worksheet",
        "kind": "Excel Spreadsheet",
        "mime": "application/vnd.ms-excel",
        "size_bytes": 63488,
        "modified": "2026-03-18",
        "account": "Meridian Freight",
        "folder": "Renewals/2026",
    },
    {
        "uuid": "3fb9c2d7-405e-41a8-b6c3",
        "name": "Meridian Freight - QBR Deck Q4 2025",
        "kind": "Slide Deck",
        "mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "size_bytes": 4194304,
        "modified": "2026-01-15",
        "account": "Meridian Freight",
        "folder": "QBR/2025",
    },
    {
        "uuid": "8e147a03-d9b2-4c65-90fd",
        "name": "Meridian Freight - Support Escalation Log 2025",
        "kind": "CSV File",
        "mime": "text/csv",
        "size_bytes": 27648,
        "modified": "2026-01-04",
        "account": "Meridian Freight",
        "folder": "Support/Logs",
    },
    {
        "uuid": "d60be3f8-27a4-4019-8c7b",
        "name": "Halcyon Labs - Renewal Agreement 2026",
        "kind": "Excel Spreadsheet",
        "mime": "application/vnd.ms-excel",
        "size_bytes": 176128,
        "modified": "2026-02-27",
        "account": "Halcyon Labs",
        "folder": "Renewals/2026",
    },
    {
        "uuid": "b7250cd1-4e8f-4a37-92e6",
        "name": "Halcyon Labs - Platform Migration Plan",
        "kind": "Word Document",
        "mime": "application/msword",
        "size_bytes": 245760,
        "modified": "2026-03-09",
        "account": "Halcyon Labs",
        "folder": "Projects/Migration",
    },
    {
        "uuid": "4a9c8e26-13db-4f70-85a1",
        "name": "Halcyon Labs - Security Questionnaire (completed)",
        "kind": "PDF Document",
        "mime": "application/pdf",
        "size_bytes": 512000,
        "modified": "2025-09-22",
        "account": "Halcyon Labs",
        "folder": "Compliance",
    },
    {
        "uuid": "e3170fa5-8b64-4d92-a70c",
        "name": "Northwind Traders - Renewal Agreement 2026",
        "kind": "Excel Spreadsheet",
        "mime": "application/vnd.ms-excel",
        "size_bytes": 168960,
        "modified": "2026-03-05",
        "account": "Northwind Traders",
        "folder": "Renewals/2026",
    },
    {
        "uuid": "97c4b0e3-d251-4a86-b3f9",
        "name": "Northwind Traders - Warehouse Integration Spec",
        "kind": "Word Document",
        "mime": "application/msword",
        "size_bytes": 331776,
        "modified": "2025-12-01",
        "account": "Northwind Traders",
        "folder": "Projects/Integration",
    },
    {
        "uuid": "2b86df41-70ca-4e13-99b5",
        "name": "Northwind Traders - Invoice History 2025",
        "kind": "CSV File",
        "mime": "text/csv",
        "size_bytes": 40960,
        "modified": "2026-01-08",
        "account": "Northwind Traders",
        "folder": "Billing",
    },
    {
        "uuid": "5c03a9b7-e648-4d21-87ef",
        "name": "Blue Harbor Foods - Renewal Agreement 2026",
        "kind": "Excel Spreadsheet",
        "mime": "application/vnd.ms-excel",
        "size_bytes": 159744,
        "modified": "2026-03-21",
        "account": "Blue Harbor Foods",
        "folder": "Renewals/2026",
    },
    {
        "uuid": "76fe2c08-a934-4b57-81d2",
        "name": "Blue Harbor Foods - Cold Chain SLA Addendum",
        "kind": "PDF Document",
        "mime": "application/pdf",
        "size_bytes": 286720,
        "modified": "2026-02-11",
        "account": "Blue Harbor Foods",
        "folder": "Contracts/Addenda",
    },
    {
        "uuid": "c1d5470b-9e3a-4826-b04f",
        "name": "Blue Harbor Foods - Onboarding Checklist",
        "kind": "Word Document",
        "mime": "application/msword",
        "size_bytes": 98304,
        "modified": "2025-08-30",
        "account": "Blue Harbor Foods",
        "folder": "Onboarding",
    },
    {
        "uuid": "0947fbc2-6d18-4e35-a8b6",
        "name": "Renewal Playbook 2026 (internal)",
        "kind": "Word Document",
        "mime": "application/msword",
        "size_bytes": 122880,
        "modified": "2026-01-02",
        "account": "internal",
        "folder": "Playbooks",
    },
    {
        "uuid": "ba6e13d9-4527-4c80-91a3",
        "name": "Pricing Matrix FY26 (internal)",
        "kind": "Excel Spreadsheet",
        "mime": "application/vnd.ms-excel",
        "size_bytes": 217088,
        "modified": "2026-02-19",
        "account": "internal",
        "folder": "Playbooks",
    },
    {
        "uuid": "38f0c72e-15ab-4d64-b9c8",
        "name": "Churn Risk Model - Methodology Notes",
        "kind": "PDF Document",
        "mime": "application/pdf",
        "size_bytes": 655360,
        "modified": "2025-07-16",
        "account": "internal",
        "folder": "Analytics",
    },
    {
        "uuid": "e79b3a15-c0d4-4f28-86e1",
        "name": "Support Handbook v9",
        "kind": "PDF Document",
        "mime": "application/pdf",
        "size_bytes": 1441792,
        "modified": "2025-06-03",
        "account": "internal",
        "folder": "Support",
    },
    {
        "uuid": "1cf4d8b6-32e7-4901-a5d7",
        "name": "Q1 2026 Board Update (draft)",
        "kind": "Slide Deck",
        "mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "size_bytes": 3670016,
        "modified": "2026-03-28",
        "account": "internal",
        "folder": "Exec",
    },
    {
        "uuid": "6d2810ce-b74f-4a53-8fc0",
        "name": "Data Retention Policy 2026",
        "kind": "PDF Document",
        "mime": "application/pdf",
        "size_bytes": 204800,
        "modified": "2026-01-19",
        "account": "internal",
        "folder": "Compliance",
    },
    {
        "uuid": "9b53e7f0-8c26-4d17-a4b2",
        "name": "Meridian Freight - Renewal Call Notes 2026-03-24",
        "kind": "Word Document",
        "mime": "application/msword",
        "size_bytes": 45056,
        "modified": "2026-03-24",
        "account": "Meridian Freight",
        "folder": "Renewals/2026",
    },
    {
        "uuid": "f4c9027a-51de-4b38-90a6",
        "name": "Meridian Freight - Usage Anomaly Report Feb 2026",
        "kind": "PDF Document",
        "mime": "application/pdf",
        "size_bytes": 372736,
        "modified": "2026-03-01",
        "account": "Meridian Freight",
        "folder": "Analytics",
    },
    {
        "uuid": "27ab6e94-d305-4c7f-b812",
        "name": "Halcyon Labs - Support Escalation Log 2025",
        "kind": "CSV File",
        "mime": "text/csv",
        "size_bytes": 33792,
        "modified": "2026-01-06",
        "account": "Halcyon Labs",
        "folder": "Support/Logs",
    },
    {
        "uuid": "50e18c37-9a2b-4d64-87f3",
        "name": "Northwind Traders - QBR Deck Q4 2025",
        "kind": "Slide Deck",
        "mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "size_bytes": 3145728,
        "modified": "2026-01-21",
        "account": "Northwind Traders",
        "folder": "QBR/2025",
    },
]


# Short stable handles, assigned once at import. The source page's example of a good tool
# response uses {"id": 1, ...} rather than a uuid, so the principled tool set has something
# to hand back when the model genuinely needs an identifier for a follow-up call.
for _number, _record in enumerate(CONTACTS, start=1):
    _record["id"] = _number
for _number, _record in enumerate(FILES, start=1):
    _record["id"] = _number


def epoch(day: str) -> int:
    """Seconds since the epoch for a YYYY-MM-DD string.

    Only the unhelpful tool set uses this: raw timestamps are one of the things the
    source page calls out as spending tokens on something the model cannot reason about.
    """
    parsed = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def daily_usage(account: str, days: int, end: date = TODAY) -> list[dict]:
    """One row per day, most recent last. Deterministic, not random.

    The numbers come from a formula seeded by the account name so different accounts
    look different but any given run is reproducible.
    """
    seed = sum(ord(character) for character in account)
    rows = []
    for offset in range(days - 1, -1, -1):
        day = end - timedelta(days=offset)
        index = (day - date(2025, 1, 1)).days
        rows.append(
            {
                "date": day.isoformat(),
                "api_calls": 1800 + (index * 37 + seed) % 640,
                "active_seats": 40 + (index + seed) % 6,
                "storage_gb": round(118.0 + index * 0.08, 1),
                "errors": (index * 7 + seed) % 13,
            }
        )
    return rows


def monthly_usage(account: str, months: int = 4, end: date = TODAY) -> list[dict]:
    """Daily rows rolled up by calendar month -- the same data, two orders of magnitude
    fewer tokens. This is what a tool that respects the context window hands back."""
    rows = daily_usage(account, days=months * 31, end=end)
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["date"][:7], []).append(row)

    summary = []
    for month in sorted(buckets)[-months:]:
        entries = buckets[month]
        summary.append(
            {
                "month": month,
                "days_observed": len(entries),
                "total_api_calls": sum(entry["api_calls"] for entry in entries),
                "avg_active_seats": round(
                    sum(entry["active_seats"] for entry in entries) / len(entries), 1
                ),
                "total_errors": sum(entry["errors"] for entry in entries),
            }
        )
    return summary
