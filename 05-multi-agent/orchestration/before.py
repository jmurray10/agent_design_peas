"""Classical cooperative pipeline: three hand-coded functions, run in sequence.

Source: reference/05-multi-agent-systems-before-after.md, "Cooperative: Orchestrated
Pipelines / BEFORE: Classical".

Each function is a single-purpose agent. Together they are an orchestrated pipeline with
no orchestrator -- the sequence is the caller's for-loop, and the contract between stages
is whatever the next function happens to look for.

Document one is the format the parser was written for. Document two carries the same two
facts in a different layout, and the parser does not survive it.
"""

from textwrap import indent


def extract_data(document: str) -> dict:
    data = {}
    for line in document.split('\n'):
        if 'name:' in line.lower():
            data['name'] = line.split(':')[1].strip()
        if 'amount:' in line.lower():
            data['amount'] = float(line.split(':')[1].strip().replace('$', ''))
    return data


def validate_data(data: dict) -> tuple[dict, list[str]]:
    errors = []
    if 'name' not in data:
        errors.append('missing name')
    if data.get('amount', 0) <= 0:
        errors.append('invalid amount')
    return data, errors


def generate_report(data: dict, errors: list[str]) -> str:
    if errors:
        return f"ERRORS: {errors}"
    return f"Processed: {data['name']} for ${data['amount']}"


DOCUMENT_ONE = "Name: John Smith\nAmount: $1500.00\nDate: 2024-01-15"

# Same two facts -- a payer and a total -- laid out the way a real invoice lays them out.
# Nothing here is unusual or adversarial. It is simply not the format the parser assumes.
DOCUMENT_TWO = "INVOICE\nBill to: Acme Manufacturing LLC\nTotal amount: $2,450.00\nTerms: Net 30"


if __name__ == "__main__":
    print("Classical pipeline: extract_data -> validate_data -> generate_report")
    print()

    print("Document 1 -- the format the parser was written for")
    print(indent(DOCUMENT_ONE, "    "))
    data = extract_data(DOCUMENT_ONE)
    data, errors = validate_data(data)
    print(f"  extracted: {data}")
    print(f"  errors:    {errors}")
    print(f"  report:    {generate_report(data, errors)}")
    print()

    print("Document 2 -- the same two facts, a different layout")
    print(indent(DOCUMENT_TWO, "    "))
    try:
        data = extract_data(DOCUMENT_TWO)
        data, errors = validate_data(data)
        print(f"  report:    {generate_report(data, errors)}")
    except ValueError as exc:
        # Deliberately not caught inside extract_data. The source page does not guard this
        # call and neither do we -- the unguarded crash is the limitation the article is
        # setting up, and hiding it behind a try/except inside the parser would hide it.
        print(f"  extract_data raised: {type(exc).__name__}: {exc}")
        print("  The pipeline stopped at the first stage. No report was produced.")
        print()
        print("  Two separate failures inside four lines of document:")
        print('    - "Bill to:" does not contain the substring "name:", so the payer is')
        print('      never seen at all -- the parser would have reported "missing name"')
        print('      even if it had survived the second failure')
        print('    - "Total amount:" does match, and hands "2,450.00" to float() with the')
        print('      thousands separator still in it')
        print()
        print("  Neither is a bug in the code. Both are the cost of a parser that encodes")
        print("  one document format. A third format needs a third parser.")
