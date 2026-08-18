# Tool design: the same three capabilities, twice

**Source:** reference/08-support-context-tools-production.md, the Tool Design section

## The claim

Tool design is not cosmetic. Two tool sets that expose identical capabilities over
identical data, differing only in the five principles from the source page, do not cost
the same to use: the unhelpful set makes the agent read whole tables to find one row,
hands back identifiers it cannot resolve, and answers two different mistakes with the same
five-word error, so the agent guesses. Run `compare.py` and the difference shows up as
tokens and calls rather than as an opinion.

## Run it

    python 08-production-patterns/tools/tools_bad.py
    python 08-production-patterns/tools/tools_good.py
    python 08-production-patterns/tools/compare.py

`tools_bad.py` and `tools_good.py` each call all of their own capabilities once and print
the raw responses. `compare.py` runs the agent task against both.

    $ python 08-production-patterns/tools/compare.py

    --- unhelpful tools -----------------------------------------
      1. ok    list_contacts()
               727 tokens  [{"id":"3e91c07b-5a44-4d18-b0f7","cn":"Dana Whitfield","eml":"d.whitfield@meridianfreight.exampl ...
      2. ok    get_files()
              1235 tokens  [{"id":"a7f3d9e2-8c4b-4f1a-9d2e","name":"Meridian Freight - Renewal Agreement 2026","mime_type": ...
      3. ERROR data(customer='Meridian Freight', days=120)
                 6 tokens  "Error: Invalid input"
      4. ERROR data(customer='b41c8f70-2d95-4e63-a118', days=120)
                 6 tokens  "Error: Invalid input"
      5. ERROR data(customer='Meridian Freight Inc', days=120)
                 6 tokens  "Error: Invalid input"
      6. final answer

    --- principled tools ----------------------------------------
      1. ok    crm_contacts_search(query='Meridian Freight', response_format='concise')
               179 tokens  {"query":"Meridian Freight","page":1,"page_size":5,"total_results":4,"showing":4,"truncated":fal ...
      2. ok    gdrive_files_search(query='Meridian Freight renewal', response_format='concise')
               135 tokens  {"query":"Meridian Freight renewal","page":1,"page_size":5,"total_results":3,"showing":3,"trunca ...
      3. ok    analytics_download_full_history(account='Meridian Freight', response_format='concise')
               129 tokens  {"account":"Meridian Freight","granularity":"calendar month","months_returned":4,"months":[{"mon ...
      4. final answer

                                         unhelpful tools  principled tools
    ----------------------------------------------------------------------
    tool definitions (tokens)                         28               688
    tool responses (tokens)                         1980               443
    prompt tokens sent, all turns                   9615              4285
    model calls                                        6                 4
    tool calls                                         5                 3
    tool calls returning an error                      3                 0
    JSON parses needing recovery                       0                 0
    ----------------------------------------------------------------------
    On this run, the unhelpful set sent 2.2x the prompt tokens of the principled set.

Those are the numbers this repo printed on the machine it was written on, against the
fixtures in `records.py`, replaying the trajectory described above. They are not a
published figure and they are not a prediction about your data.

**The unhelpful column did not finish the task.** Its final answer reports the account
owner and the renewal paperwork and then reports: "Usage summary (last 120 days): Unable
to retrieve — the usage `data` tool returned "Error: Invalid input" for every customer
identifier tried (account name, org ID, and variant name)." That is the whole argument about error
messages, made by the model rather than about it: told only that its input was invalid,
it tried the account name, then an org id, then a variant of the name, then wrote a
paragraph explaining that it had failed. The correct value was the account name with a
`days` under 90, and nothing it was shown said so.

**The principled column never tripped the `days` limit in this run.** It went straight to
`analytics_download_full_history` and got monthly rollups, so the source page's exemplary
error message does not appear in this trajectory at all. `python tools_good.py` calls it
directly if you want to read it:

    analytics_usage_report(account="Meridian Freight", days=120) ->
      The 'days' parameter must be between 1 and 90. You provided 120. To get 120 days of
      data, break into two 60-day queries or use analytics_download_full_history().

That the model routed around the mistake instead of making it is one run's worth of
evidence, not proof that a good description prevents a bad call.

## What a replay shows, and what it does not

Read this before quoting any number out of `compare.py`.

With no backend configured, `compare.py` replays a recorded run. Every tool call in both
columns below was chosen by `claude-sonnet-5` on 2026-08-04, from the prompts and tool
definitions in this directory, and stored in
`shared/transcripts/08_production_patterns__tools__compare.json`. Nobody scripted the
three failed `data()` calls in the unhelpful column. A real model made them, in that
order, and then gave up.

What that is: one run, one date, one model. Model versions move and sampling varies, so
a live run today may pick different arguments, recover where this one did not, or fail
somewhere else. Treat the trajectory as evidence, not as a constant.

What it is not: an authored story. The distinction matters because this repository used
to ship the story. Under the old hand-written mocks the call counts were decided by
whoever wrote them, which made the comparison an argument about what a model would
plausibly do. It is now a record of what one did.

The mechanical rows are true in either mode, and were true before:

- the size of the actual tool definitions, rendered the way they would be sent
- the size of the actual tool responses, produced by the actual functions in this
  directory against the same fixture data
- both counted by one estimator, on both sides, with one serializer

One more caveat on the token numbers: tokens are estimated at four characters each,
because a real tokenizer would be a dependency and this repo installs nothing. Absolute
counts are approximations. The ratio between two columns produced by the same estimator is
the part worth reading.

## The three capabilities

Both files implement the same three jobs against the same records in `records.py`. One
dataset, two designs — otherwise the comparison would be measuring fixtures.

| Capability | Unhelpful | Principled |
|---|---|---|
| Find a person | `list_contacts()` | `crm_contacts_search(query, page, page_size, response_format)` |
| Find a document | `get_files()` | `gdrive_files_search(query, page, page_size, response_format)` |
| Fetch usage data | `data(customer, days)` | `analytics_usage_report(account, days, response_format)` and `analytics_download_full_history(account, response_format)` |

The principled set lists four tools for three capabilities. Usage data gets a windowed
report and a bulk-export path because the source page's example of a good error message
points the agent at exactly such an escape hatch, and an error message that names a tool
which does not exist is worse than no message at all. The extra entry costs the principled
column tokens in the tool-definitions row. It does not save it any.

## The five principles

| Principle | Unhelpful set | Principled set |
|---|---|---|
| 1. Choose the right tools | No query parameter anywhere. The agent reads all 12 contacts and all 24 files to find one of each. | A query parameter turns a table scan into a lookup. |
| 2. Namespace your tools | `list_contacts`, `get_files`, `data`. Add a second CRM and the names collide. | `crm_`, `gdrive_`, `analytics_` — service, resource, verb. |
| 3. Return meaningful context | Uuids, MIME types, epoch timestamps, abbreviated keys, and the account as a foreign key rather than a name. | Names, human-readable file types, ISO dates. Identifiers only in `response_format="detailed"`, and short handles rather than uuids. |
| 4. Optimize for token efficiency | No pagination, no truncation indicator, 90 raw daily rows that repeat the account name on every row, and `Error: Invalid input` for two unrelated failures. | `page` / `page_size`, `total_results` and `truncated` on every result set, monthly rollups, and errors that state the constraint, the value received, and the way forward. |
| 5. Prompt-engineer descriptions | One line each, restating the function name. | Query syntax, domain vocabulary, what the tool is not for, and the mistake to avoid. |

The `days` error message in `tools_good.py` is the one printed on the source page,
verbatim except for one word: the source ends it with `download_full_history()`, and this
implementation names the namespaced tool it actually ships, `analytics_download_full_history()`.
Principle 2 and a working escape hatch beat quoting the sentence exactly.

## What the comparison actually shows

The interesting row is the first one. Prompt-engineered descriptions make the principled
tool definitions roughly twenty-five times larger than the terse ones, and those
definitions are re-sent on every single turn. The principled set still finishes the task
having sent well under half the prompt tokens, because tool *responses* are where the
context goes — the unhelpful set spent more on one call to `get_files()` (1235 tokens)
than the principled set spent on all three of its tool responses put together (443).

That is the argument worth carrying into a design review: a description is charged per
turn, a response is charged per turn for the rest of the run, and a retry loop caused by
an uninformative error is charged for both. In this run it was also charged for an answer
that did not contain the thing the task asked for.

Two published figures from the source page, cited as published figures and not measured
here: Anthropic reports that a concise response format uses 65% fewer tokens than a
detailed one, and that API-side context editing produced an 84% token reduction in long
sessions. Neither number was produced by this repo.

## What changed

Nothing about the agent loop. The same model, the same task, the same underlying records:
the only variable is how the three capabilities are described and what their responses
look like. That is the point of the pair -- if tool design were cosmetic, swapping one
tool set for the other would move nothing, and it moves prompt tokens, call counts, and
whether the final answer contains what the task asked for.

The model chooses which tool to call. The tools themselves are ordinary deterministic
Python over an in-memory store, and both sets read the same rows. A capability is not
better because it was described better; it is cheaper to use, and cheaper is what the
comparison measures.

## What it costs

Almost nothing, and that is the point — the principled set is the same three capabilities
with more thought in the signatures. The real costs are elsewhere. Pagination means the
agent can miss results it never asked for a second page of. Concise responses mean it
sometimes has to call again for an id. Namespaced names are a migration once other code
already calls the old ones. And prompt-engineered descriptions are prose in a code file:
they rot when the tool changes and nobody notices, because nothing fails when a
description lies.
