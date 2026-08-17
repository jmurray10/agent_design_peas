# Config-driven runtime

**Source:** reference/00-overview-classical-to-llm-agents.md

## The claim

The source page says the config-driven pattern is "not a framework you can install" and
that any team adopting it has to write the runtime itself. So here is the runtime, in one
file. Nine agents -- covering all five classical architectures, across insurance, fintech,
clinical research, digital health, data engineering, and the two the source pages show --
run through the same `ConfigDrivenAgent` class. They differ
in sensors, actuators, prompts, schemas, capability tier and architecture, and the only
thing that differs between the runs is the directory path handed to the constructor.

`runtime.py` contains no name of any actuator, sensor, prompt file, schema file or agent
in `agents/`, and `demo.py` proves it by searching runtime.py's string literals for all
87 of them. It searches literals rather than raw text on purpose: a method called `report`
is not agent-specific knowledge because an agent has a sensor called `report`, and a check
that constrains what an agent author may name a field enforces the coupling it exists to
forbid. That false positive appeared the moment a sixth agent arrived.

## Run it

    python 00-config-runtime/demo.py

Needs `pyyaml`, which is the only third-party dependency anywhere in this repository. It
lives here because reading YAML is the whole job. `jsonschema` is used if it is installed
and a hand-rolled key/type/enum check runs if it is not; both accept and reject the same
inputs for the schemas in `agents/`, and the first line of output says which one you got.

No API key is needed. With no backend configured the runtime replays what real models
returned to these exact prompts, one transcript per agent under `shared/transcripts/`.
Each agent's declared tier chose its model -- `claude-haiku-4-5` where a config says
`tier: small`, `claude-opus-5` where it says `frontier` -- which is the point of the tier
living in the config.

Per agent, not per file: all nine reach `llm_call` through `runtime.py`, so they originally
shared one transcript, and re-recording one agent discarded the other five. An agent is
the unit that gets recorded.

The demo drives each agent from the percepts in its own `eval/test_cases.json`. A
hardcoded percept list here would be agent-specific knowledge in the demo, and adding an
agent would mean editing it -- which is the thing this directory claims you never have to
do.

    Config-driven runtime. 9 agent directories, one ConfigDrivenAgent class.
    schema validator: jsonschema (installed)

    === uptime-triage =============================================
    driven by:    00-config-runtime/agents/uptime-triage/agent.yaml
    architecture: simple-reflex
    tier:         small
    prompts:      system
    schemas:      create_ticket, declare_incident, input_alert, page_oncall, route_to_owning_team, suppress_as_duplicate
    
    see:  {'alert_id': 'AL-3301', 'service': 'checkout-api', 'severity': 'critical', 'message': 'Payment authorisation error rate 41% over 5 minutes. Customers cannot complete orders.', 'metric': 'error_rate', 'value': 0.41, 'threshold': 0.02, 'customer_facing': True, 'owning_team': 'payments', 'open_alerts_same_service': 0, 'runbook': 'rb-payments-auth', 'oncall_team': 'payments'}
      sensor:      alert
      task prompt: (none routed)
      do:          page_oncall  {'reason': 'Payment authorization is failing at 41% error rate with direct customer impact on checkout; the payments team is on-call and can act immediately.', 'notify': 'payments'}
    
    see:  {'alert_id': 'AL-3312', 'service': 'internal-docs', 'severity': 'warning', 'message': 'TLS certificate expires in 34 days.', 'metric': 'cert_days_remaining', 'value': 34.0, 'threshold': 45.0, 'customer_facing': False, 'owning_team': 'platform', 'open_alerts_same_service': 0, 'oncall_team': 'platform'}
      sensor:      alert
      task prompt: (none routed)
      do:          create_ticket  {'reason': 'TLS certificate expires in 34 days with no customer impact; this is a planned renewal task for business hours.', 'notify': None}
    
    see:  {'alert_id': 'AL-3320', 'service': 'ml-feature-store', 'severity': 'warning', 'message': 'Nightly feature backfill job has not completed for 2 days. Model training will use stale features.', 'metric': 'job_age_hours', 'value': 51.0, 'threshold': 26.0, 'customer_facing': False, 'owning_team': 'ml-platform', 'open_alerts_same_service': 0, 'oncall_team': 'platform'}
      sensor:      alert
      task prompt: (none routed)
      do:          route_to_owning_team  {'reason': 'The ml-feature-store service is owned by ml-platform team, not the platform oncall rotation that received this alert; the 2-day backfill delay is real but not customer-facing and can wait for business hours.', 'notify': 'ml-platform'}
    
    performance, as observed by this run:
      actions dispatched: 3
        create_ticket: 1
        page_oncall: 1
        route_to_owning_team: 1
        declared metric, not measured here: pages that the on-call engineer judged necessary
        declared metric, not measured here: incidents caught before a customer reported them
        declared metric, not measured here: alerts suppressed that should not have been
    
    The same agent, handed a percept that is not a percept:
    see:  {'not_a_declared_field': 'this is not a percept for any agent here'}
      [refused] no sensor schema accepted this percept, no model called
                alert: root: 'alert_id' is a required property
    
    eval, eval/test_cases.json -- 6 of 6 cases matched:
      [pass] checkout-failing-for-customers: expected 'page_oncall', observed 'page_oncall'
      [pass] certificate-expiring-next-month: expected 'create_ticket', observed 'create_ticket'
      [pass] not-this-teams-problem: expected 'route_to_owning_team', observed 'route_to_owning_team'
      [pass] third-copy-of-one-already-open: expected 'suppress_as_duplicate', observed 'suppress_as_duplicate'
      [pass] several-services-going-together: expected 'declare_incident', observed 'declare_incident'
      [pass] severity-label-overstates-it: expected 'create_ticket', observed 'create_ticket'

    === support-bot ===============================================
    driven by:    00-config-runtime/agents/support-bot/agent.yaml
    architecture: model-based-reflex
    tier:         mid
    prompts:      escalation, reply, system
    schemas:      check_order_status, input_customer_message, input_order_lookup, issue_refund, reply_to_customer  (+ state)
    
    see:  {'message': 'Where is my order?'}
      sensor:      customer_message
      task prompt: (none routed)
      do:          request_more_info  {'message': "I'd be happy to help track your order! Could you please provide your order number so I can look up its status?"}
      state:       {'awaiting': 'order_number', 'customer_inquiry': 'order status'}
    
    see:  {'order_lookup': 'Order #7781 - delivered 3 days ago'}
      sensor:      order_lookup
      task prompt: reply
      do:          reply_to_customer  {'message': "Your order #7781 was delivered 3 days ago. If you haven't received it or notice any issues with the delivery, please let us know and we'll be happy to help further."}
      state:       {'awaiting': None, 'customer_inquiry': 'order status - resolved', 'order_number': '7781', 'order_status': 'delivered 3 days ago'}
    
    see:  {'message': 'Where is order 88214? It was due Tuesday.'}
      sensor:      customer_message
      task prompt: (none routed)
      do:          check_order_status  {'query': 'order 88214'}
      state:       {'awaiting': None, 'customer_inquiry': 'order status - order 88214, due Tuesday, checking whereabouts', 'order_number': '88214', 'order_status': 'unknown - checking'}
    
    performance, as observed by this run:
      actions dispatched: 3
        check_order_status: 1
        reply_to_customer: 1
        request_more_info: 1
        declared metric, not measured here: customer satisfaction
        declared metric, not measured here: resolution time
        declared metric, not measured here: escalation rate
    
    The same agent, handed a percept that is not a percept:
    see:  {'not_a_declared_field': 'this is not a percept for any agent here'}
      [refused] no sensor schema accepted this percept, no model called
                customer_message: root: 'message' is a required property
                order_lookup: root: 'order_lookup' is a required property
    
    does the final state still satisfy the declared state schema?
      yes
    
    eval, eval/test_cases.json -- 6 of 6 cases matched:
      [pass] vague-where-is-it: expected 'request_more_info', observed 'request_more_info'
      [pass] lookup-came-back-delivered: expected 'reply_to_customer', observed 'reply_to_customer'
      [pass] order-number-given-so-look-it-up: expected 'check_order_status', observed 'check_order_status'
      [pass] refund-inside-the-policy: expected 'issue_refund', observed 'issue_refund'
      [pass] angry-and-asking-for-a-person: expected 'escalate_to_manager', observed 'escalate_to_manager'
      [pass] thanks-that-sorted-it: expected 'close_ticket', observed 'close_ticket'

## The fence that made this look broken

Worth reading, because for a while this directory printed a runtime that appeared not to
work, and the cause was narrower than that.

`claude-haiku-4-5` returned a correct answer to every percept and wrapped each one in a
fenced code block:

    ```json
    {"action": "suck", "reason": "Current location (left) is dirty, so clean it"}
    ```

`validate_output` parsed with a bare `json.loads`, which sees a backtick in column one and
raises. The action was discarded before reaching the schema check it would have passed, so
every eval case failed with "answer was not JSON" while the model had answered correctly.
It now parses with `shared/model_json.py`, which accepts JSON inside a fence or after an
introductory sentence and still raises on anything that is not JSON at all. The refusal
path is unchanged and still reachable.

Two things about how this was found are the reason it is written up rather than quietly
patched.

It was invisible for as long as the offline responses were hand-written bare JSON. It
appeared the moment those became recordings of what a model actually sends. A fixture
written by the person writing the parser will agree with the parser.

And the first version of the fix broke the build. The explanatory comment named the agent
directory it had happened to, `demo.py`'s no-agent-specific-code check found that name in
`runtime.py`, and the run exited non-zero. That check is described below as the thing that
makes the config-driven claim mechanical rather than rhetorical; this is it doing that job
against the person maintaining it, over a comment.

The state fallback on the model-based agent is a different animal and behaves as designed:
the model returns state keys the state schema does not declare, code refuses the update,
merges the percept in by hand, and prints that it did. The run continues degraded and the
final check says so.

## Deployment: one container named peas, one service per agent

    docker compose up -d peas

    docker ps                 ->  a container called `peas`, healthy
    docker logs peas          ->  the port table, then every request
    docker exec -it peas sh   ->  a shell in it

    http://localhost:8085/docs   support-bot   Swagger UI
    http://localhost:8088/docs   uptime-triage

One container. Inside it, `serve_all.py` starts one `serve.py` per agent directory, each
listening on its own port with its own generated OpenAPI document. They are separate
processes, not one process behind path prefixes, because an agent with its own PEAS spec
and its own performance measure is a service in its own right -- and because a shared
interpreter means one agent blocking on a slow provider, exhausting memory, or raising at
import time is felt by all of them.

Ports are assigned in directory order, which is alphabetical, so `aml-alert` takes the
base port and `uptime-triage` the last one. The table is printed at start-up, so the mapping is never something you infer
from a comment that might have drifted.

Adding an agent adds a directory and the next port. It does not add a container, an image,
or a line of code. If it ever does, this directory's claim has broken.

**What happens when one agent dies.** The supervisor watches every child and exits
non-zero if any of them stops, rather than leaving a container that reports healthy while
serving less than it advertises. Verified by sending SIGKILL to one service: the supervisor
logged `uptime-triage exited with -9; stopping the rest`, the container exited, the restart
policy brought it back, and both services returned. The healthcheck polls every port for
the same reason -- a check that only probed the first one would call a half-dead container
healthy.

Without a key every agent in the container replays recordings. Add `env_file: .env` to the
service and all of them are on a live model, with nothing else changed.

**One image per agent, if they should scale and release independently.**
`Dockerfile.agent` is the base image plus a path:

    docker build -f Dockerfile.agent --build-arg AGENT=uptime-triage -t peas-uptime-triage .
    docker run --rm -p 8080:8080 peas-uptime-triage

The interesting thing about that file is how little it contains. Both agent images come out
the same size as the base, because a path is all they add. An `AGENT=` naming a directory
that does not exist fails the build with `no agent.yaml at ...`, on the grounds that a typo
should not become a crash loop.

## The API is generated from the PEAS config

`openapi.py` builds an OpenAPI 3.1 document from `agent.yaml` at start-up. Nothing is
written per agent, and there is no second place to update, so the documentation cannot
drift from the agent it documents.

The mapping is not decorative -- it is the same reading of PEAS the runtime already
performs:

| PEAS                   | OpenAPI                                              |
|------------------------|------------------------------------------------------|
| sensors                | the request body, as `oneOf` over the sensor schemas |
| actuators              | the `action` enum, and each action's argument schema |
| performance            | what the endpoint is for, in the description         |
| environment            | the description                                      |
| state                  | whether the endpoint is stateless                    |

Sensors become a `oneOf` because that is exactly how `validate_input` treats them: a
percept is accepted if any sensor schema matches. The spec says what the code does because
it was derived from the same declaration.

The two deterministic gates surface as two status codes, and that distinction is the
oscillation pattern expressed in HTTP:

    200  the model answered and the actuator schema accepted it
    422  no sensor schema accepted the percept -- the caller is wrong, and no model was
         called, so nothing was spent
    502  the model answered and the answer failed the actuator contract -- the caller is
         fine, the upstream answer is not

A server returning 500 for both would discard the only distinction that tells an operator
whose problem it is.

`serve.py` is standard library only. An HTTP framework would bring its own request
validation, and this agent already validates against the schemas its PEAS config names,
before and after the model call. A second validation layer with different rules would
obscure the one worth looking at. One thing needs saying plainly: `/docs` loads Swagger UI
from a CDN and therefore needs network access. `/openapi.json` does not, and the spec is
the artifact -- an air-gapped run still gets it.

## Adding a third agent

Everything below happens inside one new directory.

1.  Make `00-config-runtime/agents/<name>/` with three subdirectories: `prompts`,
    `schemas`, `eval`.
2.  Write `agent.yaml`: the PEAS block (performance, environment, actuators, sensors),
    plus `behavior.decision_strategy` and `behavior.tier`. Add `behavior.task_prompts` if
    more than one prompt is needed, and a `state` block if the agent is model-based --
    that block, and nothing else, is what makes the runtime keep state between turns.
3.  Write `prompts/system.md`, and any task prompts `behavior.task_prompts` routes to.
4.  Write one JSON schema per sensor that declares `input_schema` and per actuator that
    declares `output_schema`. Use only `type`, `enum`, `required`, `properties` and
    `additionalProperties`, which is what the no-dependency validator implements.
5.  Write `eval/test_cases.json`: a list of `{id, input, expect_action}`.
6.  `ConfigDrivenAgent("00-config-runtime/agents/<name>")`. With a backend configured --
    `LLM_PROVIDER`, an `ANTHROPIC_API_KEY`, or Ollama listening locally -- that is the
    last step; the agent runs.
7.  Only if you want it to run for a reader with no key: record its answers once, live.

        ANTHROPIC_API_KEY=... LLM_RECORD=1 python 00-config-runtime/demo.py

    That writes `shared/transcripts/agent__<name>.json`, keyed by the SHA-256 of each
    prompt. One file per agent, not one per script: every agent here reaches `llm_call`
    through `runtime.py`, so they originally shared a single transcript and re-recording
    one discarded the rest. There is no file of canned strings to edit and no key naming
    convention to get right. Editing a prompt afterwards changes its digest, so the next
    offline run raises with instructions to re-record rather than replaying the answer to
    a question the prompt no longer asks.

No step edits `runtime.py`. That was checked by hand rather than by a script in this
repository: `runtime.py` had an identical file hash before and after a third agent
directory was added and run. The check at the end of `demo.py` widens to cover the new
agent automatically, because it reads the names it searches for out of `agents/`.

## Notes on the code

Four, since the source listing leaves work undone. `load_tools` and `PerformanceTracker`
are named by the source but never defined; both are implemented here, and the tracker
deliberately reports only what it observed -- the declared metrics are printed as
declarations, never scored. `select_task_prompt` in the source routes hardcoded substrings
(`"pdf"`, `"review"`) onto hardcoded prompt names, which is agent-specific code in the
runtime, so the routing table moved into `agent.yaml`. `validate_input` returns the name
of the sensor whose schema matched alongside the percept, because sensors in a PEAS config
are alternatives and which one fired is worth printing. And the config files in this repo
wrap everything in a top-level `agent:` key, as the source page's YAML does, so `__init__`
unwraps it.

The two `agent.yaml` files are `01-reflex-agents/simple/config.yaml` and
`01-reflex-agents/model-based/config.yaml` with two additions: a `behavior` block, and a
`type` on each actuator. Both appear in the source page's own config and were dropped from
the copies beside those examples. Nothing else was added, and neither file contains a key
that nothing reads. A config key nothing reads is the problem this directory exists to fix.

## What changed

Nothing in the architecture. What changed is where the architecture is written down. The
LLM replaced the same component it replaced in `01-reflex-agents/` -- the rule that maps a percept
to an action -- and `decide` is the one model call per turn. Everything on both sides of
it is ordinary Python and stays that way: the percept must satisfy a declared sensor
schema before a token is spent, the answer must parse as JSON, the action must name an
actuator the config declares, the action object must satisfy that actuator's schema, and a
state update must satisfy the state schema or the config's declared fallback fires. The
difference between a simple reflex agent and a model-based one is a four-line `state` block
in YAML, not a different class.

Two more keys in that block decide what the model is allowed to touch, and they exist
because one agent here was getting it wrong. `triage-tuner` asked its model for
`outcomes_seen`, a running count of the outcomes it had been handed, inside an agent whose
own config says the reward is computed by arithmetic over what happened. Measured live over
twenty runs, the model got that count right seven times, and nothing here could tell: the
eval suite compares actions, the sequence harness compares actions, and the state schema
accepts any integer. So `counted: {outcomes_seen: outcome}` has the runtime add one every
time a percept arrives through that sensor, and `model_writable: [patterns]` drops anything
the model proposes outside the list, so a model that volunteers a count anyway cannot
overwrite the arithmetic. Both are read out of config by a runtime that still contains no
agent's name. It is the oscillation this repository argues for, reduced to four lines of
YAML: judgement to the model, addition to the adder. The capability tier is a config key too: a cheaper model
is a file edit, not a code change -- and on this run it is the config key that decided
which of the two recorded models answered each agent.

## What it costs

Indirection. A bug is now in one of five files across two directories, and reading the
code no longer tells you what the agent does. The runtime is only as good as the schemas,
and a schema nobody wrote is a validation that never runs. The parser is only as good as
the shape it expects, and a parser that was never taught to accept something ordinary is
what that costs. The fallback keeps a
run alive and degrades it. `support-bot` used to demonstrate that on every turn -- its
state schema set `additionalProperties: false` and rejected every well-formed update the
model produced, so the fallback fired constantly and the agent ran permanently degraded
while appearing to work. That schema was fixed; no fallback fires for any agent now, and
the mechanism is described here rather than shown, because manufacturing a failure to
illustrate a fallback is the kind of demonstration this repository argues against. Generality has a floor -- an
agent whose actuators need real side effects has to stop at `act` and write code again.
