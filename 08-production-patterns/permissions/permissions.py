"""Actuator authorization: not "is this action on the list" but "is it allowed now".

Every architecture page in this repository validates that the model's chosen action is in
`available_actions`. That check answers one question -- is this a real action -- and none
of the questions an incident review asks: at this amount, for this actor, how many times
this session, approved by whom, recorded where. The support-bot config in
`reference/01-reflex-agents-before-after.md` lists `issue_refund` as an actuator and says
nothing about who authorized it or up to what amount.

This module is the missing check. It sits in the same position in the oscillation loop
that action validation already sits in -- after the model names an action, before the
actuator runs -- and it is deterministic all the way down. It never calls a model, because
an authorization decision a model can be talked out of is not an authorization decision.

    percept -> [LLM]  choose action
            -> [DET]  action is in available_actions        the check that already exists
            -> [DET]  PermissionLayer.check                 this module
            -> actuator

Checks run in this order, and the first one to refuse names itself in the decision:

    actuator   is it in the spec at all, with a permissions block   (unknown -> deny)
    audit      if the spec demands a record, is there a sink        (missing -> deny)
    role       may this actor request this actuator at all
    bounds     amount present, numeric, positive, under the ceiling
    rate       has this session already spent its budget
    tier       autonomous, or does it need a confirmation or an approval

Authorization inputs come from the runtime. The actor comes from the session; approvals
come from an approval system that a human touched. None of them are ever read out of model
output: `ActionRequest.from_model_output` drops authorization-bearing keys from the parsed
object and reports which ones it dropped, because a prompt-injected model will cheerfully
set `approved_by` and mean nothing by it.

Fail closed is the rule everywhere. An actuator with no permissions block is denied, an
unparseable tier raises at load time rather than defaulting to permissive, and an
audit-required actuator with nowhere to write is denied rather than executed unrecorded.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent
SPEC_YAML = HERE / "actuators.yaml"
SPEC_JSON = HERE / "actuators.json"

ALLOW = "ALLOW"
ESCALATE = "ESCALATE"
DENY = "DENY"

TIERS = (
    "autonomous",
    "requires_confirmation",
    "requires_approval_above",
    "requires_human_approval",
)

WINDOW_SECONDS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}

# Substrings that make a key in model output authorization-bearing. Deliberately blunt:
# a false positive costs nothing, because the layer never reads authority from model
# output under any key name. A false negative costs everything.
AUTHORITY_STEMS = (
    "approv", "authoriz", "authoris", "permission", "override", "confirm",
    "bypass", "force", "admin", "sudo", "escalat", "sign_off", "signoff",
)


# -- loading the spec ---------------------------------------------------------------

def load_actuator_spec(yaml_path: Path = SPEC_YAML, json_path: Path = SPEC_JSON) -> dict:
    """Read the actuator spec, preferring YAML and falling back to the JSON mirror.

    pyyaml is not guaranteed on a reader's machine and this repository does not install
    anything, so the YAML file is the copy the article shows and the JSON file is the
    copy that always parses. The returned dict carries `_loaded_from` so a demo can say
    which one it got.
    """
    try:
        import yaml
    except ImportError:
        spec = json.loads(json_path.read_text(encoding="utf-8"))
        spec["_loaded_from"] = f"{json_path.name} (pyyaml is not installed)"
        return spec

    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    spec["_loaded_from"] = f"{yaml_path.name} (pyyaml {yaml.__version__})"
    return spec


def mirror_matches(yaml_path: Path = SPEC_YAML, json_path: Path = SPEC_JSON) -> bool | None:
    """Do the YAML spec and its JSON mirror describe the same rules?

    Returns None when pyyaml is missing and the question cannot be answered. Two files
    holding one set of authorization rules is a maintenance hazard, so this exists to be
    run rather than trusted.
    """
    try:
        import yaml
    except ImportError:
        return None
    from_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    from_json = json.loads(json_path.read_text(encoding="utf-8"))
    return from_yaml == from_json


@dataclass(frozen=True)
class RateLimit:
    count: int
    window: str
    seconds: float | None

    @classmethod
    def parse(cls, text: str) -> "RateLimit":
        """Parse a rate limit written the way the spec writes it: "3 per session"."""
        match = re.fullmatch(r"\s*(\d+)\s+per\s+(session|minute|hour|day)\s*", text)
        if not match:
            # Fail loudly at load time. A rate limit that silently parsed to "no limit"
            # because of a typo is the exact failure this module exists to prevent.
            raise ValueError(f"rate_limit {text!r} is not '<n> per session|minute|hour|day'")
        window = match.group(2)
        return cls(int(match.group(1)), window, WINDOW_SECONDS.get(window))

    def describe(self) -> str:
        return f"{self.count} per {self.window}"


@dataclass(frozen=True)
class ActuatorPermissions:
    """The permissions block of one actuator, after validation."""

    name: str
    tier: str
    reversible: bool
    audit: bool
    autonomous_limit: float | None = None
    hard_limit: float | None = None
    rate_limit: RateLimit | None = None
    roles: tuple[str, ...] = ()
    approver_roles: tuple[str, ...] = ()

    @property
    def amount_bounded(self) -> bool:
        return self.autonomous_limit is not None or self.hard_limit is not None

    @classmethod
    def from_entry(cls, entry: dict) -> "ActuatorPermissions":
        name = entry["name"]
        block = entry.get("permissions")
        if not block:
            raise ValueError(f"actuator {name!r} has no permissions block")

        tier = block["tier"]
        if tier not in TIERS:
            raise ValueError(f"actuator {name!r} has unknown tier {tier!r}; expected one of {TIERS}")
        if tier == "requires_approval_above" and block.get("autonomous_limit") is None:
            raise ValueError(f"actuator {name!r} is requires_approval_above with no autonomous_limit")

        rate = block.get("rate_limit")
        return cls(
            name=name,
            tier=tier,
            reversible=bool(block["reversible"]),
            audit=bool(block["audit"]),
            autonomous_limit=_as_float(block.get("autonomous_limit")),
            hard_limit=_as_float(block.get("hard_limit")),
            rate_limit=RateLimit.parse(rate) if rate else None,
            roles=tuple(block.get("roles", ())),
            approver_roles=tuple(block.get("approver_roles", ())),
        )


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


# -- what the runtime knows, and what the model says --------------------------------

@dataclass(frozen=True)
class Actor:
    """Who is asking. Comes from the session, never from the percept or the response."""

    identity: str
    role: str

    def __str__(self) -> str:
        return f"{self.identity}({self.role})"


@dataclass(frozen=True)
class Approval:
    """A grant issued by an approval system that a human touched.

    Constructed by runtime code only. Nothing in this module builds one from model
    output, and `from_model_output` cannot produce one no matter what the model writes.
    """

    actuator: str
    approver: str
    approver_role: str
    token: str
    max_amount: float = 0.0
    kind: str = "human_approval"      # or "confirmation"
    reference: str = ""               # ticket this grant was issued for; "" means any


@dataclass
class ActionRequest:
    """One action the agent wants to take, stripped of anything it may not assert."""

    actuator: str | None
    amount: float | None = None
    reference: str = ""
    fields: dict = field(default_factory=dict)
    ignored_fields: tuple[str, ...] = ()
    parse_note: str = ""

    @classmethod
    def from_model_output(cls, text: str, reference: str = "") -> "ActionRequest":
        """Turn a model response into a request, dropping every authority claim in it.

        A prompt-injected model returns exactly the object the attacker asked for,
        including `approved_by` and `permission_override`. Those keys are removed here
        and listed in `ignored_fields` so a demo can show what was thrown away. The
        layer would ignore them anyway -- it reads approvals from its own arguments --
        but removing them at the boundary means no later code can read them by accident.
        """
        parsed, note = _parse_json_object(text)
        if parsed is None:
            return cls(actuator=None, reference=reference, parse_note=note)

        clean, dropped = _strip_authority(parsed)
        amount, amount_note = _coerce_amount(clean.get("amount"))
        return cls(
            actuator=clean.get("action"),
            amount=amount,
            reference=reference,
            fields=clean,
            ignored_fields=dropped,
            parse_note=" ".join(part for part in (note, amount_note) if part),
        )


def _parse_json_object(text: str) -> tuple[dict | None, str]:
    """Parse a JSON object out of a model response, prose and code fences included."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        return (parsed, "") if isinstance(parsed, dict) else (None, "response was not an object")
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.S)
    if match:
        try:
            return json.loads(match.group(0)), "recovered the object from prose"
        except json.JSONDecodeError:
            pass
    return None, "no JSON object in the response"


def _strip_authority(obj: dict) -> tuple[dict, tuple[str, ...]]:
    clean, dropped = {}, []
    for key, value in obj.items():
        if any(stem in key.lower() for stem in AUTHORITY_STEMS):
            dropped.append(key)
        else:
            clean[key] = value
    return clean, tuple(dropped)


def _coerce_amount(value: Any) -> tuple[float | None, str]:
    if value is None:
        return None, ""
    if isinstance(value, bool):
        return None, "amount was a boolean"
    if isinstance(value, (int, float)):
        return float(value), ""
    text = str(value).replace("$", "").replace(",", "").strip()
    try:
        return float(text), f"read the amount out of {value!r}"
    except ValueError:
        return None, f"amount {value!r} is not a number"


# -- decisions and the record of them -----------------------------------------------

@dataclass
class Decision:
    """One authorization decision, with the rule that produced it.

    Mutable on purpose: `execute` sets `executed` and `result` on the object the log is
    already holding, so the log has one row per attempt rather than one per lifecycle
    event, and that row always reflects what finally happened.
    """

    sequence: int
    session: str
    actor: str
    actuator: str
    amount: float | None
    outcome: str
    rule: str
    reason: str
    needs: str = ""
    reference: str = ""
    reversible: bool | None = None
    audit_required: bool = False
    ignored_model_fields: tuple[str, ...] = ()
    grant: str = ""
    executed: bool = False
    result: str = ""
    at: float = 0.0

    def render(self, indent: str = "  ") -> str:
        """The decision as the demos print it: outcome, rule, reason, what would unblock."""
        head = f"{indent}[permission] {self.outcome:<9}{self.actuator}"
        if self.amount is not None:
            head += f" {self.amount:,.2f} USD"
        head += f"  actor={self.actor}"
        pad = indent + " " * 13
        lines = [head, f"{pad}rule={self.rule}  {self.reason}"]
        if self.ignored_model_fields:
            lines.append(
                f"{pad}ignored {len(self.ignored_model_fields)} authorization field(s) in "
                f"model output: {', '.join(self.ignored_model_fields)}"
            )
        if self.needs:
            lines.append(f"{pad}needs: {self.needs}")
        return "\n".join(lines)

    def as_record(self) -> dict:
        """The audit row. Everything a reviewer would ask for, and nothing derived."""
        return {
            "seq": self.sequence,
            "at": self.at,
            "session": self.session,
            "actor": self.actor,
            "actuator": self.actuator,
            "amount": self.amount,
            "reference": self.reference,
            "outcome": self.outcome,
            "rule": self.rule,
            "reason": self.reason,
            "needs": self.needs,
            "grant": self.grant,
            "reversible": self.reversible,
            "ignored_model_fields": list(self.ignored_model_fields),
            "executed": self.executed,
            "result": self.result,
        }


class AuditLog:
    """Append-only record of every decision, allowed or not.

    Denials are the interesting rows. An audit log that only records what happened
    cannot answer "what did it try", which is the question asked after an incident.
    """

    def __init__(self, path: Path | None = None):
        self.path = path
        self._decisions: list[Decision] = []

    def record(self, decision: Decision) -> None:
        self._decisions.append(decision)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(decision.as_record()) + "\n")

    def decisions(self) -> list[Decision]:
        return list(self._decisions)

    def render_table(self) -> str:
        """One row per attempt. Reasons are elided to keep the table scannable; the full
        text is in `as_record()`, which is what a durable sink writes."""
        header = (
            f"{'seq':>3}  {'actuator':<20}{'amount':>10}  {'outcome':<9}{'ran':<5}"
            f"{'rule':<30}reason"
        )
        rows = [header, "-" * (len(header) + 24)]
        for decision in self._decisions:
            amount = "" if decision.amount is None else f"{decision.amount:,.2f}"
            reason = decision.reason
            if len(reason) > 60:
                reason = reason[:57] + "..."
            rows.append(
                f"{decision.sequence:>3}  {decision.actuator:<20}{amount:>10}  "
                f"{decision.outcome:<9}{'yes' if decision.executed else 'no':<5}"
                f"{decision.rule:<30}{reason}"
            )
        return "\n".join(rows)


# -- the layer ----------------------------------------------------------------------

class PermissionLayer:
    """Deterministic authorization between action validation and action execution."""

    def __init__(
        self,
        spec: dict,
        session: str,
        audit_log: AuditLog | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.session = session
        self.audit_log = audit_log
        self.clock = clock
        self.permissions: dict[str, ActuatorPermissions] = {
            entry["name"]: ActuatorPermissions.from_entry(entry)
            for entry in spec["actuators"]
        }
        self.decisions: list[Decision] = []
        self._usage: dict[str, list[float]] = {}
        self._spent_grants: set[str] = set()
        self._sequence = 0

    # -- the public check ----------------------------------------------------------

    def check(
        self,
        request: ActionRequest,
        actor: Actor,
        approvals: Sequence[Approval] = (),
    ) -> Decision:
        """Decide whether `actor` may run `request` right now. Always logs, always returns."""
        self._sequence += 1
        perms = self.permissions.get(request.actuator or "")

        def decide(outcome: str, rule: str, reason: str, needs: str = "", grant: str = "") -> Decision:
            decision = Decision(
                sequence=self._sequence,
                session=self.session,
                actor=str(actor),
                actuator=request.actuator or "<none>",
                amount=request.amount,
                outcome=outcome,
                rule=rule,
                reason=reason,
                needs=needs,
                reference=request.reference,
                reversible=None if perms is None else perms.reversible,
                audit_required=bool(perms and perms.audit),
                ignored_model_fields=request.ignored_fields,
                grant=grant,
                at=self.clock(),
            )
            self.decisions.append(decision)
            if self.audit_log is not None:
                self.audit_log.record(decision)
            return decision

        # 1. Unknown actuator. Also where a parse failure lands, since a request with no
        #    action name is not a request for anything.
        if request.actuator is None:
            return decide(DENY, "actuator.unparsed", request.parse_note or "no action in model output")
        if perms is None:
            return decide(
                DENY, "actuator.unknown",
                f"{request.actuator!r} has no permissions block in the actuator spec",
            )

        # 2. An irreversible, audit-required action with nowhere to write the record does
        #    not run. Fail closed: an unrecorded refund is worse than a refused one.
        if perms.audit and self.audit_log is None:
            return decide(
                DENY, "audit.no_sink",
                f"{perms.name} requires an audit record and no audit log is attached",
                needs="an audit sink on the permission layer",
            )

        # 3. May this actor ask for this at all.
        if perms.roles and actor.role not in perms.roles:
            return decide(
                DENY, "role.not_permitted",
                f"role {actor.role!r} may not request {perms.name}; "
                f"permitted roles are {', '.join(perms.roles)}",
            )

        # 4. Value bounds. An amount-bounded actuator with no usable amount is refused
        #    rather than treated as zero.
        if perms.amount_bounded:
            if request.amount is None:
                return decide(
                    DENY, "bounds.no_amount",
                    f"{perms.name} is amount-bounded and the request carries no usable amount"
                    + (f" ({request.parse_note})" if request.parse_note else ""),
                )
            if request.amount <= 0:
                return decide(
                    DENY, "bounds.not_positive",
                    f"amount {request.amount:,.2f} is not a positive value",
                )
            if perms.hard_limit is not None and request.amount > perms.hard_limit:
                return decide(
                    DENY, "bounds.hard_limit",
                    f"{request.amount:,.2f} is over the {perms.hard_limit:,.2f} ceiling for "
                    f"{perms.name}; no in-session approval covers it",
                    needs="an out-of-band process this agent does not have",
                )

        # 5. Rate limit. Checked before the tier so an over-budget request is refused
        #    rather than sent to a human who could not authorize it anyway.
        if perms.rate_limit is not None:
            used = self._used(perms)
            if used >= perms.rate_limit.count:
                return decide(
                    DENY, "rate_limit.exhausted",
                    f"{used} of {perms.rate_limit.count} {perms.name} calls already used "
                    f"this {perms.rate_limit.window}",
                )

        # 6. Tier.
        if perms.tier == "autonomous":
            return decide(ALLOW, "tier.autonomous", f"{perms.name} is autonomous for {actor.role}")

        if perms.tier == "requires_approval_above":
            limit = perms.autonomous_limit or 0.0
            if request.amount is not None and request.amount <= limit:
                return decide(
                    ALLOW, "tier.autonomous_limit",
                    f"{request.amount:,.2f} is at or under the {limit:,.2f} autonomous limit",
                )
            over = f"{request.amount:,.2f} is over the {limit:,.2f} autonomous limit"
        elif perms.tier == "requires_human_approval":
            over = f"{perms.name} always requires a human approval"
        else:  # requires_confirmation
            over = f"{perms.name} requires a confirmation before it runs"

        grant, rejection = self._matching_grant(perms, request, approvals)
        if grant is not None:
            return decide(
                ALLOW, "tier.approved",
                f"{over}; {grant.kind} {grant.token} from {grant.approver} "
                f"({grant.approver_role}) covers it",
                grant=grant.token,
            )
        return decide(
            ESCALATE, f"tier.{perms.tier}", over + (f"; {rejection}" if rejection else ""),
            needs=self._needs_text(perms, request),
        )

    # -- check, then run -----------------------------------------------------------

    def execute(
        self,
        request: ActionRequest,
        actor: Actor,
        executor: Callable[[ActionRequest], str],
        approvals: Sequence[Approval] = (),
    ) -> Decision:
        """Authorize and, only on ALLOW, run `executor`. Returns the logged decision."""
        decision = self.check(request, actor, approvals)
        if decision.outcome != ALLOW:
            return decision

        perms = self.permissions[request.actuator]
        # Budget is spent before the side effect, not after. A refund that fails halfway
        # through must not come back with a free retry.
        if perms.rate_limit is not None:
            self._usage.setdefault(perms.name, []).append(self.clock())
        if decision.grant:
            self._spent_grants.add(decision.grant)

        decision.result = executor(request)
        decision.executed = True
        return decision

    # -- helpers -------------------------------------------------------------------

    def usage(self, actuator: str) -> str:
        """Human-readable budget line, e.g. "1 of 3 this session"."""
        perms = self.permissions[actuator]
        if perms.rate_limit is None:
            return "no rate limit"
        return f"{self._used(perms)} of {perms.rate_limit.count} this {perms.rate_limit.window}"

    def _used(self, perms: ActuatorPermissions) -> int:
        stamps = self._usage.get(perms.name, [])
        limit = perms.rate_limit
        if limit is None or limit.seconds is None:
            return len(stamps)
        cutoff = self.clock() - limit.seconds
        return sum(1 for stamp in stamps if stamp >= cutoff)

    def _matching_grant(
        self,
        perms: ActuatorPermissions,
        request: ActionRequest,
        approvals: Sequence[Approval],
    ) -> tuple[Approval | None, str]:
        """Find a grant that authorizes this exact request, or say why none does."""
        required = _required_grant(perms)
        rejections: list[str] = []
        for approval in approvals:
            if approval.actuator != perms.name:
                rejections.append(f"grant {approval.token} is for {approval.actuator}")
            elif approval.kind != required:
                rejections.append(f"grant {approval.token} is a {approval.kind}, not a {required}")
            elif approval.approver_role not in perms.approver_roles:
                rejections.append(
                    f"grant {approval.token} is from {approval.approver_role}, which may not "
                    f"approve {perms.name}"
                )
            elif approval.reference and approval.reference != request.reference:
                rejections.append(
                    f"grant {approval.token} was issued for {approval.reference}, not "
                    f"{request.reference or '<no reference>'}"
                )
            elif approval.token in self._spent_grants:
                rejections.append(f"grant {approval.token} was already used")
            elif (
                perms.amount_bounded
                and request.amount is not None
                and approval.max_amount < request.amount
            ):
                rejections.append(
                    f"grant {approval.token} covers {approval.max_amount:,.2f}, "
                    f"request is {request.amount:,.2f}"
                )
            else:
                return approval, ""
        return None, "; ".join(rejections)

    def _needs_text(self, perms: ActuatorPermissions, request: ActionRequest) -> str:
        required = _required_grant(perms)
        who = " or ".join(perms.approver_roles) or "an authorized approver"
        text = f"{required} from {who}"
        if perms.amount_bounded and request.amount is not None:
            text += f" covering {request.amount:,.2f}"
        if not perms.reversible:
            text += " (irreversible actuator: a customer confirmation is not enough)"
        return text


def _required_grant(perms: ActuatorPermissions) -> str:
    """What kind of grant satisfies this actuator's tier.

    Reversibility is load-bearing here rather than decorative: a confirmation is only
    ever enough for something that can be undone. An irreversible actuator needs a human
    approval no matter which tier it sits in.
    """
    if perms.tier == "requires_confirmation" and perms.reversible:
        return "confirmation"
    return "human_approval"


if __name__ == "__main__":
    # Not the demo -- see demo.py. This prints the table the spec compiles to, which is
    # the fastest way to see that a spec edit did what you meant, and checks that the
    # YAML and its JSON mirror still agree.
    spec = load_actuator_spec()
    print(f"Actuator spec: {spec['_loaded_from']}")
    print(f"Sessions are scoped to: {spec['session_scope']}\n")

    header = (
        f"{'actuator':<21}{'tier':<25}{'autonomous':>11}{'ceiling':>10}  "
        f"{'reversible':<11}{'rate limit':<16}audit"
    )
    print(header)
    print("-" * len(header))
    for entry in spec["actuators"]:
        perms = ActuatorPermissions.from_entry(entry)
        print(
            f"{perms.name:<21}{perms.tier:<25}"
            f"{'' if perms.autonomous_limit is None else f'{perms.autonomous_limit:,.2f}':>11}"
            f"{'' if perms.hard_limit is None else f'{perms.hard_limit:,.2f}':>10}  "
            f"{str(perms.reversible).lower():<11}"
            f"{perms.rate_limit.describe() if perms.rate_limit else 'none':<16}"
            f"{str(perms.audit).lower()}"
        )

    match = mirror_matches()
    print()
    if match is None:
        print("Mirror check skipped: pyyaml is not installed, so actuators.yaml cannot be read.")
    elif match:
        print("Mirror check: actuators.yaml and actuators.json describe the same rules.")
    else:
        print("Mirror check FAILED: actuators.yaml and actuators.json disagree.")
