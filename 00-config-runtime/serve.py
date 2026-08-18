"""Serve every PEAS agent from one process, each with its own generated OpenAPI document.

    python 00-config-runtime/serve.py                  every agent under agents/
    python 00-config-runtime/serve.py agents/uptime-triage  just that one, mounted at /

    GET  /                       the agents this runtime is serving
    GET  /health                 liveness, agent list, and which backend is configured
    GET  /<agent>/openapi.json   the spec, generated from that agent's agent.yaml
    GET  /<agent>/docs           Swagger UI for it
    GET  /<agent>/agent          the PEAS config being served
    POST /<agent>/act            one percept in, one action out

One process, one image, one container, many agents. That is the same claim
`00-config-runtime/` makes in Python -- one `ConfigDrivenAgent` class drives every
directory -- carried through to deployment. Adding an agent adds a directory and a URL
prefix. It does not add a process, an image, a port, or a line of code.

Standard library only. An HTTP framework would bring its own request validation, and these
agents already validate against the schemas their PEAS configs name, in `runtime.py`,
before and after the model call. A second validation layer with different rules would make
the interesting one harder to see, and the interesting one is the point of the directory.

The two deterministic gates surface as two status codes, which is the oscillation pattern
expressed in HTTP:

    422  no sensor schema accepted the percept. The caller is wrong. No model was called
         and nothing was spent.
    502  the model answered and the answer failed the actuator contract. The caller is
         fine; the upstream answer is not.

A server that returned 500 for both would be throwing away the only distinction that tells
an operator whose problem it is.

There is a third, and it belongs to how this repository runs rather than to the
architecture:

    503  the percept is valid and there is no recorded answer for it, on a deployment
         with no backend configured. Nobody is wrong. The server will not invent a
         response, so it says what it would need instead.

That one exists because the first thing anyone does with a live endpoint is post their
own JSON, which is a prompt no recording has ever seen. Without it the handler raised,
the connection closed with no status and no body, and a visitor's reasonable first
request looked like a broken server.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from openapi import build_spec  # noqa: E402
from runtime import ActionRejected, ConfigDrivenAgent, PerceptRejected  # noqa: E402

# Swagger UI is loaded from a CDN. Said plainly rather than discovered: /docs needs network
# access and /openapi.json does not. The spec is the artifact; the UI is a convenience for
# reading it, and an air-gapped run still gets the spec.
SWAGGER_CDN = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5"

# How many named sessions one agent keeps at once. Small on purpose: this is a demo
# endpoint, and the number exists so that inventing session ids cannot hold memory open.
MAX_SESSIONS = 64

DOCS_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="{cdn}/swagger-ui.css"></head>
<body><div id="swagger-ui"></div>
<script src="{cdn}/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({{url:'{spec_url}',dom_id:'#swagger-ui'}});</script>
</body></html>"""


class Mounted:
    """One agent, its generated spec, and the prefix it answers on."""

    def __init__(self, prefix: str, agent_dir: Path):
        self.prefix = prefix
        self.agent_dir = agent_dir
        self.agent = ConfigDrivenAgent(agent_dir)
        self.spec = build_spec(agent_dir)
        self.name = self.agent.config["name"]
        self.architecture = self.agent.config["architecture"]
        # The generated spec describes paths relative to the agent. Mounted under a prefix,
        # the paths a caller actually calls are prefixed too, so the document has to say
        # so -- a spec whose paths do not resolve is worse than no spec.
        if prefix:
            self.spec = dict(self.spec)
            self.spec["paths"] = {f"{prefix}{p}": v for p, v in self.spec["paths"].items()}

        # One agent per named session. See `for_request`.
        self._sessions: dict[str, ConfigDrivenAgent] = {}
        self._lock = threading.Lock()

    def for_request(self, session: str | None) -> ConfigDrivenAgent:
        """The agent instance this request should run on.

        A stateful agent carries what it learned into the next prompt. Served from one
        long-lived instance, that state is process-wide and permanent: every caller shares
        one conversation, and the second caller's answer is shaped by the first caller's
        order number. The configs here already say what the scope should be --
        per-conversation, per-investigation, per-session -- and the server was honouring
        none of them.

        It also broke the documentation. The examples in each agent's Swagger page are its
        eval cases, which were recorded with state reset between them; replayed in sequence
        against one accumulating instance, the fourth example builds a prompt no recording
        has ever seen, and a reader clicking Try it out gets a 503 on a request the page
        told them to make. Eight of the forty-nine examples did exactly that.

        So: no session named, no shared state. Construction is about 9ms, which is nothing
        beside a model call, and it makes the default case both correct and thread-safe --
        two concurrent requests never touch one object. Name a session with the
        `X-Session-Id` header to get the multi-turn behaviour back, which is what the
        sequence claims in `sequence_eval.py` are about.
        """
        if not session:
            return ConfigDrivenAgent(self.agent_dir)
        with self._lock:
            agent = self._sessions.get(session)
            if agent is None:
                # Bounded, so a public endpoint cannot be made to hold memory open by
                # inventing session ids. Oldest out first; a dropped session starts fresh
                # rather than erroring, which is the behaviour a caller can recover from.
                if len(self._sessions) >= MAX_SESSIONS:
                    self._sessions.pop(next(iter(self._sessions)))
                agent = self._sessions[session] = ConfigDrivenAgent(self.agent_dir)
            return agent


class AgentHandler(BaseHTTPRequestHandler):
    mounts: dict[str, Mounted] = {}
    server_version = "peas-runtime"
    sys_version = ""

    # -- routing ---------------------------------------------------------------------

    def _resolve(self, path: str) -> tuple[Mounted | None, str]:
        """Split a request path into the agent it addresses and the route within it."""
        if "" in self.mounts:                       # single-agent mode, mounted at root
            return self.mounts[""], path
        head, _, rest = path.lstrip("/").partition("/")
        mount = self.mounts.get(f"/{head}")
        return (mount, "/" + rest) if mount else (None, path)

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            self._health()
            return
        if path == "/" and "" not in self.mounts:
            self._index()
            return

        mount, route = self._resolve(path)
        if mount is None:
            self._json(404, {"error": "not_found",
                             "reason": f"no agent serves {path}",
                             "detail": {"agents": sorted(m.name for m in self.mounts.values())}})
            return

        route = route.rstrip("/") or "/"
        if route == "/openapi.json":
            self._json(200, mount.spec)
        elif route == "/docs":
            body = DOCS_PAGE.format(title=mount.spec["info"]["title"], cdn=SWAGGER_CDN,
                                    spec_url=f"{mount.prefix}/openapi.json")
            self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/agent":
            self._json(200, mount.agent.config)
        elif route == "/":
            self._json(200, {"agent": mount.name,
                             "architecture": mount.architecture,
                             "docs": f"{mount.prefix}/docs",
                             "spec": f"{mount.prefix}/openapi.json",
                             "act": f"POST {mount.prefix}/act"})
        elif route == "/act":
            # The route exists; the method does not. 404 here would say this agent has no
            # /act, which is the one thing it certainly does have, and would send someone
            # looking for a typo instead of at their verb. The same distinction the 422
            # and 502 codes make: say whose problem it is.
            self._send_405("POST", path)
        else:
            self._json(404, {"error": "not_found", "reason": f"no route {path}"})

    def _send_405(self, allow: str, path: str) -> None:
        body = json.dumps({
            "error": "method_not_allowed",
            "reason": f"{path} exists and does not accept {self.command}",
            "detail": {"allow": allow},
        }, indent=2).encode("utf-8")
        self.send_response(405)
        self.send_header("Allow", allow)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        mount, route = self._resolve(path)
        if mount is not None and route.rstrip("/") in ("/openapi.json", "/docs", "/agent", "/"):
            self._send_405("GET", path)
            return
        if mount is None or route.rstrip("/") != "/act":
            self._json(404, {"error": "not_found", "reason": f"no route {path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            percept = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as broken:
            self._json(400, {"error": "bad_json", "reason": str(broken)})
            return
        if not isinstance(percept, dict):
            self._json(400, {"error": "bad_json", "reason": "body must be a JSON object"})
            return

        # No session named means no shared state: this request gets its own agent. See
        # Mounted.for_request.
        agent = mount.for_request(self.headers.get("X-Session-Id"))
        try:
            self._json(200, agent.run(percept))
        except PerceptRejected as refusal:
            # Every sensor schema that rejected it comes back, because "invalid" without
            # saying what it was measured against is not an error a caller can act on.
            self._json(422, {"error": "percept_rejected",
                             "reason": "no sensor schema accepted this percept, no model was called",
                             "detail": refusal.errors})
        except ActionRejected as refusal:
            self._json(502, {"error": "action_rejected",
                             "reason": str(refusal),
                             "detail": {"errors": getattr(refusal, "errors", None) or []}})
        except LookupError as unrecorded:
            # A valid percept with no recording behind it. shared/transcript.py raises
            # rather than answering, which is the rule this repository is built on, and
            # over HTTP that has to arrive as a status rather than as a dropped socket.
            self._json(503, {
                "error": "no_recorded_response",
                "reason": "this percept is valid, and no backend is configured, and no "
                          "recording exists for the prompt it produces. Nothing is "
                          "invented here, so there is no answer to return.",
                "detail": {
                    "how_to_get_an_answer": [
                        "set a backend key (see .env.example) and this agent will call a "
                        "model for real",
                        "or post one of the percepts in this agent's eval/test_cases.json, "
                        "which are recorded",
                    ],
                    "transcript_error": str(unrecorded).splitlines()[0],
                },
            })

    # -- shared routes ---------------------------------------------------------------

    def _index(self) -> None:
        self._json(200, {
            "runtime": "one ConfigDrivenAgent class, one process, one container",
            "agents": [
                {"name": m.name, "architecture": m.architecture, "prefix": m.prefix,
                 "docs": f"{m.prefix}/docs", "spec": f"{m.prefix}/openapi.json",
                 "act": f"POST {m.prefix}/act"}
                for m in sorted(self.mounts.values(), key=lambda m: m.name)
            ],
        })

    def _health(self) -> None:
        import shared.llm as shim

        self._json(200, {"status": "ok",
                         "provider": shim._select_provider(),
                         "agents": sorted(m.name for m in self.mounts.values())})

    # -- plumbing --------------------------------------------------------------------

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, indent=2).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - name fixed by base class
        # One line per request on stdout, which is where a container logs.
        sys.stdout.write("  %s %s\n" % (self.address_string(), format % args))
        sys.stdout.flush()


def discover(agents_root: Path) -> list[Path]:
    """Every directory under `agents_root` that declares an agent."""
    return sorted(d for d in agents_root.iterdir() if (d / "agent.yaml").is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("agent_dir", nargs="?",
                        help="serve one agent at / instead of every agent under agents/")
    parser.add_argument("--agents-root", default=str(HERE / "agents"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if args.agent_dir:
        AgentHandler.mounts = {"": Mounted("", Path(args.agent_dir))}
    else:
        found = discover(Path(args.agents_root))
        if not found:
            raise SystemExit(f"no agent.yaml under {args.agents_root}")
        AgentHandler.mounts = {f"/{d.name}": Mounted(f"/{d.name}", d) for d in found}

    print(f"peas runtime, {len(AgentHandler.mounts)} agent(s), one process")
    for mount in sorted(AgentHandler.mounts.values(), key=lambda m: m.name):
        schemas = len(mount.spec["components"]["schemas"])
        where = mount.prefix or "/"
        print(f"  {mount.name:<14} {mount.architecture:<20} {where:<16} "
              f"{schemas} schemas generated from agent.yaml")
    print(f"  listening on http://{args.host}:{args.port}/")
    ThreadingHTTPServer((args.host, args.port), AgentHandler).serve_forever()


if __name__ == "__main__":
    main()
