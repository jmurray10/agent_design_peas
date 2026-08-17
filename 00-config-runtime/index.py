"""One page listing every agent the container is serving.

Nine services on nine ports is nine things to remember. This is the page that makes the
port table in the logs unnecessary: what each agent is, which architecture it implements,
and a link to the Swagger page generated from its own config.

Nothing here is hand-maintained per agent. The rows come from the same `agent.yaml` files
the runtime reads and the OpenAPI documents are generated from, so an agent that is added
appears here, and one whose architecture or actuators change says so without anyone
editing this file. That is the same argument the rest of `00-config-runtime/` makes,
applied to its own front door.

Standard library only, and the markup is written out rather than templated, because a
template engine would be a dependency this page does not need.
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Read once at start-up: the configs cannot change under a running container, and a page
# that re-read them per request would be slower and no more correct.
ARCHITECTURE_NOTE = {
    "simple-reflex":
        "One percept in, one action out, no memory between calls.",
    "model-based-reflex":
        "Carries state between percepts, so what it saw earlier changes what it does now.",
    "goal-based":
        "The goal is a constraint problem. A solver decides, and can prove there is no "
        "answer rather than inventing one.",
    "utility-based":
        "Weighs outcomes that cost differently in different directions, with no single "
        "correct answer available at decision time.",
    "learning":
        "Improves from the outcomes of its own actions, and has an action for buying "
        "information it does not have.",
}


def collect(agents_root: Path, base_port: int) -> list[dict]:
    """One row per agent directory, in the order the ports were assigned."""
    import yaml

    rows = []
    directories = sorted(d for d in agents_root.iterdir() if (d / "agent.yaml").is_file())
    for offset, directory in enumerate(directories):
        config = yaml.safe_load((directory / "agent.yaml").read_text(encoding="utf-8"))["agent"]
        rows.append({
            "name": config["name"],
            "architecture": config["architecture"],
            "port": base_port + offset,
            "tier": config.get("behavior", {}).get("tier", "default"),
            "stateful": "state" in config,
            "actuators": [a["name"] for a in config.get("actuators", [])],
            "sensors": [s["name"] for s in config.get("sensors", [])],
            "metrics": config.get("performance", {}).get("metrics", []),
        })
    return rows


def render(rows: list[dict]) -> str:
    architectures = {}
    for row in rows:
        architectures.setdefault(row["architecture"], []).append(row)

    parts = [
        "<!doctype html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>peas: agents being served</title>",
        "<style>",
        "  :root { color-scheme: light dark; }",
        "  body { font: 15px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;",
        "         max-width: 62rem; margin: 2.5rem auto; padding: 0 1.25rem; }",
        "  h1 { font-size: 1.35rem; margin-bottom: 0.3rem; }",
        "  h2 { font-size: 1rem; margin: 2rem 0 0.15rem; }",
        "  p.note { opacity: 0.75; margin: 0.15rem 0 0.9rem; }",
        "  table { border-collapse: collapse; width: 100%; margin-bottom: 0.5rem; }",
        "  td, th { text-align: left; padding: 0.35rem 0.7rem 0.35rem 0; vertical-align: top; }",
        "  th { border-bottom: 1px solid currentColor; opacity: 0.6; font-weight: normal; }",
        "  tr + tr td { border-top: 1px solid rgba(128,128,128,0.25); }",
        "  code { opacity: 0.8; }",
        "  .small { font-size: 0.85em; opacity: 0.7; }",
        "</style></head><body>",
        "<h1>peas</h1>",
        f"<p class='note'>{len(rows)} agents, one process each, one container. Every page "
        "linked below is generated from that agent's <code>agent.yaml</code> at start-up, "
        "so it cannot describe an agent the runtime is not running.</p>",
    ]

    for architecture in sorted(architectures):
        group = architectures[architecture]
        note = ARCHITECTURE_NOTE.get(architecture, "")
        parts.append(f"<h2>{html.escape(architecture)}</h2>")
        if note:
            parts.append(f"<p class='note'>{html.escape(note)}</p>")
        parts.append("<table><tr><th>agent</th><th>port</th><th>what it decides</th>"
                     "<th>tier</th><th>docs</th></tr>")
        for row in group:
            actions = ", ".join(row["actuators"])
            state = " &middot; carries state" if row["stateful"] else ""
            parts.append(
                "<tr>"
                f"<td>{html.escape(row['name'])}</td>"
                f"<td>{row['port']}</td>"
                f"<td>{html.escape(actions)}<span class='small'>{state}</span></td>"
                f"<td>{html.escape(row['tier'])}</td>"
                f"<td><a href='http://localhost:{row['port']}/docs'>/docs</a></td>"
                "</tr>"
            )
        parts.append("</table>")

    parts += [
        "<h2>trying one</h2>",
        "<p class='note'>Open any <code>/docs</code> above and use the Examples dropdown: "
        "the request examples are that agent's own evaluation cases, so each one is a "
        "percept it is asserted to handle. Or from a shell:</p>",
        "<pre><code>curl -s localhost:8080/act -H 'content-type: application/json' \\",
        "     -d '{\"not_a_declared_field\": 1}'</code></pre>",
        "<p class='note'>That returns 422. No sensor schema accepts it, so the request is "
        "refused before a model is called and nothing is spent. A 200 means the model "
        "answered and the actuator contract accepted the answer; a 502 means the model "
        "answered and the contract refused it. The three codes are the two deterministic "
        "gates either side of the model call.</p>",
        "<p class='small'>With no key configured every agent here replays what a real "
        "model returned to that exact prompt. Nothing invents a response.</p>",
        "</body></html>",
    ]
    return "\n".join(parts)


def make_handler(rows: list[dict]):
    page = render(rows).encode("utf-8")
    payload = json.dumps(rows, indent=2).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/agents.json":
                body, kind = payload, "application/json"
            elif path in ("/", "/index.html"):
                body, kind = page, "text/html; charset=utf-8"
            else:
                self.send_error(404, "only / and /agents.json are served here")
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - base class name
            # The agents log their own requests. An index that logged every hit would
            # bury them.
            return

    return Handler


def serve(agents_root: Path, base_port: int, index_port: int) -> HTTPServer:
    rows = collect(Path(agents_root), base_port)
    return HTTPServer(("0.0.0.0", index_port), make_handler(rows))
