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
        "Looks at one thing and decides. It remembers nothing, so the same input always "
        "gets the same treatment. Start here.",
    "model-based-reflex":
        "Remembers what it has already seen, so the answer to the same question can "
        "differ depending on what came before it.",
    "goal-based":
        "Has a goal and works out how to reach it. The plan comes from a solver rather "
        "than from the model, so when a request cannot be met at all it can say so and "
        "be right, instead of inventing a schedule that does not work.",
    "utility-based":
        "Chooses when there is no right answer -- only options that cost differently, "
        "and where being wrong in one direction costs more than the other.",
    "learning":
        "Gets better from how its own decisions turned out. It can also deliberately "
        "choose a worse option now to find something out that helps later.",
}

# The order the textbook introduces them in, which is also simplest first. Sorting these
# alphabetically put goal-based at the top, so the first agent anyone met was a clinical
# trial scheduler -- the narrowest domain here and the least obvious architecture. That is
# an accident of the letter g, and a reader arriving cold deserves better than an accident.
ARCHITECTURE_ORDER = [
    "simple-reflex", "model-based-reflex", "goal-based", "utility-based", "learning",
]


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

    ordered = ([a for a in ARCHITECTURE_ORDER if a in architectures]
               + sorted(a for a in architectures if a not in ARCHITECTURE_ORDER))
    for architecture in ordered:
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
        "<p class='note'>Pick any <code>/docs</code> link above. Press <b>Try it out</b>, "
        "choose an entry from the <b>Examples</b> dropdown, press <b>Execute</b>. Those "
        "examples are real inputs that agent is meant to handle, so you do not have to "
        "invent one. You need no API key and no network.</p>",
        "<p class='note'>Then try giving one something it does not handle:</p>",
        "<pre><code>curl -s localhost:8080/act -H 'content-type: application/json' \\",
        "     -d '{\"not_a_declared_field\": 1}'</code></pre>",
        "<p class='note'>You get a <b>422</b>, and no model was asked. That is the part "
        "worth noticing: the agent checked the request against the shapes it accepts and "
        "refused it in ordinary code, before spending anything. A <b>200</b> means a model "
        "answered <i>and</i> the answer passed a second check on the way out. A <b>502</b> "
        "means the model answered and that second check refused it.</p>",
        "<p class='note'>So a model decides one thing here, in the middle, and code decides "
        "what it may be asked and what it may answer. That is the whole argument, and these "
        "nine agents are it running rather than described.</p>",
        "<p class='small'>With no key configured every agent replays what a real model "
        "returned to that exact request, on a recorded date. Nothing invents a response.</p>",
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
