"""Run every PEAS agent as its own service, inside one container.

    python 00-config-runtime/serve_all.py

One container named `peas`. Inside it, one process per agent, each listening on its own
port, each with its own generated OpenAPI document. `docker exec peas ps` shows them; a
crash in one does not take the others down, and the supervisor exits non-zero if any of
them dies, so the container's health reflects the health of every agent in it.

This is deliberately not the same thing as one process serving several agents under path
prefixes. That version shares an interpreter: one agent exhausting memory, blocking on a
slow provider, or raising at import time is felt by all of them. Separate processes make
each agent a service in its own right, which is what an agent with its own PEAS spec and
its own performance measure already is on paper.

Ports are assigned in directory order from --base-port, and the table is printed at
start-up so the mapping is never something you have to infer.

Splitting one container into several processes is not the usual Docker advice, and the
usual advice assumes the processes are unrelated. These are one runtime driving one
directory each, deployed and versioned together. `Dockerfile.agent` builds a genuinely
separate image per agent for when they should be scaled and released independently; this
is for when they should not.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import index  # noqa: E402 - both imports need HERE on sys.path first
from serve import discover  # noqa: E402


class Service:
    def __init__(self, agent_dir: Path, port: int):
        self.agent_dir = agent_dir
        self.name = agent_dir.name
        self.port = port
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, str(HERE / "serve.py"), str(self.agent_dir),
             "--port", str(self.port), "--host", "0.0.0.0"],
            stdout=sys.stdout, stderr=sys.stderr,
        )

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--agents-root", default=str(HERE / "agents"))
    parser.add_argument("--base-port", type=int, default=8080)
    # One port below the agents, so the index sits immediately before them and the block
    # stays contiguous however many agents there turn out to be.
    parser.add_argument("--index-port", type=int, default=8079)
    args = parser.parse_args()

    found = discover(Path(args.agents_root))
    if not found:
        raise SystemExit(f"no agent.yaml under {args.agents_root}")

    services = [Service(d, args.base_port + i) for i, d in enumerate(found)]

    print(f"peas: {len(services)} agent(s), one service each, one container")
    print(f"  start here      :{args.index_port}  http://localhost:{args.index_port}/")
    for service in services:
        print(f"  {service.name:<16} :{service.port}  "
              f"http://localhost:{service.port}/docs")
    print(flush=True)

    # The index is a thread rather than a tenth process: it reads the same configs at
    # start-up and then serves one static page, so there is nothing in it worth
    # supervising, and a dead index should not fail a container whose agents are fine.
    index_server = index.serve(Path(args.agents_root), args.base_port, args.index_port)
    threading.Thread(target=index_server.serve_forever, daemon=True).start()

    for service in services:
        service.start()

    def shutdown(signum, _frame):
        # Forward the signal rather than letting docker stop wait out its timeout and
        # then kill. Each child is a server that stops cleanly when asked.
        print(f"peas: signal {signum}, stopping {len(services)} service(s)", flush=True)
        for service in services:
            if service.alive() and service.process is not None:
                service.process.terminate()
        for service in services:
            if service.process is not None:
                try:
                    service.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    service.process.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # If any single agent dies the container is no longer serving what it claims to serve.
    # Reporting that as a failure beats staying up with a hole in it, and lets a restart
    # policy or an orchestrator do something about it.
    while True:
        for service in services:
            if not service.alive():
                code = service.process.returncode if service.process else "unknown"
                print(f"peas: {service.name} exited with {code}; stopping the rest",
                      file=sys.stderr, flush=True)
                for other in services:
                    if other is not service and other.alive() and other.process is not None:
                        other.process.terminate()
                raise SystemExit(1)
        time.sleep(1)


if __name__ == "__main__":
    main()
