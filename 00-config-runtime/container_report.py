"""Is the container running the code you have?

    python 00-config-runtime/container_report.py

An image is a copy of the source taken at one moment. Nothing tells you when that moment
was, so a container can serve yesterday's agents against today's configs and look
perfectly healthy doing it -- the healthcheck asks whether the services answer, not
whether they answer with what you just wrote.

This compares the two. It digests every file that goes into the image on disk, asks the
running services what they are serving, and says whether those agree.

It needs no key and calls no model. Digests and HTTP, nothing else.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INDEX_PORT = 8079
TIMEOUT = 10

# The trees that decide agent behaviour. A change anywhere in these is a change the
# running container may not have.
WATCHED = ["00-config-runtime", "shared"]


def source_digest() -> tuple[str, int]:
    """One digest over every watched file, and the count that went into it."""
    digest = hashlib.sha256()
    count = 0
    for directory in WATCHED:
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
            digest.update(path.read_bytes())
            count += 1
    return digest.hexdigest()[:16], count


def agent_fingerprint() -> dict[str, str]:
    """Per agent, a digest of the config the runtime actually reads."""
    out = {}
    agents = ROOT / "00-config-runtime" / "agents"
    for directory in sorted(d for d in agents.iterdir() if (d / "agent.yaml").is_file()):
        digest = hashlib.sha256()
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(path.read_bytes())
        out[directory.name] = digest.hexdigest()[:12]
    return out


def get(url: str):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.load(response)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=ROOT, timeout=15).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    digest, file_count = source_digest()
    fingerprints = agent_fingerprint()

    print("Container report")
    print()
    print("  ON DISK")
    print(f"    commit           {git('rev-parse', '--short', 'HEAD') or 'unknown'}"
          f"{'  (dirty)' if git('status', '--porcelain') else '  (clean)'}")
    print(f"    watched files    {file_count} across {', '.join(WATCHED)}")
    print(f"    source digest    {digest}")
    print(f"    agents           {len(fingerprints)}")
    print()

    try:
        served = get(f"http://127.0.0.1:{INDEX_PORT}/agents.json")
    except (urllib.error.URLError, OSError) as err:
        print("  SERVING")
        print(f"    nothing answered on :{INDEX_PORT} ({err.__class__.__name__}).")
        print()
        print("    Start it with:  docker compose up -d peas")
        print("    That is not a failure of this report -- there is simply no container")
        print("    to compare against.")
        raise SystemExit(2)

    print("  SERVING")
    print(f"    agents           {len(served)} on :{served[0]['port']}-{served[-1]['port']}")
    print()

    disk_names = set(fingerprints)
    served_names = {row["name"] for row in served}
    drift = []

    if disk_names != served_names:
        for name in sorted(disk_names - served_names):
            drift.append(f"on disk but not being served: {name}")
        for name in sorted(served_names - disk_names):
            drift.append(f"served but not on disk: {name}")

    # Per agent, compare what the service reports against what the config says now.
    import yaml

    for row in sorted(served, key=lambda r: r["name"]):
        directory = ROOT / "00-config-runtime" / "agents" / row["name"]
        if not directory.is_dir():
            continue
        config = yaml.safe_load((directory / "agent.yaml").read_text(encoding="utf-8"))["agent"]
        declared = [a["name"] for a in config.get("actuators", [])]
        if sorted(declared) != sorted(row["actuators"]):
            drift.append(f"{row['name']}: actuators differ from the config on disk")
        if config["architecture"] != row["architecture"]:
            drift.append(f"{row['name']}: architecture differs from the config on disk")

        try:
            spec = get(f"http://127.0.0.1:{row['port']}/openapi.json")
        except (urllib.error.URLError, OSError):
            drift.append(f"{row['name']}: not answering on :{row['port']}")
            continue

        body = spec["paths"]["/act"]["post"]["requestBody"]["content"]["application/json"]
        served_examples = set(body.get("examples", {}))
        eval_path = directory / config["performance"]["eval"]
        raw = json.loads(eval_path.read_text(encoding="utf-8"))
        cases = raw["cases"] if isinstance(raw, dict) else raw
        disk_examples = {c["id"] for c in cases}
        if served_examples != disk_examples:
            missing = disk_examples - served_examples
            extra = served_examples - disk_examples
            detail = []
            if missing:
                detail.append(f"{len(missing)} case(s) on disk not in the served docs")
            if extra:
                detail.append(f"{len(extra)} served case(s) no longer on disk")
            drift.append(f"{row['name']}: {'; '.join(detail)}")

    print("  VERDICT")
    if drift:
        print("    STALE. The container is not serving what is on disk:")
        for line in drift:
            print(f"      - {line}")
        print()
        print("    Rebuild and restart:")
        print("      docker build -t peas . && docker compose up -d --force-recreate peas")
        raise SystemExit(1)

    print("    CURRENT. Every agent being served matches its config on disk, and every")
    print("    evaluation case on disk appears as a request example in the served docs.")
    print()
    print("    That is a comparison of content, not of timestamps: an image rebuilt from")
    print("    unchanged sources reports current, and one built before an edit reports")
    print("    stale even if it was built seconds ago.")


if __name__ == "__main__":
    main()
