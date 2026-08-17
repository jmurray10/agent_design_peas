# The runtime, containerised. The image is the agent runtime; an agent directory is the
# payload. That is the same claim `00-config-runtime/` makes in Python -- new agent, new
# directory, no runtime change -- and a container is where it stops being a claim: mount a
# directory that this image has never seen and it runs, because the config says what the
# agent is and the runtime reads the config.
#
#   docker build -t peas .
#   docker run --rm peas                                   replay, no key, no network
#   docker run --rm --env-file .env peas                   live, against a real model
#   docker run --rm peas python 02-goal-based/csp/after.py  any example
#
# Pinned by digest, not by tag. A tag is a moving pointer: `python:3.13-slim` is a
# different image this month than last, so a build that succeeds today can fail tomorrow
# with no change on this side of the Dockerfile. A digest is the image, and "the same
# image next month" is the only reason to containerise a reference repository at all.
#
# This resolves to Python 3.13.14, the version every result in this repository was
# recorded and measured against. To move it: `docker pull python:3.13-slim` then
# `docker inspect --format='{{index .RepoDigests 0}}' python:3.13-slim`, and re-run the
# live sweep afterwards rather than assuming a patch release changed nothing.
FROM python@sha256:bf503bb2243c5aad0aa951544dd60d165f992646441d35dea90893703fc26251

# Fail fast and log straight through, which is what you want from a container whose whole
# job is to print what an agent did.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Installed as a separate layer, before the source, so editing an example does not
# reinstall anything.
#
# Every one of these is optional to the repository and required by the image, which is a
# distinction worth keeping straight. `before.py` files import none of them and never
# will. The rest degrade rather than crash when a package is missing -- the orchestration
# example falls back to a hand-rolled schema check, the parallelisation benchmark falls
# back to stdlib arithmetic. Installing all four means the image exercises the real path
# instead of the fallback, and the fallbacks stay reachable for a reader running on a bare
# Python install outside the container.
RUN pip install --no-cache-dir --root-user-action=ignore \
        anthropic==0.97.0 \
        pyyaml==6.0.2 \
        jsonschema==4.23.0 \
        numpy==2.4.2

# Non-root. Nothing here needs write access to anything but /tmp, and a container that
# runs an agent against a live API is exactly the sort of thing that should not be root.
RUN useradd --create-home --uid 10001 agent
COPY --chown=agent:agent . /app

# The build context came off a Windows filesystem, where directories map to mode 555 --
# readable and traversable, writable by nobody, owner included. `snapshot.py` writes a
# baseline and a recording pass writes a transcript, so both failed inside the container
# with PermissionError while working fine on the host, which is the whole reason to test
# in the image rather than beside it.
#
# Only these two directories get the write bit. Everything else stays read-only, which is
# the posture you want for the code an agent with a live API key is executing: a container
# that cannot rewrite its own examples cannot be talked into doing so.
RUN chmod -R a-w /app \
 && chmod -R u+w /app/10-drift/baselines /app/shared/transcripts

USER agent

# Replay is the default because it needs no key, no network, and no arguments, so
# `docker run peas` does something useful on a machine that has never seen this project.
# Supplying an API key through --env-file switches the same command to a live model
# without changing the image.
CMD ["python", "00-config-runtime/demo.py"]

# `docker run -p 8080:8080 peas python 00-config-runtime/serve.py` serves every agent
# from this one image, each under its own path prefix. See docker-compose.yml.
