# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems. Use
GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository, or email **jeff.murray@alumni.upenn.edu** with:

- A description of the issue and steps to reproduce
- Which example or shared module it affects
- Your proposed fix or mitigation, if any

You'll get an acknowledgement within 72 hours.

## Scope

This repository is a set of demonstrations, not a library or a service. There is
no server to attack and nothing here is intended to run in production. What is in
scope:

- **Credential handling.** `shared/llm.py` and `shared/embeddings.py` read API keys
  from the environment. A path by which a key is logged, written to
  `shared/transcripts/`, baked into a Docker layer, or otherwise persisted is a
  real finding and worth reporting.
- **The permissions example.** `08-production-patterns/permissions/` demonstrates
  that authorization belongs in code rather than in a prompt. If its enforcement
  layer can be bypassed by a crafted percept, that is a bug in the demonstration
  and undermines the point it makes.
- **Container hardening.** The image runs as a non-root user with a read-only
  application tree. An escape from those constraints is in scope.
- **Recorded transcripts.** `shared/transcripts/` holds real model responses. If
  any of them contain something that should not be published, that is worth
  reporting privately rather than in an issue.

What is out of scope:

- Prompt injection against the example agents. `08-production-patterns/permissions/injection_demo.py`
  exists to show that the model agrees to far more than the deterministic layer
  allows, and that bounding blast radius is the answer rather than detection. A
  demonstration that a model can be talked into proposing a forbidden action is
  the documented behaviour, not a vulnerability.
- Anything requiring the reader to supply their own API key and then misuse it.
