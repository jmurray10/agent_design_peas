# From Classical AI Agents to LLM-Powered Agents

A reference for teams building LLM agent systems who want to ground their work in proven AI architecture patterns rather than ad-hoc prompt engineering.

---

## The Problem

Most "AI agent" projects start with a prompt and some tool calls. There is no formal definition of what the agent should do, what success looks like, what its boundaries are, or how it should behave in edge cases. When things break, there is nothing to debug against because there was never a spec.

Classical AI solved this decades ago with a framework called **PEAS** and a taxonomy of five agent architectures. These frameworks are not outdated -- they are exactly what LLM agents need and are missing.

This guide shows how every modern LLM agent pattern maps to a classical agent architecture, and how PEAS can become the **configuration file** that defines and deploys agents systematically.

---

## How This Is Organized

| Page | What It Covers |
|------|---------------|
| **This page** | PEAS framework, config-driven agents, architecture selection, the oscillation pattern |
| **01 - Reflex Agents** | Simple reflex and model-based reflex -- before/after with code |
| **02 - Goal-Based Agents** | Search (A*, BFS) and constraint satisfaction -- before/after with code |
| **03 - Utility-Based Agents** | MDPs, value iteration, policy approximation -- before/after with code |
| **04 - Learning Agents** | Q-learning components mapped to LLM feedback loops -- before/after with code |
| **05 - Multi-Agent Systems** | Adversarial search and cooperative orchestration -- before/after with code |
| **06 - Support: GPU Parallelization** | Why parallel computation matters for agents, five patterns with real performance data |
| **07 - Support: NLP Foundations** | Language models, embeddings, vectors, RAG, and why LLMs work as agent functions |
| **08 - Support: Context, Tools, and Production** | Anthropic best practices for context engineering, tool design (5 principles), evaluation, and scaling |

Pages 01-05 show classical code, the LLM-powered equivalent, a comparison table, and a config-driven YAML spec. Pages 06-08 are supplementary reference material.

---

## What Is an Agent?

An **agent** perceives its environment through **sensors** and acts on it through **actuators**. It is defined by an agent function `f: P -> a` that maps percept sequences to actions (Russell & Norvig, AIMA).

`agent = architecture + program`

A **rational agent** picks actions that maximize the expected value of its **performance measure**. Rationality does not require omniscience -- it requires doing the best you can with the information you have.

Nothing in this definition constrains the program to be a lookup table, a set of rules, or a search algorithm. An LLM qualifies as the program (or a component of it) as long as the agent produces rational actions.

---

## The PEAS Framework

Every agent needs a well-defined task environment. PEAS provides the spec:

- **P** - Performance measure: the objective criteria for success
- **E** - Environment: the world the agent operates in
- **A** - Actuators: the actions the agent can take
- **S** - Sensors: the information the agent receives

**Examples:**

| Agent | Performance | Environment | Actuators | Sensors |
|-------|------------|-------------|-----------|---------|
| Self-driving taxi | Safe, fast, legal, profitable | Roads, traffic, pedestrians | Steering, accelerator, brake | Cameras, LiDAR, GPS |
| Delivery drone | On-time, correct location, no damage | Airspace, obstacles, weather | Propellers, payload release | GPS, radar, altitude sensor |
| Support bot | Satisfaction score, resolution time | Tickets, knowledge base, customer history | Reply, escalate, refund | Ticket text, order data |

PEAS defines the problem, not the solution. The spec is identical whether the agent is rule-based or LLM-powered.

---

## Environment Classification

After writing the PEAS spec, classify the environment along these dimensions:

- **Fully observable** vs **partially observable** -- can the agent see the complete state?
- **Deterministic** vs **stochastic** -- are outcomes guaranteed or probabilistic?
- **Episodic** vs **sequential** -- do past actions affect future states?
- **Static** vs **dynamic** -- does the environment change while the agent thinks?
- **Discrete** vs **continuous** -- finite states/actions or smooth ranges?
- **Single-agent** vs **multi-agent** -- are there other entities with their own goals?

The hardest case is: partially observable, stochastic, sequential, dynamic, continuous, multi-agent. That is also the case that benefits most from LLMs.

These classifications drive architecture selection -- they are not academic categories.

---

## The Five Agent Architectures

Every LLM agent maps to one of these:

### 1. Simple Reflex Agent
Current percept -> action. No memory, no planning. Use when: inputs are self-contained and the right action is obvious from what you see right now.

### 2. Model-Based Reflex Agent
Maintains internal state across interactions. Use when: the environment is partially observable and context from prior interactions matters.

### 3. Goal-Based Agent
Has explicit goals and uses search/planning to reach them. Use when: the agent needs to find a sequence of actions, not just react.

### 4. Utility-Based Agent
Maximizes expected utility across uncertain outcomes. Use when: there are tradeoffs, conflicting goals, or probabilistic outcomes.

### 5. Learning Agent
Improves through experience. Four components: performance element (picks actions), critic (measures results), learning element (updates behavior), problem generator (suggests exploration). Use when: the optimal behavior is not known upfront.

---

## LLMs Upgrade Components, Not Architectures

The architecture stays the same. What changes is which internal component is powered by the LLM:

| Component | Traditional | LLM-Powered |
|-----------|-------------|-------------|
| Percept interpretation | Regex, parsers, structured input | LLM reads natural language and unstructured data |
| Action selection | Rule table, search, policy lookup | LLM reasons about what to do |
| State maintenance | Explicit data structures | LLM summarizes and updates state from context |
| Outcome evaluation | Hand-coded utility function | LLM judges state quality |
| Output generation | Templates, structured response | LLM produces natural language |

---

## Config-Driven Agents: PEAS as Your Config File

If the PEAS spec defines the problem and the architecture defines the structure, then PEAS can be the **configuration** that deploys an agent.

**What this section is:** a reference specification -- the target structure for how agent configs should be organized. The YAML examples define the *shape* of a config-driven system. They are a design pattern, not a drop-in runtime. Any team implementing this pattern would build the runtime that reads these configs and wires up the prompts, schemas, tools, and LLM calls. The examples throughout pages 01-05 show what that runtime code looks like for each agent architecture.

**What this section is not:** a framework you can install. The value is in the structure itself -- PEAS as a schema for agent definition, with modular prompt loading, schema validation, and externalized domain assets.

### Design Principles

The critical design choice: **the config references external resources, it does not contain them.** Prompts, schemas, and domain assets are separate files loaded at runtime. This makes agents modular -- swap a prompt without changing the agent definition, version schemas independently, reuse prompt templates across agents.

### Agent Directory Structure

```
agents/
  insurance-processor/
    agent.yaml              # PEAS config -- references everything, contains nothing
    prompts/
      system.md             # system prompt (the agent's identity and rules)
      extraction.md         # task prompt template for field extraction
      validation.md         # task prompt template for validation checks
      escalation.md         # prompt for deciding when to escalate
    schemas/
      input_percept.json    # validates raw sensor input before LLM sees it
      action_output.json    # validates LLM action output before execution
      state.json            # defines internal state structure (model-based agents)
      acord_fields.json     # domain-specific field definitions
    eval/
      test_cases.json       # evaluation cases tied to performance spec
```

This mirrors Anthropic's Agent Skills pattern: a SKILL.md (the config) plus scripts, references, and assets loaded progressively. The agent does not carry everything in memory -- it loads what it needs when it needs it.

### The Config (References, Not Contents)

```yaml
agent:
  name: "insurance-form-processor"
  architecture: "goal-based"

  # P - Performance Measure
  performance:
    metrics:
      - "accuracy of field extraction"
      - "percentage of forms completed without human review"
      - "processing time per form"
    success_threshold: 0.95
    eval: "eval/test_cases.json"         # external test cases

  # E - Environment
  environment:
    type: "partially-observable, stochastic, episodic"
    inputs:
      - "ACORD PDF forms (variable layouts)"
      - "free-text agent notes"
      - "client emails"

  # A - Actuators (tools the agent can use)
  actuators:
    - name: "extract_pdf_fields"
      type: "deterministic"
      output_schema: "schemas/action_output.json"   # validates LLM output before execution
    - name: "fill_form"
      type: "deterministic"
      output_schema: "schemas/action_output.json"
    - name: "request_human_review"
      type: "escalation"

  # S - Sensors (inputs the agent receives)
  sensors:
    - name: "pdf_content"
      type: "document"
      input_schema: "schemas/input_percept.json"    # validates/normalizes raw input
    - name: "field_schema"
      type: "structured"
      source: "schemas/acord_fields.json"           # domain knowledge loaded from file
    - name: "client_context"
      type: "text"

  # Prompt loading (separate from PEAS -- this is the implementation layer)
  prompts:
    system: "prompts/system.md"                     # loaded once, cached
    extraction: "prompts/extraction.md"             # loaded per-task as needed
    validation: "prompts/validation.md"
    escalation: "prompts/escalation.md"

  # State schema (for model-based agents)
  state:
    schema: "schemas/state.json"                    # defines what internal state looks like
    persistence: "per-session"

  # Behavior
  behavior:
    decision_strategy: "interpret -> map -> validate -> execute or escalate"
    max_steps: 10
```

### What Gets Loaded vs What Gets Referenced

| Resource | Loaded When | Cached | Why External |
|----------|-------------|--------|-------------|
| System prompt | Agent startup | Yes (prompt caching) | Version independently, A/B test, reuse across agents |
| Task prompts | Per-task as needed | Per-session | Different tasks use different prompts within one agent |
| Input schemas | Agent startup | Yes | Deterministic validation does not change per-call |
| Output schemas | Agent startup | Yes | Same -- validation is stable infrastructure |
| State schema | Agent startup | Yes | Defines the model-based agent's internal state shape |
| Domain assets | On-demand (JIT) | Optional | Large files loaded only when referenced by a task |
| Eval test cases | Evaluation time only | No | Never loaded during production runs |

### The Generic Runtime (Illustrative)

The code below shows what a runtime that reads these configs would look like. It is not a production implementation -- it is a reference for how the pieces connect. Each before/after page (01-05) shows the architecture-specific version of this pattern.

```python
import json
import yaml
from pathlib import Path

class ConfigDrivenAgent:
    def __init__(self, agent_dir: str):
        self.base = Path(agent_dir)
        self.config = yaml.safe_load((self.base / "agent.yaml").read_text())

        # Load prompts from files
        self.prompts = {}
        for name, path in self.config.get("prompts", {}).items():
            self.prompts[name] = (self.base / path).read_text()

        # Load schemas from files
        self.schemas = {}
        for actuator in self.config.get("actuators", []):
            if "output_schema" in actuator:
                schema_path = self.base / actuator["output_schema"]
                self.schemas[actuator["name"]] = json.loads(schema_path.read_text())

        # Load input validation schemas
        for sensor in self.config.get("sensors", []):
            if "input_schema" in sensor:
                schema_path = self.base / sensor["input_schema"]
                self.schemas[f"input_{sensor['name']}"] = json.loads(schema_path.read_text())

        # Load state schema if model-based
        if "state" in self.config:
            state_path = self.base / self.config["state"]["schema"]
            self.state_schema = json.loads(state_path.read_text())
            self.state = {}

        # Load tools
        self.tools = load_tools(self.config["actuators"])
        self.performance = PerformanceTracker(self.config["performance"])

    def run(self, input_data):
        # DETERMINISTIC: validate input against sensor schema
        percept = self.validate_input(input_data)

        # LLM: decide action (system prompt + task prompt + percept)
        action = self.decide(percept)

        # DETERMINISTIC: validate action against output schema
        validated_action = self.validate_output(action)

        # DETERMINISTIC: execute
        result = self.act(validated_action)

        # DETERMINISTIC: measure
        self.performance.record(result)
        return result

    def decide(self, percept):
        # Assemble context from loaded prompts
        system = self.prompts["system"]
        task_prompt = self.select_task_prompt(percept)

        prompt = f"""{system}

{task_prompt}

Available actions: {[t['name'] for t in self.config['actuators']]}
Current observation: {percept}
Strategy: {self.config['behavior']['decision_strategy']}

Pick the next action. Return valid JSON matching the output schema."""

        return llm_call(prompt)

    def select_task_prompt(self, percept):
        """Pick the right task prompt based on the current situation."""
        # Simple routing -- could also be LLM-driven
        if "pdf" in str(percept).lower():
            return self.prompts.get("extraction", "")
        if "review" in str(percept).lower():
            return self.prompts.get("escalation", "")
        return self.prompts.get("extraction", "")

    def validate_input(self, input_data):
        """DETERMINISTIC: validate raw input against sensor schemas."""
        # jsonschema.validate(input_data, self.schemas["input_pdf_content"])
        return input_data  # pass-through if valid, raise if not

    def validate_output(self, action):
        """DETERMINISTIC: validate LLM output against actuator schema."""
        parsed = json.loads(action)
        tool_name = parsed.get("action", "")
        if tool_name in self.schemas:
            pass  # jsonschema.validate(parsed, self.schemas[tool_name])
        return parsed
```

### Why Modular Loading Matters

**Prompts are not code.** They change more often than architecture. A prompt tweak should not require redeploying the agent. External prompts can be:
- Version-controlled independently (prompt v2.3 with agent config v1.0)
- A/B tested (swap `prompts/system.md` for `prompts/system_v2.md` in the config)
- Shared across agents (multiple agents reference the same escalation prompt)
- Cached efficiently (Anthropic's prompt caching works best with stable prefixes -- external system prompts are the stable prefix)

**Schemas are the deterministic layer.** They enforce the oscillation pattern: the LLM generates output, schemas validate it before execution. External schemas mean:
- Validation logic is inspectable and testable without running the LLM
- Domain experts can update field definitions without touching agent code
- Input schemas normalize messy sensor data before the LLM sees it
- Output schemas guarantee the action is structurally valid before the actuator fires

**Domain assets are loaded on demand.** A 500-field ACORD schema should not live in the config or the system prompt. It is loaded from `schemas/acord_fields.json` only when the extraction task prompt needs it. This is JIT retrieval from page 08 applied to the agent's own configuration.

### What This Pattern Gets You

When a team builds a runtime against this spec, the payoff is:

- **New agent = new directory.** Config + prompts + schemas + eval cases. No runtime code changes.
- **PEAS enforces completeness.** Cannot ship without performance, environment, actuators, sensors.
- **Prompts are swappable.** Change behavior without changing structure.
- **Schemas enforce contracts.** Deterministic validation wraps every LLM call.
- **Eval is built in.** Test cases live next to the agent they test.
- **Agents are portable.** Zip the directory, hand it to another team, it runs on the same runtime.

---

## The Oscillation Pattern

The key design pattern for LLM-powered agents:

**LLMs are nondeterministic. Do not try to make them deterministic. Build strong deterministic infrastructure around them.**

```
1. [DETERMINISTIC] Sensors gather and validate raw input
2. [LLM]           Interpret percepts, update state, decide action
3. [DETERMINISTIC] Validate the chosen action, execute it
4. [DETERMINISTIC] Environment provides feedback
5. [LLM]           Interpret feedback, decide next action
6. Repeat
```

This is the standard agent loop. Sensors perceive, agent function selects action, actuators execute, environment changes, repeat. The LLM handles interpretation and decision-making. Deterministic code handles everything else.

---

## Anthropic Patterns Mapped to Architectures

Anthropic's published agent patterns map directly:

| Anthropic Pattern | Classical Architecture | When to Use |
|------------------|----------------------|-------------|
| Workflows | Goal-based with fixed plan | Task decomposes into predictable steps |
| Routing | Model-based reflex | Distinct input categories, specialized handlers |
| Parallelization | Multi-agent system | Independent subtasks, voting/aggregation |
| Autonomous agents | Utility-based in a loop | Open-ended problems, unknown step count |

Token cost scales with complexity (from Anthropic): ~4x for single agents, ~15x for multi-agent systems vs basic chat. Match architecture to problem -- do not over-engineer.

---

## Architecture Selection Checklist

1. **Write the PEAS spec.** Do not touch code until Performance, Environment, Actuators, and Sensors are defined.
2. **Classify the environment.** The dimensions above determine your architecture.
3. **Pick the simplest architecture that works.** Reflex before goal-based. Goal-based before utility-based.
4. **For each component, ask: LLM or deterministic?** Default to deterministic. Use LLM only where formalization is hard.
5. **Make it config-driven.** PEAS spec in YAML. Runtime reads config. New agents are new configs.
6. **Measure against the performance spec.** The P in PEAS is not decoration.

---

## References

- Russell & Norvig, *Artificial Intelligence: A Modern Approach* (4th ed.)
- Anthropic, "Building Effective Agents"
- Anthropic, "Agent Skills" -- progressive disclosure and modular capabilities
