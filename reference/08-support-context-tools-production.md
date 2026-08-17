# Support: Context Engineering, Tool Design, and Production Patterns

Anthropic-specific best practices for making config-driven agents work in production. This is supplementary material for the agent architecture pages -- it covers the implementation patterns that sit between "I designed my PEAS agent" and "it works reliably at scale."

---

## Why This Matters for Agents

The before/after pages (01-05) show what the LLM replaces in each architecture. This page covers the infrastructure that makes those replacements reliable:

- **Context engineering** -- how to manage what the LLM sees, because context is finite and every wasted token degrades performance
- **Tool design** -- how to build the actuators in your PEAS spec so the LLM actually uses them correctly
- **Evaluation** -- how to measure whether your agent meets its performance spec
- **Scaling** -- prompt caching, batching, and parallelization for production workloads

These are the production patterns that turn a prototype into a system.

---

## Context Engineering

### The Problem

Every agent call packs a context window: system prompt, PEAS config, conversation history, retrieved documents, tool definitions, and the current percept. The context window is finite, and performance degrades as it fills. Each token creates n-squared attention relationships in the transformer -- at 100K tokens, that is 10 billion relationships.

**Context engineering** is the discipline of curating what goes into each call so the LLM has exactly what it needs and nothing more.

### From Prompt Engineering to Context Engineering

| | Prompt Engineering | Context Engineering |
|--|-------------------|-------------------|
| Scope | Write effective instructions | Curate the entire information environment |
| Timing | Once (at design time) | Every turn (at runtime) |
| Content | System prompt | System prompt + tools + retrieved docs + history + tool results + current input |

The PEAS config from the overview page is a prompt engineering artifact. Context engineering is what happens at runtime to assemble each specific call.

### Three Strategies for Long-Running Agents

**1. Compaction (summarize and condense)**

As a workflow progresses, intermediate results accumulate. Compaction periodically summarizes and discards raw data.

```python
def compact_context(history, max_tokens=50000):
    """
    Deterministic check: are we over budget?
    LLM call: summarize older entries.
    Keep recent entries intact.
    """
    if estimate_tokens(history) < max_tokens:
        return history  # no compaction needed

    recent = history[-5:]          # always keep last 5 entries
    older = history[:-5]

    summary = llm_call(f"Summarize these results concisely:\n{older}")

    return [{"role": "summary", "content": summary}] + recent
```

Best for: research agents accumulating search results, data processing with large intermediate outputs, any workflow where history grows unboundedly.

**2. Structured note-taking**

Instead of keeping raw conversation history, maintain an organized state document that gets updated incrementally.

```xml
<agent_state>
  <current_task>Processing ACORD form batch</current_task>
  <completed>
    <step>Extracted 12 of 15 forms</step>
    <step>3 flagged for human review</step>
  </completed>
  <pending>
    <action>Process remaining 3 forms</action>
    <action>Generate completion report</action>
  </pending>
  <key_findings>
    <finding>Policy #AUTO-2024-001: missing VIN field</finding>
    <finding>Policy #HOME-2024-003: address mismatch</finding>
  </key_findings>
</agent_state>
```

This is the model-based reflex agent's internal state (page 01) made explicit and token-efficient.

**3. Context editing (API-side automatic pruning)**

Anthropic's API can automatically clear stale tool results when approaching token limits:

```python
context_editing = {
    "type": "clear_tool_uses_20250919",
    "trigger": {"input_tokens": 100000},   # when to activate
    "keep": 5,                              # recent tool interactions to preserve
    "clear_at_least": {"input_tokens": 15000},
    "exclude_tools": ["get_risk_factors"],  # never clear these
}
```

Result: 84% token reduction in long sessions with no manual intervention. The agent keeps working without hitting context limits.

### Token Efficiency Hierarchy

What to prioritize when assembling context:

| Priority | Content | Rationale |
|----------|---------|-----------|
| Always include | Current task, recent tool results, critical instructions | Required for correct action selection |
| When space permits | Examples, historical decisions, reference docs | Improves quality but not strictly required |
| On-demand only | Full file contents, extended logs, archived data | Fetch via tools (JIT retrieval) only when needed |

This maps to the PEAS config: the Performance spec and Actuators list are always in context. Sensor data (retrieved documents, tool results) is managed dynamically.

---

## Tool Design (Actuators That Work)

In the PEAS framework, tools are **actuators** -- the actions the agent can take. Anthropic's research identifies five principles for designing tools that LLMs actually use correctly.

### Principle 1: Choose the Right Tools

More tools does not mean better outcomes. Agents have limited context but abundant reasoning. Design tools that match how agents think, not how APIs are structured.

**Anti-pattern:** `list_contacts` returns ALL contacts. The agent reads token-by-token (brute-force search through the entire list).

**Better:** `search_contacts(query)` skips directly to relevant results.

**Consolidation pattern:** instead of separate `list_users`, `list_events`, `create_event` tools, provide one `schedule_event` tool that finds availability and schedules in a single call. Fewer tool calls = less context consumed.

Start with few high-impact tools targeting specific workflows. Scale up based on evaluation data.

### Principle 2: Namespace Your Tools

When agents access multiple services (Slack, Drive, database, etc.), tool names must be unambiguous.

```
# Namespaced by service and resource
slack_messages_search
slack_channels_list
gdrive_files_search
gdrive_folders_create
```

Without namespacing, agents confuse similar tools across services. Namespacing reduces context consumed by tool descriptions and reduces agent mistakes.

### Principle 3: Return Meaningful Context

Tool responses should contain semantic information, not technical identifiers.

```python
# Bad: cryptic, wastes tokens on things the LLM cannot use
{"id": "a7f3d9e2-8c4b-4f1a-9d2e", "mime_type": "application/vnd.ms-excel"}

# Good: meaningful, the LLM can reason about this
{"id": 1, "name": "Quarterly Report Q3 2024", "file_type": "Excel Spreadsheet"}
```

Resolving UUIDs to natural language identifiers significantly improves retrieval precision and reduces hallucinations.

**Flexible response formats:** provide a `response_format` parameter (concise vs detailed). Concise for reading/display (65% fewer tokens). Detailed when the agent needs IDs for follow-up tool calls.

### Principle 4: Optimize for Token Efficiency

Every tool response consumes context. Add controls:

- **Pagination:** `search_documents(query, page=1, page_size=10)`
- **Range selection:** `read_file(path, start_line=1, end_line=50)`
- **Filtering:** `get_logs(level="ERROR", limit=100)`
- **Truncation with indicators:** `{"truncated": true, "total_results": 450, "showing": 100}`

Provide actionable error messages, not stack traces:

```
# Bad
Error: Invalid input

# Good
The 'days' parameter must be between 1 and 90. You provided 120.
To get 120 days of data, break into two 60-day queries or use download_full_history().
```

### Principle 5: Prompt-Engineer Tool Descriptions

Tool descriptions are the most effective lever for improving tool use. Think of it as onboarding a new team member -- make implicit context explicit:

- Specialized query formats and syntax
- Domain terminology definitions
- Relationships between resources
- Constraints and edge cases
- Common pitfalls and how to avoid them

```markdown
## search_knowledge_base

Best practices:
- Make multiple small, targeted searches rather than one broad search
- Use specific keywords for better precision
- Refine based on initial results rather than requesting more

Avoid:
- Broad searches like "kubernetes" that return thousands of results
```

### Connecting Tools to PEAS Config

The config-driven approach from the overview page already structures tools as actuators. The five principles above refine how those actuators are implemented:

```yaml
actuators:
  - name: "gdrive_files_search"          # Principle 2: namespaced
    type: "deterministic"
    description: |                         # Principle 5: prompt-engineered
      Search Google Drive for files by name or content.
      Use specific keywords. Returns top 10 results by default.
      Use page parameter for more results.
    parameters:
      query: { type: "string" }
      page_size: { type: "int", default: 10 }          # Principle 4: pagination
      response_format: { type: "enum", values: ["concise", "detailed"] }  # Principle 3: flexible
```

---

## Evaluation

The P in PEAS defines what success looks like. Evaluation measures whether the agent achieves it.

### Start Small

Anthropic's guidance: 20 well-designed test cases can reveal most major issues. Do not wait for a massive test suite before evaluating.

### Strong vs Weak Test Cases

**Strong** (use these):
- Grounded in real-world complexity
- Require multiple tool calls
- Use realistic data
- Have verifiable outcomes

Example: "Customer ID 9182 reported triple-charging for a single purchase. Find all relevant log entries and determine if other customers are affected."

**Weak** (avoid these):
- Pre-specify exact tool calls
- Use simplified sandbox data
- Have trivially verifiable outcomes

Example: "Search payment logs for purchase_complete and customer_id=9182"

The weak version tests tool invocation. The strong version tests whether the agent can solve a problem.

### Metrics to Track

| Metric | What It Measures | Maps to PEAS |
|--------|-----------------|-------------|
| Task success rate | Does the agent achieve the goal? | Performance measure |
| Token consumption per task | How efficiently does it use context? | Cost / scalability |
| Tool call count per task | How many actions does it take? | Actuator efficiency |
| Tool error rate | How often do tool calls fail? | Actuator reliability |
| Latency (end to end) | How long does it take? | Performance measure |
| Escalation rate | How often does it need human help? | Confidence calibration |

### LLM-as-Judge

For subjective outputs where automatic verification is hard, use a second LLM call to evaluate quality. This is cheaper than human evaluation and scales to thousands of test cases.

```python
def evaluate_response(agent_output, expected_criteria):
    prompt = f"""Rate this agent output on a scale of 1-5.
Output: {agent_output}
Criteria: {expected_criteria}
Return JSON: {{"score": int, "reasoning": str}}"""
    return json.loads(llm_call(prompt))
```

### The Evaluation Loop

```
1. Define test cases from PEAS performance spec
2. Run agent on test cases
3. Collect metrics (deterministic) + LLM judge scores
4. Analyze: where does the agent fail?
5. Fix: adjust tools, prompts, context strategy
6. Re-run on same test cases + held-out set
7. Repeat until performance spec is met
```

---

## Scaling Patterns

### Prompt Caching

Cache the stable parts of the context (system prompt, PEAS config, tool definitions) so they are not re-processed on every call.

- **Cost reduction:** 90% on cached tokens
- **Latency reduction:** up to 85%
- **TTL:** 5 minutes (refreshed on each use)
- **Minimum:** 1024 tokens to cache

What to cache:
1. System prompts and PEAS config (stable across all calls)
2. Tool definitions (stable across all calls)
3. Few-shot examples (stable per agent type)
4. Long reference documents (stable per session)

This is the agent equivalent of shared memory tiling from the GPU parallelization page (06) -- load shared data once, reuse across many operations.

### Batch Processing

For non-real-time workloads, submit up to 10,000 requests in a single batch.

- **Cost reduction:** 50% vs real-time API
- **Processing window:** 24 hours
- **Use case:** bulk document processing, batch evaluations, data enrichment

The config-driven approach makes batching natural: same config, many inputs, one batch submission.

### Parallelization

Run independent agent calls concurrently. From page 06 (GPU parallelization), the same rules apply:

- Only parallelize when tasks are genuinely independent
- Overhead (API latency, context assembly) dominates for small tasks
- Match concurrency to throughput capacity, not maximum possible
- Keep parallel agents homogeneous to avoid "warp divergence" (fast agents waiting for slow ones)

### Cost Multipliers (From Anthropic)

| Pattern | Token Cost vs Chat | When Justified |
|---------|-------------------|---------------|
| Basic chat | 1x | Simple Q&A |
| Single agent with tools | ~4x | Multi-step tasks |
| Multi-agent system | ~15x | Parallel subtasks, context overflow |

Apply caching and batching before scaling to multi-agent. A cached single agent is often cheaper and faster than an uncached multi-agent system.

---

## How These Patterns Connect to Agent Configs

The config-driven PEAS approach from the overview page can incorporate all of these patterns:

```yaml
agent:
  name: "document-processor"
  architecture: "goal-based"

  performance:
    metrics: ["accuracy", "completion_rate", "processing_time"]
    evaluation:
      test_cases: "eval/test_cases.json"
      judge: "llm"
      threshold: 0.95

  environment:
    type: "partially-observable, stochastic, episodic"

  actuators:
    - name: "extract_fields"
      type: "deterministic"
      description: "prompts/tool_extract_fields.md"   # Principle 5: description loaded from file
      output_schema: "schemas/action_extract.json"     # Principle 3: structured output
      response_format: ["concise", "detailed"]         # Principle 4: token control
      pagination: true

  sensors:
    - name: "document_text"
      type: "text"
      input_schema: "schemas/percept_document.json"
    - name: "knowledge_base"
      type: "vector_store"
      top_k: 5

  prompts:
    system: "prompts/system.md"                        # loaded once, cached
    extraction: "prompts/extraction.md"                # loaded per-task
    validation: "prompts/validation.md"

  context:
    strategy: "compaction"
    max_tokens: 80000
    cache: ["prompts/system.md", "tool_definitions"]   # stable prefix for prompt caching
    editing:
      trigger_tokens: 100000
      keep_recent: 5

  scaling:
    caching: true
    batch_eligible: true
    max_parallel: 4

  behavior:
    decision_strategy: "extract -> validate -> output"
```

Every section maps to a concept from this page: `prompts` are loaded from files and cached, `schemas` enforce deterministic validation (the oscillation pattern), `context` drives context engineering, actuator descriptions follow the five tool design principles, `performance.evaluation` defines the test framework, and `scaling` configures caching and batching.

---

## References

- Anthropic, "Building Effective Agents" -- agent patterns, tool design, evaluation
- Anthropic, "Prompt Caching" -- caching mechanics, cost analysis
- Anthropic, "Batch API" -- bulk processing patterns
- Anthropic, "Context Engineering" -- JIT retrieval, compaction, structured note-taking
