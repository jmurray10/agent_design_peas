# 05 - Multi-Agent Systems: Before and After LLMs

---

## Architecture Overview

An entity should be modeled as a separate agent when its behavior is best described as having **its own performance measure** (Russell & Norvig). Multi-agent environments can be competitive (adversarial search, minimax) or cooperative (orchestrated pipelines).

Adversarial search algorithms (minimax, alpha-beta pruning) handle competitive scenarios with game trees. Cooperative multi-agent systems coordinate specialized agents toward a shared objective.

From Anthropic: single agents use ~4x tokens vs basic chat. Multi-agent systems use ~15x. Only go multi-agent when justified -- genuine parallelization, context window overflow, or complex tool orchestration.

---

## Adversarial: Game Playing

### BEFORE: Classical Minimax with Alpha-Beta

Minimax computes the optimal strategy assuming the opponent plays optimally. Alpha-beta pruning cuts branches that cannot affect the result.

```python
def alpha_beta(state, depth, alpha, beta, is_maximizing, game):
    if depth == 0 or game.is_terminal(state):
        return game.evaluate(state), None

    if is_maximizing:
        best_val, best_act = float('-inf'), None
        for action in game.get_actions(state):
            val, _ = alpha_beta(game.result(state, action),
                                depth-1, alpha, beta, False, game)
            if val > best_val: best_val, best_act = val, action
            alpha = max(alpha, best_val)
            if beta <= alpha: break
        return best_val, best_act
    else:
        best_val, best_act = float('inf'), None
        for action in game.get_actions(state):
            val, _ = alpha_beta(game.result(state, action),
                                depth-1, alpha, beta, True, game)
            if val < best_val: best_val, best_act = val, action
            beta = min(beta, best_val)
            if beta <= alpha: break
        return best_val, best_act


class TicTacToe:
    def is_terminal(self, state):
        return self.check_winner(state) is not None or len(self.get_actions(state)) == 0

    def get_actions(self, state):
        return [i for i in range(9) if state[i] == '.']

    def result(self, state, action):
        s = list(state)
        s[action] = 'X' if state.count('X') == state.count('O') else 'O'
        return tuple(s)

    def check_winner(self, state):
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in lines:
            if state[a] == state[b] == state[c] != '.': return state[a]
        return None

    def evaluate(self, state):
        w = self.check_winner(state)
        if w == 'X': return 1
        if w == 'O': return -1
        return 0

game = TicTacToe()
value, action = alpha_beta(tuple('.'*9), 9, float('-inf'), float('inf'), True, game)
print(f"Best first move: {action}, value: {value}")
```

**Strengths:** provably optimal in perfect-information games. Alpha-beta prunes efficiently.

**Limitations:** exponential game tree growth. Needs depth limit + evaluation function for complex games. Hand-coding good evaluation functions is hard. Imperfect-information games (poker, negotiation) are not directly solvable with minimax.

---

### AFTER: LLM Replaces the Evaluation Function

Search structure and pruning stay. The LLM replaces the hardest part: evaluating non-terminal positions.

```python
import json

class LLMGameAgent:
    """
    Alpha-beta search with LLM-powered evaluation.
    Deterministic: tree structure, pruning, terminal evaluation.
    LLM: non-terminal position evaluation (the hard part).
    """
    def __init__(self, game, max_depth=4):
        self.game = game
        self.max_depth = max_depth

    def get_action(self, state):
        _, action = self.search(state, self.max_depth,
                                float('-inf'), float('inf'), True)
        return action

    def search(self, state, depth, alpha, beta, is_max):
        if depth == 0 or self.game.is_terminal(state):
            return self.evaluate(state), None
        if is_max:
            best_v, best_a = float('-inf'), None
            for a in self.game.get_actions(state):
                v, _ = self.search(self.game.result(state, a),
                                   depth-1, alpha, beta, False)
                if v > best_v: best_v, best_a = v, a
                alpha = max(alpha, best_v)
                if beta <= alpha: break
            return best_v, best_a
        else:
            best_v, best_a = float('inf'), None
            for a in self.game.get_actions(state):
                v, _ = self.search(self.game.result(state, a),
                                   depth-1, alpha, beta, True)
                if v < best_v: best_v, best_a = v, a
                beta = min(beta, best_v)
                if beta <= alpha: break
            return best_v, best_a

    def evaluate(self, state):
        # Terminal: deterministic, exact
        if self.game.is_terminal(state):
            return self.game.evaluate(state)

        # Non-terminal: LLM estimates
        board = '\n'.join(' '.join(state[i:i+3]) for i in range(0, 9, 3))
        prompt = f"""Evaluate this game position from -1.0 to 1.0.
+1.0 = X winning, -1.0 = O winning, 0.0 = even.
{board}
Return just a number."""
        try:
            return max(-1.0, min(1.0, float(llm_call(prompt).strip())))
        except ValueError:
            return 0.0
```

---

## Cooperative: Orchestrated Pipelines

### BEFORE: Classical

Pipelines of specialized functions. Each function is essentially a single-purpose agent.

```python
def extract_data(document):
    data = {}
    for line in document.split('\n'):
        if 'name:' in line.lower():
            data['name'] = line.split(':')[1].strip()
        if 'amount:' in line.lower():
            data['amount'] = float(line.split(':')[1].strip().replace('$',''))
    return data

def validate_data(data):
    errors = []
    if 'name' not in data: errors.append('missing name')
    if data.get('amount', 0) <= 0: errors.append('invalid amount')
    return data, errors

def generate_report(data, errors):
    if errors: return f"ERRORS: {errors}"
    return f"Processed: {data['name']} for ${data['amount']}"

# Pipeline
doc = "Name: John Smith\nAmount: $1500.00\nDate: 2024-01-15"
data = extract_data(doc)
data, errors = validate_data(data)
print(generate_report(data, errors))
```

**Strengths:** predictable, testable, fast.

**Limitations:** every parser, validator, and formatter is hand-coded. New document types require new parsing rules. Cannot handle unstructured input.

---

### AFTER: LLM Multi-Agent System

Each agent has its own PEAS spec and performance measure. An orchestrator coordinates them. Maps to Anthropic's workflow/routing/parallel patterns.

```python
import json
import asyncio

class AgentConfig:
    def __init__(self, config: dict):
        self.name = config['name']
        self.role = config['role']
        self.performance = config['performance']
        self.actions = config['actuators']
        self.system_prompt = config['behavior']['system_prompt']


class WorkerAgent:
    """Single agent with its own PEAS spec and performance measure."""
    def __init__(self, config: AgentConfig):
        self.config = config
        self.performance_log = []

    async def process(self, input_data: dict) -> dict:
        prompt = f"""{self.config.system_prompt}
Input: {json.dumps(input_data)}
Available actions: {self.config.actions}
Process the input. Return JSON."""

        response = llm_call(prompt)
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {'raw_output': response, 'parse_error': True}

        score = self._evaluate(result)
        self.performance_log.append(score)
        return result

    def _evaluate(self, result: dict) -> float:
        score = 0.0
        if not result.get('parse_error'): score += 0.5
        if result.get('confidence', 0) > 0.8: score += 0.5
        return score


class Orchestrator:
    """
    Coordinates agents. Patterns (mapped to architectures):
    - workflow: sequential pipeline (goal-based, fixed plan)
    - routing: classify then dispatch (reflex-based)
    - parallel: concurrent workers (separate performance measures)
    """
    def __init__(self, agents: list, pattern: str = 'workflow'):
        self.agents = agents
        self.pattern = pattern

    async def run(self, input_data: dict) -> dict:
        if self.pattern == 'workflow':
            return await self._workflow(input_data)
        elif self.pattern == 'parallel':
            return await self._parallel(input_data)
        elif self.pattern == 'routing':
            return await self._routing(input_data)

    async def _workflow(self, data: dict) -> dict:
        current = data
        results = []
        for agent in self.agents:
            result = await agent.process(current)
            if result.get('parse_error'):
                return {'error': f'{agent.config.name} failed', 'partial': results}
            results.append({'agent': agent.config.name, 'result': result})
            current = result
        return {'status': 'complete', 'results': results, 'final': current}

    async def _parallel(self, data: dict) -> dict:
        tasks = [a.process(data) for a in self.agents]
        results = await asyncio.gather(*tasks)
        return {'status': 'complete', 'results': [
            {'agent': a.config.name, 'result': r}
            for a, r in zip(self.agents, results)
        ]}

    async def _routing(self, data: dict) -> dict:
        names = [a.config.name for a in self.agents]
        prompt = f"""Classify to one handler: {names}
Input: {json.dumps(data)}
Return just the handler name."""
        category = llm_call(prompt).strip()
        agent_map = {a.config.name: a for a in self.agents}
        target = agent_map.get(category, self.agents[0])
        result = await target.process(data)
        return {'routed_to': target.config.name, 'result': result}


# Build from configs
agents = [
    WorkerAgent(AgentConfig({
        'name': 'extractor', 'role': 'data extraction',
        'performance': {'metrics': ['accuracy', 'completeness']},
        'actuators': ['extract_fields', 'flag_unclear'],
        'behavior': {'system_prompt': 'Extract structured data from the document. Return JSON.'}
    })),
    WorkerAgent(AgentConfig({
        'name': 'validator', 'role': 'data validation',
        'performance': {'metrics': ['error_detection', 'false_positive_rate']},
        'actuators': ['approve', 'reject', 'flag'],
        'behavior': {'system_prompt': 'Validate extracted data for completeness and correctness. Return JSON.'}
    })),
    WorkerAgent(AgentConfig({
        'name': 'formatter', 'role': 'output formatting',
        'performance': {'metrics': ['format_compliance']},
        'actuators': ['format_report', 'format_json'],
        'behavior': {'system_prompt': 'Format validated data into the requested output. Return JSON.'}
    })),
]

system = Orchestrator(agents, pattern='workflow')
result = asyncio.run(system.run({'document': 'Name: John Smith\nAmount: $1500\nPolicy: AUTO-2024-001'}))
print(json.dumps(result, indent=2))
```

---

### Comparison

| Aspect | Classical Pipeline | LLM Multi-Agent |
|--------|-------------------|-----------------|
| Agent definition | Functions with hardcoded logic | PEAS-configured agents with LLM reasoning |
| Performance measure | Implicit | Explicit per-agent metrics |
| Coordination | Function calls | Orchestrator with patterns |
| Input handling | Regex, parsers | LLM interprets unstructured input |
| Adding agents | Write new functions | Write new config |
| Error handling | Try/except | Deterministic validation between agents |

---

### Config

```yaml
system:
  name: "document-processing-pipeline"
  pattern: "workflow"

  agents:
    - name: "extractor"
      config_dir: "agents/extractor/"              # each agent is its own directory
      architecture: "goal-based"
      performance:
        metrics: ["extraction accuracy", "completeness"]
      actuators:
        - name: "extract_fields"
          output_schema: "agents/extractor/schemas/output.json"
        - name: "flag_unclear"
      sensors:
        - name: "document_text"
          input_schema: "agents/extractor/schemas/input.json"
      prompts:
        system: "agents/extractor/prompts/system.md"
        extraction: "agents/extractor/prompts/extraction.md"

    - name: "validator"
      config_dir: "agents/validator/"
      architecture: "simple-reflex"
      performance:
        metrics: ["error detection rate", "false positive rate"]
      actuators:
        - name: "approve"
          output_schema: "agents/validator/schemas/output.json"
        - name: "reject"
          output_schema: "agents/validator/schemas/output.json"
        - name: "flag"
      sensors:
        - name: "extracted_data"
          input_schema: "agents/validator/schemas/input.json"
      prompts:
        system: "agents/validator/prompts/system.md"

    - name: "formatter"
      config_dir: "agents/formatter/"
      architecture: "simple-reflex"
      performance:
        metrics: ["format compliance"]
      actuators:
        - name: "format_report"
          output_schema: "agents/formatter/schemas/output.json"
        - name: "format_json"
          output_schema: "agents/formatter/schemas/output.json"
      sensors:
        - name: "validated_data"
          input_schema: "agents/formatter/schemas/input.json"
      prompts:
        system: "agents/formatter/prompts/system.md"

  orchestration:
    type: "sequential"
    validation_between_agents: true                # validate output schema of agent N against input schema of agent N+1
    error_handling: "retry_once_then_fail"
    max_total_steps: 15
```

Each agent is its own directory with its own prompts, schemas, and eval cases. The orchestration config wires them together. Schema validation between agents is deterministic -- the output schema of the extractor must match the input schema of the validator. New agent = new directory. New pipeline = new orchestration config referencing existing agent directories.
