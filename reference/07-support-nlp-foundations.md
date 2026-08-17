# Support: NLP Foundations and Why LLMs Can Be Agent Functions

A reference for understanding the natural language processing concepts that make LLM-powered agents possible. This is supplementary material for the agent architecture pages -- it explains why an LLM is a viable implementation of the agent function `f: P -> a` when percepts and actions involve human language.

---

## Why This Matters for Agents

Every agent architecture page shows LLMs handling natural language inputs (sensors) and generating natural language outputs (actuators). The LLM is not magic -- it is a probabilistic language model built on decades of NLP research. Understanding the foundations helps explain both what LLMs are good at and where they break down.

Key connections to the agent work:
- **Language models as policy functions** -- the utility-based agent page (03) describes LLMs approximating pi(s). This section explains what "predict the next token" means mathematically.
- **Percept interpretation** -- every before/after comparison shows LLMs parsing unstructured text into structured state. This section explains how that works.
- **Tool use and structured output** -- agents need LLMs to output valid JSON, function calls, and constrained action names. This connects to how language models assign probabilities to sequences.
- **RAG and agent grounding** -- agents that use knowledge bases, vector stores, and retrieval depend on the vector math described here. This is how an agent "finds relevant context" for a percept.

---

## From N-grams to LLMs: The Core Idea

A **language model** assigns a probability to a sequence of words:

$P(W) = P(w_1, w_2, w_3, \ldots, w_n)$

Or equivalently, predicts the next word given context:

$P(w_n | w_1, w_2, \ldots, w_{n-1})$

This is computed using the **chain rule of probability**:

$P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdot \ldots \cdot P(w_n|w_1, \ldots, w_{n-1})$

The problem: estimating $P(w_n | w_1, \ldots, w_{n-1})$ requires counting every possible word sequence, which is impossible for real language.

### The Markov Assumption

The simplifying assumption that made early language models practical: approximate by looking at only the last k words.

| Model | Context Window | Example |
|-------|---------------|---------|
| Unigram | 0 words | P(word) -- no context at all |
| Bigram | 1 word | P(word \| previous word) |
| Trigram | 2 words | P(word \| previous 2 words) |
| N-gram | N-1 words | P(word \| previous N-1 words) |

N-gram models estimate probabilities by counting: how often does "the cat" appear divided by how often does "the" appear gives you P(cat | the).

**Limitations:** language has long-distance dependencies. "The computer(s) which I had just put into the machine room on the fifth floor is (are) crashing" -- the verb form depends on a noun many words back. N-gram models miss this.

### From N-grams to Transformers

The progression:
1. **N-gram models** (1990s-2000s) -- count-based, Markov assumption, limited context
2. **Neural language models** (2010s) -- learned representations, longer context, but still sequential processing (RNNs)
3. **Transformer architecture** (2017-present) -- attention mechanism allows every token to attend to every other token. No Markov assumption needed. Context windows of thousands to millions of tokens.

Modern LLMs (Claude, GPT, etc.) are transformer-based language models trained on massive corpora. They still do the same fundamental thing -- predict the next token given context -- but the "context" can now be the entire conversation, document, or prompt.

---

## Vectors, Embeddings, and Similarity

This section explains the math that powers three critical agent capabilities: understanding meaning, finding relevant context, and routing inputs to the right handler.

### Feature Vectors: Representing the World as Numbers

Before any model can process text, it needs a numeric representation. A **feature vector** is a list of numbers where each dimension encodes something about the input.

For text classification (like spam detection), the feature vector might be:

```
email = {
    "free_count": 2,
    "your_name_present": 0,
    "misspelled_words": 2,
    "from_friend": 0,
    ...
}
```

This is the same concept as the percept data structure in the agent architectures. A percept comes in as messy real-world data. A feature vector is the structured numeric representation a model can work with. The LLM handles this conversion internally through its embedding layers.

### Sparse vs Dense Vectors

**Sparse vectors** (TF-IDF, bag-of-words): high-dimensional (one dimension per word in vocabulary), mostly zeros. A 50,000-word vocabulary means a 50,000-dimensional vector where maybe 20 entries are nonzero.

- **TF (term frequency):** how often does this word appear in this document?
- **IDF (inverse document frequency):** how rare is this word across all documents? Common words (the, is, a) get low weight. Rare words get high weight.
- **TF-IDF** = TF x IDF. High value = frequent in this document but rare overall.

**Dense vectors** (word2vec, transformer embeddings): low-dimensional (256-4096 dimensions), every entry is nonzero. Learned during training to capture semantic relationships.

**Why dense beats sparse for agents:**
- Dense embeddings are shorter and computationally cheaper to process
- Dense embeddings generalize better -- they capture that "refund" and "money back" mean similar things even though they share no words
- Sparse vectors represent words explicitly but cannot generalize across synonyms

### Dot Products and Similarity

The **dot product** of two vectors measures how aligned they are:

$\text{dot}(\mathbf{a}, \mathbf{b}) = \sum_i a_i \cdot b_i$

**Cosine similarity** normalizes by vector length, giving a value from -1 (opposite) to +1 (identical direction):

$\text{cosine}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \cdot |\mathbf{b}|}$

This is the foundation of how agents find relevant information. When an agent needs to search a knowledge base, it:
1. Embeds the query (the current percept) into a dense vector
2. Compares it via cosine similarity to pre-embedded documents
3. Returns the most similar documents

The word2vec skip-gram model trains embeddings so that words appearing in similar contexts get similar vectors. Given "The quick brown fox jumps over the lazy dog" with a context window of +/- 2 around "fox": the positive training pairs are (fox, quick), (fox, brown), (fox, jumps), (fox, over). After training on billions of such pairs, semantically related words cluster together in vector space.

### Non-Linear Separability and Higher-Dimensional Spaces

Some problems are not linearly separable in their original feature space. The solution: map to a higher-dimensional space where they become separable.

$\Phi: \mathbf{x} \rightarrow \phi(\mathbf{x})$

This is exactly what embedding layers in a neural network do. Raw text tokens exist in a space where "similar meaning" is not geometrically close. The embedding layer maps them into a space where it is. Each layer of the transformer further refines this mapping.

For agents, this means: even when percepts look different on the surface ("I want my money back" vs "please process a refund"), the LLM's internal embedding space puts them close together, enabling correct action selection.

### The Neural Network Stack (How Embeddings Get Computed)

The building blocks inside every LLM layer:

**Perceptron / linear classifier:** inputs are feature values, each has a weight, sum is the activation. If activation is positive, output +1; negative, output -1. A single perceptron defines a hyperplane decision boundary in feature space.

**Multi-layer networks:** stacking perceptrons lets the network learn non-linear decision boundaries. The XOR problem (not solvable by a single perceptron) is solvable with two layers.

**Activation functions** apply non-linearity after each layer:
- **Sigmoid:** maps output to [0, 1]. Used in older networks and for probability outputs.
- **Tanh:** maps to [-1, 1]. Centers the output around zero.
- **ReLU:** max(0, x). The standard for modern deep networks. Simple, fast, avoids vanishing gradients.

**Backpropagation:** computes gradients for updating every weight in the network. This is how embedding vectors get trained -- the error signal propagates backward through the network, adjusting weights so that similar inputs produce similar embeddings.

A transformer is a deep network of these components, with the addition of the **attention mechanism** that lets every token attend to every other token. The key insight: attention computes a weighted dot product between query and key vectors (both derived from embeddings), then uses those weights to aggregate value vectors. It is dot-product similarity all the way down.

---

## From Vectors to RAG: How Agents Find Relevant Context

**Retrieval-Augmented Generation (RAG)** is the practical application of everything above. It is how agents ground their responses in actual data rather than relying solely on training knowledge.

### The RAG Pipeline

```
1. OFFLINE (ahead of time):
   - Take your knowledge base (docs, FAQs, policies, etc.)
   - Chunk each document into passages
   - Embed each passage into a dense vector using an embedding model
   - Store vectors in a vector database (Pinecone, Weaviate, pgvector, etc.)

2. ONLINE (at agent runtime):
   - Agent receives a percept (e.g., customer question)
   - Embed the percept into a dense vector using the same embedding model
   - Query the vector database: find top-k passages by cosine similarity
   - Inject those passages into the agent's context window
   - LLM generates a response grounded in the retrieved passages
```

### Why This Matters for Each Agent Architecture

| Agent Type | How RAG Helps |
|-----------|---------------|
| Simple reflex | Not typically needed (stateless, immediate response) |
| Model-based reflex | Retrieved context supplements internal state -- the agent "remembers" things it has not seen directly by finding similar past situations |
| Goal-based | Retrieved passages help the LLM interpret ambiguous goals by providing domain-specific examples |
| Utility-based | Retrieved context provides the information needed to estimate utilities in novel states |
| Learning | Past experience can be stored as embeddings and retrieved for few-shot examples, improving the performance element |
| Multi-agent | Each agent retrieves only the context relevant to its own PEAS spec, keeping per-agent context windows lean |

### Vector Databases as Agent Sensors

In the PEAS framework, a vector database is a **sensor**. It provides the agent with information about the environment that is not directly in the current percept. The config-driven approach handles this naturally:

```yaml
sensors:
  - name: "customer_message"
    type: "text"
  - name: "knowledge_base"
    type: "vector_store"
    embedding_model: "text-embedding-3-small"
    top_k: 5
    similarity_threshold: 0.75
```

The agent runtime embeds the percept, queries the vector store, and injects retrieved passages into the LLM context -- all deterministic infrastructure around the nondeterministic LLM call. This is the oscillation pattern applied to retrieval.

### Similarity Search for Routing

The routing pattern from page 05 (classify input, dispatch to specialist agent) can be implemented with embeddings instead of an LLM classifier:

1. Embed example inputs for each agent category
2. When a new input arrives, embed it and find the nearest category centroid
3. Route to that agent

This is faster and cheaper than an LLM classification call, and it is deterministic. The tradeoff: it requires pre-defined categories with example inputs, so it does not handle novel categories as flexibly as LLM routing.

---

## Language Models as Agent Functions

A rational agent needs a function `f: P -> a` that maps percept sequences to actions. A language model provides exactly this:

- **Percept** = the prompt (system instructions + conversation history + current input + retrieved context)
- **Action** = the generated text (which can be a tool call, a response, or structured output)
- **f** = the language model's next-token prediction, applied autoregressively until a complete response is generated

The model assigns probabilities to every possible next token, then samples from that distribution. The system prompt and PEAS configuration constrain which outputs are likely.

### Why This Is Nondeterministic

The language model samples from a probability distribution. With temperature > 0, the same input can produce different outputs. This is why the agent architecture pages emphasize deterministic validation around LLM calls -- the LLM is inherently stochastic.

With temperature = 0, the model always picks the highest-probability token. This is more deterministic but still not guaranteed identical across API calls (due to floating-point nondeterminism in GPU computation).

### Why Structured Output Works

When the agent config specifies available actions as `["approve", "reject", "escalate"]` and the prompt says "return just the action name," the model is doing constrained generation. The probability distribution over next tokens heavily favors the action names because the prompt context makes them the most likely continuation.

Modern APIs support explicit structured output constraints (JSON schemas, function calling) that force the output to conform to a format at the decoding level, removing the need to hope the model complies.

---

## Practical Issues for Agent Systems

### Log Probabilities

All probability computations happen in log space to avoid numerical underflow. Multiplying many small probabilities produces numbers too small for floating-point representation. Log space converts multiplication to addition.

API responses include `logprobs` -- log probabilities of generated tokens. Agent systems can use logprobs as a confidence signal: low logprob tokens indicate the model is uncertain, which maps to the "flag for human review" pattern in the agent configs.

### Out-of-Vocabulary and Unknown Tokens

Classical N-gram models fail on words not seen in training data (zero probability for the entire sentence). Modern tokenizers use subword units (BPE, SentencePiece) that can represent any string by breaking it into known subword pieces. This is why LLMs can handle typos, domain jargon, and code -- they decompose unknown words into familiar subword tokens.

**For agents:** this means LLM-powered sensors can handle messy, misspelled, domain-specific input that would break a regex-based parser.

### Context Window as Working Memory

The context window is the maximum number of tokens the model can attend to. It functions as the agent's working memory. Everything the agent needs to reason about -- the PEAS config, conversation history, retrieved documents, current percept -- must fit in this window. Each token creates n-squared attention relationships, so context efficiency matters exponentially.

This is why the config-driven approach keeps configs concise, why RAG retrieves only the top-k most relevant passages, and why multi-agent systems (page 05) exist: when the context for a single task exceeds what one agent call can hold, you split across agents.

### Evaluation: Perplexity

Language models are evaluated using **perplexity** on a held-out test set. Perplexity measures how "surprised" the model is by the test data. Lower perplexity = better model.

**For agents:** perplexity does not directly measure agent quality (task success rate does). But it explains why larger, better-trained models tend to be better agent functions -- they model language more accurately, which means they interpret percepts and generate actions more reliably.

---

## Summary: The NLP Stack Under Agent Systems

| Layer | What It Does | Agent Connection |
|-------|-------------|-----------------|
| Tokenization | Converts text to token IDs via subword units | Handles messy, unstructured percepts |
| Embeddings | Maps tokens to dense vectors in semantic space | Enables meaning-based similarity, powers RAG retrieval |
| Dot product / cosine similarity | Measures vector alignment | Powers vector search, routing, and nearest-neighbor lookup |
| Attention / Transformer | Models relationships across full context via weighted dot products | Powers the agent function f: P -> a |
| Next-token prediction | Generates output autoregressively | Produces actions, tool calls, structured output |
| Structured decoding | Constrains output format (JSON schema, function calling) | Ensures valid action selection from allowed list |
| Logprobs | Measures model confidence per token | Confidence-based escalation ("flag for review") |
| RAG pipeline | Embeds query, retrieves similar passages, injects into context | Grounds agent responses in actual data |

The LLM is not a black box. It is a probabilistic language model that predicts the most likely continuation of a text sequence. When that sequence is a PEAS-configured prompt with a current percept and retrieved context, the continuation is an action. That is the agent function.

---

## References

- Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), Ch. 3 (N-grams), Ch. 6 (Embeddings), Ch. 10 (Transformers)
- Russell & Norvig, *AIMA* (4th ed.), Ch. 24 (NLP), Ch. 19 (Learning)
- Vaswani et al., "Attention Is All You Need" (2017)
- Anthropic, "Building Effective Agents" -- tool use and structured output patterns
