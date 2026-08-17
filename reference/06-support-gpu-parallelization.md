# Support: GPU Parallelization and Why It Matters for Agent Systems

A reference for understanding how parallel computation works at the hardware level, why it matters for LLM-powered agents, and what the performance tradeoffs look like. This is supplementary material for the agent architecture pages -- it explains the "how" behind parallel execution patterns.

---

## Why This Matters for Agents

The multi-agent systems page (05) describes parallelization as running multiple agents simultaneously. Anthropic's parallelization pattern (sectioning and voting) depends on the ability to process independent subtasks concurrently. At the infrastructure level, this is the same problem GPU computing solves: take a large task, decompose it into independent units of work, and run them at the same time.

Understanding GPU parallelization also explains why LLMs themselves are fast enough to power real-time agents. Every forward pass through an LLM is a massive matrix multiplication running on GPU hardware. The concepts below -- threads, blocks, memory hierarchy, arithmetic intensity -- are what make that possible.

---

## The Core Model: Threads, Blocks, and Grids

GPUs execute thousands of threads simultaneously, organized into a hierarchy:

- **Thread** -- the smallest unit of execution. Each thread runs the same code (a "kernel") on different data.
- **Block** -- a group of threads that can share fast on-chip memory and synchronize with each other. Typical block sizes: 256 or 512 threads.
- **Grid** -- a collection of blocks. Blocks within a grid cannot directly communicate -- they run independently.

This maps to agent parallelization: each block is like an independent agent instance, each thread is like a subtask within that agent, and the grid is the orchestrator launching all of them.

---

## Memory Hierarchy (Why Data Movement Matters More Than Compute)

GPU memory has layers, from slow to fast:

| Memory Type | Size | Speed | Scope |
|-------------|------|-------|-------|
| Global memory (DRAM) | GBs | ~500 GB/s | All threads |
| Shared memory (on-chip) | ~48 KB per block | ~10 TB/s | Threads within one block |
| Registers | KBs per thread | Fastest | Single thread |

The key insight: **moving data is almost always the bottleneck, not computing on it.** This is measured by **arithmetic intensity** -- the ratio of compute operations to bytes transferred.

For agent systems, the analogy holds: the bottleneck is usually not the LLM inference itself but the data movement -- API calls, context assembly, tool responses, inter-agent communication.

---

## Five Parallel Patterns (With Real Performance Numbers)

These patterns show up in both GPU programming and agent system design.

### 1. Element-wise Operations (SAXPY)

Each element is processed independently. No communication between threads.

`z[i] = a * x[i] + y[i]`

| Problem Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 10K | 0.015 ms | 0.014 ms | 1.07x |
| 1M | 1.5 ms | 0.05 ms | 30x |
| 100M | 150 ms | 4.5 ms | 33x |

**Arithmetic intensity:** 0.167 FLOPS/byte (very low -- memory-bound). Only 2 operations per 12 bytes transferred. Performance is limited by how fast you can feed data, not how fast you can compute.

**Agent parallel:** this is the pattern when multiple agents process independent inputs with no shared state. Each agent is like a thread -- runs the same logic on different data. Scales linearly until you hit the I/O bottleneck (API rate limits, context assembly time).

**Key lesson:** below 10K elements, kernel launch overhead makes GPU *slower* than CPU. Same with agents -- spinning up parallel LLM calls has overhead. Only parallelize when the task is large enough to justify it.

### 2. Matrix Multiplication (Tiled Shared Memory)

The core operation behind every LLM forward pass. Each output element requires reading an entire row and column. Massive data reuse opportunity.

Three implementations:

| Matrix Size | Naive GPU | Shared Memory Tiled | Speedup vs CPU |
|------------|-----------|-------------------|----------------|
| 128x128 | 12.8x | 32x | - |
| 512x512 | 25.6x | 136x | - |
| 2048x2048 | 46.8x | 437x | - |

**Why tiling works:** in the naive version, each thread reads the same row/column data from slow global memory repeatedly. Tiled shared memory loads a 16x16 chunk cooperatively into fast on-chip memory, then all threads in the block reuse it. This reduces global memory accesses by a factor of the tile size (16x).

**Arithmetic intensity:** O(n) FLOPS/byte -- compute-bound. This is why GPUs are so good at matrix multiplication: there is enough compute per byte transferred to keep the hardware busy.

**Agent parallel:** this is the pattern when agents need shared context. Instead of each agent fetching the same knowledge base independently (naive), you load shared context once and distribute it (tiled). In practice: prompt caching, shared vector store lookups, batched API calls.

### 3. Reduction (Parallel Aggregation)

Combine many values into one (sum, max, vote). Requires coordination between threads.

Three progressive optimizations:

| Approach | Speedup (16M elements) | Issue |
|----------|----------------------|-------|
| Interleaved addressing | 21.9x | Warp divergence, bank conflicts |
| Sequential addressing | 27.4x | Eliminates divergence |
| Add-on-load | 47.8x | Processes multiple elements per thread during load |

**Why sequential addressing wins:** interleaved addressing causes "warp divergence" -- threads within the same group take different code paths, so some sit idle while others work. Sequential addressing ensures all active threads execute the same instruction.

**Agent parallel:** this is the **voting** pattern from Anthropic -- multiple agents solve the same problem, then results are aggregated. The reduction teaches that aggregation strategy matters. Naive voting (wait for all, then pick majority) is like interleaved addressing. Smarter aggregation (early termination when consensus is clear, weighted by confidence) is like the optimized versions.

### 4. Convolution (Sliding Window with Reuse)

Each output depends on a local neighborhood of inputs. High data reuse because input elements contribute to multiple outputs.

| Image Size | Filter | Speedup |
|-----------|--------|---------|
| 400x400 | 3x3 | 228x |
| 400x400 | 5x5 | 344x |
| 4096x4096 | 5x5 | 433x |

**Why it scales so well:** high arithmetic intensity. Each input pixel participates in (filter_width)^2 output calculations. The ratio of compute to memory access is excellent.

**Agent parallel:** this is the pattern when agents process overlapping contexts -- like a document processing pipeline where each agent handles a section but needs surrounding context for coherence. The overlap (like the convolution filter) enables better results at the boundary.

### 5. Scan (Prefix Sum -- Dependent Operations)

Each output depends on all previous inputs. Inherently sequential... except it is not. Parallel scan algorithms exist but are counterintuitive.

| Approach | Work Complexity | Actual Performance |
|----------|----------------|-------------------|
| Naive (tree-based) | O(n log n) | 25.7x speedup |
| Work-efficient (Blelloch) | O(n) | 23.5x speedup |

**The scan paradox:** the theoretically more efficient algorithm (O(n) work) is *slower* in practice than the O(n log n) version. The Blelloch scan has complex up-sweep/down-sweep phases that create irregular memory access patterns and warp divergence. The naive version has simple, predictable access patterns that GPUs handle better.

**Agent parallel:** this is the sequential workflow pattern -- each step depends on the previous one. The scan teaches that sometimes sequential dependencies cannot be fully parallelized, and that a simpler approach with slightly more total work can outperform a complex "optimal" one. Anthropic's guidance: start with simple workflows before complex orchestration.

---

## Block Size Selection (Configuration Matters)

Choosing the right thread block size is critical:

| Block Size | Threads | Typical Occupancy | Best For |
|-----------|---------|-------------------|----------|
| 8x8 (64) | 64 | 25% | Underutilizes hardware |
| 16x16 (256) | 256 | 75% | Matrix multiplication, convolution |
| 32x32 (1024) | 1024 | 50% | Register pressure reduces efficiency |
| 512 (1D) | 512 | Full | Reduction, scan |

The sweet spot is usually 256-512 threads per block. Too few threads wastes hardware. Too many creates resource pressure (registers, shared memory) that reduces the number of blocks that can run simultaneously.

**Agent parallel:** same tradeoff. Too few parallel agents underutilizes your API throughput. Too many creates context window pressure, rate limiting, and coordination overhead. The config-driven approach from the overview page lets you tune this per-agent-system.

---

## Cross-Pattern Performance Summary

| Pattern | Best Speedup | Bottleneck | Agent Equivalent |
|---------|-------------|-----------|-----------------|
| Element-wise (SAXPY) | 33x | Memory bandwidth | Independent parallel agents |
| Matrix multiply (tiled) | 437x | Compute | Shared-context batch processing |
| Reduction | 48x | Synchronization | Voting/aggregation |
| Convolution | 433x | Compute | Overlapping-context pipelines |
| Scan | 26x | Sequential dependency | Sequential workflows |

Memory-bound operations (SAXPY, reduction) hit 33-48x speedup. Compute-bound operations (matrix multiply, convolution) hit 433-437x. The lesson: if your agent system is I/O-bound (waiting on API calls), parallelization helps linearly. If it is compute-bound (LLM inference itself), batching and caching help dramatically more.

---

## Connecting to the Agent Architecture

These GPU concepts map directly to agent system design decisions:

- **Arithmetic intensity -> Task complexity per agent call.** If each agent does minimal work per API call, you are memory-bound (I/O-bound). Batch more work per call.
- **Shared memory -> Shared context.** Loading the same knowledge base into every agent call is like reading from global memory every time. Prompt caching is the equivalent of shared memory tiling.
- **Warp divergence -> Agent heterogeneity.** If agents in a parallel batch need very different instructions, some will finish fast and wait for others. Keep parallel agents homogeneous.
- **Block size -> Parallelism degree.** Match the number of concurrent agents to your throughput capacity, not to the maximum possible.
- **Kernel launch overhead -> Agent startup cost.** Do not parallelize tiny tasks. The overhead of spinning up an LLM call (context assembly, API latency) dominates for small inputs.

---

## References

- Kirk & Hwu, *Programming Massively Parallel Processors* (4th ed.)
- NVIDIA CUDA Programming Guide
- Anthropic, "Building Effective Agents" -- parallelization patterns
