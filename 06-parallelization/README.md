# Parallelization: finding the floor

**Source:** reference/06-support-gpu-parallelization.md

## The claim

Every parallel execution strategy has a size below which it is slower than doing the work
in order, and the crossover is measurable rather than a matter of taste. Run the two
scripts and they find that floor on the machine you run them on. If parallelism were free
below some threshold, both would show the parallel line winning at every size, and neither
does.

Two scripts, one argument. Every parallel execution strategy has a size below which it
is slower than doing the work in order, because the setup cost is paid per launch while
the saving is proportional to the work. GPUs make this vivid because their setup cost is
large and their per-element cost is tiny. Agent systems have the same shape with
different constants: the launch cost is a planning call and a fan-out, and the per-task
saving is a network round trip you no longer wait for serially.

This directory is not a `before.py` / `after.py` pair. There is no classical algorithm
being upgraded here. It is the measurement that makes the floor concrete, and it is
split in two because the two halves have very different evidentiary status.

## Run it

    python 06-parallelization/benchmark_floor.py
    python 06-parallelization/agent_floor.py
    python 06-parallelization/agent_floor.py --latency 0.5

`benchmark_floor.py` took 10.2 seconds and `agent_floor.py` 8.7 seconds on the machine
described below. `benchmark_floor.py` is standard library only; it will use numpy for one
extra reference line if numpy happens to be installed, and prints a note instead if it is
not. It calls no model at all.

`agent_floor.py` needs no API key: with no backend configured it replays the model
responses recorded in `shared/transcripts/06_parallelization__agent_floor.json` -- the
routing calls from `claude-haiku-4-5` at tier `small`, the planning calls from
`claude-sonnet-5` at tier `mid`, all recorded 2026-08-04. Those are real answers a real
model gave to these exact prompts on that date, not strings anybody wrote to make the
demo come out. They are also one run, on one day, against model versions that move; a
live run will not reproduce them exactly.

The `--latency 0.5` variant is closer to a real API round trip and takes well over a
minute, because the sequential column has to actually wait.

---

## Section A -- PUBLISHED FIGURES. NOT MEASURED HERE.

Everything in this section is quoted from `reference/06-support-gpu-parallelization.md`,
which in turn cites Kirk and Hwu's *Programming Massively Parallel Processors* (4th ed.),
the NVIDIA CUDA Programming Guide, and Anthropic's "Building Effective Agents".

**No GPU was involved in the CPU measurements below.** These published numbers were not
reproduced here, not verified here, and not measured on any machine this code has
executed on. No hardware, driver version, or CUDA toolkit is specified for them on the
source page, so they cannot be reproduced from it either. They are claims belonging to
that page. Attribute them to it, not to this repo.

Element-wise SAXPY, as published:

| Problem size | CPU time | GPU time | Speedup |
|---|---|---|---|
| 10K | 0.015 ms | 0.014 ms | 1.07x |
| 1M | 1.5 ms | 0.05 ms | 30x |
| 100M | 150 ms | 4.5 ms | 33x |

Cross-pattern summary, as published:

| Pattern | Best speedup | Bottleneck |
|---|---|---|
| Element-wise (SAXPY) | 33x | Memory bandwidth |
| Matrix multiply (tiled) | 437x | Compute |
| Reduction | 48x | Synchronization |
| Convolution | 433x | Compute |
| Scan | 26x | Sequential dependency |

The source page also publishes figures for block-size occupancy, three reduction
strategies, and a context-reduction claim. They are on that page. They are not repeated
here, because repeating a number is how it stops being attributed.

The one thing this repo takes from Section A is the *shape* of the claim -- "below 10K
elements, kernel launch overhead makes GPU slower than CPU" -- and that shape is what
Section B tests, on hardware it can actually describe.

---

## Section B -- MEASURED HERE.

Every number in this section came out of the two scripts in this directory, or out of the
Space in `hf-space/` for the GPU ones. Run them and you will get different numbers, because
you have a different machine. That is the point, and the scripts say so in their own output
rather than only here.

### On a real GPU, which this machine does not have

Section A quotes 33x for SAXPY and 437x for a tiled matrix multiply, as published figures.
This directory could not check them, because there is no GPU here. `hf-space/` is that
directory's missing half: the same measurement on CUDA, deployed as a Hugging Face Space
on ZeroGPU, with the code in this repository.

    https://huggingface.co/spaces/jmurray10/peas-parallelization-floor

One run, 2026-08-09, on `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb`
(compute capability 12.0, 47.4 GiB visible, torch 2.11.0+cu130), against numpy 2.5.1 on
the Space's CPU. ZeroGPU hands out a slice of a card rather than a card.

    ELEMENT-WISE  y = a*x + y

        elements     cpu ms     gpu ms   kernel ms   speedup   kernel-only
           1,000      0.002      0.059       0.016     0.03x         0.12x
         100,000      0.025      0.160       0.015     0.15x         1.64x
       1,000,000      0.573      0.886       0.018     0.65x        32.26x
      10,000,000      6.891     28.511       0.175     0.24x        39.43x
      30,000,000     37.733     94.223       0.499     0.40x        75.69x

    MATRIX MULTIPLY  C = A @ B, square, float32

               n     cpu ms     gpu ms   kernel ms   speedup   kernel-only
              64      0.006      0.071       0.022     0.09x         0.27x
             256      0.043      0.124       0.021     0.35x         2.03x
             512      0.172      0.355       0.035     0.48x         4.89x
            1024      1.014      0.975       0.088     1.04x        11.55x
            2048      7.413     12.317       0.492     0.60x        15.07x

The element-wise operation never crossed over. At every size tested, counting the two
transfers, the CPU finished first -- and the kernel-only column reached 75x on the same
row where the honest column reads 0.40x. That gap is the whole argument of this directory
stated in one line: a kernel-only speedup is a real number about the kernel and not an
available number about your program, unless your data already lives on the device and
stays there.

The matrix multiply did cross over, at n = 1024. Arithmetic grows as n^3 while transfers
grow as n^2, so the ratio moves in the GPU's favour as the problem grows. That is the same
mechanism as the source page's tiling lesson, arriving from the other direction.

Two runs minutes apart on the same Space gave 54x and 76x kernel-only at 30 million
elements. Shared hardware, one sample each. Neither is a benchmark, and the crossover
structure -- element-wise never, matmul at about a thousand -- held across both, which is
the part worth carrying away.

### Hardware for the sample runs below

| | |
|---|---|
| CPU | Intel Core i7-10875H, 8 cores / 16 logical |
| OS | Windows 11, 10.0.26200 |
| Python | 3.13.14, `spawn` start method |
| GPU used | none |

This is one laptop. It is not a claim about your machine, and it is not a claim about
this laptop on a different day.

### benchmark_floor.py -- CPU, element-wise, processes

SAXPY over a range of sizes: one process versus a `multiprocessing.Pool` of 8 workers.
The kernel is the same pure-Python function in both columns so the comparison is one
variable wide. The pool is started once before timing and reused; its startup cost is
measured, printed, and deliberately not charged to any row.

    pool startup      613.9 ms (paid once, before timing, not charged to any row below)

            size    sequential      parallel    speedup  verdict
    ------------------------------------------------------------
           1,000       0.22 ms       2.14 ms      0.10x  parallel LOSES
           5,000       0.86 ms       2.90 ms      0.30x  parallel LOSES
          10,000       1.36 ms       3.53 ms      0.39x  parallel LOSES
          25,000       5.10 ms       4.61 ms      1.11x  parallel wins
          50,000      11.85 ms       6.70 ms      1.77x  parallel wins
         100,000      20.69 ms       9.89 ms      2.09x  parallel wins
       1,000,000     183.69 ms      96.60 ms      1.90x  parallel wins
       4,000,000     772.77 ms     455.62 ms      1.70x  parallel wins

    Crossover on this machine: between 10,000 and 25,000 elements.

A floor exists and parallel loses below it. Three things worth reading off this table
that were not designed into it:

- The speedup never approaches 8x on 8 workers. Element-wise SAXPY is memory-bound, and
  the chunks have to be pickled out to the workers and results pickled back. Data
  movement dominates, exactly as the source page argues, and it dominates on a CPU with
  no GPU anywhere near it.
- The speedup peaks at 100,000 elements in this run and falls off from there. That is
  not a second crossover, it is bandwidth saturation.
- Pool startup, at 613.9 ms in this run, is roughly 2,800 times the cost of the entire
  1,000-element sequential run, and longer than the first six rows of the table put
  together -- both columns, about 70 ms. A pool created per batch rather than reused
  moves the floor up by orders of magnitude. Reuse is not an optimization here, it is the
  difference between the technique working and not.

The script also times a numpy vectorized SAXPY at the largest size, in one process, no
pool. In the run above that took **38.54 ms** against 455.62 ms for the eight-process
version and 772.77 ms sequential. Vectorizing one process beat parallelizing across
eight. Before reaching for a process pool, check whether the single-process version can
be made to stop being the bottleneck.

### agent_floor.py -- agent calls, threads

The same shape one layer up. Independent support-ticket routing calls through
`shared/llm.py`, sequential versus a `ThreadPoolExecutor`. Threads rather than processes
because the work is I/O-bound by construction -- an agent call is nearly all waiting.

The concurrent path pays a fixed cost the sequential path does not: an orchestrator
planning call that decides the fan-out width. That fixed cost is what puts a floor under
it.

      tasks    sequential    concurrent    speedup  width  verdict
    --------------------------------------------------------------
          1       55.5 ms      190.3 ms      0.29x      1  concurrent LOSES
          2      115.2 ms      191.0 ms      0.60x      2  concurrent LOSES
          3      169.5 ms      185.4 ms      0.91x      3  concurrent LOSES
          4      223.0 ms      187.8 ms      1.19x      4  concurrent wins
          6      333.0 ms      184.0 ms      1.81x      6  concurrent wins
          8      430.5 ms      189.9 ms      2.27x      8  concurrent wins
         12      644.7 ms      248.8 ms      2.59x      8  concurrent wins
         16      854.1 ms      251.2 ms      3.40x      8  concurrent wins
         24     1281.6 ms      303.1 ms      4.23x      8  concurrent wins
         32     1716.9 ms      358.3 ms      4.79x      8  concurrent wins

    Crossover at these settings: between 3 and 4 tasks.

**What is assumed and what is measured, in this table specifically.** The per-call
latency is a fixed 50 ms `sleep` injected by `agent_floor.py`, standing in for a network
round trip and a remote forward pass. It is an assumption, not a measurement of any
provider's API, and a replayed response returns in microseconds, so without the sleep
there would be nothing to measure. The planning call is charged 2.5x a worker call on the
assumption that a fan-out prompt and its output are longer. Both constants are printed by
the script and the first is settable with `--latency`.

What is measured is every millisecond in the table: real wall clock, real thread
creation, real queue handoff, real result collection. What is replayed is the routing
label each call comes back with -- a real model's answer, recorded, and identical on
every run, which is what makes the two columns comparable at all. A live run pays real
latency instead of the simulated 50 ms and can route a ticket differently.

Scaling the latency does not move the floor much, which is worth understanding rather
than assuming. Both the planning call and the worker calls scale together, so the
crossover is set by their *ratio*, not by the absolute latency. Push the latency down far
enough and something else takes over: at `--latency 0.001` on this machine the crossover
sat between 2 and 3 tasks in two consecutive runs, and the ceiling collapsed -- the best
speedup over all ten batch sizes fell from 4.79x to about 3x, because real thread and
queue overhead stopped being negligible next to a one-millisecond call. Concurrency
started winning marginally sooner and stopped paying nearly as well.

That is the useful version of the claim. A floor exists because a fixed orchestration
cost divided by a growing per-task saving crosses one somewhere, and cutting your
per-call latency does not remove the problem -- it just changes which fixed cost is the
one that matters. The number to take from this table is not "four". It is "find yours,
and know which constant is holding it there".

### The fan-out width is decided by code, not by the model

The planning call asks a mid-tier model for a concurrency number. The script then
refuses to trust it. Both paths are exercised before timing starts, and the script
prints both:

    Control-flow check on the planner, before timing anything:
      well-formed JSON    -> concurrency=8  (model proposed 16, clamped to 8)
      prose, not JSON     -> concurrency=8  (model proposed 16, clamped to 8)

Read those two lines carefully, because the second one no longer does what its label
says. Both calls build the same prompt, and both used to be steered to different canned
strings by `mock_key`. Recordings are keyed by the content of the prompt, so `mock_key`
now selects nothing: the same prompt replays the same recorded answer twice, and the
prose branch of `plan_concurrency` is not reached offline. The branch is still there,
still returns `min(MAX_CONCURRENCY, task_count)`, and is still what runs against a live
model that answers in prose. The offline run can no longer show it to you.

What the offline run does show is the clamp. `claude-sonnet-5` proposed 16 for a batch of
16; `MAX_CONCURRENCY` cut it to 8. The model proposes; `MAX_CONCURRENCY` and the batch
size dispose. This is the source page's block-size lesson wearing different clothes:
match parallelism to your throughput capacity, not to the maximum anything will accept.

## What changed

The model proposes a fan-out width and nothing else. `plan_concurrency` asks a mid-tier
model how many tasks to run at once, and that is the whole of the LLM's involvement in
this directory.

Everything that decides what actually happens stayed deterministic: `MAX_CONCURRENCY`
clamps whatever the model proposes, the batch size bounds it again, the fan-out is a
thread pool, and the timing is `time.perf_counter`. The recorded run has the model
proposing 16 for a batch of 16 and the clamp cutting it to 8 -- the model proposes,
the constants dispose. A model that answered 10,000 would change nothing about the run.

## What it costs

Parallelism buys wall-clock time back and spends everything else. Ordering becomes
non-deterministic, so anything that depends on sequence has to be reimposed
deliberately. Failures stop being one failure and become a partial result set that some
code now has to reason about. Debugging gets worse in proportion to the width. Cost goes
up before latency comes down, since the planning call is spent whether the fan-out was
worth it or not. And below the floor, all of that is paid for a program that is slower
than the loop it replaced.
