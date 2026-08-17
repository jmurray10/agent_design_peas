---
title: The Parallelization Floor, Measured
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
hardware: zero-a10g
python_version: "3.12"
pinned: false
license: mit
short_description: Where a GPU stops being worth the transfer
---

# The parallelization floor, measured on real CUDA

Companion to [jmurray10/agent_design_peas](https://github.com/jmurray10/agent_design_peas), which
argues that classical agent architectures survive the arrival of LLMs and that an LLM
replaces one component inside them rather than the architecture.

One directory in that repository is about parallelism rather than agents. Its claim: every
parallel execution strategy has a size below which it is slower than doing the work in
order, because setup is paid per launch while the saving is proportional to the work. It
measures that on a CPU, and at the agent layer with a planning call standing in for a
kernel launch.

It could not measure it on a GPU, because there is no GPU on the machine it runs on. That
is what this Space is for.

## What it prints

Two operations, element-wise `y = a*x + y` and a square matrix multiply, each at six sizes,
timed three ways:

- the CPU, via numpy
- the GPU including both transfers, which is what offloading actually costs
- the GPU counting the kernel alone

The third column is the one that gets quoted. The second is the one you get if your data
starts on the host and has to go back, which for most workloads it does. The distance
between them is the setup cost, and the size at which the second column finally beats the
CPU is the floor.

## What it is not

It is not a benchmark of the card. ZeroGPU hands out a slice rather than a whole GPU, the
Space is shared, and one run is one sample.

It does not reproduce any published figure. The source material behind the repository
quotes 33x for SAXPY and 437x for a tiled matrix multiply; those are cited there as
published figures and were not produced by that repository or by this Space. Whatever this
prints is what this hardware did on this run, which is a different kind of claim and the
only kind either project makes.

numpy reaches several cores through BLAS, so the CPU column is not a single-threaded
baseline and the ratios are not core-count ratios.

No model is called anywhere in this Space. It is arithmetic, timed.
