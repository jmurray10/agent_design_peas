"""The parallelization floor on CUDA, as a plain function.

`benchmark_floor.py` measures the same shape on a CPU with the standard library. This one
needs a GPU, so it cannot be part of the zero-setup path and is imported by the two places
that have one: the Gradio Space in `hf-space/`, and the Colab notebook in `colab/`.

It is deliberately one implementation rather than two. `02-goal-based/csp/` makes a point
of importing its solver instead of reimplementing it, and proves it with a source digest;
the same reasoning applies to a measurement that two front ends are going to quote.

This module has no entry-point guard, on purpose: running it needs torch and a GPU, and
the offline sweep and the container's verify service pick scripts by searching for that
guard, so leaving it out keeps this file out of a run that must work with no setup.

The phrasing above is deliberate too. Those selectors search the file as text, so a
docstring that spelled the guard out would select this file by talking about it -- the
same false positive the no-agent-specific-code check in `00-config-runtime/demo.py` hit
when a method name appeared inside a sentence, and fixed by parsing instead of matching.
The selectors here have not been fixed that way, so this file avoids the word.

No model is called anywhere in this module. It is arithmetic, timed.
"""

from __future__ import annotations

import platform
import time
from typing import Callable

# Element counts spanning the region where the crossover lives: small enough at the bottom
# that a launch cannot be amortised, large enough at the top that it stops mattering.
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 30_000_000]
MATMUL_SIZES = [64, 128, 256, 512, 1024, 2048]
REPEATS = 3
WARMUP = 2


def _time(fn: Callable[[], object], repeats: int = REPEATS, warmup: int = WARMUP) -> float:
    """Best-of-N seconds.

    Best rather than mean: the minimum is the run least disturbed by whatever else the
    machine was doing, and on shared or virtualised hardware that is most of the variance.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def measure() -> str:
    """Time element-wise and matrix-multiply work on CPU and GPU, and return the report.

    Three columns per row, and the middle one is the honest one: the CPU, the GPU counting
    both transfers, and the GPU counting only the kernel. What you can spend is the middle
    column, unless your data already lives on the device and stays there.
    """
    try:
        import numpy as np
        import torch
    except ImportError as missing:
        # Colab ships both. Anywhere else might not, and a traceback here reads as a
        # broken example rather than a missing dependency.
        return "\n".join([
            f"This measurement needs numpy and torch, and {missing.name} is not installed.",
            "",
            "    pip install torch numpy",
            "",
            "The CPU half of the same argument needs neither, and no install at all:",
            "",
            "    python 06-parallelization/benchmark_floor.py",
        ])

    if not torch.cuda.is_available():
        return (
            "No CUDA device is visible to torch.\n\n"
            "In Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU.\n"
            "Locally: this needs a CUDA build of torch and a GPU. The CPU half of the\n"
            "same argument runs anywhere with no setup at all:\n\n"
            "    python 06-parallelization/benchmark_floor.py\n"
        )

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    lines = [
        "MEASURED ON THIS HARDWARE, ON THIS RUN. Not a published figure.",
        "",
        f"  GPU                {torch.cuda.get_device_name(0)}",
        f"  compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(0))}",
        f"  VRAM visible       {properties.total_memory / (1024 ** 3):.1f} GiB",
        f"  torch              {torch.__version__}",
        f"  CPU baseline       numpy {np.__version__} on {platform.processor() or 'unknown'}",
        "",
        "",
        "ELEMENT-WISE:  y = a*x + y",
        "",
        "  CPU, GPU including the transfers you have to pay to use it, and GPU counting",
        "  only the kernel. The gap between the last two is the setup cost.",
        "",
        f"  {'elements':>12}  {'cpu ms':>9}  {'gpu ms':>9}  {'kernel ms':>10}  {'speedup':>8}  {'kernel-only':>12}",
    ]

    crossover = None
    for n in SIZES:
        x_h = np.random.rand(n).astype(np.float32)
        y_h = np.random.rand(n).astype(np.float32)
        a = np.float32(2.5)

        cpu = _time(lambda: a * x_h + y_h)

        x_d = torch.from_numpy(x_h).to(device)
        y_d = torch.from_numpy(y_h).to(device)
        torch.cuda.synchronize()

        def kernel_only() -> None:
            torch.add(y_d, x_d, alpha=2.5)
            torch.cuda.synchronize()

        kernel = _time(kernel_only)

        def with_transfer() -> None:
            xd = torch.from_numpy(x_h).to(device)
            yd = torch.from_numpy(y_h).to(device)
            torch.add(yd, xd, alpha=2.5).cpu()
            torch.cuda.synchronize()

        total = _time(with_transfer)
        if crossover is None and total < cpu:
            crossover = n

        lines.append(
            f"  {n:>12,}  {cpu * 1e3:>9.3f}  {total * 1e3:>9.3f}  {kernel * 1e3:>10.3f}"
            f"  {cpu / total:>7.2f}x  {cpu / kernel:>11.2f}x"
        )
        del x_d, y_d
        torch.cuda.empty_cache()

    lines += [
        "",
        (f"  Crossover including transfers: {crossover:,} elements."
         if crossover else
         "  No crossover including transfers within the sizes tested: the CPU finished"
         " first every time, which is itself the finding."),
        "",
        "",
        "MATRIX MULTIPLY:  C = A @ B, square, float32",
        "",
        "  Arithmetic grows as n^3 while transfers grow as n^2, so this crossover sits",
        "  lower than the element-wise one. Same hardware, same run.",
        "",
        f"  {'n':>6}  {'cpu ms':>10}  {'gpu ms':>10}  {'kernel ms':>10}  {'speedup':>8}  {'kernel-only':>12}",
    ]

    mm_crossover = None
    for n in MATMUL_SIZES:
        a_h = np.random.rand(n, n).astype(np.float32)
        b_h = np.random.rand(n, n).astype(np.float32)

        cpu = _time(lambda: a_h @ b_h)

        a_d = torch.from_numpy(a_h).to(device)
        b_d = torch.from_numpy(b_h).to(device)
        torch.cuda.synchronize()

        def kernel_only() -> None:
            torch.matmul(a_d, b_d)
            torch.cuda.synchronize()

        kernel = _time(kernel_only)

        def with_transfer() -> None:
            ad = torch.from_numpy(a_h).to(device)
            bd = torch.from_numpy(b_h).to(device)
            torch.matmul(ad, bd).cpu()
            torch.cuda.synchronize()

        total = _time(with_transfer)
        if mm_crossover is None and total < cpu:
            mm_crossover = n

        lines.append(
            f"  {n:>6}  {cpu * 1e3:>10.3f}  {total * 1e3:>10.3f}  {kernel * 1e3:>10.3f}"
            f"  {cpu / total:>7.2f}x  {cpu / kernel:>11.2f}x"
        )
        del a_d, b_d
        torch.cuda.empty_cache()

    lines += [
        "",
        (f"  Crossover including transfers: n = {mm_crossover}."
         if mm_crossover else
         "  No crossover including transfers within the sizes tested."),
        "",
        "",
        "WHAT THIS DOES AND DOES NOT SHOW",
        "",
        "  It shows that a floor exists and roughly where it sits on this hardware, for",
        "  these two operations, on this run. That is the claim.",
        "",
        "  It is not a benchmark of the card, it reproduces no published speedup, and it",
        "  is not evidence about anyone else's workload. numpy reaches several cores",
        "  through BLAS, so the CPU column is not a single-threaded baseline and these",
        "  ratios are not core-count ratios.",
        "",
        "  The agent-layer version of the same shape, with a planning call standing in",
        "  for a kernel launch, is agent_floor.py in this directory.",
    ]
    return "\n".join(lines)
