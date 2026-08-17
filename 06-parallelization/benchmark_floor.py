"""Where does parallelizing element-wise work start to pay for itself?

The source page for this directory reports GPU numbers. This machine has no GPU and
this script does not pretend otherwise. It measures a different thing that has the same
shape: SAXPY (`z[i] = a * x[i] + y[i]`) run sequentially in one process versus split
across a `multiprocessing.Pool`, over a range of problem sizes, on whatever CPU you
happen to be sitting in front of.

The claim under test is structural, not numeric: there is a size below which the
parallel version is slower, because dispatch overhead is paid per launch while the
saving is proportional to the work. The size itself is a property of this machine and
this run. Every number printed below was produced by this script, here, now.

Two deliberate choices, both stated in the output so nobody has to read the source to
know how the numbers were made:

  1. The worker pool is started once, before timing, and reused. Process startup is
     measured and reported separately rather than charged to any row. Charging it per
     row would be measuring "how long does Windows take to spawn interpreters", which
     is a real cost but not the one this benchmark is about.
  2. The kernel is the same pure-Python function in both columns, so the comparison is
     one variable wide. Vectorizing it with numpy would make the sequential column much
     faster -- which is itself worth knowing, so the script measures that separately at
     the end when numpy is importable.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from array import array
from multiprocessing import Pool, cpu_count, get_start_method

A = 2.0

# Deliberately dense between 10K and 100K: on the machines this was developed on the
# crossover lands somewhere in that band, and a table that steps straight from 10K to
# 1M hides it. Yours may land elsewhere; that is the point of running it yourself.
SIZES: tuple[int, ...] = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 1_000_000, 4_000_000)


def saxpy_chunk(job: tuple[float, array, array]) -> array:
    """Compute `a * x + y` element-wise over one chunk.

    Module scope is not a style preference here. Windows (and macOS) start worker
    processes with `spawn`, which re-imports this module in each child and looks the
    function up by name -- a closure or a nested def would fail to pickle.
    """
    a, x, y = job
    return array("d", [a * xi + yi for xi, yi in zip(x, y)])


def saxpy_sequential(a: float, x: array, y: array) -> array:
    """One process, one pass. Same kernel the workers run, called once."""
    return saxpy_chunk((a, x, y))


def saxpy_parallel(pool: Pool, a: float, x: array, y: array, workers: int) -> array:
    """Split, dispatch, collect, reassemble.

    Splitting and reassembly are inside this function on purpose. They are not
    bookkeeping you can wish away -- they are the price of admission for the parallel
    path, and a benchmark that times only the dispatch is measuring a program nobody
    can actually run.
    """
    n = len(x)
    step = (n + workers - 1) // workers
    jobs = [(a, x[i:i + step], y[i:i + step]) for i in range(0, n, step)]
    out = array("d")
    for part in pool.map(saxpy_chunk, jobs):
        out.extend(part)
    return out


def best_of(fn, repeats: int) -> float:
    """Fastest of `repeats` runs, in seconds.

    Minimum rather than mean: the noise on a shared desktop is one-sided. Background
    work can only make a run slower, never faster, so the fastest observation is the
    closest thing to a measurement of the code rather than of the machine's mood.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def repeats_for(n: int) -> int:
    """Fewer repeats on the big sizes so the whole script stays under ~10 seconds."""
    if n <= 10_000:
        return 5
    if n <= 1_000_000:
        return 3
    return 2


def numpy_reference(a: float, n: int) -> tuple[float, str] | None:
    """Time a vectorized single-process SAXPY, if numpy is installed.

    Returned rather than printed so the caller decides how to label it. `None` means
    numpy is absent, which is not an error -- everything above this line is standard
    library only, and the script is expected to run without it.
    """
    try:
        import numpy
    except ImportError:
        return None

    x = numpy.arange(n, dtype=numpy.float64)
    y = numpy.arange(n, dtype=numpy.float64)
    seconds = best_of(lambda: a * x + y, repeats_for(n))
    return seconds, numpy.__version__


def main() -> None:
    workers = min(cpu_count() or 1, 8)

    print("Parallelization floor -- element-wise SAXPY, z[i] = a*x[i] + y[i]")
    print()
    print("This machine, this run. Nothing below is copied from the source page.")
    print(f"  platform          {platform.platform()}")
    print(f"  python            {sys.version.split()[0]}")
    print(f"  logical CPUs      {os.cpu_count()}")
    print(f"  pool workers      {workers}")
    print(f"  start method      {get_start_method()}")
    print()

    # Started once and reused. The cost of starting it is measured, printed, and then
    # kept out of the table -- see the caveats at the bottom for why that matters.
    warmup_job = (A, array("d", [1.0]), array("d", [1.0]))
    start = time.perf_counter()
    with Pool(processes=workers) as pool:
        pool.map(saxpy_chunk, [warmup_job] * workers)
        pool_startup = time.perf_counter() - start
        print(f"  pool startup      {pool_startup * 1000:.1f} ms "
              f"(paid once, before timing, not charged to any row below)")
        print()

        header = f"{'size':>12}  {'sequential':>12}  {'parallel':>12}  {'speedup':>9}  verdict"
        print(header)
        print("-" * len(header))

        crossover_from: int | None = None
        crossover_to: int | None = None
        last_losing_size: int | None = None
        every_size_lost = True

        for n in SIZES:
            x = array("d", range(n))
            y = array("d", range(n))
            repeats = repeats_for(n)

            sequential = best_of(lambda: saxpy_sequential(A, x, y), repeats)
            parallel = best_of(lambda: saxpy_parallel(pool, A, x, y, workers), repeats)
            speedup = sequential / parallel

            won = speedup > 1.0
            if won:
                every_size_lost = False
                # Freeze the boundary at the FIRST win. Memory-bound work can lose
                # again at larger sizes once bandwidth saturates, and that later loss
                # is not a second crossover -- it is a different wall.
                if crossover_to is None and last_losing_size is not None:
                    crossover_from, crossover_to = last_losing_size, n
            else:
                last_losing_size = n

            print(f"{n:>12,}  {sequential * 1000:>9.2f} ms  {parallel * 1000:>9.2f} ms  "
                  f"{speedup:>8.2f}x  {'parallel wins' if won else 'parallel LOSES'}")

    print()
    if crossover_to is not None:
        print(f"Crossover on this machine: between {crossover_from:,} and {crossover_to:,} "
              f"elements. Below it, splitting the work costs more than doing it.")
    elif every_size_lost:
        print(f"No crossover inside the sizes tested -- parallel lost at every size up to "
              f"{SIZES[-1]:,}. The floor on this machine is above the top of this table.")
    else:
        print("Parallel won at every size tested, including the smallest. The floor on "
              "this machine is below the bottom of this table.")

    reference = numpy_reference(A, SIZES[-1])
    print()
    if reference is None:
        print("numpy not installed, so the vectorized reference row was skipped. "
              "Everything above is standard library only.")
    else:
        seconds, version = reference
        print(f"Reference, same size ({SIZES[-1]:,}), one process, no pool: "
              f"numpy {version} vectorized SAXPY ran in {seconds * 1000:.2f} ms.")
        print("Measured here too. Worth sitting with before reaching for a process pool: "
              "for memory-bound element-wise work, vectorizing one process can beat "
              "splitting the work across several.")

    print()
    print("Caveats, and they are not boilerplate:")
    print("  - No GPU was involved in anything this script ran. The speedup figures on")
    print("    the source page are published GPU results, not these. See README.md.")
    print("  - The crossover size is a property of this CPU, this Python build, and what")
    print("    else the machine was doing. Run it twice and the boundary may move.")
    print("  - Pool startup, printed above, is excluded from the table. Include it and the")
    print("    floor rises sharply: a pool you start per batch has to amortize that cost")
    print("    before it can amortize anything else.")
    print("  - The kernel is pure Python in both columns so the comparison is one variable")
    print("    wide. A C or numpy kernel changes the sequential column, and therefore the")
    print("    crossover, without changing the shape of the argument.")


if __name__ == "__main__":
    # Required on any platform that starts workers with `spawn`: without it, each child
    # re-imports this module, re-runs main(), and forks its own pool, forever.
    main()
