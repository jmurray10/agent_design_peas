# Colab notebooks

Three notebooks. One is the argument itself, run end to end without installing anything.
The other two are for what a README cannot show you: a GPU this repository does not have,
and a live model call you have not paid for.

    oscillation.ipynb   the series' central claim: before.py and after.py side by side
                        across five architectures, same algorithm, one component swapped
    gpu_floor.ipynb     the parallelization floor, measured on whatever GPU Colab gives you
    agents_live.ipynb   nine agents through one runtime, replaying or live

Start with `oscillation.ipynb`. It is the argument itself, needs no key and no GPU, and
every pair in it prints both halves so the claim is visible or it is false.

Open them from the repository page on GitHub -- Colab has a "Open in Colab" path for any
notebook in a public repo:

    https://colab.research.google.com/github/jmurray10/agent_design_peas/blob/main/colab/oscillation.ipynb
    https://colab.research.google.com/github/jmurray10/agent_design_peas/blob/main/colab/gpu_floor.ipynb
    https://colab.research.google.com/github/jmurray10/agent_design_peas/blob/main/colab/agents_live.ipynb

**Those links need the repository to be public.** While it is private they will 404 for
everyone including you, and the notebooks' first cell -- a `git clone` -- will fail the
same way. Nothing else about them changes when the repository flips.

## Why a notebook and not just the Space

`06-parallelization/hf-space/` runs the GPU measurement too, and there is a real
difference. The Space shows you a number measured on a slice of a shared card, by me,
minutes ago. The notebook measures on the GPU Colab allocated to you, now, and you can
edit the sizes and run it again.

For a repository whose argument is "clone it and check it rather than take my word for
it", the second one is the point and the first one is a convenience.

Both import the same `06-parallelization/gpu_floor.py`. Two front ends, one
implementation, so a number quoted from either came out of the same code -- the same
reason `02-goal-based/csp/` imports its solver instead of reimplementing it.

## What each one needs

`gpu_floor.ipynb` needs a GPU runtime and no key. Runtime -> Change runtime type ->
Hardware accelerator -> GPU. The first cell prints what you were given, and the
measurement says so itself if CUDA is not visible. No model is called anywhere in it.

`agents_live.ipynb` needs neither, and takes an optional Anthropic key from Colab's
Secrets panel. Without one it replays recorded real responses; with one it calls the
model. The percept in the last cell ships with a recording, so it works either way --
and editing it without a key raises rather than replaying an answer to a question nobody
asked, which is the behaviour the repository argues for rather than a limitation of the
notebook.
