"""A button in front of the measurement in gpu_floor.py.

Everything that decides a number lives in `gpu_floor.py`, which is one file in the
repository this Space belongs to and is uploaded beside this one rather than copied into
it. The Colab notebook in `colab/` imports the same module. Two front ends, one
implementation, so a result quoted from either is the same measurement.

There is no model in this Space. It is arithmetic, timed.
"""

import gradio as gr
import spaces

from gpu_floor import measure


@spaces.GPU(duration=90)
def run() -> str:
    return measure()


with gr.Blocks(title="The parallelization floor, on real CUDA") as demo:
    gr.Markdown(
        "# The parallelization floor, measured\n\n"
        "Every parallel execution strategy has a size below which it is slower than "
        "doing the work in order, because setup is paid per launch while the saving is "
        "proportional to the work. This runs that measurement on a real GPU and prints "
        "where the crossover falls.\n\n"
        "Three columns per row: the CPU, the GPU counting the two transfers you pay to "
        "use it, and the GPU counting only the kernel. The middle one is what you can "
        "spend, unless your data already lives on the device and stays there.\n\n"
        "The same measurement runs in Colab on whatever GPU you are given, and the CPU "
        "half of the argument runs anywhere with no setup: "
        "[jmurray10/agent_design_peas](https://github.com/jmurray10/agent_design_peas).\n\n"
        "Nothing here is a published figure and no model is called."
    )
    button = gr.Button("Measure", variant="primary")
    output = gr.Textbox(label="Result", lines=44, max_lines=60)
    button.click(run, inputs=None, outputs=output)

demo.launch()
