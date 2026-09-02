"""The host side (Stage 7): image in, digit out.

Everything the accelerator does not do lives here: normalizing and quantizing the
image, loading the program, starting the core, and taking the argmax of the logits.
The same code drives the simulator today and the FPGA later; only `backend` changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .program import Program
from .quant import quantize_input
from .sim import Machine


class SimBackend:
    """Runs a Program on the Python simulator."""

    def run(self, prog: Program, x_q: np.ndarray) -> np.ndarray:
        prog.write_input(x_q)
        m = Machine(prog.image)
        m.run()
        self.last_machine = m
        return Program.read_output(m.dram, prog.regions["output"])


def predict(prog: Program, pixels: np.ndarray, backend=None) -> tuple[int, np.ndarray]:
    """Classify one uint8 image (784,) -> (digit, real-valued logits)."""
    backend = backend or SimBackend()
    # --- SOLUTION(stage=7): quantize the pixels, run the backend, dequantize with prog.output_scale, take argmax ---
    x_q = quantize_input(pixels)
    logits_q = backend.run(prog, x_q)
    logits = logits_q.astype(np.float32) / prog.output_scale
    digit = int(np.argmax(logits))
    # --- END SOLUTION ---
    return digit, logits


def ascii_digit(pixels: np.ndarray) -> str:
    """A 28x28 image as text, for the terminal demo."""
    ramp = " .:-=+*#%@"
    rows = pixels.reshape(28, 28)
    return "\n".join("".join(ramp[min(9, int(p) * 10 // 256)] for p in row) for row in rows)


def load_program(path: str | Path = "artifacts/mnist.cub") -> Program:
    return Program.load(path)
