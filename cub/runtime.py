"""The host side: image in, digit out (Stage 5).

Everything the accelerator does not do lives here: normalizing and quantizing the
image, writing it into main memory, telling the core to start, waiting for it to
finish, and reading the answer back out. The same code drives the Python simulator
today and could drive real hardware tomorrow; only `backend` changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .program import Program
from .quantization import quantize_input
from .simulator import Machine


class SimulatorBackend:
    """Runs a Program on the Python simulator.

    This is the software stand-in for the four things a host does to real hardware:
    write the memory image, write the input, set the start bit, poll the done bit.
    """

    def run(self, program: Program, quantized_pixels: np.ndarray) -> np.ndarray:
        program.write_input(quantized_pixels)
        machine = Machine(program.image)
        machine.run()
        self.last_machine = machine
        return Program.read_output(machine.main_memory, program.regions["output"])


def predict(program: Program, pixels: np.ndarray, backend=None) -> tuple[int, np.ndarray]:
    """Classify one 8-bit image (784,). Returns (digit, decimal-valued logits)."""
    backend = backend or SimulatorBackend()
    quantized_pixels = quantize_input(pixels)
    quantized_logits = backend.run(program, quantized_pixels)
    logits = quantized_logits.astype(np.float32) / program.output_scale
    digit = int(np.argmax(logits))
    return digit, logits


def ascii_digit(pixels: np.ndarray) -> str:
    """A 28x28 image as text, for the terminal demo."""
    ramp = " .:-=+*#%@"
    rows = pixels.reshape(28, 28)
    return "\n".join(
        "".join(ramp[min(9, int(pixel) * 10 // 256)] for pixel in row) for row in rows
    )


def load_program(path: str | Path = "artifacts/mnist.cub") -> Program:
    return Program.load(path)
