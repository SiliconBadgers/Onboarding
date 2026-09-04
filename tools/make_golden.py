#!/usr/bin/env python3
"""Regenerate artifacts/golden.npz from the trained weights and the test slice.

golden.npz is what every test compares against: the same 1000 images at every point
in the stack, from decimal PyTorch logits to the whole-number logits the chip
produces. Run this after retraining (python -m cub train) and recompiling
(python -m cub compile).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cub.compiler import compile_from_artifacts  # noqa: E402
from cub.model import decimal_forward, load_test_images, load_trained_weights  # noqa: E402
from cub.quantization import int8_forward, quantize_input  # noqa: E402
from cub.runtime import SimulatorBackend  # noqa: E402


def main() -> None:
    program, quantized_model = compile_from_artifacts()
    images, labels = load_test_images()
    weights = load_trained_weights()
    quantized_pixels = quantize_input(images)
    reference = int8_forward(quantized_model, quantized_pixels)
    simulated = np.stack(
        [SimulatorBackend().run(program, quantized_pixels[i]) for i in range(len(images))]
    )
    assert (simulated == reference).all(), "simulator and NumPy reference disagree"
    np.savez_compressed(
        ROOT / "artifacts" / "golden.npz",
        images=images,
        labels=labels,
        quantized_pixels=quantized_pixels,
        decimal_logits=decimal_forward(weights, images).astype(np.float32),
        int8_logits=reference.astype(np.int32),
        shift1=np.int32(quantized_model.layers[0].shift),
        output_scale=np.float32(quantized_model.output_scale),
        weights1=quantized_model.layers[0].weights,
        biases1=quantized_model.layers[0].biases,
        weights2=quantized_model.layers[1].weights,
        biases2=quantized_model.layers[1].biases,
    )
    accuracy = (reference.argmax(1) == labels).mean()
    print(f"golden.npz written; whole-number accuracy {accuracy:.1%}")


if __name__ == "__main__":
    main()
