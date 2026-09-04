"""Turning the network's decimal numbers into whole numbers.

The scheme, in one paragraph. A real number `r` is stored as a whole number `q` plus a
per-tensor scale `S`, with `r` approximately `q / S`. The image and each layer's
weights get their own `S`, chosen so the largest magnitude in that tensor lands exactly
on 127, the largest value a signed 8-bit number can hold. Multiplying two such tensors
produces 32-bit sums at scale `input_scale * weight_scale`. The bias is pre-scaled to
that same scale so it can be added straight in. A right shift then brings the sums back
into 8-bit range for the next layer, whose input scale is therefore
`input_scale * weight_scale / 2**shift`. No decimal number survives past this module:
the accelerator only ever sees whole numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The normalization PyTorch training used. The host has to apply the same transform
# before quantizing, so the constants live here rather than in the model file.
MNIST_MEAN = 0.1307
MNIST_STANDARD_DEVIATION = 0.3081

# The brightest possible pixel, after normalization, maps to exactly 127.
INPUT_SCALE = 127.0 / ((1.0 - MNIST_MEAN) / MNIST_STANDARD_DEVIATION)


def quantize_input(pixels: np.ndarray) -> np.ndarray:
    """8-bit unsigned pixels (784,) or (N, 784) -> signed 8-bit values at INPUT_SCALE.

    Scale to [0, 1], normalize the way training did, multiply by INPUT_SCALE, round to
    nearest, clamp to [-128, 127].
    """
    values = pixels.astype(np.float32) / 255.0
    values = (values - MNIST_MEAN) / MNIST_STANDARD_DEVIATION
    quantized = np.rint(values * INPUT_SCALE)
    quantized = np.clip(quantized, -128, 127)
    return quantized.astype(np.int8)


def weight_scale(weights: np.ndarray) -> float:
    """Per-tensor symmetric scale: the largest magnitude maps to 127."""
    return 127.0 / float(np.max(np.abs(weights)))


def quantize_weights(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Decimal weights (outputs, inputs) -> (signed 8-bit weights, scale)."""
    scale = weight_scale(weights)
    quantized = np.clip(np.rint(weights * scale), -128, 127).astype(np.int8)
    return quantized, scale


def quantize_bias(biases: np.ndarray, input_scale: float, weight_scale: float) -> np.ndarray:
    """Decimal biases -> 32-bit whole numbers at the accumulator's scale."""
    return np.rint(biases * input_scale * weight_scale).astype(np.int32)


def choose_shift(largest_accumulator: int) -> int:
    """The smallest right shift that brings an accumulator peak back into 8-bit range.

    After the rectified linear step the largest value the next layer will see is
    `largest_accumulator >> shift`, and that has to be at most 127. A rare input may
    exceed `largest_accumulator` and saturate; that is what the clamp is for.
    """
    shift = 0
    while (largest_accumulator >> shift) > 127:
        shift += 1
    return shift


@dataclass
class QuantizedLayer:
    weights: np.ndarray      # signed 8-bit, shape (outputs, inputs)
    biases: np.ndarray       # signed 32-bit, shape (outputs,)
    weight_scale: float
    input_scale: float       # scale of the 8-bit input this layer consumes
    shift: int               # right shift applied to this layer's output
    rectify: bool            # whether negatives are zeroed (False for the last layer)

    @property
    def accumulator_scale(self) -> float:
        return self.input_scale * self.weight_scale

    @property
    def output_scale(self) -> float:
        return self.accumulator_scale / (1 << self.shift)


@dataclass
class QuantizedModel:
    layers: list[QuantizedLayer]

    @property
    def output_scale(self) -> float:
        return self.layers[-1].output_scale


def int8_forward(model: QuantizedModel, quantized_pixels: np.ndarray, trace: bool = False):
    """Run the quantized network in NumPy, using exactly the arithmetic the chip uses.

    quantized_pixels: signed 8-bit (784,) or (N, 784). Returns 32-bit logits (N, 10),
    or (10,) for a single image. This is the reference the simulator and the hardware
    both have to match bit for bit.
    """
    squeeze = quantized_pixels.ndim == 1
    activations = np.atleast_2d(quantized_pixels).astype(np.int32)
    all_accumulators = []
    for layer in model.layers:
        accumulators = (
            activations @ layer.weights.astype(np.int32).T + layer.biases
        )                                                     # (N, outputs), 32-bit
        all_accumulators.append(accumulators)
        if layer is model.layers[-1]:
            activations = accumulators
        else:
            value = np.maximum(accumulators, 0) if layer.rectify else accumulators
            value = value >> layer.shift                      # arithmetic, on int32
            activations = np.clip(value, -128, 127).astype(np.int32)
    out = activations[0] if squeeze else activations
    return (out, all_accumulators) if trace else out


def quantize_model(
    weights1: np.ndarray,
    biases1: np.ndarray,
    weights2: np.ndarray,
    biases2: np.ndarray,
    calibration_pixels: np.ndarray,
) -> QuantizedModel:
    """Turn the decimal network into a QuantizedModel.

    Layer 1's shift is chosen from real images: run the layer, see how big the
    accumulators actually get, and pick the smallest shift that fits them into 8 bits.
    """
    quantized_weights1, scale1 = quantize_weights(weights1)
    quantized_biases1 = quantize_bias(biases1, INPUT_SCALE, scale1)

    quantized_pixels = quantize_input(calibration_pixels).astype(np.int32)
    accumulators1 = (
        quantized_pixels @ quantized_weights1.astype(np.int32).T + quantized_biases1
    )
    shift1 = choose_shift(int(np.max(np.maximum(accumulators1, 0))))
    layer1 = QuantizedLayer(
        quantized_weights1, quantized_biases1, scale1, INPUT_SCALE, shift1, rectify=True
    )

    quantized_weights2, scale2 = quantize_weights(weights2)
    quantized_biases2 = quantize_bias(biases2, layer1.output_scale, scale2)
    layer2 = QuantizedLayer(
        quantized_weights2, quantized_biases2, scale2, layer1.output_scale,
        shift=0, rectify=False,
    )
    return QuantizedModel([layer1, layer2])


def accumulator_bits_needed(inputs: int) -> int:
    """How many bits a dot product of `inputs` signed 8-bit pairs can need.

    One product needs 16 bits; summing `inputs` of them needs log2(inputs) more.
    """
    return 16 + math.ceil(math.log2(inputs))
