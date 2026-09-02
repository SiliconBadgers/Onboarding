"""Float -> INT8 quantization for the Cub pipeline (Stage 2).

The scheme, in one paragraph. A real number r is stored as an integer q with a
per-tensor scale S: r ~= q / S. Inputs and weights get their own S, chosen so the
largest magnitude lands on 127. A matmul of two such tensors produces INT32 sums at
scale S_in * S_w. The bias is pre-scaled to that same scale so it can be added
directly. Then a right shift by `shift` bits brings the sums back into INT8 range for
the next layer, whose input scale is therefore S_in * S_w / 2**shift. No floating
point survives past this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The normalization PyTorch training used. The host must apply the same transform
# before quantizing (docs/isa.md section 3), so the constants live here, not in the
# model file.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# The brightest possible pixel, normalized, maps to exactly 127.
INPUT_SCALE = 127.0 / ((1.0 - MNIST_MEAN) / MNIST_STD)


def quantize_input(pixels: np.ndarray) -> np.ndarray:
    """uint8 pixels (784,) or (N, 784) -> INT8 at INPUT_SCALE.

    Steps: scale to [0, 1], normalize with MNIST_MEAN/STD, multiply by INPUT_SCALE,
    round to nearest, clamp to [-128, 127].
    """
    x = pixels.astype(np.float32) / 255.0
    # TODO(onboard, stage 2): normalize, scale, round, clamp, cast to int8
    raise NotImplementedError("stage 2 blank, see quant.py:36")
    return q.astype(np.int8)


def weight_scale(w: np.ndarray) -> float:
    """Per-tensor symmetric scale: the largest |w| maps to 127."""
    return 127.0 / float(np.max(np.abs(w)))


def quantize_weights(w: np.ndarray) -> tuple[np.ndarray, float]:
    """float weights (out, in) -> (INT8 weights, scale)."""
    s = weight_scale(w)
    # TODO(onboard, stage 2): scale, round, clamp, cast
    raise NotImplementedError("stage 2 blank, see quant.py:52")
    return q, s


def quantize_bias(b: np.ndarray, in_scale: float, w_scale: float) -> np.ndarray:
    """float biases -> INT32 at the accumulator scale in_scale * w_scale."""
    return np.rint(b * in_scale * w_scale).astype(np.int32)


def choose_shift(acc_max: int) -> int:
    """Smallest right shift that brings a non-negative INT32 accumulator peak into INT8.

    After ReLU the largest value the next layer will see is acc_max >> shift, and we
    need that to be <= 127. Some rare inputs may exceed acc_max and saturate; that is
    what the clamp in RELU is for.
    """
    # TODO(onboard, stage 2): the smallest shift such that (acc_max >> shift) <= 127
    raise NotImplementedError("stage 2 blank, see quant.py:70")


@dataclass
class QuantLayer:
    w: np.ndarray        # int8 (out, in)
    b: np.ndarray        # int32 (out,)
    w_scale: float
    in_scale: float      # scale of the INT8 input this layer consumes
    shift: int           # right shift applied to this layer's output by RELU
    relu: bool           # whether RELU zeroes negatives (False for the last layer)

    @property
    def acc_scale(self) -> float:
        return self.in_scale * self.w_scale

    @property
    def out_scale(self) -> float:
        return self.acc_scale / (1 << self.shift)


@dataclass
class QuantModel:
    layers: list[QuantLayer]

    @property
    def output_scale(self) -> float:
        return self.layers[-1].out_scale


def int8_forward(model: QuantModel, x_q: np.ndarray, trace: bool = False):
    """Run the quantized MLP in NumPy, using exactly the arithmetic the ISA specifies.

    x_q: int8 (784,) or (N, 784). Returns int32 logits (N, 10) (or (10,)).
    This is the NumPy reference the simulator must match bit for bit.
    """
    squeeze = x_q.ndim == 1
    a = np.atleast_2d(x_q).astype(np.int32)
    accs = []
    for layer in model.layers:
        acc = a @ layer.w.astype(np.int32).T + layer.b        # (N, out), int32
        accs.append(acc)
        if layer is model.layers[-1]:
            a = acc
        else:
            v = np.maximum(acc, 0) if layer.relu else acc
            v = v >> layer.shift                              # arithmetic on int32
            a = np.clip(v, -128, 127).astype(np.int32)
    out = a[0] if squeeze else a
    return (out, accs) if trace else out


def quantize_model(w1, b1, w2, b2, calib_pixels: np.ndarray) -> QuantModel:
    """Turn the float MLP into a QuantModel, choosing layer-1's shift from calibration data."""
    w1_q, s1 = quantize_weights(w1)
    b1_q = quantize_bias(b1, INPUT_SCALE, s1)
    # Calibrate: run layer 1 on real images to find the accumulator peak.
    x_q = quantize_input(calib_pixels).astype(np.int32)
    acc1 = x_q @ w1_q.astype(np.int32).T + b1_q
    shift1 = choose_shift(int(np.max(np.maximum(acc1, 0))))
    layer1 = QuantLayer(w1_q, b1_q, s1, INPUT_SCALE, shift1, relu=True)

    w2_q, s2 = quantize_weights(w2)
    b2_q = quantize_bias(b2, layer1.out_scale, s2)
    layer2 = QuantLayer(w2_q, b2_q, s2, layer1.out_scale, shift=0, relu=False)
    return QuantModel([layer1, layer2])


def compute_shift_bits(k: int) -> int:
    """How many bits an INT8 x INT8 dot product of length k needs. Used in the Stage 2 guide."""
    return 16 + math.ceil(math.log2(k))
