"""The Cub compiler (Stage 6): a quantized MLP -> a Program.

There is no graph, no IR, and no scheduler: an MLP is a list of layers, and each layer
becomes the same five-instruction pattern. The interesting decisions are *where*
things live (the memory plan) and *what shift* each layer uses (already decided by
cub.quant). The compiler's job is to write those decisions down as instructions.
"""

from __future__ import annotations

import numpy as np

from . import isa
from .isa import make
from .program import Program
from .quant import QuantModel

# Scratchpad plan. SPAD_A holds the input image at 0 and each layer's activations
# after it; SPAD_W holds every layer's weights back to back; SPAD_B every bias.
# Everything fits, so the plan is fixed rather than computed (docs/isa.md section 2).
SPAD_A_INPUT = 0
SPAD_A_HIDDEN = 1024


def compile_mlp(model: QuantModel) -> Program:
    prog = Program.new()

    # 1. Reserve room for the instructions at DRAM offset 0. We do not know how many
    #    there will be yet; 64 is far more than the MLP needs.
    prog.place("insns", 64 * isa.INSN_BYTES)

    # 2. Place the parameters in DRAM and remember where they landed.
    dram = {}
    for i, layer in enumerate(model.layers, start=1):
        dram[f"w{i}"] = prog.place(f"w{i}", layer.w)
        dram[f"b{i}"] = prog.place(f"b{i}", layer.b)
    in_len = model.layers[0].w.shape[1]
    out_len = model.layers[-1].w.shape[0]
    dram["input"] = prog.place("input", in_len)
    dram["output"] = prog.place("output", out_len * 4)

    # 3. Emit the program. The scratchpad cursors advance as each layer's parameters
    #    are loaded, so layer 2's weights land right after layer 1's.
    w_cursor = 0
    b_cursor = 0
    a_in = SPAD_A_INPUT
    prog.insns.append(make("LOAD", mem=isa.MEM_SPAD_A, dram=dram["input"].offset, spad=a_in, count=in_len))
    for i, layer in enumerate(model.layers, start=1):
        n, k = layer.w.shape
        last = i == len(model.layers)
        # TODO(onboard, stage 6): emit LOAD w, LOAD b, MATMUL, ADD_BIAS, then RELU (or STORE for the last layer)
        raise NotImplementedError("stage 6 blank, see compiler.py:51")
        w_cursor += n * k
        b_cursor += n
        a_in = SPAD_A_HIDDEN
    prog.insns.append(make("HALT"))

    prog.output_scale = model.output_scale
    prog.finalize()
    return prog


def compile_from_artifacts() -> tuple[Program, QuantModel]:
    """The whole front half of the pipeline: float weights + calibration images -> Program."""
    from .model import load_float_weights, load_test_1k
    from .quant import quantize_model

    w = load_float_weights()
    images, _ = load_test_1k()
    model = quantize_model(w["w1"], w["b1"], w["w2"], w["b2"], images)
    return compile_mlp(model), model
