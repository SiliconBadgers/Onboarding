"""The compiler: a quantized network in, a runnable program out (Stage 3).

There is no graph and no intermediate representation. A network like this one is
already a list of layers, and each layer becomes the same five instructions. The
interesting decisions are *where* things live (the memory plan) and *what shift* each
layer uses (already decided in python/quantization.py). The compiler's job is to write
those decisions down as instructions.
"""

from __future__ import annotations

from . import instruction_set
from .instruction_set import make
from .program import Program
from .quantization import QuantizedModel

# Where things sit in the scratchpads. The activation scratchpad holds the input image
# at index 0 and the hidden activations at index 1024; the weight scratchpad holds
# every layer's weights back to back; the bias scratchpad every bias. Everything fits
# at once, so the plan is fixed rather than computed.
ACTIVATION_INPUT_INDEX = 0
ACTIVATION_HIDDEN_INDEX = 1024

# The instruction region is reserved before we know how many instructions there are.
INSTRUCTION_SLOTS = 64


def compile_model(model: QuantizedModel) -> Program:
    program = Program.new()

    # 1. Reserve room for the instructions at main-memory address 0.
    program.place("instructions", INSTRUCTION_SLOTS * instruction_set.INSTRUCTION_BYTES)

    # 2. Place the parameters in main memory and remember where they landed.
    regions = {}
    for number, layer in enumerate(model.layers, start=1):
        regions[f"weights{number}"] = program.place(f"weights{number}", layer.weights)
        regions[f"biases{number}"] = program.place(f"biases{number}", layer.biases)
    input_length = model.layers[0].weights.shape[1]
    output_length = model.layers[-1].weights.shape[0]
    regions["input"] = program.place("input", input_length)
    regions["output"] = program.place("output", output_length * 4)

    # 3. Emit the instructions. The scratchpad cursors advance as each layer's
    #    parameters are loaded, so layer 2's weights land right after layer 1's.
    weight_cursor = 0
    bias_cursor = 0
    activation_input = ACTIVATION_INPUT_INDEX

    program.instructions.append(make(
        "LOAD",
        space=instruction_set.SPACE_ACTIVATION_SCRATCHPAD,
        memory=regions["input"].offset,
        index=activation_input,
        count=input_length,
    ))

    for number, layer in enumerate(model.layers, start=1):
        outputs, inputs = layer.weights.shape
        is_last_layer = number == len(model.layers)

        program.instructions.append(make(
            "LOAD",
            space=instruction_set.SPACE_WEIGHT_SCRATCHPAD,
            memory=regions[f"weights{number}"].offset,
            index=weight_cursor,
            count=outputs * inputs,
        ))
        program.instructions.append(make(
            "LOAD",
            space=instruction_set.SPACE_BIAS_SCRATCHPAD,
            memory=regions[f"biases{number}"].offset,
            index=bias_cursor,
            count=outputs,
        ))
        program.instructions.append(make(
            "MATRIX_MULTIPLY",
            input=activation_input,
            weights=weight_cursor,
            accumulator=0,
            outputs=outputs,
            inputs=inputs,
        ))
        program.instructions.append(make(
            "ADD_BIAS", accumulator=0, bias=bias_cursor, count=outputs,
        ))
        if is_last_layer:
            # The last layer's raw 32-bit accumulators are the answer, so they go
            # straight back to main memory for the host to read.
            program.instructions.append(make(
                "STORE",
                space=instruction_set.SPACE_ACCUMULATORS,
                memory=regions["output"].offset,
                index=0,
                count=outputs,
            ))
        else:
            program.instructions.append(make(
                "RECTIFIED_LINEAR",
                accumulator=0,
                destination=ACTIVATION_HIDDEN_INDEX,
                count=outputs,
                shift=layer.shift,
                rectify=int(layer.rectify),
            ))

        weight_cursor += outputs * inputs
        bias_cursor += outputs
        activation_input = ACTIVATION_HIDDEN_INDEX

    program.instructions.append(make("HALT"))

    program.output_scale = model.output_scale
    program.finalize()
    return program


def compile_from_artifacts() -> tuple[Program, QuantizedModel]:
    """The whole front half of the stack: trained weights + images -> a Program."""
    from .model import load_test_images, load_trained_weights
    from .quantization import quantize_model

    weights = load_trained_weights()
    images, _ = load_test_images()
    model = quantize_model(
        weights["weights1"], weights["biases1"],
        weights["weights2"], weights["biases2"],
        images,
    )
    return compile_model(model), model
