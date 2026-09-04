# Stage 3 — The compiler

**Goal:** follow a trained PyTorch network all the way into a list of instructions and
a picture of memory, with nothing typed by hand.

**Files to read:** `cub/quantization.py`, `cub/compiler.py`, `cub/program.py`.
**Test:** `pytest tests/test_03_compiler.py -v` (it already passes).

---

## What a compiler does here

You have a trained network on one side and an instruction set on the other. The
compiler is the thing in between. For a network like this one it does three jobs:

1. **Convert every number.** Decimal weights and biases become whole numbers, using the
   scaling from [Stage 1](01-pytorch.md).
2. **Decide where everything lives.** Which byte of main memory holds the first weight;
   which scratchpad index the biases land on; where the input image will be written and
   where the answer will be left.
3. **Emit instructions.** The same five instructions per layer, with the addresses from
   step 2 filled into their operands.

There is no graph and no intermediate representation, because a network like this one
*is already* a list of layers. A production compiler that has to handle convolutions,
reshapes and branching needs one; the final step, where a layer becomes instructions,
still looks like this one.

Run it and watch:

```bash
python -m cub compile
```

```
12 instructions, 104104 byte image, layer-1 shift 12 -> artifacts/mnist.cub
  instructions  0x00000 .. 0x00400  (1024 bytes)
  weights1      0x00400 .. 0x18C00  (100352 bytes)
  biases1       0x18C00 .. 0x18E00  (512 bytes)
  weights2      0x18E00 .. 0x19300  (1280 bytes)
  biases2       0x19300 .. 0x19328  (40 bytes)
  input         0x19340 .. 0x19650  (784 bytes)
  output        0x19680 .. 0x196A8  (40 bytes)
```

## Job 1: converting the numbers

`cub/quantization.py`, and `quantize_model` in particular, is the whole of it:

```python
quantized_weights1, scale1 = quantize_weights(weights1)          # decimals -> 8-bit
quantized_biases1 = quantize_bias(biases1, INPUT_SCALE, scale1)  # decimals -> 32-bit

quantized_pixels = quantize_input(calibration_pixels)            # images -> 8-bit
accumulators1 = quantized_pixels @ quantized_weights1.T + quantized_biases1
shift1 = choose_shift(int(np.max(np.maximum(accumulators1, 0))))
```

Three things worth noticing.

**The bias scale is not the weight scale.** A bias is added to the *result* of the
multiply, so it must be at the result's scale, which is `input_scale * weight_scale`.
Scale it by the weight scale alone and every bias is wrong by a factor of about 45.

**The shift is measured, not derived.** `choose_shift` needs to know how large layer
1's accumulators actually get, and the only honest way to find out is to run layer 1 on
real images. That is what the `calibration_pixels` argument is for. This step is called
*calibration*, and every quantizing compiler has one.

**`int8_forward` is the reference.** It is ten lines of NumPy that perform the network
using exactly the arithmetic the instruction set specifies — the same shifts, the same
clamps, the same 32-bit wrapping. Everything downstream is checked against it. Read it;
you will recognize every line from the instruction descriptions in Stage 2.

## Job 2: the memory plan

The compiler produces a **memory image**: the exact bytes that must be sitting in main
memory before the chip starts. `Program.place()` in `cub/program.py` appends a region
at the next 64-byte boundary and reports where it landed.

| Region | Contents |
|---|---|
| `instructions` | the instructions themselves, 16 bytes each, starting at address 0 |
| `weights1` | layer-1 weights, 8-bit, row-major `(128, 784)` |
| `biases1` | layer-1 biases, 32-bit, 128 of them |
| `weights2` | layer-2 weights, 8-bit, `(10, 128)` |
| `biases2` | layer-2 biases, 32-bit, 10 of them |
| `input` | 784 bytes of nothing — the host writes the image here before each run |
| `output` | 40 bytes of nothing — the program's last `STORE` writes the answer here |

The instruction region is reserved *first*, at address 0, before the compiler knows how
many instructions there will be. It reserves 64 slots, far more than the network needs,
because the alternative is emitting everything twice: once to count, once to place. A
kilobyte of padding is a much better trade than a two-pass compiler.

Alongside the memory plan there is a **scratchpad plan** — where things go *inside* the
chip. It is three cursors:

```python
weight_cursor = 0        # advances by outputs*inputs after each layer
bias_cursor = 0          # advances by outputs after each layer
activation_input = 0     # 0 for layer 1, then 1024 for the hidden activations
```

Layer 2's weights therefore land at weight scratchpad index 100,352, right after layer
1's, and its biases at index 128. Those numbers are not typed anywhere: they are what
the cursors hold by the time layer 2 is emitted.

## Job 3: emitting instructions

The loop in `compile_model` is the entire code generator:

```python
for number, layer in enumerate(model.layers, start=1):
    outputs, inputs = layer.weights.shape
    ...
    LOAD             this layer's weights -> WEIGHT_SCRATCHPAD[weight_cursor]
    LOAD             this layer's biases  -> BIAS_SCRATCHPAD[bias_cursor]
    MATRIX_MULTIPLY  from activation_input, by those weights, into ACCUMULATORS[0]
    ADD_BIAS         those accumulators
    RECTIFIED_LINEAR into ACTIVATION_SCRATCHPAD[1024]   ... or, on the last layer,
    STORE            the accumulators to the output region
```

Five instructions per layer, plus one `LOAD` for the image at the start and one `HALT`
at the end. Two layers, so twelve instructions.

The last layer is the only special case, and it is special for one reason: its raw
32-bit accumulators *are* the answer. There is no next layer to feed, so there is
nothing to rectify and nothing to scale down; the ten numbers go straight back to main
memory for the host to read.

## The program file

`artifacts/mnist.cub` is a 64-byte header followed by the memory image:

| Offset | Type | Field |
|---|---|---|
| 0 | 4 characters | magic, `"CUB1"` |
| 4 | 32-bit | instruction set version |
| 8 | 32-bit | number of instructions |
| 12 | 32-bit | image length in bytes |
| 16, 20 | 32-bit | `input` region offset and length |
| 24, 28 | 32-bit | `output` region offset and length |
| 32 | decimal | output scale: `real_value = stored_value / output_scale` |

The header exists so the host can find the input and output regions without being told.
That is the whole reason: the chip does not need it, and it is stripped off before the
image reaches memory.

## Check it yourself

```bash
pytest tests/test_03_compiler.py -v
```

One of these tests requires the compiler's output to be **byte-identical** to the
committed `artifacts/mnist.cub`, instructions and image both. Another runs the result
on the simulator and compares against the golden answers. So:

```bash
python -m cub compile && git status artifacts/
```

should report nothing changed.

## Things to try

- In `cub/compiler.py`, change `ACTIVATION_HIDDEN_INDEX` from 1024 to 2048. Recompile
  and rerun. Everything still works — why? Now change it to 4000. What breaks, and
  which error message tells you?
- Reserve 8 instruction slots instead of 64. What is the error, and at what point in
  the compile does it appear?
- What would have to change to support a three-layer network? A layer with 5,000 hidden
  units? (One of those is a one-line change and one is not.)

## Read next

[04-mnist-by-hand.md](04-mnist-by-hand.md), where you write this program yourself.
