# Stage 3 — The compiler

**Goal:** follow a trained PyTorch network into a list of instructions and a picture of
memory, with nothing typed by hand.

**Files to read:** `python/quantization.py`, `python/compiler.py`, `python/program.py`.
**Test:** `pytest tests/test_03_compiler.py -v` (it already passes).

---

## What a compiler does here

A trained network on one side, an instruction set on the other. The compiler does
three jobs:

1. **Convert every number**, using the scaling from [Stage 1](01-pytorch.md).
2. **Decide where everything lives**: which byte of main memory holds the first weight,
   which scratchpad index the biases land on, where the image is written and the answer
   left.
3. **Emit instructions**, with those addresses in their operands.

There is no graph and no intermediate representation, because this network *is already*
a list of layers. A production compiler handling convolutions and branching needs one,
but its final step still looks like this.

Run it:

```bash
python -m python compile
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

`python/quantization.py`, and `quantize_model` in particular, is the whole of it:

```python
quantized_weights1, scale1 = quantize_weights(weights1)          # decimals -> 8-bit
quantized_biases1 = quantize_bias(biases1, INPUT_SCALE, scale1)  # decimals -> 32-bit

quantized_pixels = quantize_input(calibration_pixels)            # images -> 8-bit
accumulators1 = quantized_pixels @ quantized_weights1.T + quantized_biases1
shift1 = choose_shift(int(np.max(np.maximum(accumulators1, 0))))
```

Three things worth noticing.

**The bias scale is not the weight scale.** A bias is added to the multiply's *result*,
so it must be at that result's scale, `input_scale * weight_scale`. Use the weight
scale alone and every bias is off by a factor of about 45.

**The shift is measured, not derived.** The only honest way to know how large layer 1's
accumulators get is to run layer 1 on real images — that is what `calibration_pixels`
is for. Every quantizing compiler has this *calibration* step.

**`int8_forward` is the reference.** Ten lines of NumPy using exactly the arithmetic the
instruction set specifies: the same shifts, clamps and 32-bit wrapping. Everything
downstream is checked against it, and you will recognize every line from Stage 2.

## Job 2: the memory plan

The compiler produces a **memory image**: the exact bytes that must be in main memory
before the chip starts. `Program.place()` appends a region at the next 64-byte boundary
and reports where it landed.

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
many instructions there will be. It takes 64 slots, far more than needed, because the
alternative is emitting everything twice — once to count, once to place. A kilobyte of
padding beats a two-pass compiler.

Alongside it is the **scratchpad plan** — where things go *inside* the chip. Three
cursors:

```python
weight_cursor = 0        # advances by outputs*inputs after each layer
bias_cursor = 0          # advances by outputs after each layer
activation_input = 0     # 0 for layer 1, then 1024 for the hidden activations
```

So layer 2's weights land at weight scratchpad index 100,352, right after layer 1's,
and its biases at index 128. Neither number is typed anywhere — they are what the
cursors hold by the time layer 2 is emitted.

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

Five instructions per layer, plus one `LOAD` for the image and one `HALT`. Two layers,
twelve instructions.

The last layer is the only special case, for one reason: its raw 32-bit accumulators
*are* the answer. There is no next layer to feed, so nothing to rectify and nothing to
scale down — the ten numbers go straight back to main memory.

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
The chip does not need it, and it is stripped off before the image reaches memory.

## Check it yourself

```bash
pytest tests/test_03_compiler.py -v
```

One test requires the compiler's output to be **byte-identical** to the committed
`artifacts/mnist.cub`; another runs it on the simulator against the golden answers. So:

```bash
python -m python compile && git status artifacts/
```

should report nothing changed.

## Things to try

- Change `ACTIVATION_HIDDEN_INDEX` from 1024 to 2048 and recompile. Still works — why?
  Now 4000. What breaks, and which error tells you?
- Reserve 8 instruction slots instead of 64. What is the error, and when does it appear?
- What would it take to support a three-layer network? A layer with 5,000 hidden units?
  (One is a one-line change; one is not.)

## Read next

[04-mnist-by-hand.md](04-mnist-by-hand.md), where you write this program yourself.
