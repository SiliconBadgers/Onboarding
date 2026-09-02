# Stage 2 — Quantization

**Goal:** run the network with INT8 weights and activations and INT32 sums, losing
almost no accuracy, and understand where every scale factor comes from.

**File:** `cub/quant.py`. **Test:** `pytest tests/test_02_quantization.py`.

## Why integers

An FPGA multiplies two 8-bit integers in one DSP slice in one cycle. A 32-bit float
multiply-add takes several slices and several cycles, and the accumulator needs more.
For inference, INT8 weights and activations lose very little accuracy on a network
like this, so the trade is overwhelmingly worth it. The accelerator therefore never
sees a float. Converting to integers is the compiler's job, and this stage is that
conversion.

## The scheme

A real number `r` is stored as an integer `q` and a scale `S`, with `r ~= q / S`.
Bigger `S` means finer resolution but smaller range. We use one `S` per tensor,
chosen so the largest magnitude in the tensor lands exactly on 127:

```
S = 127 / max|r|          q = round(r * S)          q in [-127, 127]
```

This is called *symmetric* quantization: zero maps to zero, and there is no offset.
It matters more than it looks. A dot product of symmetric tensors is a plain sum of
products; with an offset you would have correction terms in the hardware.

### Inputs

The image is normalized the way training saw it, `(pixel/255 - mean) / std`, and then
scaled. `INPUT_SCALE` is chosen so a fully white pixel maps to exactly 127. A black
pixel then maps to about -19 (it is below zero after mean subtraction).

### Weights

Per layer, `S_w = 127 / max|W|`, `W_q = round(W * S_w)`.

### The dot product

```
acc = sum_k  x_q[k] * W_q[n][k]
```

`x_q` has scale `S_in`, `W_q` has scale `S_w`, so `acc` represents the real value at
scale `S_in * S_w`. Each product is at most 127 * 128, which fits in 16 bits. Summing
784 of them needs at most 16 + 10 = 26 bits. INT32 has room to spare. (Compute this
yourself: `compute_shift_bits(784)`.)

### Biases

The bias is added to `acc`, so it must be at the same scale: `b_q = round(b * S_in * S_w)`,
stored as INT32.

### The shift

`acc` is a big number, around 2^19 for this network. The next layer needs INT8
inputs. We divide by a power of two, `acc >> shift`, and clamp to `[-128, 127]`. The
next layer's input scale is then `S_in * S_w / 2^shift`.

How to choose `shift`? Run layer 1 on a few hundred real images, find the largest
accumulator value after ReLU, and pick the smallest shift that brings it to 127 or
less. A rare input that exceeds it will saturate at 127, which is harmless.

That is the whole scheme. Badger replaces the shift with a fixed-point multiply so it
can hit any scale ratio, not just powers of two, and gets a bit more accuracy from
per-channel scales. Same idea, more bits.

## Your task

Three blanks in `cub/quant.py`:

1. `quantize_input`: normalize, scale by `INPUT_SCALE`, round to nearest, clamp, cast.
   Use `np.rint` for rounding (round half to even, which is what NumPy and PyTorch
   both do), and `np.clip` for the clamp.
2. `quantize_weights`: scale, round, clamp, cast. The scale is already computed.
3. `choose_shift`: the smallest `shift` such that `acc_max >> shift <= 127`. A
   `while` loop is fine.

`quantize_bias`, `quantize_model`, and `int8_forward` are written for you. Read
`int8_forward`: it is the NumPy reference that your simulator (Stage 4) must match
bit for bit. It is about ten lines.

## Check

```bash
pytest tests/test_02_quantization.py -v
```

The last test runs the whole quantized network on 1000 images and requires at least
96%. The float model gets 97.4%; INT8 gets 97.3%. If you get 10%, your rounding or
clamp is wrong; if you get 90%, your shift is probably off by one.

## Questions to be able to answer

- Why does the bias need to be scaled by `S_in * S_w` and not just `S_w`?
- What would go wrong if `shift` were one too small? One too large?
- What is the scale of the final logits, and does it matter for argmax?

## Read next

`docs/isa.md`, all of it, then `docs/03-encoding.md`.
