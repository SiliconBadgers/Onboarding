# Stage 1 — PyTorch, and turning decimals into whole numbers

**Goal:** know which operations the chip must perform, and why every number ends up an
integer.

**Files to read:** `python/model.py`, `python/quantization.py`.
**Test:** `pytest tests/test_01_pytorch.py -v` (it already passes).

---

## Most of you will not need the machine learning

We do need some people who understand machine learning and PyTorch. But the majority of
you will never need to know how any of it works in order to design the chip.

What you need is the **operations** the complicated machine learning breaks down into —
matrix multiply, rectified linear, and a couple of others. Every one of them abstracts
down to plain addition and multiplication, and that is what the chip executes.

If you are interested in learning how this machine learning stuff works, give this
prompt to an AI and have it teach you. Keep the constraints: asked plainly, an AI will
explain MNIST with convolutions, softmax and a training loop, none of which this
project uses.

> Explain how weights, biases, the first layer, the second layer, and ReLU work using
> MNIST as an example. Start out with just explaining it on a 3x3 grid with a line
> going down the middle, and then explain it with MNIST. I have 0 knowledge about
> machine learning, PyTorch, etc.
>
> Stay inside these constraints, because they are exactly what the project I am
> learning uses:
>
> - Inference only. The weights and biases already exist. Do not explain training,
>   backpropagation, gradients, loss functions or optimizers.
> - Fully connected layers only. No convolutions and no pooling.
> - Exactly two layers: 784 inputs (a 28x28 image flattened into one row), then 128,
>   then 10 outputs — one score per digit.
> - ReLU is the only activation function, and it appears only between the two layers.
> - No softmax and no probabilities. The prediction is whichever of the 10 scores is
>   largest.
> - One image at a time. No batches.
>
> Show me the actual arithmetic on the 3x3 example, and give the shape of every weight
> matrix and bias vector.

## The network

```
input   784 numbers   (a 28x28 grayscale image, flattened into one row)
layer 1 784 -> 128    hidden      = image  x  weights1 + biases1
        rectify       hidden      = max(hidden, 0)
layer 2 128 -> 10     ten_outputs = hidden x  weights2 + biases2
        argmax        the digit is the position of the largest of the ten
```

Four operations.

### Weights — a matrix multiply

`weights1` is a table of 128 x 784 = 100,352 numbers. For each of the 128 outputs, walk
down that output's row of 784 weights, multiply each by the matching pixel, and add the
784 products. One output, one number.

That is 100,352 multiplies and as many additions — by far the bulk of the work, and why
`MATRIX_MULTIPLY` is the instruction the chip is built around.

### Biases — an add

`biases1` is 128 numbers, one per output. Add each to its output. 128 additions.

### Rectified linear — a comparison against zero

Usually written "ReLU". Replace negatives with zero, leave the rest alone:

```
if value < 0: value = 0
```

In hardware, a look at the sign bit.

### Argmax — find the largest

The ten outputs are scores, one per digit; the largest wins. The chip does not do this
— it hands back the ten numbers and the host compares them. Ten comparisons are not
worth a circuit.

### Deliberately absent

- **Softmax.** It never changes *which* score is largest, so argmax on the raw scores
  gives the same answer for free.
- **Dropout and batch normalization.** Real models have them, and they would need
  folding into the weights before compiling. Ours does not.
- **Convolutions.** A matrix multiply with more complicated indexing in front of it.

`python/model.py`'s `forward` is three lines, one per operation above.

---

## From decimal numbers to whole numbers

```bash
python -c "
import numpy as np
w = np.load('artifacts/trained_weights.npz')['weights1']
print(w[0][:6]); print('largest magnitude:', abs(w).max())"
```

Decimals either side of zero, stored as 32-bit floating point. Hardware wants integers
instead, for two reasons:

- **Multiplying is cheaper.** An 8-bit integer multiply is one small circuit finishing
  in a cycle; a 32-bit floating-point multiply is much larger and takes several.
- **Memory is cheaper.** Those 100,352 weights are 401,408 bytes as decimals and
  100,352 as 8-bit integers — a quarter of the storage, and a quarter of the time spent
  moving them onto the chip. Moving weights is often an accelerator's biggest cost.

So every number is converted before it reaches the chip. This is **quantization**, and
it is one line of arithmetic.

### Scaling

```
S = 127 / (the largest magnitude in the tensor)
whole_number = round(decimal_number * S)
```

The largest value now lands on 127, the biggest a signed 8-bit integer holds, and the
rest land in `[-127, 127]`. Only precision below the rounding step is lost. Dividing by
`S` recovers the decimal — but nothing on the chip ever does.

Zero maps to zero and there is no offset. That is *symmetric* quantization, and it
matters: a dot product of two symmetric tensors is a plain sum of products. Add an
offset and the hardware needs correction terms.

### Where the scales come from

- **The image.** Pixels are 0 to 255, but training normalized them, so the host applies
  the same normalization and scales a fully white pixel to exactly 127. A black pixel
  becomes about -19, since it is below zero after the mean is subtracted.
- **Each layer's weights.** `127 / largest magnitude`, one scale per layer.
- **Each layer's biases.** A bias is added to the multiply's result, so it must be at
  that result's scale: `input_scale * weight_scale`. Stored as 32-bit, since the
  product can be large.

### Why the sums need 32 bits

Two 8-bit numbers multiply to at most 127 x 128, needing 16 bits. Summing 784 of those
needs about 10 more: 26 in total. So the **accumulators** are 32-bit, with room to
spare. That is the whole reason accumulators are wider than their inputs.

The next layer wants 8-bit inputs again, so after the biases are added the accumulator
is divided by a power of two — a right shift, which in hardware is just wires — and
clamped:

```
value = max(value, 0)          the rectified linear step
value = value >> shift         scale back down to 8 bits
value = clamp(value, -128, 127)
```

`shift` is chosen by running layer 1 on a few hundred real images and taking the
smallest shift that brings the largest accumulator to 127 or below. Here it is 12. A
rare image that exceeds it saturates, which is what the clamp is for.

### This is what a real accelerator does

Google's Tensor Processing Unit runs inference in 8-bit integers with 32-bit
accumulators, for exactly those two reasons. The first generation was a big grid of
8-bit multiply-accumulate units and little else — what you are about to build, with the
grid shrunk to one multiplier.

A production chip's one refinement is replacing the power-of-two shift with a
fixed-point multiply, so it can hit any scale ratio rather than only powers of two.

## Check it yourself

```bash
pytest tests/test_01_pytorch.py -v
```

The decimal network gets 97.4% on 1000 test images; the whole-number network gets
97.3%. Losing one image in a thousand to buy a four-times-smaller, much faster chip is
the trade this entire field is built on.

## Things to try

- In `python/quantization.py`, make `quantize_input` use `np.floor` instead of `np.rint`.
  Which test fails, and by how much?
- Make `choose_shift` return one more than it does. Accuracy barely moves — why? Now
  one less. Why is that so much worse?
- `python -c "from python.quantization import accumulator_bits_needed as f; print(f(784))"`
  — what would happen with a 16-bit accumulator?

## Retraining (optional)

```bash
python -m python train
```

Three epochs take about a minute and reach roughly 97.3%. It overwrites everything in
`artifacts/`, and every expected value downstream came from the committed weights, so
**do not commit the result** — `git checkout artifacts/` when you are done.

## Read next

[02-instruction-set.md](02-instruction-set.md).
