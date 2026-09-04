# Stage 1 — PyTorch, and turning decimals into whole numbers

**Goal:** know exactly which operations the chip has to be able to perform, and why
every number in the network ends up as an integer.

**Files to read:** `cub/model.py`, `cub/quantization.py`.
**Test:** `pytest tests/test_01_pytorch.py -v` (it already passes).

---

## You do not need to understand machine learning

This is the most important sentence in the guide, so it gets its own heading.

Nothing on this track requires you to know *why* a neural network recognizes digits, or
how it was trained, or what a gradient is. Those are real and interesting questions and
they belong to somebody else's job.

What you need is the **list of operations** a network performs when it runs. That list
is short, every item on it is arithmetic you learned in school, and it is the exact
specification of what the chip must do. Read this stage as "here is the small set of
things my hardware will be asked to do", not as "here is how machine learning works".

## The network

```
input   784 numbers   (a 28x28 grayscale image, flattened into one row)
layer 1 784 -> 128    hidden      = image  x  weights1 + biases1
        rectify       hidden      = max(hidden, 0)
layer 2 128 -> 10     ten_outputs = hidden x  weights2 + biases2
        argmax        the digit is the position of the largest of the ten
```

That is the whole network. Four operations. Here is what each one is.

### Weights — a matrix multiply

`weights1` is a table of 128 x 784 = 100,352 numbers. The image is a row of 784
numbers. Multiplying them means: for each of the 128 outputs, walk down that output's
row of 784 weights, multiply each weight by the matching pixel, and add all 784
products together. One output, one number.

Do that 128 times and you have the hidden layer. That is 100,352 multiplies and
100,352 additions, and it is by far the bulk of the work — which is why
`MATRIX_MULTIPLY` is the one instruction the chip is actually built around.

**Why it works** is machine learning. **What it is** is "multiply pairs of numbers and
add up the results", and that is all the hardware cares about.

### Biases — an add

`biases1` is a list of 128 numbers, one per output. After the matrix multiply, add the
matching bias to each output. 128 additions. That is the entire operation.

### Rectified linear — a comparison against zero

Usually written "ReLU" and pronounced "ray-loo". It replaces every negative number with
zero and leaves positive numbers alone:

```
if value < 0: value = 0
```

That is it. In hardware it is a single comparison, or really just a look at the sign
bit. It is on the list of operations because a network of pure matrix multiplies would
collapse into one big matrix multiply and could not learn anything interesting — but
again, that is the machine-learning reason, and you do not need it. You need "check the
sign, maybe write zero".

### Argmax — find the largest

The ten final numbers are scores, one per digit. The largest one wins. The chip does
not do this at all: it hands the ten numbers back and the host program compares them.
Ten comparisons on a general-purpose processor are not worth building a circuit for.

### What is deliberately not here

- **No softmax.** Softmax turns the ten scores into probabilities. It never changes
  *which* score is largest, so taking the argmax of the raw scores gives the same
  answer for free.
- **No dropout or batch normalization.** Real models have them; they would have to be
  folded into the weights before compiling. Ours does not have them, so we skip the
  problem.
- **No convolutions.** A fully connected network is the smallest thing that is still a
  real network. Convolutions are a matrix multiply with a more complicated indexing
  pattern in front of it.

Look at `cub/model.py`. The `forward` method is three lines, and each line is one of
the operations above.

---

## From decimal numbers to whole numbers

Open `artifacts/trained_weights.npz` and look at a weight:

```bash
python -c "
import numpy as np
w = np.load('artifacts/trained_weights.npz')['weights1']
print(w[0][:6]); print('largest magnitude:', abs(w).max())"
```

They are decimal numbers, a bit either side of zero, stored as 32-bit floating point.
Hardware does not want them that way, for two reasons:

- **Multiplying is expensive.** An 8-bit integer multiply is one small circuit on a
  chip, and it finishes in one cycle. A 32-bit floating-point multiply is a much larger
  circuit that takes several.
- **Memory is expensive.** Those 100,352 weights are 401,408 bytes as 32-bit decimals
  and 100,352 bytes as 8-bit integers. A quarter of the memory, and a quarter of the
  time spent moving them onto the chip. On a real accelerator, moving weights is often
  the thing that costs the most.

So before the network ever reaches the chip, every decimal number is turned into a
whole number. This is called **quantization**, and it is one line of arithmetic.

### Scaling

Pick a scale factor `S`, multiply every number by it, and round:

```
S = 127 / (the largest magnitude in the tensor)
whole_number = round(decimal_number * S)
```

The largest value in the tensor now lands exactly on 127, which is the biggest value a
signed 8-bit integer can hold, and everything else lands somewhere in `[-127, 127]`.
Nothing is thrown away except precision below the rounding step. To read a whole number
back as a decimal you divide by `S` again — but nothing on the chip ever does, because
the chip does not deal in decimals at all.

Note that zero maps to zero, and the scale has no offset added to it. That is called
*symmetric* quantization, and it matters more than it looks: a dot product of two
symmetric tensors is a plain sum of products. Add an offset and the hardware needs
correction terms.

### Where the scales come from

- **The image.** Pixels are already whole numbers, 0 to 255, but training normalized
  them, so the host applies the same normalization and then scales so that a fully
  white pixel becomes exactly 127. A black pixel becomes about -19, because it is below
  zero once the mean is subtracted.
- **Each layer's weights.** One scale per layer: `127 / largest magnitude`.
- **Each layer's biases.** The bias is added to the result of the multiply, so it has
  to be at the *same* scale as that result — which is `input_scale * weight_scale`. The
  biases are stored as 32-bit integers because that product can be large.

### Why the sums need 32 bits, and how they come back down

Multiply two 8-bit numbers and you get at most 127 x 128, which needs 16 bits. Add 784
of those together and you need about 10 bits more: 26 in total. So the running sums —
the **accumulators** — are 32-bit. There is room to spare, and that is the whole reason
accumulators are wider than the values that feed them.

But the next layer wants 8-bit inputs again. So after the biases are added, the
accumulator is divided by a power of two — a right shift, which in hardware is free,
just wires — and clamped to `[-128, 127]`:

```
value = max(value, 0)          the rectified linear step
value = value >> shift         scale back down to 8 bits
value = clamp(value, -128, 127)
```

How is `shift` chosen? Run layer 1 on a few hundred real images, see how large the
accumulators actually get, and pick the smallest shift that brings the largest of them
to 127 or below. For this network that turns out to be 12. A rare image that exceeds it
saturates at 127, which is harmless — that is what the clamp is for.

### This is what a real accelerator does

Google's Tensor Processing Unit — the chip that made this style of hardware famous —
runs inference in 8-bit integers with 32-bit accumulators, for exactly the two reasons
above: integer multipliers are small, and 8-bit weights are cheap to move. The first
generation was built around a big grid of 8-bit multiply-accumulate units and very
little else. Everything you are about to build is the same idea with the grid shrunk to
a single multiplier.

The one refinement a production chip adds is replacing the power-of-two shift with a
fixed-point multiply, so it can hit any scale ratio instead of only powers of two, and
recover a little accuracy. Same idea, more bits.

## Check it yourself

```bash
pytest tests/test_01_pytorch.py -v
```

The tests load the committed trained weights, check that the PyTorch network reproduces
the reference answers, and then check the whole-number version of the same network. The
decimal network gets 97.4% on 1000 test images. The whole-number network gets 97.3%.
Losing one image in a thousand to buy a four-times-smaller, much faster chip is the
trade that this entire field is built on.

## Things to try

Break something and watch a specific number change:

- In `cub/quantization.py`, change `quantize_input` to use `np.floor` instead of
  `np.rint`. Run the tests. Which one fails, and by how much?
- Change `choose_shift` to always return one more than it does. Accuracy barely moves —
  why? Now one less. Why is that so much worse?
- `python -c "from cub.quantization import accumulator_bits_needed as f; print(f(784))"`
  — how many bits does layer 1's dot product actually need? What would happen with a
  16-bit accumulator?

## Retraining (optional)

```bash
python -m cub train
```

Three epochs on a laptop processor take about a minute and reach roughly 97.3%. This
overwrites everything in `artifacts/`, and every expected value downstream was
generated from the committed weights, so **do not commit the result**. Run
`git checkout artifacts/` when you are done looking.

## Read next

[02-instruction-set.md](02-instruction-set.md).
