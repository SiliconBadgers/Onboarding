# Stage 4 — MNIST by hand

**Goal:** write the entire MNIST program yourself, in assembly, and run it.

**File you edit:** `programs/mnist_by_hand.cubasm`.
**Test:** `pytest tests/test_04_mnist_by_hand.py -v`.

This is the first of the two stages where you write something.

---

## Why do it by hand

The compiler in Stage 3 is forty lines, and every one of them makes more sense once you
have done the job manually. More importantly: when a compiler emits something wrong —
and it will — you need to be able to read its output and see the mistake. This stage is
that skill.

## The two operands everyone gets wrong

Before you write a line, get these two straight. Almost every failure in this stage is
one of them.

### `index` is an element index, not an address

`memory=` is a **byte address in main memory**. It is where the compiler physically put
something, and it counts bytes: `memory=0x18C00` means byte number 101,376.

`index=` is an **element index inside a scratchpad**. Scratchpads are not addressed in
bytes at all. They are arrays, and `index=` is the array subscript. `index=128` means
"the 129th element", whatever size that element happens to be.

The reason for the difference is physical. Main memory is one wide byte-addressed
memory shared with the host, so a byte address is the only thing that makes sense.
Each scratchpad is a separate array inside the chip with a fixed element width baked
into its wiring, so an element index is the only thing that makes sense — the hardware
literally has no way to address half of a bias.

```
main memory (bytes)          bias scratchpad (32-bit elements)

0x18C00 [ b0 byte 0 ]  ->    index 0  [ b0 ]
0x18C01 [ b0 byte 1 ]
0x18C02 [ b0 byte 2 ]
0x18C03 [ b0 byte 3 ]
0x18C04 [ b1 byte 0 ]  ->    index 1  [ b1 ]
...
```

### `count` counts elements, never bytes

`count=128` moves 128 **elements**. How many bytes that touches in main memory depends
on which scratchpad you are talking to:

| Space | Element width | `count=128` reads/writes |
|---|---|---|
| `ACTIVATION_SCRATCHPAD` | 1 byte | 128 bytes of main memory |
| `WEIGHT_SCRATCHPAD` | 1 byte | 128 bytes of main memory |
| `BIAS_SCRATCHPAD` | 4 bytes | **512** bytes of main memory |
| `ACCUMULATORS` | 4 bytes | **512** bytes of main memory |

So loading 128 biases is `count=128`. Writing `count=512` there would try to load 512
biases and run off the end of a 256-element scratchpad, and the simulator would tell
you so.

For a weight matrix, `count` is the number of individual weights, not the number of
rows: layer 1's `weights1` is 128 rows of 784, so `count = 128 * 784 = 100352`.

## What you are writing

Open `programs/mnist_by_hand.cubasm`. The comment block at the top has every number you
need. The first `LOAD` (the input image) and the final `HALT` are already written; fill
in the two blanks between them.

### Layer 1: 784 inputs, 128 outputs

1. `LOAD` the weights into `WEIGHT_SCRATCHPAD` at index 0. How many elements?
2. `LOAD` the 128 biases into `BIAS_SCRATCHPAD` at index 0. Remember they are 32-bit
   but `count` is still in elements.
3. `MATRIX_MULTIPLY` the input vector at `input=0` by those weights, `outputs=128`,
   `inputs=784`, into `accumulator=0`.
4. `ADD_BIAS` those 128 accumulators.
5. `RECTIFIED_LINEAR` them into `ACTIVATION_SCRATCHPAD` at `destination=1024`, with
   `shift=12`.

### Layer 2: 128 inputs, 10 outputs

The same shape with different numbers, and two differences.

- Layer 2's parameters go *after* layer 1's in the scratchpads, so its `index=` operands
  are not zero. Where exactly? Layer 1 used elements 0 through 100,351 of the weight
  scratchpad and 0 through 127 of the bias scratchpad.
- The last layer has no rectified linear step. Its raw 32-bit accumulators are the
  answer, so it ends with a `STORE` from `ACCUMULATORS` to the output region instead.

And its `input=` is 1024 — the hidden activations layer 1 just wrote — not 0.

Every number you need is in the comment block at the top of the file, or is the product
of two of them.

## Check

```bash
pytest tests/test_04_mnist_by_hand.py -v
```

The test assembles your file, replaces the compiler's instructions in
`artifacts/mnist.cub` with yours, runs ten images through the simulator, and requires
the ten answers to match exactly. The last test checks that your instruction *sequence*
matches the compiler's; that is not required for correctness, but you should be able to
explain any difference.

### When it fails

- **A range error** (`LOAD WEIGHT_SCRATCHPAD: [0, 802816) exceeds size 131072`) means a
  `count` or an `index` is wrong. The message names the space and the range, which is
  usually enough to spot the mistake — 802,816 is 100,352 times eight, so someone
  multiplied by the wrong thing.
- **It runs but the answers are wrong.** Something is pointed at the wrong place. Check
  layer 2's `index=` operands first: those are the ones that are not zero.
- **Still stuck?** `python -m cub disassemble artifacts/mnist.cub` prints the
  compiler's version of the same program. Compare it against yours line by line. Try to
  find the bug yourself first — that is the entire point of this stage.

You can also run your program on the simulator directly:

```bash
python -m cub assemble programs/mnist_by_hand.cubasm -o /tmp/mine.bin
python -m cub disassemble /tmp/mine.bin
```

## Stretch: splitting a matrix multiply

`ACTIVATION_SCRATCHPAD` holds 4,096 elements, so a 784-wide input fits comfortably.
Pretend it only held 512.

Write layer 1's multiply as two `MATRIX_MULTIPLY` instructions over `inputs=392` each,
the second one with `accumulate=1` so it adds to the first one's result instead of
overwriting it. You will need two `LOAD`s for the input and two for the weights — and
you will discover that a row-major weight matrix is inconvenient to split this way,
because each row's two halves are not next to each other in memory.

That inconvenience is exactly why real accelerators have a `LOAD` with a stride.

## Read next

[05-registers-and-memory.md](05-registers-and-memory.md).
