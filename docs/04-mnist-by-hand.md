# Stage 4 — MNIST by hand

**Goal:** write the entire MNIST program yourself, in assembly, and run it.

**File you edit:** `programs/mnist_by_hand.cubasm`.
**Test:** `pytest tests/test_04_mnist_by_hand.py -v`.

---

## Why by hand

Stage 3's compiler makes more sense once you have done its job manually. And when a
compiler emits something wrong — it will — you need to read its output and spot the
mistake. This stage is that skill.

## The two operands everyone gets wrong

Almost every failure in this stage is one of these.

### `index` is an element index, not an address

`memory=` is a **byte address in main memory**: `memory=0x18C00` is byte 101,376.

`index=` is an **element index inside a scratchpad**. Scratchpads are not byte-addressed
at all — they are arrays, and `index=` is the subscript. `index=128` is the 129th
element, whatever size that element is.

The difference is physical. Main memory is one byte-addressed memory shared with the
host. Each scratchpad is a separate array with a fixed element width baked into its
wiring, so the hardware has no way to address half of a bias.

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

`count=128` moves 128 **elements**. How many bytes of main memory that touches depends
on the scratchpad:

| Space | Element width | `count=128` reads/writes |
|---|---|---|
| `ACTIVATION_SCRATCHPAD` | 1 byte | 128 bytes of main memory |
| `WEIGHT_SCRATCHPAD` | 1 byte | 128 bytes of main memory |
| `BIAS_SCRATCHPAD` | 4 bytes | **512** bytes of main memory |
| `ACCUMULATORS` | 4 bytes | **512** bytes of main memory |

So 128 biases is `count=128`. `count=512` would try to load 512 biases and run off the
end of a 256-element scratchpad; the simulator will say so.

For a weight matrix, `count` is the number of weights, not rows: layer 1 is 128 rows of
784, so `count = 128 * 784 = 100352`.

## What you are writing

Open `programs/mnist_by_hand.cubasm`. The comment block at the top has every number you
need. The first `LOAD` and the final `HALT` are written; fill in the two blanks between
them.

### Layer 1: 784 inputs, 128 outputs

1. `LOAD` the weights into `WEIGHT_SCRATCHPAD` at index 0. How many elements?
2. `LOAD` the 128 biases into `BIAS_SCRATCHPAD` at index 0 — 32-bit, but `count` is
   still in elements.
3. `MATRIX_MULTIPLY` from `input=0` by those weights, `outputs=128`, `inputs=784`, into
   `accumulator=0`.
4. `ADD_BIAS` those 128 accumulators.
5. `RECTIFIED_LINEAR` them into `destination=1024` with `shift=12`.

### Layer 2: 128 inputs, 10 outputs

The same shape, three differences.

- Its parameters go *after* layer 1's in the scratchpads, so its `index=` operands are
  not zero. Layer 1 used weight scratchpad elements 0 to 100,351 and bias scratchpad
  elements 0 to 127.
- Its `input=` is 1024 — the hidden activations layer 1 just wrote.
- There is no rectified linear step. The raw 32-bit accumulators are the answer, so it
  ends with a `STORE` from `ACCUMULATORS` to the output region.

Every number is in the comment block at the top of the file, or the product of two of
them.

## Check

```bash
pytest tests/test_04_mnist_by_hand.py -v
```

The test assembles your file, drops your instructions into `artifacts/mnist.cub` in
place of the compiler's, runs ten images, and requires the answers to match exactly. A
last test compares your instruction *sequence* to the compiler's — not required for
correctness, but you should be able to explain any difference.

### When it fails

- **A range error** (`LOAD WEIGHT_SCRATCHPAD: [0, 802816) exceeds size 131072`) means a
  `count` or `index` is wrong. The message names the space and the range: 802,816 is
  100,352 times eight, so something was multiplied by the wrong thing.
- **It runs but the answers are wrong.** Check layer 2's `index=` operands first —
  those are the ones that are not zero.
- **Still stuck?** `python -m python disassemble artifacts/mnist.cub` prints the
  compiler's version; compare line by line. Try to find it yourself first.

You can also assemble and read back your own file:

```bash
python -m python assemble programs/mnist_by_hand.cubasm -o /tmp/mine.bin
python -m python disassemble /tmp/mine.bin
```

## Stretch: splitting a matrix multiply

`ACTIVATION_SCRATCHPAD` holds 4,096 elements, so a 784-wide input fits easily. Pretend
it only held 512.

Write layer 1's multiply as two `MATRIX_MULTIPLY` instructions over `inputs=392` each,
the second with `accumulate=1` so it adds rather than overwrites. You will need two
`LOAD`s for the input and two for the weights — and you will find a row-major matrix
awkward to split this way, since each row's halves are not adjacent in memory.

That awkwardness is why real accelerators have a `LOAD` with a stride.

## Read next

[05-registers-and-memory.md](05-registers-and-memory.md).
