# Stage 5 — MNIST by hand

**Goal:** write the entire MNIST program yourself, in assembly, and run it.

**File:** `programs/mnist_by_hand.cubasm`. **Test:** `pytest tests/test_05_mnist_by_hand.py`.

## Why by hand

The compiler in Stage 6 is about forty lines, and every one of them is easier to
write once you have done the job manually. More importantly: when the compiler emits
something wrong, you need to be able to read its output and see the mistake. This
stage is that skill.

## The plan

The file already contains the DRAM layout (where the compiler put the weights,
biases, input, and output) and the scratchpad plan (where you will put things inside
the core). The first `LOAD` and the final `HALT` are written. Fill in the rest.

Layer 1 needs, in order:

1. `LOAD` the 128x784 weights into `SPAD_W`. How many elements is that?
2. `LOAD` the 128 biases into `SPAD_B`. Remember they are INT32, but `count` is in
   elements, not bytes.
3. `MATMUL` with the input vector at `SPAD_A[0]`, `n=128`, `k=784`.
4. `ADD_BIAS`.
5. `RELU` into `SPAD_A[1024]` with the layer-1 shift, which is 12.

Layer 2 is the same shape with different numbers, except that the last layer has no
ReLU: its raw INT32 accumulators are the logits, so it ends with a `STORE` from `ACC`
to the output region instead. Layer 2's weights and biases go *after* layer 1's in
the scratchpads, so its `spad=` and `w=`/`bias=` operands are not zero.

Every number you need is in the comment block at the top of the file, or is a
product of two of them.

## Check

```bash
pytest tests/test_05_mnist_by_hand.py -v
```

The test assembles your file, replaces the compiler's instructions in
`artifacts/mnist.cub` with yours, runs ten images through the simulator, and requires
the logits to match exactly. The last test checks that your instruction *sequence*
matches the compiler's; it is not required for correctness but you should be able to
explain any difference.

If the simulator raises a range error, you have a `count` or address wrong; the
message tells you which memory and which range. If it runs but the logits are wrong,
compare your file against `python -m cub disasm artifacts/mnist.cub` one line at a
time. The point of this stage is to find that bug yourself, so try before you diff.

## Stretch: tiling

`SPAD_A` has 4096 entries, so a 784-wide input fits. Pretend it only had 512. Write
the layer-1 matmul as two `MATMUL`s over `k=392` each, the second with
`accumulate=1`. You will need two `LOAD`s for the input and two for the weights, and
you will discover that a row-major weight matrix is inconvenient to split along `k`.
That inconvenience is why Badger's `LOAD` has a stride.

## Milestone video

1. `pytest tests/test_05_mnist_by_hand.py -v` passing.
2. Scroll through your program and explain why layer 2's `w=` is 100352.

Submit at **[INSERT_GOOGLE_FORM_LINK_HERE]**.

## Read next

`docs/06-compiler.md`.
