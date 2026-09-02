# Stage 6 — The compiler

**Goal:** make the program from Stage 5 come out of a function, from the quantized
weights, with no numbers typed by hand.

**File:** `cub/compiler.py`. **Test:** `pytest tests/test_06_compiler.py`.

## What a compiler is here

For an MLP, "compiling" is three things:

1. **Placing** the parameters in DRAM and remembering the addresses.
2. **Deciding** where things live in the scratchpads.
3. **Emitting** the same five-instruction pattern per layer, with those addresses.

There is no graph and no intermediate representation. A list of layers is already
the graph. Badger's compiler has to handle convolutions, pooling, and reshapes, so it
does have an IR; the emit step at the end still looks like this one.

Read `compile_mlp` top to bottom. Steps 1 and 2 are written: `prog.place(...)`
appends a region to the DRAM image at the next 64-byte boundary and returns where it
landed. The scratchpad plan is three cursors (`w_cursor`, `b_cursor`, `a_in`) that
advance after each layer.

## Your task

One blank inside the per-layer loop. Emit, using `make(...)`:

1. `LOAD` this layer's weights from `dram[f"w{i}"].offset` to `SPAD_W[w_cursor]`.
2. `LOAD` this layer's biases to `SPAD_B[b_cursor]`.
3. `MATMUL` from `a_in` with the weights you just loaded, into `ACC[0]`.
4. `ADD_BIAS`.
5. If this is the last layer, `STORE` `n` INT32 values from `ACC` to the output
   region. Otherwise `RELU` into `SPAD_A_HIDDEN` with `layer.shift` and `layer.relu`.

`make("LOAD", mem=isa.MEM_SPAD_W, dram=..., spad=..., count=...)` is the shape. Look
at Stage 5 for the numbers each operand should come out to; here you are naming them
instead of typing them.

## Check

```bash
pytest tests/test_06_compiler.py -v
```

One of the tests requires your compiler's output to be byte-identical to the
committed `artifacts/mnist.cub`, both instructions and image. Another runs it on the
simulator. Then regenerate the artifact with your own compiler and confirm nothing
changed:

```bash
python -m cub compile && git status artifacts/
```

## Questions to be able to answer

- Why does the compiler reserve 64 instruction slots instead of counting first?
- What would have to change to support a three-layer MLP? A layer with 5000 hidden
  units?

## Stretch

Make the compiler tile `MATMUL` over `k` when `k` exceeds a `max_k` parameter, using
the `accumulate` flag. Add a test with `max_k=392` that still matches the golden
logits. (This is the Stage 5 stretch, automated.)

## Read next

`docs/07-runtime.md`.
