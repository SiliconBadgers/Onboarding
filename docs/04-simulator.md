# Stage 4 — The simulator

**Goal:** a software model of the core that executes Cub programs exactly as the
hardware will. This is the *golden model*: when the RTL disagrees with it in Stage 8,
the RTL is wrong.

**File:** `cub/sim.py`. **Test:** `pytest tests/test_04_simulator.py`.
**Read first:** `docs/isa.md` sections 2 to 4.

## Why a simulator before hardware

Hardware bugs are expensive to find. A simulator is cheap to write, cheap to run, and
lets you settle every question about *what the result should be* before you have to
worry about *how many cycles it takes*. The main project's roadmap has the same rule:
no RTL until there is something bit-exact to compare it against.

For that to work the simulator must be **bit-exact**, not approximately right. Every
rounding, every wraparound, every clamp is specified in `docs/isa.md`, and the
simulator implements it literally. Write it the slow, obvious way. Loops, not NumPy
tricks. It should read like the pseudocode in the spec, and later like the RTL.

## The machine

`Machine` holds the five memories from `docs/isa.md` section 2 as NumPy arrays of the
exact element type, plus a program counter. `step()` fetches 16 bytes at `pc`,
decodes them (strictly), and dispatches to `_exec_<name>`. `run()` steps until `HALT`.

`LOAD`, `STORE`, `NOP`, and `HALT` are written for you. Read them; they show the
pattern (range check first, then the work, then the cycle estimate).

## Your task

Three blanks, in this order:

1. `_exec_add_bias`. Five lines. Add each bias to its accumulator. Use `wrap_i32`
   so a sum past 2^31 wraps the way a 32-bit adder does, instead of NumPy raising an
   overflow warning or silently promoting.
2. `_exec_matmul`. For each output row `n`, compute the dot product of the input
   vector with row `n` of the weight matrix (row-major, `w + n*k + col`), add the
   existing accumulator if `accumulate` is set, and write it back wrapped. Convert
   NumPy scalars to `int` before multiplying, or NumPy will do the arithmetic in
   INT8 and overflow.
3. `_exec_relu`. Read the accumulator, zero it if negative and the `relu` flag is
   set, shift right (Python's `>>` on a negative `int` is already arithmetic), clamp
   to INT8 with `clamp_i8`, write to `SPAD_A`.

## Check

```bash
pytest tests/test_04_simulator.py -v
```

The tests go instruction by instruction with tiny programs whose answers you can
check by hand, and then run the full compiled MNIST program on 20 images and require
the logits to match the NumPy reference from Stage 2 exactly. The RELU test table is
worth reading closely: it covers saturation on both sides, a negative number shifted
right, and shift 31.

Then:

```bash
python -m cub eval --count 200
```

should print about 97%, and `python -m cub run --index 3` should draw a zero and
predict 0.

## Questions to be able to answer

- `wrap_i32(2**31)` is `-2**31`. Where in the MNIST program could that actually
  happen, if anywhere?
- The simulator counts one "cycle" per MAC. How many cycles does one MNIST inference
  take? (`m.cycles` after `run()`.) At 100 MHz, how many inferences per second is
  that?

## Milestone video

1. `pytest tests/test_04_simulator.py -v` passing.
2. `python -m cub run --index 3`.
3. Walk through your `_exec_matmul` and explain what `w + row * k + col` is indexing.

Submit at **[INSERT_GOOGLE_FORM_LINK_HERE]**.

## Read next

`docs/05-mnist-by-hand.md`.
