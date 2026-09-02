# Stage 8 — RTL

**Goal:** finish the hardware core and prove, with a cocotb testbench, that it
computes exactly what your simulator computes.

**File:** `rtl/src/cub_core.sv`. **Test:** `pytest tests/test_08_rtl.py` (or
`make -C rtl`). **Read first:** `docs/isa.md` section 7, and your own `cub/sim.py`.

## Tools

You need [Icarus Verilog](https://steveicarus.github.io/iverilog/) and cocotb:

```bash
brew install icarus-verilog            # macOS
sudo apt-get install iverilog          # Debian/Ubuntu
pip install -e ".[rtl]"
```

cocotb lets you write the testbench in Python. That matters here because the
reference the hardware is checked against *is* Python: the testbench builds a
program with `cub.asm`, runs it on `cub.sim.Machine`, runs the same bytes through
the RTL, and compares. There is no second copy of the expected values to get wrong.

## The core

`cub_core` is a single state machine with one multiply-accumulate unit. It is slow on
purpose. One instruction at a time, one byte of DRAM per cycle, one MAC per cycle.
Readable first, fast later.

```
          +-----------+          +----------------------+
 start -->|           |  addr    |                      |
 done  <--| cub_core  |<-------->|  DRAM (byte-wide,    |
          |           |  rdata   |  1-cycle read)       |
          +-----------+  wdata   +----------------------+
           |  |  |  |
        SPAD_A SPAD_W SPAD_B ACC      (inside the core)
```

Read `rtl/src/cub_core.sv` top to bottom before touching it. The states map onto the
simulator's methods one to one:

| State | Simulator | What it does |
|---|---|---|
| `FETCH` | `fetch()` | reads 16 bytes from `pc` into the instruction register |
| `DECODE` | `decode()` | slices the fields out of the 128-bit register |
| `LOAD` / `STORE` | `_exec_load` / `_exec_store` | one byte per cycle, assembling INT32s for `SPAD_B` and `ACC` |
| `MATMUL` | `_exec_matmul` | two nested counters, one MAC per cycle |
| `ADD_BIAS` | `_exec_add_bias` | one add per cycle |
| `RELU` | `_exec_relu` | one element per cycle |
| `HALT` | `_exec_halt` | raises `done` |

The memories are plain SystemVerilog arrays. A synthesis tool infers block RAM from
them; that is why they are read and written in the simple synchronous pattern you see
rather than through anything fancier.

## Your task

Two blanks, both marked `TODO(onboard, stage 8)`:

1. **The MAC.** `mac_sum_next` must become `mac_sum` plus the product of the
   activation byte and the weight byte currently coming out of the scratchpads. Both
   are declared `signed`; keep them that way. If you cast either to unsigned, or
   introduce an unsigned intermediate, Verilog multiplies unsigned and every
   negative weight is wrong. The tests with negative values will catch it.
2. **RELU.** Starting from `relu_tmp = relu_in`: zero it if the `relu` flag
   (`f_flag8`) is set and it is negative, shift right *arithmetically* (`>>>` on a
   signed value, not `>>`) by the low five bits of the shift field, then set
   `relu_out` to the value saturated to `[-128, 127]`.

Above each blank is a safe default (`mac_sum_next = mac_sum`, `relu_out = 0`), so
the unfilled core compiles and simply gets every answer wrong. That is on purpose:
you get a running simulation and a failing comparison, not a syntax error.

Both are things you already wrote in Python in Stage 4. The test that fails first
will tell you which one is wrong.

## Check

```bash
pytest tests/test_08_rtl.py -v
```

This compiles the core with Icarus and runs the cocotb tests in `rtl/tb/test_cub.py`.
The directed tests cover every instruction with small programs against the simulator.
The final test runs the complete MNIST program on one image, baked into a hex file by
`python -m cub hex`, and compares the ten logits against `artifacts/golden.npz`.
The whole suite takes a few seconds under Icarus. The MNIST program is 205 134
cycles of a byte-at-a-time machine, which is 1.6 ms at the board's 125 MHz.

To run the same tests without pytest, or to get waveforms:

```bash
make -C rtl                 # same tests, plain cocotb flow
make -C rtl WAVES=1         # also dump a waveform, see rtl/build/sim_build/
make -C rtl clean-all
```

Open the dump with GTKWave or Surfer and find the `MATMUL` state: you can watch
`mac_sum` grow one product per cycle.

## Questions to be able to answer

- How many cycles does the layer-1 `MATMUL` take? The `LOAD` of its weights? Which
  would you speed up first, and how?
- Why does `SPAD_B` get loaded four bytes at a time, and what would go wrong if the
  byte order were reversed?
- The core ignores unused instruction bits; the simulator rejects them. Why the
  difference?

## Stretch

Make `MATMUL` do four MACs per cycle by reading four weight bytes at once (widen
`SPAD_W` to 32 bits). Keep the cocotb tests passing. Then measure the cycle count
again.

## Milestone video

1. `pytest tests/test_08_rtl.py -v` passing, including the MNIST test.
2. Show your two blanks and explain why `$signed` and `>>>` are there.

Submit at **[INSERT_GOOGLE_FORM_LINK_HERE]**.

## Read next

`docs/09-fpga.md`.
