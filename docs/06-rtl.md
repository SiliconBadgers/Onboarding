# Stage 6 — The chip

**Goal:** finish the chip and prove it computes exactly what the Python simulator
computes.

**File you edit:** `rtl/cub_core.sv`.
**Test:** `pytest tests/test_06_rtl.py -v`, or `make -C rtl`.
**Read first:** [05-registers-and-memory.md](05-registers-and-memory.md), and
`python/simulator.py`.

---

## What "RTL" means

Hardware is described at **register transfer level**, shortened to RTL. The name is
literal: a set of registers, plus the rule for what value each transfers into on each
clock edge. That is all a synchronous digital circuit is.

Two constructs matter:

```systemverilog
always_ff @(posedge clk)     // "on each rising clock edge, these registers update"
always_comb                  // "these wires are always this function of their inputs"
```

`always_ff` is storage — flip-flops. `always_comb` is plain logic with no memory: gates
and wires, computing a result the instant its inputs change. A design is those two
things and nothing else. Assignments inside `always_ff` use `<=`, meaning "take this
value at the *end* of the cycle", so the whole block updates together from the old
values.

SystemVerilog looks like a programming language and is not one: no sequence, no calls,
no allocation. Every line in a module happens simultaneously, every cycle, forever.

## Why hardware after a simulator

Hardware bugs are expensive to find. A simulator is cheap and settles every question
about *what the answer should be* before you worry about *how many cycles it takes*. So:
no hardware until there is something bit-exact to compare against.

That is why the testbench is Python. It builds a program with `python.assembler`, runs it
on `python.simulator.Machine`, runs the same bytes through the SystemVerilog, and compares
— no second copy of the expected answers to get wrong. [cocotb](https://www.cocotb.org/)
is what lets Python drive a hardware simulation cycle by cycle.

## The whole chip is one state machine

Read `rtl/cub_core.sv` top to bottom before touching it. Its nine states map onto
the simulator's methods one for one:

| State | Simulator | What it does |
|---|---|---|
| `STATE_IDLE` | — | wait for the start bit |
| `STATE_FETCH` | `fetch()` | read 16 bytes into the instruction register |
| `STATE_DECODE` | `decode()` | slice the operands out and pick the next state |
| `STATE_LOAD` / `STATE_STORE` | `_execute_load` / `_execute_store` | one byte per cycle |
| `STATE_MATRIX_MULTIPLY` | `_execute_matrix_multiply` | two counters, one multiply per cycle |
| `STATE_ADD_BIAS` | `_execute_add_bias` | one add per cycle |
| `STATE_RECTIFIED_LINEAR` | `_execute_rectified_linear` | one element per cycle |
| `STATE_HALT` | `_execute_halt` | raise `done` |

Slow on purpose: one instruction at a time, one byte per cycle, one multiply per cycle.
Readable first; fast later, if ever.

## It really is this simple

### Decoding is slicing

There is no decoder. The operands are bit ranges of the instruction register, and naming
them is the entire decode step:

```systemverilog
wire logic [7:0]  opcode                = instruction[7:0];
wire logic [15:0] field_multiply_inputs = instruction[103:88];
wire logic [15:0] field_rectify_count   = instruction[63:48];
```

`decode()` in `python/instruction_set.py` shifts and masks the same ranges. The hardware
does not even need the shift — the wires come off those bit positions.

### Fetching is a shift register

Sixteen bytes arrive one per cycle, each shifted in from the top, so after sixteen
shifts byte 0 sits in `instruction[7:0]`:

```systemverilog
STATE_FETCH: begin
    if (fetch_count != 5'd0)
        instruction <= {memory_read_data, instruction[127:8]};
    if (fetch_count == 5'd16) begin
        fetch_count     <= '0;
        program_counter <= program_counter + MEMORY_ADDRESS_WIDTH'(16);
        state           <= STATE_DECODE;
    end else begin
        fetch_count <= fetch_count + 5'd1;
    end
end
```

Hence 17 cycles: sixteen bytes plus one for the memory's read latency.

### `ADD_BIAS` is an adder and a counter

```systemverilog
STATE_ADD_BIAS: begin
    accumulators[bias_accumulator_index[7:0]] <=
        accumulators[bias_accumulator_index[7:0]] + bias_scratchpad[bias_source_index[7:0]];
    if (element_index == {10'd0, field_bias_count} - 26'd1)
        state <= STATE_FETCH;
    else
        element_index <= element_index + 26'd1;
end
```

Five lines, one of which is the work. `_execute_add_bias` is a `for` loop over the same
addition; the loop is the difference. Python gets a counter for free, hardware keeps
`element_index` in a register and increments it itself.

There is no overflow handling because a 32-bit adder wraps on its own — exactly what
`wrap_to_int32` imitates in the simulator. Here the simulator works to match the
hardware, not the other way around.

### The memories are arrays

```systemverilog
logic signed [7:0]  activation_scratchpad [0:4095];
logic signed [7:0]  weight_scratchpad     [0:131071];
logic signed [31:0] bias_scratchpad       [0:255];
logic signed [31:0] accumulators          [0:255];
```

Plain arrays. A synthesis tool recognizes the pattern and builds them from the chip's
block memory. That is why the two large ones are read *synchronously* — address out one
cycle, data back the next:

```systemverilog
always_ff @(posedge clk) begin
    activation_read_data <= activation_scratchpad[activation_read_index];
    weight_read_data     <= weight_scratchpad[weight_read_index];
end
```

That one-cycle delay is why `MATRIX_MULTIPLY` is a small pipeline: half the state
issues addresses, half consumes the data from the addresses issued last cycle, and
`product_valid` connects them.

## Your task

Two blanks, both marked `TODO(onboard, stage 6)`. The simulator already does both, in
Python.

### 1. The multiplier

In the `always_comb` block near the top, `running_sum_next` must become `running_sum`
plus the product of the two bytes arriving this cycle. Variables:

- `running_sum` — `logic signed [31:0]`, the dot product so far
- `activation_read_data` — `logic signed [7:0]`, this cycle's activation
- `weight_read_data` — `logic signed [7:0]`, this cycle's weight
- `running_sum_next` — `logic signed [31:0]`, what you assign

Both inputs are already `signed`, so a plain `*` is a signed multiply and widens
correctly. **Do not cast either one, and do not introduce an unsigned intermediate.**
SystemVerilog picks signed or unsigned multiplication from the operand types; make
either one unsigned and every negative weight becomes a large positive number. The
tests with negative values catch it, but the failure looks like nonsense until you know
to look for this.

### 2. The activation function

In the second `always_comb` block, starting from `rectify_value = rectify_input`:

1. If `flag_bit` is set and `rectify_input` is negative (bit 31 is 1), set
   `rectify_value` to zero.
2. Shift `rectify_value` right by `field_rectify_shift[4:0]` using `>>>`, the
   *arithmetic* shift, not `>>`. On a signed value `>>>` copies the sign bit in, as the
   instruction set specifies; `>>` shifts in zeros and turns negatives into large
   positives.
3. Set `rectify_output` to `rectify_value` clamped: `127` above 127, `-128` below -128,
   otherwise its low 8 bits.

Variables:

- `rectify_input` — `logic signed [31:0]`, the accumulator, already assigned for you
- `flag_bit` — the instruction's `rectify` flag, bit 8
- `field_rectify_shift[4:0]` — the shift amount, low five bits
- `rectify_value` — `logic signed [31:0]`, your scratch variable
- `rectify_output` — `logic signed [7:0]`, what gets written to the scratchpad

Each blank has a safe default above it (`running_sum_next = running_sum`,
`rectify_output = 0`), so an unfilled core compiles and simply gets every answer wrong.
That is deliberate — you get a running simulation and a failing comparison, not a wall
of syntax errors.

## Check

```bash
pytest tests/test_06_rtl.py -v
```

This compiles the core with Icarus Verilog and runs `rtl/test_cub.py`: directed
tests for every instruction against the simulator, then the whole MNIST program on one
image against `artifacts/golden.npz`. A few seconds.

The first failure tells you which blank is wrong.
`test_matrix_multiply_small` uses negative weights, so it fails immediately if the
multiply lost its sign. `test_rectified_linear_shift_and_saturate` covers saturation
both ways, a negative number shifted right, and a shift of 31.

Without pytest, or for waveforms:

```bash
make -C rtl                 # the same tests, plain cocotb
make -C rtl WAVES=1         # also dump a waveform into rtl/build/sim_build/
make -C rtl clean-all
```

Open the waveform in [GTKWave](https://gtkwave.sourceforge.net/) or
[Surfer](https://surfer-project.org/) and find `STATE_MATRIX_MULTIPLY`. Watch
`running_sum` grow by one product per cycle, 784 times, then land in an accumulator and
reset. Two minutes, and the most useful thing you can do to make the design feel real.

## Questions to be able to answer

- How many cycles does layer 1's `MATRIX_MULTIPLY` take? The `LOAD` of its weights?
  Which would you speed up first, and how?
- Why does the bias scratchpad get loaded four bytes at a time, and what would go wrong
  if the byte order were reversed?
- The core ignores instruction bits it does not use; the simulator rejects a program
  that sets them. Why is that difference the right way round?
- `STATE_MATRIX_MULTIPLY` has both `output_row` and `product_row`, one cycle apart. What
  breaks if you use `output_row` when writing the accumulator?

## Stretch

Make `MATRIX_MULTIPLY` do four multiplies per cycle by widening the weight
scratchpad's read port to 32 bits. Keep every test passing, then measure the cycle count
again — and notice which of the two big costs it does *not* improve. That one needs a
wider memory port instead.

## Where you are now

You have followed one network from PyTorch through quantization, a compiler, an
instruction set, and into hardware, with the answers agreeing bit for bit at every step.
That agreement is the whole discipline: a golden model, a specification both sides are
written against, and a test that compares them on every change.

Ask about a first task on the main project whenever you are ready.
