# Stage 6 — The hardware

**Goal:** finish the chip and prove, with a testbench, that it computes exactly what the
Python simulator computes.

**File you edit:** `rtl/src/cub_core.sv`.
**Test:** `pytest tests/test_06_rtl.py -v`, or `make -C rtl`.
**Read first:** [05-registers-and-memory.md](05-registers-and-memory.md), and
`cub/simulator.py`.

This is the second and last stage where you write something.

---

## What "RTL" means

Hardware is described at a level called **register transfer level**, almost always
shortened to RTL. The name is literal: you describe the design as a set of registers,
and the rules for what value each register transfers into on each clock edge. That is
all a synchronous digital circuit is.

The two constructs that matter:

```systemverilog
always_ff @(posedge clk)     // "on each rising clock edge, these registers update"
always_comb                  // "these wires are always this function of their inputs"
```

`always_ff` describes storage — flip-flops, hence the name. `always_comb` describes
plain logic with no memory: gates and wires, computing a result the instant its inputs
change. A design is those two things and nothing else. Assignments inside `always_ff`
use `<=`, which means "take this value at the *end* of the cycle", so everything in the
block updates together from the old values.

SystemVerilog is a language for writing this down. It looks like a programming language
and it is not one: there is no sequence, no calling, no allocating. Every line you write
inside a module is happening simultaneously, every cycle, forever.

## Why hardware after a simulator

Hardware bugs are expensive to find. A simulator is cheap to write, cheap to run, and
lets you settle every question about *what the answer should be* before you have to
worry about *how many cycles it takes*. So the rule is: no hardware until there is
something bit-exact to compare it against.

That is why the testbench is written in Python. It builds a program with
`cub.assembler`, runs it on `cub.simulator.Machine`, runs the same bytes through the
SystemVerilog, and compares. There is no second copy of the expected answers to get
wrong. The tool that makes this possible is [cocotb](https://www.cocotb.org/), which
lets Python drive a hardware simulation cycle by cycle.

## The whole chip is one state machine

Open `rtl/src/cub_core.sv` and read it top to bottom before you touch anything. Its
nine states map onto the simulator's methods one for one:

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

It is slow on purpose. One instruction at a time, one byte of memory per cycle, one
multiply per cycle. Readable first; fast later, if ever.

## It really is this simple

Four excerpts, each doing exactly what its name says.

### Decoding is slicing

There is no decoder. The operands are just bit ranges of the instruction register, and
naming them is the entire decode step:

```systemverilog
wire logic [7:0]  opcode                = instruction[7:0];
wire logic [15:0] field_multiply_inputs = instruction[103:88];
wire logic [15:0] field_rectify_count   = instruction[63:48];
```

Compare that with `decode()` in `cub/instruction_set.py`, which shifts and masks the
same ranges. The hardware does not even need the shift: the wires simply come off those
bit positions.

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

That is why fetch costs 17 cycles: sixteen bytes, plus one for the memory's read
latency.

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

Five lines, and one of them is the actual work. Compare `_execute_add_bias` in the
simulator: it is a `for` loop over the same addition. The `for` loop is the difference —
Python gets a loop counter for free, and the hardware has to keep `element_index` in a
register and increment it itself.

Note there is no overflow handling. A 32-bit adder wraps on overflow because that is
what a 32-bit adder physically does, which is exactly the behaviour `wrap_to_int32`
models in the simulator. The simulator has to work to imitate the hardware here, not
the other way around.

### The memories are arrays

```systemverilog
logic signed [7:0]  activation_scratchpad [0:4095];
logic signed [7:0]  weight_scratchpad     [0:131071];
logic signed [31:0] bias_scratchpad       [0:255];
logic signed [31:0] accumulators          [0:255];
```

Plain arrays. A synthesis tool recognizes the pattern and builds them out of the block
memory the chip has. That is why the two large ones are read *synchronously* — the
address goes out on one cycle, the data comes back the next:

```systemverilog
always_ff @(posedge clk) begin
    activation_read_data <= activation_scratchpad[activation_read_index];
    weight_read_data     <= weight_scratchpad[weight_read_index];
end
```

That one-cycle delay is the reason `MATRIX_MULTIPLY` is written as a small pipeline:
one half of the state issues addresses, the other half consumes the data that arrived
from the addresses issued last cycle. The `product_valid` register is what connects
them.

## Your task

Two blanks, both marked `TODO(onboard, stage 6)`. You already wrote both of them in
Python; the simulator is right there.

### 1. The multiplier

Inside the `always_comb` block near the top, `running_sum_next` has to become
`running_sum` plus the product of the two bytes arriving from the scratchpads this
cycle. The variables you need:

- `running_sum` — `logic signed [31:0]`, the dot product so far
- `activation_read_data` — `logic signed [7:0]`, this cycle's activation
- `weight_read_data` — `logic signed [7:0]`, this cycle's weight
- `running_sum_next` — `logic signed [31:0]`, what you assign

Both inputs are already declared `signed`, so a plain `*` between them is a signed
multiply and the result widens correctly. **Do not cast either one, and do not
introduce an unsigned intermediate.** SystemVerilog decides signed versus unsigned
multiplication from the operand types, and if either one is unsigned the whole
multiply becomes unsigned — at which point every negative weight comes out as a large
positive number. The tests with negative values will catch it, but the failure looks
like nonsense until you know to look for this.

### 2. The activation function

In the second `always_comb` block, starting from `rectify_value = rectify_input`, do
three things:

1. If `flag_bit` is set and `rectify_input` is negative (its bit 31 is 1), set
   `rectify_value` to zero.
2. Shift `rectify_value` right by `field_rectify_shift[4:0]` — with `>>>`, the
   *arithmetic* shift, not `>>`. On a signed value `>>>` copies the sign bit in, which
   is what the instruction set specifies; `>>` shifts in zeros and turns negative
   numbers into large positive ones.
3. Set `rectify_output` to `rectify_value` clamped: `127` if it is above 127, `-128` if
   it is below -128, otherwise its low 8 bits.

The variables you need:

- `rectify_input` — `logic signed [31:0]`, the accumulator, already assigned for you
- `flag_bit` — the instruction's `rectify` flag, bit 8
- `field_rectify_shift[4:0]` — the shift amount, low five bits
- `rectify_value` — `logic signed [31:0]`, your scratch variable
- `rectify_output` — `logic signed [7:0]`, what gets written to the scratchpad

Above each blank is a safe default (`running_sum_next = running_sum` and
`rectify_output = 0`), so an unfilled core still compiles and simply gets every answer
wrong. That is deliberate: you get a running simulation and a failing comparison, not a
wall of syntax errors.

## Check

```bash
pytest tests/test_06_rtl.py -v
```

This compiles the core with Icarus Verilog and runs the tests in `rtl/tb/test_cub.py`.
The directed tests cover every instruction with small programs checked against the
simulator; the last one runs the whole MNIST program on one image and compares the ten
answers against `artifacts/golden.npz`. The suite takes a few seconds.

The test that fails first tells you which blank is wrong.
`test_matrix_multiply_small` uses negative weights, so it fails immediately if the
multiply lost its sign. `test_rectified_linear_shift_and_saturate` covers saturation in
both directions, a negative number shifted right, and a shift of 31.

To run the same tests without pytest, or to get waveforms:

```bash
make -C rtl                 # the same tests, plain cocotb
make -C rtl WAVES=1         # also dump a waveform into rtl/build/sim_build/
make -C rtl clean-all
```

Open the waveform with [GTKWave](https://gtkwave.sourceforge.net/) or
[Surfer](https://surfer-project.org/) and find `STATE_MATRIX_MULTIPLY`. You can watch
`running_sum` grow by one product per cycle, 784 times, and then be written into an
accumulator and reset to zero. This is the single most useful thing you can do to make
the design feel real, and it takes two minutes.

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

Make `MATRIX_MULTIPLY` do four multiplies per cycle by reading four weight bytes at
once (widen the weight scratchpad's read port to 32 bits). Keep every test passing, then
measure the cycle count again. Notice which of the two big costs it does *not* improve —
that is the one a wider memory port would fix instead.

## Where you are now

You have followed one network from PyTorch, through quantization, through a compiler,
into an instruction set, and into hardware — and the answers agree bit for bit at every
step. That agreement is the whole discipline: a golden model, a specification both
sides are written against, and a test that compares them on every change.

Ask about a first task on the main project whenever you are ready.
