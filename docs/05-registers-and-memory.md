# Stage 5 — Talking to the chip

**Goal:** understand how anything gets *into* the chip, how the answer gets back *out*,
and what all its memory is for.

**Files to read:** `cub/runtime.py`, `rtl/tb/test_cub.py` (the `run_program` helper),
and the port list at the top of `rtl/src/cub_core.sv`.
**Test:** `pytest tests/test_05_host_interface.py -v` (it already passes).

---

## The chip cannot reach out

Software trains you to think in function calls. There is no function call here. The
chip is a box on the end of some wires: it cannot open a file, take an argument, or
return a value. It has two ways of learning anything:

1. **Registers** — a few named storage locations the host writes and reads directly.
   This is how the host says "go" and the chip says "done".
2. **Memory** — a large shared array both sides can read and write. This is how bulk
   data moves: the image in, the weights in, the answer out.

Everything the chip does follows from something the host put in one of those two
places.

## What a register is

A **register** is a small piece of storage built into the chip — typically 32 bits of
flip-flops, readable and writable in one cycle. Unlike memory it has a name rather than
an address range, and it usually *means* something rather than just holding data.

Two groups, worth keeping apart.

### Registers the host can see

The chip's control panel. On a real accelerator these are **memory-mapped**: the chip
owns a range of addresses, and a host write to one of them lands in a register instead
of memory, so a driver looks like ordinary pointer writes.

A minimal control panel is two registers:

| Register | Bit | Direction | Meaning |
|---|---|---|---|
| `CONTROL` | 0 | host writes | write 1 to start the program at memory address 0 |
| `STATUS` | 0 | host reads | busy: the core is working |
| `STATUS` | 1 | host reads | done: the core reached `HALT` |

Here those three bits are not wrapped in a bus interface — they are three wires on
`cub_core`, because the testbench *is* the host and can drive wires directly:

```systemverilog
input  wire logic start,   // pulse high for one cycle: run the program at address 0
output      logic busy,    // high while the core is working
output      logic done     // high after HALT, until the next start
```

Turning them into two memory-mapped registers is about fifteen lines. This is an
illustration of the shape, not code in this repository:

```systemverilog
// A host writing to address 0x00 sets the start bit; reading 0x04 returns status.
always_ff @(posedge clk) begin
    start <= 1'b0;                                  // one-cycle pulse, self-clearing
    if (host_write && host_address == 8'h00)
        start <= host_write_data[0];
end

always_comb begin
    host_read_data = 32'd0;
    if (host_address == 8'h04)
        host_read_data = {30'd0, done, busy};       // the STATUS register
end
```

The start bit clears itself after one cycle — a common pattern: writing a 1 means "do
this once", not "stay in this state".

### Registers inside the chip

The chip's own working state, declared together near the top of `rtl/src/cub_core.sv`:

| Register | What it holds |
|---|---|
| `state` | which of nine states the state machine is in |
| `program_counter` | the main-memory address of the instruction being fetched |
| `instruction` | 128 bits: the instruction currently being executed |
| `fetch_count` | how many of the instruction's 16 bytes have arrived |
| `element_index` | the loop counter for `LOAD`, `STORE`, `ADD_BIAS`, `RECTIFIED_LINEAR` |
| `output_row`, `input_column`, `weight_pointer` | the three counters `MATRIX_MULTIPLY` walks |
| `running_sum` | the dot product accumulated so far for the current output |

The host sees none of these and does not need to. They exist because a state machine
has to remember where it was between clock cycles.

`cub/simulator.py` has the same registers in Python, but fewer, because Python's `for`
loops keep the loop counters for it. Hardware has no `for` loop: every counter is a
register you declare and increment yourself. That difference is most of what makes
hardware feel strange at first.

## The five memory spaces

Registers hold a few dozen values; a network needs a hundred thousand. That is what the
memories are for.

```
             +-------------------------------------------------+
  host <---> |  MAIN MEMORY   256 KiB, byte-addressed           |
             |  program | weights | biases | input | output     |
             +-------------------------------------------------+
                     |  one byte per cycle, both directions
                     |  (LOAD reads it, STORE writes it, fetch reads it)
             +-------|-----------------------------------------+
             |       v                        the chip          |
             |  ACTIVATION_SCRATCHPAD   4,096 x 8-bit           |
             |  WEIGHT_SCRATCHPAD     131,072 x 8-bit           |
             |  BIAS_SCRATCHPAD           256 x 32-bit          |
             |  ACCUMULATORS              256 x 32-bit          |
             +-------------------------------------------------+
```

**Main memory** is shared ground: the program, the weights, the image the host writes,
the answer the chip writes. Byte-addressed, because the host thinks in bytes.

**The four scratchpads** are inside the chip, one job each:

| Scratchpad | Holds | Why that size, why that width |
|---|---|---|
| `ACTIVATION_SCRATCHPAD` | the image, then each layer's outputs | 8-bit, because that is what a multiplier input is. 4,096 elements holds the 784-pixel image and the 128 hidden values with room to spare |
| `WEIGHT_SCRATCHPAD` | every layer's weights | 8-bit for the same reason. 131,072 elements because layer 1 alone is 100,352, and everything must fit at once |
| `BIAS_SCRATCHPAD` | every layer's biases | 32-bit, since a bias is added to a 32-bit accumulator at the same scale. Only 138 are needed |
| `ACCUMULATORS` | running sums, then the answers | 32-bit, since summing 784 products of two 8-bit numbers needs 26 bits ([Stage 1](01-pytorch.md)) |

Three things follow.

**The scratchpads are managed by software.** No cache, no automatic fetching, no
virtual memory: if the program does not `LOAD` it, it is not on the chip. The compiler
decides what to bring in and when, and that is the whole memory hierarchy. Predictable,
compiler-planned data movement is a large part of why accelerators beat
general-purpose processors here.

**Scratchpads are addressed in elements, main memory in bytes.** See
[Stage 4](04-mnist-by-hand.md) if that is still fuzzy.

**Nothing moves directly between scratchpads.** No instruction copies weights to
activations. Every path goes through the accumulators, or out to main memory and back.

## Every instruction, as a data movement

That is all the instruction set is: rules for moving numbers between the boxes above.

| Instruction | From | Through | To |
|---|---|---|---|
| `LOAD` | main memory | — | a scratchpad |
| `STORE` | activations or accumulators | — | main memory |
| `MATRIX_MULTIPLY` | activations + weights | the multiplier | accumulators |
| `ADD_BIAS` | accumulators + biases | an adder | accumulators |
| `RECTIFIED_LINEAR` | accumulators | rectify, shift, clamp | activations |
| `HALT` | — | — | sets the done bit |

Follow one image through the twelve instructions with that table in hand:

```
host writes the image into main memory at 0x19340
  LOAD              main memory 0x19340  ->  activations[0 .. 783]
  LOAD              main memory 0x400    ->  weights[0 .. 100351]
  LOAD              main memory 0x18C00  ->  biases[0 .. 127]
  MATRIX_MULTIPLY   activations x weights -> accumulators[0 .. 127]
  ADD_BIAS          accumulators += biases[0 .. 127]
  RECTIFIED_LINEAR  accumulators -> activations[1024 .. 1151]
  LOAD              main memory 0x18E00  ->  weights[100352 .. 101631]
  LOAD              main memory 0x19300  ->  biases[128 .. 137]
  MATRIX_MULTIPLY   activations[1024..] x weights[100352..] -> accumulators[0 .. 9]
  ADD_BIAS          accumulators += biases[128 .. 137]
  STORE             accumulators[0 .. 9] -> main memory 0x19680
  HALT              done goes high
host reads ten 32-bit numbers from main memory at 0x19680
```

## The host's four steps

In `cub/runtime.py`, against the simulator:

```python
class SimulatorBackend:
    def run(self, program, quantized_pixels):
        program.write_input(quantized_pixels)          # 1. put the image in memory
        machine = Machine(program.image)               # 2. hand memory to the chip
        machine.run()                                  # 3. start, and wait for done
        return Program.read_output(                    # 4. read the answer back
            machine.main_memory, program.regions["output"])
```

And in `rtl/tb/test_cub.py`, against the state machine, one wire at a time:

```python
dut.start.value = 1          # set the start bit
await RisingEdge(dut.clk)
dut.start.value = 0          # ... for exactly one cycle
while dut.done.value != 1:   # poll the done bit
    await RisingEdge(dut.clk)
```

Same four steps. A driver for a real board would be the same again, with the register
writes going over a bus. That is why `predict()` takes a `backend` argument: everything
outside those four steps — normalizing, scaling, comparing the ten answers — is
identical whatever is on the other end.

## What the host still has to do

The chip does integer arithmetic between the first `LOAD` and the last `STORE`. The
rest is the host's:

1. Normalize the pixels the way training did, and scale them to 8-bit whole numbers.
2. Write them into the input region.
3. Start the core; wait for done.
4. Read the ten 32-bit values.
5. Divide by `output_scale` to get back to decimal numbers — only needed for display —
   and take the argmax.

Step 5 is worth a thought. The ten numbers are at a scale nobody chose deliberately —
it fell out of the input scale, the two weight scales and the shift. That does not
matter: dividing all ten by the same positive constant cannot change which is largest,
so the chip is free to hand back numbers in units nobody has a name for.

## Timing: one thing at a time

The core is **synchronous** in the strictest sense: one instruction fetched, decoded
and executed to completion before the next fetch begins, nothing overlapping. That is
what makes the design comprehensible, and it costs performance you can measure:

| Operation | Cycles |
|---|---|
| fetch | 17 per instruction (16 bytes, plus one for the memory's read latency) |
| decode | 1 |
| `LOAD` / `STORE` | one per byte moved, plus 1 |
| `MATRIX_MULTIPLY` | `outputs * inputs`, plus 2 |
| `ADD_BIAS` / `RECTIFIED_LINEAR` | one per element |

Twelve instructions at 18 cycles of overhead is 216 cycles of fetch and decode against
roughly 205,000 cycles of work. Fetch is free here, because one instruction covers
100,352 multiplies.

The whole program takes **205,134 cycles**, which the Stage 6 testbench prints. Half is
the `LOAD` of layer 1's weights — 100,352 bytes at one per cycle — and half is layer 1's
`MATRIX_MULTIPLY`, 100,352 multiplies at one per cycle. At 125 MHz that is 1.6
milliseconds per image, about 600 images a second.

Both halves are the first things a real design attacks: a wider memory port so `LOAD`
moves eight bytes a cycle, and a grid of multipliers instead of one. Neither changes the
instruction set, which is the point of having one.

## Things to try

```bash
pytest tests/test_05_host_interface.py -v
python -m cub run --index 12
python -m cub accuracy --count 500
```

- Comment out the divide by `output_scale` in `cub/runtime.py`. Does `predict` still
  return the right digit? Why?
- `LOAD` from the output region before the program has written it. What comes back?
- Print `machine.program_counter` at the top of `step()` and run one image. Twelve
  values, sixteen apart — the whole control flow of the chip.

## Read next

[06-rtl.md](06-rtl.md).
