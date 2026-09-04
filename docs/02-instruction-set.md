# Stage 2 — The instruction set

**Goal:** know every instruction, what each one does, and how each one is written
down as sixteen bytes.

**Files to read:** `cub/instruction_set.py`, `cub/assembler.py`, `cub/simulator.py`.
**Test:** `pytest tests/test_02_instruction_set.py -v` (it already passes).

This document is the **specification**. The assembler, the simulator, the compiler and
the SystemVerilog are all written against it. When two of them disagree, this file
decides which one is wrong.

---

## What an instruction set architecture is

An **instruction set architecture** — usually shortened to ISA — is the contract
between software and hardware. It says:

1. **What operations exist.** The complete list. If it is not on the list, the chip
   cannot do it.
2. **What each operation does**, precisely enough that two people implementing it
   separately get bit-for-bit identical results.
3. **How each operation is written down as bits**, so that the hardware can read a
   program out of memory.

x86 and ARM are instruction set architectures. So is this one. Theirs have well over a
thousand instructions; ours has six that do something (plus `NO_OPERATION`), because
ours only has to run neural networks.

The point of having a contract at all is that the two sides can then be built
independently. In this repository the Python simulator and the SystemVerilog core were
written by different means at different times, and they agree exactly, because they
were both written against this file.

## The chip is simple. The software is not.

Here is the idea that makes the rest of the track make sense.

Running a neural network is a big, structured, complicated computation. Executing one
instruction is not. The complexity lives entirely in the software: the compiler knows
the shape of every layer, and it decides that layer 1 is one `MATRIX_MULTIPLY` with 128
outputs and 784 inputs, that its weights need to be at scratchpad index 0, that the
result needs a shift of 12. It writes all of that down as twelve instructions.

The chip knows none of that. It reads sixteen bytes, sees `MATRIX_MULTIPLY`, performs
100,352 multiply-and-adds in the most obvious possible way, writes the answers, and
reads the next sixteen bytes. Then it does that eleven more times and stops.

And it does it **synchronously**: one instruction at a time, each one finished
completely before the next one starts, in the order they appear in memory. No
overlapping, no reordering, no running ahead. There are no branches and no loops — the
compiler knows every shape in advance, so it unrolls the entire network into a straight
line of instructions.

So the only thing you have to hold in your head to understand the hardware is: *what
does one instruction do, from start to finish?* Six answers, and you are done.

## Execution model

A program is a list of 128-bit (16-byte) instructions stored in main memory starting at
byte address 0. The core fetches them in order, executes each one to completion, and
stops at `HALT`.

One run of the program classifies one image. There is no batching.

## The memory spaces

The chip has one connection to the outside world — a byte-wide port to **main memory** —
and four small memories of its own, called **scratchpads**.
[Stage 5](05-registers-and-memory.md) goes into these properly; here is what you need
in order to read the instruction descriptions.

| Space | Element | How many | Addressed by | Written by | Read by |
|---|---|---|---|---|---|
| `MAIN_MEMORY` | byte | 262,144 (256 KiB) | byte address | the host, `STORE` | fetch, `LOAD` |
| `ACTIVATION_SCRATCHPAD` | 8-bit | 4,096 | element index | `LOAD`, `RECTIFIED_LINEAR` | `MATRIX_MULTIPLY`, `STORE` |
| `WEIGHT_SCRATCHPAD` | 8-bit | 131,072 | element index | `LOAD` | `MATRIX_MULTIPLY` |
| `BIAS_SCRATCHPAD` | 32-bit | 256 | element index | `LOAD` | `ADD_BIAS` |
| `ACCUMULATORS` | 32-bit | 256 | element index | `MATRIX_MULTIPLY`, `ADD_BIAS` | `RECTIFIED_LINEAR`, `STORE` |

Two rules that catch everyone once:

- **Main memory is addressed in bytes. Scratchpads are addressed in elements.** A
  `LOAD` of 128 biases uses `count=128`, not 512, even though it reads 512 bytes of
  main memory.
- **The scratchpads are managed by software.** There is no cache. If the program does
  not `LOAD` something, it is not on the chip. The sizes were chosen so the entire
  network fits at once, so no program here ever has to load a matrix in pieces.

Values larger than a byte are stored in main memory little-endian: lowest byte first.

## Numbers

- Activations and weights are signed 8-bit: `-128` to `127`.
- Biases and accumulators are signed 32-bit.
- 32-bit arithmetic **wraps** on overflow, the way a 32-bit adder does. (It never
  actually overflows in the MNIST program: an 8-bit by 8-bit product needs 16 bits, and
  summing 784 of them needs 26.)
- Right shifts are **arithmetic**: the sign bit is copied in, so a shift is a floor
  division by a power of two. `-9 >> 1` is `-5`, not `-4`.

The chip never sees a decimal number. Converting the image to whole numbers on the way
in and interpreting the ten answers on the way out are the host's job (Stage 5).

## The instructions

Below, `A` means the activation scratchpad, `W` the weight scratchpad, `B` the bias
scratchpad, and `ACC` the accumulators.

### `LOAD` — opcode `0x01`

Copy `count` consecutive elements from main memory into a scratchpad.

| Operand | Meaning |
|---|---|
| `space` | destination: `ACTIVATION_SCRATCHPAD`, `WEIGHT_SCRATCHPAD` or `BIAS_SCRATCHPAD` |
| `memory` | source **byte** address in main memory |
| `index` | destination **element** index in the scratchpad |
| `count` | how many **elements** |

Element width follows the destination: one byte for the 8-bit scratchpads, four bytes
(little-endian) for the bias scratchpad. Loading into `ACCUMULATORS` is not allowed.

### `STORE` — opcode `0x02`

Copy `count` consecutive elements from a scratchpad back into main memory. Same four
operands, with `space` naming the *source*, which must be `ACTIVATION_SCRATCHPAD` or
`ACCUMULATORS`. The MNIST program ends with a `STORE` of ten 32-bit accumulators, which
is 40 bytes.

### `MATRIX_MULTIPLY` — opcode `0x10`

Multiply an activation vector by a weight matrix. This is the instruction the chip
exists for.

```
for n in 0 .. outputs-1:
    total = 0
    for k in 0 .. inputs-1:
        total += A[input + k] * W[weights + n*inputs + k]
    ACC[accumulator + n] = (ACC[accumulator + n] if accumulate else 0) + total
```

| Operand | Meaning |
|---|---|
| `input` | activation scratchpad index of the input vector |
| `weights` | weight scratchpad index of the matrix |
| `accumulator` | accumulator index where the results go |
| `outputs` | length of the output vector (rows of the matrix) |
| `inputs` | length of the input vector (columns of the matrix) |
| `accumulate` | `0` = overwrite the accumulators, `1` = add to what is there (default `0`) |

The matrix is stored **row-major, one row per output** — exactly the layout of a
PyTorch `Linear` layer's weight tensor, shape `(outputs, inputs)`. That is on purpose:
the compiler never has to transpose anything.

`accumulate` exists so a reduction too long for the activation scratchpad can be split
into several `MATRIX_MULTIPLY` instructions over slices of the input. The MNIST program
never needs it.

### `ADD_BIAS` — opcode `0x20`

```
for i in 0 .. count-1:
    ACC[accumulator + i] += B[bias + i]        (32-bit, wrapping)
```

| Operand | Meaning |
|---|---|
| `accumulator` | accumulator index |
| `bias` | bias scratchpad index |
| `count` | how many |

### `RECTIFIED_LINEAR` — opcode `0x30`

The activation instruction. It reads 32-bit accumulators, zeroes the negatives, scales
down by a power of two, saturates to 8 bits, and writes the result where the next
layer's `MATRIX_MULTIPLY` can read it.

```
for i in 0 .. count-1:
    value = ACC[accumulator + i]
    if rectify: value = max(value, 0)
    value = value >> shift                     (arithmetic)
    A[destination + i] = clamp(value, -128, 127)
```

| Operand | Meaning |
|---|---|
| `accumulator` | accumulator index to read |
| `destination` | activation scratchpad index to write |
| `count` | how many |
| `shift` | right shift amount, `0` to `31` |
| `rectify` | `1` = zero the negatives (the normal case), `0` = shift and saturate only (default `1`) |

Three separate jobs in one instruction — rectify, rescale, narrow — because the pass
that already has every accumulator in hand is the cheapest place to do all three. The
compiler picks `shift` per layer; see [Stage 1](01-pytorch.md).

### `NO_OPERATION` — `0x00` and `HALT` — `0xFF`

`NO_OPERATION` does nothing. `HALT` stops the core and raises its done signal. Every
program ends with one, and anything after it is never fetched.

## How an instruction is written down as bits

Every instruction is exactly 128 bits — 16 bytes — stored little-endian. A fixed width
means the fetch logic in the hardware is trivial: read 16 bytes, done. No length
decoding, no alignment. The cost is that a `HALT` wastes 15 bytes, and at twelve
instructions per program nobody cares.

```
 127                                                     16 15     8 7      0
+--------------------------------------------------------+---------+--------+
|                     operands (112 bits)                 |  flags  | opcode |
+--------------------------------------------------------+---------+--------+
```

The opcode is bits `[7:0]`, so it is the first byte you see in a hex dump. Each operand
occupies a fixed bit range `[high:low]`:

| Instruction | Operand bit ranges |
|---|---|
| `LOAD`, `STORE` | `space` `[9:8]`, `memory` `[47:16]`, `index` `[71:48]`, `count` `[95:72]` |
| `MATRIX_MULTIPLY` | `accumulate` `[8]`, `input` `[31:16]`, `weights` `[55:32]`, `accumulator` `[71:56]`, `outputs` `[87:72]`, `inputs` `[103:88]` |
| `ADD_BIAS` | `accumulator` `[31:16]`, `bias` `[47:32]`, `count` `[63:48]` |
| `RECTIFIED_LINEAR` | `rectify` `[8]`, `accumulator` `[31:16]`, `destination` `[47:32]`, `count` `[63:48]`, `shift` `[71:64]` |

Memory space codes, used by the `space` field: `0` activation, `1` weight, `2` bias,
`3` accumulators.

To place a value in its field, shift it left by `low` and OR it in; to read it back,
shift right by `low` and mask off the width. That is all `encode` and `decode` in
`cub/instruction_set.py` do, and it is all the `wire` declarations near the top of
`rtl/src/cub_core.sv` do. Unused bits must be zero.

Notice that field widths are chosen to be at least as wide as the largest thing they
have to name: `index` and `count` in `LOAD` are 24 bits because the weight scratchpad
has 131,072 elements, which does not fit in 16. Sizing a field against the largest
memory it can address is a check you will make every time you design an encoding.

Work one by hand.
`LOAD space=WEIGHT_SCRATCHPAD memory=0x400 index=0 count=100352`: opcode `0x01` in byte
0; `space=1` in bit 8, so byte 1 is `0x01`; `memory=0x400` in bits 47 to 16, so bytes 2
to 5 are `00 04 00 00`; `index=0`; `count=100352 = 0x18800` in bits 95 to 72, so bytes
9 to 11 are `00 88 01`. Then check yourself:

```bash
xxd -s 80 -l 16 artifacts/mnist.cub
```

(The file has a 64-byte header, and this is the second instruction, so the offset is
64 + 16.)

## The assembly syntax

One instruction per line: the opcode, then `name=value` operands in any order. `;`
starts a comment. Numbers may be decimal or `0x` hexadecimal. Memory spaces are written
by name.

```
; MNIST, layer 1
LOAD             space=WEIGHT_SCRATCHPAD      memory=0x400    index=0  count=100352
LOAD             space=BIAS_SCRATCHPAD        memory=0x18C00  index=0  count=128
LOAD             space=ACTIVATION_SCRATCHPAD  memory=0x19340  index=0  count=784
MATRIX_MULTIPLY  input=0  weights=0  accumulator=0  outputs=128  inputs=784
ADD_BIAS         accumulator=0  bias=0  count=128
RECTIFIED_LINEAR accumulator=0  destination=1024  count=128  shift=12
HALT
```

`cub/assembler.py` turns that text into instructions, and `disassemble` turns
instructions back into that text:

```bash
python -m cub disassemble artifacts/mnist.cub
```

## Strict simulator, lenient hardware

The simulator refuses to run a bad program. The hardware quietly does something
harmless. That is deliberate: the simulator's job is to catch a broken compiler, and
the hardware's job is to be small.

| Situation | Simulator | Hardware |
|---|---|---|
| unknown opcode | error | does nothing |
| unused bits set | error | ignored |
| `LOAD` into `ACCUMULATORS`, `STORE` from a weight or bias scratchpad | error | does nothing |
| index past the end of a scratchpad | error | the address wraps |
| `shift` above 31 | error | only the low five bits are used |
| a `count` of zero | does nothing | does nothing |

A program the simulator accepts behaves identically on both. A program it rejects is
not a valid program, and what the hardware does with it is not defined.

## What this instruction set deliberately does not have

- No convolution or pooling. A fully connected network does not need them.
- No strided `LOAD` or `STORE`. Everything here is contiguous.
- No per-channel scales or rounding multipliers. A power-of-two shift is enough for
  MNIST and can be built out of wires.
- No batching. One image per run.
- No overlapping of loads and compute. Everything is in order.

Each of those is a real feature of a production accelerator, and each one is now
something you can explain the cost of.

## Things to try

```bash
printf 'MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs=128 inputs=784\nHALT\n' > /tmp/try.cubasm
python -m cub assemble /tmp/try.cubasm -o /tmp/try.bin && xxd /tmp/try.bin
```

- Change one operand and watch which byte moves.
- In `cub/instruction_set.py`, give `count` in `LOAD` a 16-bit range instead of 24 and
  run the tests. What is the first thing that breaks, and why is it a real limit rather
  than an arbitrary one?
- Encode a `HALT` with bit 127 set. What does the simulator do? What does the hardware
  do (see the table above)?

## Read next

[03-compiler.md](03-compiler.md).
