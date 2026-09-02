# Cub ISA v1 — the teaching instruction set

Cub is the small instruction set you will build a compiler, a simulator, and a
hardware core for during onboarding. It is a deliberately simplified relative of the
real Badger ISA in the main
[AI-Inference-Chip](https://github.com/SiliconBadgers/AI-Inference-Chip) repository.
Section 9 maps every Cub instruction onto its Badger equivalent so that when you finish
onboarding, the real ISA reads as "Cub, plus the things we left out".

This document is **normative**. The assembler, the simulator, the compiler, and the
RTL are all written against it. When two of them disagree, this file decides who is
wrong.

---

## 1. Execution model

A program is a list of fixed-width 128-bit instructions stored in host memory
starting at byte address 0. The core fetches them in order, one at a time, executes
each one to completion, and stops at `HALT`. There is no branching and no loop
instruction: the compiler knows every tensor shape ahead of time, so it unrolls the
whole network into a straight line of instructions.

Batch size is 1. One program run classifies one image.

## 2. Memory spaces

| Space | Element type | Elements | Addressed by | Written by | Read by |
|---|---|---|---|---|---|
| `DRAM`   | byte  | 262 144 (256 KiB) | byte address    | host, `STORE` | fetch, `LOAD` |
| `SPAD_A` | INT8  | 4 096   | element index | `LOAD`, `RELU` | `MATMUL`, `STORE` |
| `SPAD_W` | INT8  | 131 072 | element index | `LOAD` | `MATMUL` |
| `SPAD_B` | INT32 | 256     | element index | `LOAD` | `ADD_BIAS` |
| `ACC`    | INT32 | 256     | element index | `MATMUL`, `ADD_BIAS` | `RELU`, `STORE` |

`DRAM` is the memory the host can see. In the simulator it is a Python `bytearray`.
On the FPGA it is a block RAM preloaded with the program image. Everything else is
private to the core.

The scratchpads are **software managed**. There is no cache. If the program does not
`LOAD` something, it is not there. The sizes were chosen so that the whole MNIST MLP
(a 784x128 weight matrix plus a 128x10 one) fits at once, so a V1 program never has
to tile. Tiling is the stretch exercise in Stage 6.

Multi-byte values in `DRAM` are little-endian.

## 3. Numbers

- Activations (`SPAD_A`) are signed INT8.
- Weights (`SPAD_W`) are signed INT8.
- Biases (`SPAD_B`) and accumulators (`ACC`) are signed INT32.
- INT32 arithmetic wraps on overflow, the way a hardware adder does. (It never
  overflows in the MNIST program: an INT8 x INT8 product needs 16 bits, and summing
  784 of them needs 26. See Stage 2.)
- Right shifts are **arithmetic** (the sign bit is copied in). A shift is a floor
  division by a power of two, not a rounding one.

The core never sees a floating-point number. Converting the image to INT8 and turning
the final INT32 logits back into a digit are the host's job (Stage 7).

## 4. Instruction encoding

Every instruction is exactly 128 bits (16 bytes), stored little-endian. Bit ranges
below are `[high:low]` inclusive, bit 0 being the least significant bit of the
16-byte little-endian integer.

```
 127                                                     16 15     8 7      0
+--------------------------------------------------------+---------+--------+
|                     operands (112 bits)                 |  flags  | opcode |
+--------------------------------------------------------+---------+--------+
```

Unused bits must be zero. The simulator rejects an instruction with unused bits set;
the hardware ignores them.

### Opcode map

| Opcode | Name | What it does |
|---|---|---|
| `0x00` | `NOP`      | nothing |
| `0x01` | `LOAD`     | copy DRAM -> scratchpad |
| `0x02` | `STORE`    | copy scratchpad -> DRAM |
| `0x10` | `MATMUL`   | vector x matrix into `ACC` |
| `0x20` | `ADD_BIAS` | `ACC += SPAD_B` |
| `0x30` | `RELU`     | `ACC` -> ReLU -> shift -> saturate -> `SPAD_A` |
| `0xFF` | `HALT`     | stop |

The opcode numbers are the same ones Badger uses for the corresponding instruction
(section 9), which is why they are not consecutive.

### Memory-space codes

Used by the `mem` flag field of `LOAD` and `STORE`.

| Code | Space |
|---|---|
| `0` | `SPAD_A` |
| `1` | `SPAD_W` |
| `2` | `SPAD_B` |
| `3` | `ACC` |

### `LOAD` — `0x01`

Copies `count` contiguous elements from `DRAM` into a scratchpad.

| Field | Bits | Meaning |
|---|---|---|
| `mem`       | `[9:8]`   | destination: `0`, `1`, or `2` (`ACC` is not loadable) |
| `dram_addr` | `[47:16]` | source byte address in `DRAM` |
| `spad_addr` | `[71:48]` | destination element index |
| `count`     | `[95:72]` | number of elements |

Element width follows the destination: 1 byte for `SPAD_A` and `SPAD_W`, 4 bytes
(little-endian) for `SPAD_B`. `spad_addr` and `count` are 24 bits wide because
`SPAD_W` has 131 072 elements, which does not fit in 16 bits. A field has to be at
least as wide as the largest memory it can address; that is a recurring check when
you size an encoding. So a `LOAD` into `SPAD_B` with `count=128` reads 512
bytes of `DRAM`.

### `STORE` — `0x02`

Copies `count` contiguous elements from a scratchpad into `DRAM`.

| Field | Bits | Meaning |
|---|---|---|
| `mem`       | `[9:8]`   | source: `0` (`SPAD_A`) or `3` (`ACC`) |
| `dram_addr` | `[47:16]` | destination byte address in `DRAM` |
| `spad_addr` | `[71:48]` | source element index |
| `count`     | `[95:72]` | number of elements |

Element width is 1 byte for `SPAD_A` and 4 bytes for `ACC`. The MNIST program ends by
storing 10 INT32 logits from `ACC`, which is 40 bytes of `DRAM`.

### `MATMUL` — `0x10`

Multiplies an activation vector by a weight matrix.

```
for n in 0 .. N-1:
    s = 0
    for k in 0 .. K-1:
        s += SPAD_A[a_addr + k] * SPAD_W[w_addr + n*K + k]
    ACC[acc_addr + n] = (ACC[acc_addr + n] if accumulate else 0) + s
```

| Field | Bits | Meaning |
|---|---|---|
| `accumulate` | `[8]`     | `0` = overwrite `ACC`, `1` = add to what is there |
| `a_addr`     | `[31:16]` | `SPAD_A` index of the input vector |
| `w_addr`     | `[55:32]` | `SPAD_W` index of the weight matrix |
| `acc_addr`   | `[71:56]` | `ACC` index of the output vector |
| `N`          | `[87:72]` | output length (rows of the weight matrix) |
| `K`          | `[103:88]` | input length (columns of the weight matrix) |

The weight matrix is stored **row-major, one row per output**, exactly the layout of
a PyTorch `nn.Linear.weight` tensor of shape `(out_features, in_features)`. The
compiler therefore never transposes anything.

The `accumulate` flag exists so a reduction too long for `SPAD_A` can be split into
several `MATMUL`s over slices of `K`. The V1 program never needs it.

### `ADD_BIAS` — `0x20`

```
for i in 0 .. count-1:
    ACC[acc_addr + i] += SPAD_B[bias_addr + i]        (INT32, wrapping)
```

| Field | Bits | Meaning |
|---|---|---|
| `acc_addr`  | `[31:16]` | `ACC` index |
| `bias_addr` | `[47:32]` | `SPAD_B` index |
| `count`     | `[63:48]` | number of elements |

### `RELU` — `0x30`

The activation instruction. It reads INT32 accumulators, applies ReLU, scales down by
a power of two, saturates to INT8, and writes the result into `SPAD_A` where the next
layer's `MATMUL` can read it.

```
for i in 0 .. count-1:
    v = ACC[acc_addr + i]
    if relu: v = max(v, 0)
    v = v >> shift                       (arithmetic shift)
    SPAD_A[dst_addr + i] = clamp(v, -128, 127)
```

| Field | Bits | Meaning |
|---|---|---|
| `relu`     | `[8]`     | `1` = zero the negatives (the normal case), `0` = shift and saturate only |
| `acc_addr` | `[31:16]` | `ACC` index to read |
| `dst_addr` | `[47:32]` | `SPAD_A` index to write |
| `count`    | `[63:48]` | number of elements |
| `shift`    | `[71:64]` | right shift amount, `0`..`31` |

Why the shift lives here: after a layer, `ACC` holds numbers about 2^20 in size, and
the next layer needs INT8 inputs. Something has to scale them down, and the pass that
already has every accumulator in hand is the cheapest place to do it. The compiler
picks `shift` per layer (Stage 2). Badger does the same thing with a full
fixed-point multiplier instead of a shift, and calls the instruction `REQUANT`.

### `HALT` — `0xFF`

Stops the core and raises `done`. Every program ends with one. Anything after it is
never fetched.

### Illegal programs: simulator versus hardware

The simulator is strict and the hardware is lenient. That is deliberate: the
simulator's job is to catch a bad compiler, and the hardware's job is to be small.

| Situation | Simulator | Hardware |
|---|---|---|
| unknown opcode | error | treated as `NOP` |
| unused bits set | error | ignored |
| `LOAD` into `ACC`, `STORE` from `SPAD_W` or `SPAD_B` | error | treated as `NOP` |
| scratchpad index past the end | error | index wraps (address bits truncated) |
| `RELU` with `shift` above 31 | error | only the low 5 bits are used |
| `count`, `N`, or `K` of zero | does nothing | does nothing |

A program the simulator accepts behaves identically on both. A program it rejects is
not a Cub program, and what the hardware does with it is unspecified.

## 5. Assembly syntax

One instruction per line, opcode first, then `key=value` operands in any order.
Comments start with `;`. Numbers may be decimal or `0x` hex. Memory spaces are
written by name.

```
; MNIST, layer 1
LOAD     mem=SPAD_W  dram=0x1000   spad=0  count=100352
LOAD     mem=SPAD_B  dram=0x19800  spad=0  count=128
LOAD     mem=SPAD_A  dram=0x1A000  spad=0  count=784
MATMUL   a=0  w=0  acc=0  n=128  k=784
ADD_BIAS acc=0  bias=0  count=128
RELU     acc=0  dst=1024  count=128  shift=9
HALT
```

Field name aliases accepted by the assembler:

| Instruction | Fields (`key=`) |
|---|---|
| `LOAD`, `STORE` | `mem`, `dram`, `spad`, `count` |
| `MATMUL` | `a`, `w`, `acc`, `n`, `k`, optional `accumulate` (default `0`) |
| `ADD_BIAS` | `acc`, `bias`, `count` |
| `RELU` | `acc`, `dst`, `count`, `shift`, optional `relu` (default `1`) |

## 6. Program image

The compiler emits a single **DRAM image**: the bytes that must be in `DRAM` before
the core starts. Instructions are at offset 0. The compiler places everything else
after them at 64-byte-aligned offsets and records where:

| Region | Contents |
|---|---|
| `insns`  | the instructions, 16 bytes each, starting at 0 |
| `w1`     | layer-1 weights, INT8, row-major `(128, 784)` |
| `b1`     | layer-1 biases, INT32 x 128 |
| `w2`     | layer-2 weights, INT8, row-major `(10, 128)` |
| `b2`     | layer-2 biases, INT32 x 10 |
| `input`  | the INT8 image, 784 bytes, written by the host before each run |
| `output` | the INT32 logits, 40 bytes, written by the program's final `STORE` |

The `.cub` file format (Stage 6) is a 64-byte header followed by the image:

| Offset | Type | Field |
|---|---|---|
| 0  | `char[4]` | magic `"CUB1"` |
| 4  | `u32` | ISA version (`1`) |
| 8  | `u32` | number of instructions |
| 12 | `u32` | image length in bytes |
| 16 | `u32` | `input` offset |
| 20 | `u32` | `input` length |
| 24 | `u32` | `output` offset |
| 28 | `u32` | `output` length |
| 32 | `f32` | output scale: `logit_real = logit_int32 / output_scale` |
| 36 | — | zero padding to 64 |

The host runtime (Stage 7) reads the header, copies the image into `DRAM`, writes the
quantized image at `input`, starts the core, waits for `done`, and reads `output`.

## 7. The hardware contract

The RTL core (Stage 8) has one memory port to `DRAM` plus `start` and `done`:

- `DRAM` is byte-wide with a one-cycle synchronous read.
- The core fetches from byte address 0 when `start` is pulsed.
- `done` goes high after `HALT` and stays high until the next `start`.
- The core executes one instruction at a time and one multiply-accumulate per cycle.

Speed is not a V1 goal. Matching the simulator bit for bit is.

Cycle costs of the V1 core, for when you want to reason about where the time goes:

| Operation | Cycles |
|---|---|
| fetch + decode | 17 per instruction |
| `LOAD` / `STORE` | one per byte moved, plus 1 |
| `MATMUL` | `N * K`, plus 2 |
| `ADD_BIAS` / `RELU` | `count` |

The MNIST program is about 205 000 cycles, roughly half of it the `LOAD` of the
layer-1 weights and half the layer-1 `MATMUL`. At 125 MHz that is 1.6 ms per image.

## 8. What the ISA does not have, on purpose

- No convolution, pooling, or im2col. The MLP does not need them.
- No strided `LOAD`/`STORE`. Everything in V1 is contiguous.
- No per-channel scales or rounding multipliers. A power-of-two shift is enough for
  MNIST and can be built with wires.
- No batch dimension. One image per run.
- No decoupling of loads and compute. Everything is in order.

Each of these is a real feature of Badger, and each one is something you can now
explain the cost of.

## 9. Cub to Badger

| Cub | Badger | What changes |
|---|---|---|
| `LOAD` `0x01`     | `LOAD` `0x01`    | Badger adds `run`/`stride` for strided DMA |
| `STORE` `0x02`    | `STORE` `0x02`   | same |
| `MATMUL` `0x10`   | `MATMUL` `0x10`  | Badger has an `M` field (many rows at once) and feeds a 16x16 MAC array |
| `ADD_BIAS` `0x20` | *(fused into `REQUANT`)* | Badger reserves `0x20` for a standalone `ADD` used by residual connections |
| `RELU` `0x30`     | `REQUANT` `0x30` | Badger replaces the shift with a Q0.31 multiplier plus rounding shift, and reads bias from a parameter table |
| `HALT` `0xFF`     | `HALT` `0xFF`    | same |
| —                 | `SETCFG`, `IM2COL`, `MAXPOOL`, `AVGPOOL`, `FENCE` | convolution, pooling, and performance features |

The encoding layout (opcode in `[7:0]`, flags in `[15:8]`, operands above) and the
memory model (software-managed scratchpads, element-indexed) are identical. If you can
read a Cub program, you can read a Badger one.
