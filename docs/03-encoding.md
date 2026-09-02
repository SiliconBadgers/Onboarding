# Stage 3 — Encoding and the assembler

**Goal:** turn an instruction into 16 bytes and back, and turn a line of text into an
instruction.

**Files:** `cub/isa.py`, `cub/asm.py`. **Test:** `pytest tests/test_03_encoding.py`.
**Read first:** `docs/isa.md` section 4.

## Why 128 fixed bits

A fixed-width instruction means the hardware's fetch stage is trivial: read 16 bytes,
done. No length decoding, no alignment logic. The cost is that a `HALT` wastes 15
bytes. At twelve instructions per program, nobody cares. Badger made the same call.

## The layout

```
 127                                                     16 15     8 7      0
+--------------------------------------------------------+---------+--------+
|                     operands (112 bits)                 |  flags  | opcode |
+--------------------------------------------------------+---------+--------+
```

Treat the whole instruction as one 128-bit integer. The opcode occupies bits 7 to 0.
Every operand has a bit range `[hi:lo]`, listed per instruction in `docs/isa.md` and
transcribed into the `FIELDS` table in `cub/isa.py`. To place a value in its field,
shift it left by `lo` and OR it in. To read it back, shift right by `lo` and mask
with `(1 << width) - 1`.

The 128-bit integer is then written as 16 **little-endian** bytes, so the opcode is
the first byte you see in a hex dump. Look at `artifacts/mnist.cub` with
`xxd -s 64 -l 64 artifacts/mnist.cub`: byte 64 of the file is the opcode of the first
instruction (`01`, a `LOAD`), because the file has a 64-byte header.

Work one example by hand before writing code. `LOAD mem=SPAD_W dram=0x400 spad=0
count=100352`: opcode `0x01` in byte 0; `mem=1` in bit 8, so byte 1 is `0x01`;
`dram=0x400` in bits 47 to 16, so bytes 2 to 5 are `00 04 00 00`; `spad=0`;
`count=100352 = 0x18800` in bits 95 to 72, so bytes 9 to 11 are `00 88 01`. The test
file has this exact case.

## Your task

1. `encode` in `cub/isa.py`: the `word` already holds the opcode. OR each field into
   it at `f.lo`. The `Instruction` constructor has already validated that every value
   fits, so you do not need to check.
2. `parse_line` in `cub/asm.py`: the opcode has been recognized and `parts[1:]` holds
   strings like `dram=0x400`. Split each on `=`, lower-case the key, and convert the
   value with `_parse_value` (which knows that `mem=SPAD_W` means `1`). Raise
   `AsmError(lineno, ...)` for a part with no `=`.

`decode` and `format_insn` are written for you. Read `decode` before you write
`encode`; it is the mirror image, and it also shows the strict check that rejects any
set bit no field claims. The simulator uses strict decoding, so a wrong `encode` fails
at the first fetch instead of running garbage.

## Check

```bash
pytest tests/test_03_encoding.py -v
```

The golden table in the test was checked by hand against the spec. If one fails,
print `encode(...).hex()` and compare byte by byte; the first differing byte tells you
which field is misplaced.

Then try the tools:

```bash
python -m cub disasm artifacts/mnist.cub
printf 'MATMUL a=0 w=0 acc=0 n=128 k=784\nHALT\n' > /tmp/t.cubasm
python -m cub asm /tmp/t.cubasm -o /tmp/t.bin && xxd /tmp/t.bin
```

## Questions to be able to answer

- Why are `spad` and `count` 24 bits in `LOAD` but `a` is 16 bits in `MATMUL`?
- What happens in the simulator if you encode `HALT` with bit 127 set? In hardware?

## Read next

`docs/04-simulator.md`.
