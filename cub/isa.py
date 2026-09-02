"""Cub ISA v1: instruction definitions, encoding, and decoding.

This module is the executable form of docs/isa.md section 4. Nothing here does any
computation; it only knows what an instruction *is* and how it is laid out in 128 bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INSN_BYTES = 16
INSN_BITS = 128

# --- memory spaces (docs/isa.md section 2) -------------------------------------

DRAM_BYTES = 256 * 1024
SPAD_A_SIZE = 4096      # INT8 elements
SPAD_W_SIZE = 131072    # INT8 elements
SPAD_B_SIZE = 256       # INT32 elements
ACC_SIZE = 256          # INT32 elements

MEM_SPAD_A = 0
MEM_SPAD_W = 1
MEM_SPAD_B = 2
MEM_ACC = 3

MEM_NAMES = {MEM_SPAD_A: "SPAD_A", MEM_SPAD_W: "SPAD_W", MEM_SPAD_B: "SPAD_B", MEM_ACC: "ACC"}
MEM_CODES = {v: k for k, v in MEM_NAMES.items()}
MEM_ELEM_BYTES = {MEM_SPAD_A: 1, MEM_SPAD_W: 1, MEM_SPAD_B: 4, MEM_ACC: 4}

# --- opcodes ----------------------------------------------------------------------

OP_NOP = 0x00
OP_LOAD = 0x01
OP_STORE = 0x02
OP_MATMUL = 0x10
OP_ADD_BIAS = 0x20
OP_RELU = 0x30
OP_HALT = 0xFF

OP_NAMES = {
    OP_NOP: "NOP",
    OP_LOAD: "LOAD",
    OP_STORE: "STORE",
    OP_MATMUL: "MATMUL",
    OP_ADD_BIAS: "ADD_BIAS",
    OP_RELU: "RELU",
    OP_HALT: "HALT",
}
OP_CODES = {v: k for k, v in OP_NAMES.items()}


@dataclass(frozen=True)
class Field:
    """One operand field: its name and the bit range [hi:lo] it occupies."""

    name: str
    hi: int
    lo: int
    default: int | None = None   # None means the field is required

    @property
    def width(self) -> int:
        return self.hi - self.lo + 1

    @property
    def max(self) -> int:
        return (1 << self.width) - 1


# The field tables below are a direct transcription of the per-instruction tables in
# docs/isa.md. If you change one, change the other.
FIELDS: dict[int, tuple[Field, ...]] = {
    OP_NOP: (),
    OP_LOAD: (
        Field("mem", 9, 8),
        Field("dram", 47, 16),
        Field("spad", 71, 48),
        Field("count", 95, 72),
    ),
    OP_STORE: (
        Field("mem", 9, 8),
        Field("dram", 47, 16),
        Field("spad", 71, 48),
        Field("count", 95, 72),
    ),
    OP_MATMUL: (
        Field("accumulate", 8, 8, default=0),
        Field("a", 31, 16),
        Field("w", 55, 32),
        Field("acc", 71, 56),
        Field("n", 87, 72),
        Field("k", 103, 88),
    ),
    OP_ADD_BIAS: (
        Field("acc", 31, 16),
        Field("bias", 47, 32),
        Field("count", 63, 48),
    ),
    OP_RELU: (
        Field("relu", 8, 8, default=1),
        Field("acc", 31, 16),
        Field("dst", 47, 32),
        Field("count", 63, 48),
        Field("shift", 71, 64),
    ),
    OP_HALT: (),
}


@dataclass
class Instruction:
    """A decoded instruction: an opcode plus a dict of operand values."""

    op: int
    fields: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return OP_NAMES[self.op]

    def __getitem__(self, key: str) -> int:
        return self.fields[key]

    def __post_init__(self) -> None:
        if self.op not in FIELDS:
            raise ValueError(f"unknown opcode 0x{self.op:02X}")
        spec = {f.name: f for f in FIELDS[self.op]}
        unknown = set(self.fields) - set(spec)
        if unknown:
            raise ValueError(f"{self.name}: unknown field(s) {sorted(unknown)}")
        for f in spec.values():
            if f.name not in self.fields:
                if f.default is None:
                    raise ValueError(f"{self.name}: missing field '{f.name}'")
                self.fields[f.name] = f.default
            v = self.fields[f.name]
            if not 0 <= v <= f.max:
                raise ValueError(
                    f"{self.name}: field '{f.name}'={v} does not fit in {f.width} bits"
                )


def make(name: str, **fields: int) -> Instruction:
    """Convenience constructor: make("MATMUL", a=0, w=0, acc=0, n=128, k=784)."""
    return Instruction(OP_CODES[name], dict(fields))


def encode(insn: Instruction) -> bytes:
    """Pack an Instruction into its 16-byte little-endian form (docs/isa.md section 4).

    The opcode goes in bits [7:0]. Each operand goes in the bit range its Field says.
    The result is the 128-bit integer converted to 16 little-endian bytes.
    """
    word = insn.op
    # --- SOLUTION(stage=3): OR each field's value into the word at its bit position ---
    for f in FIELDS[insn.op]:
        word |= insn.fields[f.name] << f.lo
    # --- END SOLUTION ---
    return word.to_bytes(INSN_BYTES, "little")


def decode(raw: bytes, strict: bool = True) -> Instruction:
    """Unpack 16 bytes into an Instruction. The inverse of encode().

    With strict=True (the simulator's setting), any set bit that no field claims is an
    error, so a mis-encoded instruction fails loudly instead of running wrong.
    """
    if len(raw) != INSN_BYTES:
        raise ValueError(f"an instruction is {INSN_BYTES} bytes, got {len(raw)}")
    word = int.from_bytes(raw, "little")
    op = word & 0xFF
    if op not in FIELDS:
        raise ValueError(f"unknown opcode 0x{op:02X}")
    fields = {}
    claimed = 0xFF
    for f in FIELDS[op]:
        fields[f.name] = (word >> f.lo) & f.max
        claimed |= f.max << f.lo
    if strict and word & ~claimed:
        raise ValueError(f"{OP_NAMES[op]}: unused bits are set: 0x{word & ~claimed:032X}")
    return Instruction(op, fields)
