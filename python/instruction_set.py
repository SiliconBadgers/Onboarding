"""The instruction set: what an instruction is, and how it is laid out in 128 bits.

This module is the executable form of docs/02-instruction-set.md. Nothing here does
any arithmetic; it only knows what an instruction *is* and how to turn one into
sixteen bytes and back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INSTRUCTION_BYTES = 16
INSTRUCTION_BITS = 128

# --- memory spaces (docs/02-instruction-set.md) --------------------------------

MAIN_MEMORY_BYTES = 256 * 1024
ACTIVATION_SCRATCHPAD_SIZE = 4096      # 8-bit elements
WEIGHT_SCRATCHPAD_SIZE = 131072        # 8-bit elements
BIAS_SCRATCHPAD_SIZE = 256             # 32-bit elements
ACCUMULATOR_COUNT = 256                # 32-bit elements

SPACE_ACTIVATION_SCRATCHPAD = 0
SPACE_WEIGHT_SCRATCHPAD = 1
SPACE_BIAS_SCRATCHPAD = 2
SPACE_ACCUMULATORS = 3

SPACE_NAMES = {
    SPACE_ACTIVATION_SCRATCHPAD: "ACTIVATION_SCRATCHPAD",
    SPACE_WEIGHT_SCRATCHPAD: "WEIGHT_SCRATCHPAD",
    SPACE_BIAS_SCRATCHPAD: "BIAS_SCRATCHPAD",
    SPACE_ACCUMULATORS: "ACCUMULATORS",
}
SPACE_CODES = {name: code for code, name in SPACE_NAMES.items()}

# How many bytes one element of each space occupies in main memory.
SPACE_ELEMENT_BYTES = {
    SPACE_ACTIVATION_SCRATCHPAD: 1,
    SPACE_WEIGHT_SCRATCHPAD: 1,
    SPACE_BIAS_SCRATCHPAD: 4,
    SPACE_ACCUMULATORS: 4,
}

# --- opcodes -------------------------------------------------------------------

OPCODE_NO_OPERATION = 0x00
OPCODE_LOAD = 0x01
OPCODE_STORE = 0x02
OPCODE_MATRIX_MULTIPLY = 0x10
OPCODE_ADD_BIAS = 0x20
OPCODE_RECTIFIED_LINEAR = 0x30
OPCODE_HALT = 0xFF

OPCODE_NAMES = {
    OPCODE_NO_OPERATION: "NO_OPERATION",
    OPCODE_LOAD: "LOAD",
    OPCODE_STORE: "STORE",
    OPCODE_MATRIX_MULTIPLY: "MATRIX_MULTIPLY",
    OPCODE_ADD_BIAS: "ADD_BIAS",
    OPCODE_RECTIFIED_LINEAR: "RECTIFIED_LINEAR",
    OPCODE_HALT: "HALT",
}
OPCODE_VALUES = {name: code for code, name in OPCODE_NAMES.items()}


@dataclass(frozen=True)
class Field:
    """One operand field: its name and the bit range [high_bit:low_bit] it occupies."""

    name: str
    high_bit: int
    low_bit: int
    default: int | None = None   # None means the operand must be written out

    @property
    def width(self) -> int:
        return self.high_bit - self.low_bit + 1

    @property
    def largest(self) -> int:
        """The largest value that fits in this field."""
        return (1 << self.width) - 1


# The tables below are a direct transcription of the per-instruction tables in
# docs/02-instruction-set.md. If you change one, change the other.
FIELDS: dict[int, tuple[Field, ...]] = {
    OPCODE_NO_OPERATION: (),
    OPCODE_LOAD: (
        Field("space", 9, 8),
        Field("memory", 47, 16),
        Field("index", 71, 48),
        Field("count", 95, 72),
    ),
    OPCODE_STORE: (
        Field("space", 9, 8),
        Field("memory", 47, 16),
        Field("index", 71, 48),
        Field("count", 95, 72),
    ),
    OPCODE_MATRIX_MULTIPLY: (
        Field("accumulate", 8, 8, default=0),
        Field("input", 31, 16),
        Field("weights", 55, 32),
        Field("accumulator", 71, 56),
        Field("outputs", 87, 72),
        Field("inputs", 103, 88),
    ),
    OPCODE_ADD_BIAS: (
        Field("accumulator", 31, 16),
        Field("bias", 47, 32),
        Field("count", 63, 48),
    ),
    OPCODE_RECTIFIED_LINEAR: (
        Field("rectify", 8, 8, default=1),
        Field("accumulator", 31, 16),
        Field("destination", 47, 32),
        Field("count", 63, 48),
        Field("shift", 71, 64),
    ),
    OPCODE_HALT: (),
}


@dataclass
class Instruction:
    """A decoded instruction: an opcode plus a dictionary of operand values."""

    opcode: int
    operands: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return OPCODE_NAMES[self.opcode]

    def __getitem__(self, key: str) -> int:
        return self.operands[key]

    def __post_init__(self) -> None:
        if self.opcode not in FIELDS:
            raise ValueError(f"unknown opcode 0x{self.opcode:02X}")
        spec = {f.name: f for f in FIELDS[self.opcode]}
        unknown = set(self.operands) - set(spec)
        if unknown:
            raise ValueError(f"{self.name}: unknown operand(s) {sorted(unknown)}")
        for definition in spec.values():
            if definition.name not in self.operands:
                if definition.default is None:
                    raise ValueError(f"{self.name}: missing operand '{definition.name}'")
                self.operands[definition.name] = definition.default
            value = self.operands[definition.name]
            if not 0 <= value <= definition.largest:
                raise ValueError(
                    f"{self.name}: operand '{definition.name}'={value} "
                    f"does not fit in {definition.width} bits"
                )


def make(name: str, **operands: int) -> Instruction:
    """Convenience constructor.

    make("MATRIX_MULTIPLY", input=0, weights=0, accumulator=0, outputs=128, inputs=784)
    """
    return Instruction(OPCODE_VALUES[name], dict(operands))


def encode(instruction: Instruction) -> bytes:
    """Pack an Instruction into its 16-byte little-endian form.

    The opcode goes in bits [7:0]. Each operand goes in the bit range its Field says.
    The result is that 128-bit integer written out as 16 little-endian bytes.
    """
    word = instruction.opcode
    for definition in FIELDS[instruction.opcode]:
        word |= instruction.operands[definition.name] << definition.low_bit
    return word.to_bytes(INSTRUCTION_BYTES, "little")


def decode(raw: bytes, strict: bool = True) -> Instruction:
    """Unpack 16 bytes into an Instruction. The inverse of encode().

    With strict=True (what the simulator uses), any set bit that no operand claims is
    an error, so a mis-encoded instruction fails loudly instead of running wrong.
    """
    if len(raw) != INSTRUCTION_BYTES:
        raise ValueError(f"an instruction is {INSTRUCTION_BYTES} bytes, got {len(raw)}")
    word = int.from_bytes(raw, "little")
    opcode = word & 0xFF
    if opcode not in FIELDS:
        raise ValueError(f"unknown opcode 0x{opcode:02X}")
    operands = {}
    claimed = 0xFF
    for definition in FIELDS[opcode]:
        operands[definition.name] = (word >> definition.low_bit) & definition.largest
        claimed |= definition.largest << definition.low_bit
    if strict and word & ~claimed:
        raise ValueError(
            f"{OPCODE_NAMES[opcode]}: unused bits are set: 0x{word & ~claimed:032X}"
        )
    return Instruction(opcode, operands)
