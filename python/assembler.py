"""Assembler and disassembler for the accelerator's text format.

    LOAD            space=WEIGHT_SCRATCHPAD memory=0x400 index=0 count=100352
    MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs=128 inputs=784
    HALT

One instruction per line, ';' starts a comment, operands are name=value in any order.
The syntax is described in docs/02-instruction-set.md.
"""

from __future__ import annotations

from .instruction_set import (
    FIELDS,
    INSTRUCTION_BYTES,
    OPCODE_VALUES,
    SPACE_CODES,
    SPACE_NAMES,
    Instruction,
    decode,
    encode,
)


class AssemblyError(ValueError):
    def __init__(self, line_number: int, message: str):
        super().__init__(f"line {line_number}: {message}")
        self.line_number = line_number


def _parse_value(name: str, text: str) -> int:
    """'0x1000' -> 4096, '784' -> 784, 'WEIGHT_SCRATCHPAD' -> 1 (only for space=)."""
    if name == "space":
        if text.upper() in SPACE_CODES:
            return SPACE_CODES[text.upper()]
        raise ValueError(f"unknown memory space '{text}'")
    return int(text, 0)


def parse_line(line: str, line_number: int = 0) -> Instruction | None:
    """One line of assembly -> an Instruction, or None for a blank or comment line."""
    text = line.split(";", 1)[0].strip()
    if not text:
        return None
    parts = text.split()
    name = parts[0].upper()
    if name not in OPCODE_VALUES:
        raise AssemblyError(line_number, f"unknown instruction '{parts[0]}'")
    operands: dict[str, int] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise AssemblyError(line_number, f"expected name=value, got '{part}'")
        operand_name, value = part.split("=", 1)
        operand_name = operand_name.lower()
        try:
            operands[operand_name] = _parse_value(operand_name, value)
        except ValueError as error:
            raise AssemblyError(line_number, str(error)) from error
    try:
        return Instruction(OPCODE_VALUES[name], operands)
    except ValueError as error:
        raise AssemblyError(line_number, str(error)) from error


def assemble(text: str) -> list[Instruction]:
    """Assemble a whole program. Returns the instruction list (not yet encoded)."""
    out = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        instruction = parse_line(line, line_number)
        if instruction is not None:
            out.append(instruction)
    return out


def assemble_bytes(text: str) -> bytes:
    return b"".join(encode(i) for i in assemble(text))


def format_instruction(instruction: Instruction) -> str:
    """An Instruction -> one line of assembly (the inverse of parse_line)."""
    parts = [f"{instruction.name:<16}"]
    for definition in FIELDS[instruction.opcode]:
        value = instruction.operands[definition.name]
        if definition.default is not None and value == definition.default:
            continue
        if definition.name == "space":
            parts.append(f"space={SPACE_NAMES[value]}")
        elif definition.name == "memory":
            parts.append(f"memory=0x{value:X}")
        else:
            parts.append(f"{definition.name}={value}")
    return " ".join(parts).rstrip()


def disassemble(raw: bytes) -> str:
    """16*n bytes -> n lines of assembly."""
    if len(raw) % INSTRUCTION_BYTES:
        raise ValueError("program length is not a multiple of 16 bytes")
    lines = []
    for offset in range(0, len(raw), INSTRUCTION_BYTES):
        lines.append(format_instruction(decode(raw[offset : offset + INSTRUCTION_BYTES])))
    return "\n".join(lines) + "\n"
