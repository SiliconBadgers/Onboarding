"""Assembler and disassembler for the Cub text format (docs/isa.md section 5).

    LOAD     mem=SPAD_W dram=0x1000 spad=0 count=100352
    MATMUL   a=0 w=0 acc=0 n=128 k=784
    HALT

One instruction per line, ';' starts a comment, operands are key=value in any order.
"""

from __future__ import annotations

from .isa import FIELDS, MEM_CODES, MEM_NAMES, OP_CODES, Instruction, encode, decode, INSN_BYTES


class AsmError(ValueError):
    def __init__(self, lineno: int, msg: str):
        super().__init__(f"line {lineno}: {msg}")
        self.lineno = lineno


def _parse_value(key: str, text: str) -> int:
    """'0x1000' -> 4096, '784' -> 784, 'SPAD_W' -> 1 (only for mem=)."""
    if key == "mem":
        if text.upper() in MEM_CODES:
            return MEM_CODES[text.upper()]
        raise ValueError(f"unknown memory space '{text}'")
    return int(text, 0)


def parse_line(line: str, lineno: int = 0) -> Instruction | None:
    """Turn one line of assembly into an Instruction, or None for a blank/comment line."""
    text = line.split(";", 1)[0].strip()
    if not text:
        return None
    parts = text.split()
    name = parts[0].upper()
    if name not in OP_CODES:
        raise AsmError(lineno, f"unknown instruction '{parts[0]}'")
    fields: dict[str, int] = {}
    # TODO(onboard, stage 3): parse each 'key=value' operand into the fields dict
    raise NotImplementedError("stage 3 blank, see asm.py:40")
    try:
        return Instruction(OP_CODES[name], fields)
    except ValueError as e:
        raise AsmError(lineno, str(e)) from e


def assemble(text: str) -> list[Instruction]:
    """Assemble a whole program. Returns the instruction list (not yet encoded)."""
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        insn = parse_line(line, lineno)
        if insn is not None:
            out.append(insn)
    return out


def assemble_bytes(text: str) -> bytes:
    return b"".join(encode(i) for i in assemble(text))


def format_insn(insn: Instruction) -> str:
    """Instruction -> one line of assembly (the inverse of parse_line)."""
    parts = [f"{insn.name:<8}"]
    for f in FIELDS[insn.op]:
        v = insn.fields[f.name]
        if f.default is not None and v == f.default:
            continue
        if f.name == "mem":
            parts.append(f"mem={MEM_NAMES[v]}")
        elif f.name == "dram":
            parts.append(f"dram=0x{v:X}")
        else:
            parts.append(f"{f.name}={v}")
    return " ".join(parts).rstrip()


def disassemble(raw: bytes) -> str:
    """16*n bytes -> n lines of assembly."""
    if len(raw) % INSN_BYTES:
        raise ValueError("program length is not a multiple of 16 bytes")
    lines = []
    for off in range(0, len(raw), INSN_BYTES):
        lines.append(format_insn(decode(raw[off : off + INSN_BYTES])))
    return "\n".join(lines) + "\n"
