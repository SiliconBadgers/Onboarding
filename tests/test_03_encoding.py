"""Stage 3: the 128-bit encoding and the assembler."""

import pytest

from cub.stages import skip_unless_started

pytestmark = pytest.mark.stage(3)


@pytest.fixture(autouse=True)
def _started():
    skip_unless_started(3)


# Hand-checked against docs/isa.md section 4. Bytes are little-endian, so the opcode
# is the first byte you see.
GOLDEN = [
    ("HALT", "ff000000000000000000000000000000"),
    ("NOP", "00000000000000000000000000000000"),
    ("LOAD mem=SPAD_W dram=0x400 spad=0 count=100352", "01010004000000000000880100000000"),
    ("LOAD mem=SPAD_B dram=0x18C00 spad=128 count=10", "0102008c01008000000a000000000000"),
    ("STORE mem=ACC dram=0x19680 spad=0 count=10", "0203809601000000000a000000000000"),
    ("MATMUL a=0 w=0 acc=0 n=128 k=784", "10000000000000000080001003000000"),
    ("MATMUL a=1024 w=100352 acc=0 n=10 k=128 accumulate=1", "1001000400880100000a008000000000"),
    ("ADD_BIAS acc=0 bias=128 count=10", "2000000080000a000000000000000000"),
    ("RELU acc=0 dst=1024 count=128 shift=12", "30010000000480000c00000000000000"),
    ("RELU acc=0 dst=1024 count=128 shift=12 relu=0", "30000000000480000c00000000000000"),
]


@pytest.mark.parametrize("text,hexstr", GOLDEN)
def test_encode_golden(text, hexstr):
    from cub.asm import parse_line
    from cub.isa import encode

    assert encode(parse_line(text)).hex() == hexstr


@pytest.mark.parametrize("text,hexstr", GOLDEN)
def test_decode_roundtrip(text, hexstr):
    from cub.asm import format_insn, parse_line
    from cub.isa import decode, encode

    insn = parse_line(text)
    again = decode(encode(insn))
    assert again == insn
    assert parse_line(format_insn(insn)) == insn


def test_parse_line_operands():
    from cub.asm import parse_line

    i = parse_line("  load  MEM=spad_a  DRAM=0x10 spad=3 count=784 ; a comment")
    assert i.name == "LOAD"
    assert i.fields == {"mem": 0, "dram": 16, "spad": 3, "count": 784}
    assert parse_line("; only a comment") is None
    assert parse_line("") is None


def test_parse_line_errors():
    from cub.asm import AsmError, parse_line

    with pytest.raises(AsmError):
        parse_line("JUMP 0", 1)
    with pytest.raises(AsmError):
        parse_line("LOAD mem=SPAD_A dram=0", 1)                 # missing fields
    with pytest.raises(AsmError):
        parse_line("LOAD mem=SPAD_A dram=0 spad=0 count=784 foo=1", 1)
    with pytest.raises(AsmError):
        parse_line("MATMUL a=0 w=0 acc=0 n=70000 k=1", 1)      # n does not fit 16 bits


def test_strict_decode_rejects_junk():
    from cub.isa import decode

    raw = bytearray(16)
    raw[0] = 0xFF
    raw[15] = 0x80          # a bit no HALT field claims
    with pytest.raises(ValueError):
        decode(bytes(raw))
    assert decode(bytes(raw), strict=False).name == "HALT"


def test_disassemble_compiled_program(compiled_program):
    from cub.asm import disassemble
    from cub.isa import encode

    raw = b"".join(encode(i) for i in compiled_program.insns)
    names = [line.split()[0] for line in disassemble(raw).splitlines()]
    assert names == ["LOAD", "LOAD", "LOAD", "MATMUL", "ADD_BIAS", "RELU",
                     "LOAD", "LOAD", "MATMUL", "ADD_BIAS", "STORE", "HALT"]
