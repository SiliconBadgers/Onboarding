"""Stage 2: the instruction set.

Three things, all described in docs/02-instruction-set.md: an instruction is 16 bytes
with a fixed layout, a line of text turns into one of those, and the simulator does
exactly what the guide says each instruction does.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(2)


# --- the 128-bit layout -------------------------------------------------------------

# Hand-checked against docs/02-instruction-set.md. Bytes are little-endian, so the
# opcode is the first byte you see.
GOLDEN_ENCODINGS = [
    ("HALT",
     "ff000000000000000000000000000000"),
    ("NO_OPERATION",
     "00000000000000000000000000000000"),
    ("LOAD space=WEIGHT_SCRATCHPAD memory=0x400 index=0 count=100352",
     "01010004000000000000880100000000"),
    ("LOAD space=BIAS_SCRATCHPAD memory=0x18C00 index=128 count=10",
     "0102008c01008000000a000000000000"),
    ("STORE space=ACCUMULATORS memory=0x19680 index=0 count=10",
     "0203809601000000000a000000000000"),
    ("MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs=128 inputs=784",
     "10000000000000000080001003000000"),
    ("MATRIX_MULTIPLY input=1024 weights=100352 accumulator=0 outputs=10 inputs=128 accumulate=1",
     "1001000400880100000a008000000000"),
    ("ADD_BIAS accumulator=0 bias=128 count=10",
     "2000000080000a000000000000000000"),
    ("RECTIFIED_LINEAR accumulator=0 destination=1024 count=128 shift=12",
     "30010000000480000c00000000000000"),
    ("RECTIFIED_LINEAR accumulator=0 destination=1024 count=128 shift=12 rectify=0",
     "30000000000480000c00000000000000"),
]


@pytest.mark.parametrize("text,hex_string", GOLDEN_ENCODINGS)
def test_encode_golden(text, hex_string):
    from python.assembler import parse_line
    from python.instruction_set import encode

    assert encode(parse_line(text)).hex() == hex_string


@pytest.mark.parametrize("text,hex_string", GOLDEN_ENCODINGS)
def test_decode_roundtrip(text, hex_string):
    from python.assembler import format_instruction, parse_line
    from python.instruction_set import decode, encode

    instruction = parse_line(text)
    assert decode(encode(instruction)) == instruction
    assert parse_line(format_instruction(instruction)) == instruction


def test_strict_decode_rejects_junk():
    """A bit no operand claims is an error, so a bad encoding fails at the first fetch."""
    from python.instruction_set import decode

    raw = bytearray(16)
    raw[0] = 0xFF
    raw[15] = 0x80          # a bit no HALT operand claims
    with pytest.raises(ValueError):
        decode(bytes(raw))
    assert decode(bytes(raw), strict=False).name == "HALT"


# --- the assembly text format -------------------------------------------------------


def test_parse_line_operands():
    from python.assembler import parse_line

    instruction = parse_line(
        "  load  SPACE=activation_scratchpad  MEMORY=0x10 index=3 count=784 ; a comment"
    )
    assert instruction.name == "LOAD"
    assert instruction.operands == {"space": 0, "memory": 16, "index": 3, "count": 784}
    assert parse_line("; only a comment") is None
    assert parse_line("") is None


def test_parse_line_errors():
    from python.assembler import AssemblyError, parse_line

    with pytest.raises(AssemblyError):
        parse_line("JUMP 0", 1)
    with pytest.raises(AssemblyError):
        parse_line("LOAD space=ACTIVATION_SCRATCHPAD memory=0", 1)         # missing operands
    with pytest.raises(AssemblyError):
        parse_line("LOAD space=ACTIVATION_SCRATCHPAD memory=0 index=0 count=784 extra=1", 1)
    with pytest.raises(AssemblyError):
        parse_line("MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs=70000 inputs=1", 1)


def test_disassemble_compiled_program(compiled_program):
    """The twelve instructions that classify a digit."""
    from python.assembler import disassemble
    from python.instruction_set import encode

    raw = b"".join(encode(i) for i in compiled_program.instructions)
    names = [line.split()[0] for line in disassemble(raw).splitlines()]
    assert names == [
        "LOAD", "LOAD", "LOAD", "MATRIX_MULTIPLY", "ADD_BIAS", "RECTIFIED_LINEAR",
        "LOAD", "LOAD", "MATRIX_MULTIPLY", "ADD_BIAS", "STORE", "HALT",
    ]


# --- what each instruction does -----------------------------------------------------


def run(assembly: str, data: dict[int, bytes] | None = None):
    """Assemble, place at address 0, drop `data` at the given addresses, run to HALT."""
    from python.assembler import assemble_bytes
    from python.simulator import Machine

    machine = Machine(bytearray(assemble_bytes(assembly)))
    for address, raw in (data or {}).items():
        machine.main_memory[address : address + len(raw)] = raw
    machine.run()
    return machine


def int32_bytes(*values) -> bytes:
    return np.array(values, dtype="<i4").tobytes()


def test_load_into_each_scratchpad():
    machine = run("""
        LOAD space=ACTIVATION_SCRATCHPAD memory=0x1000 index=5 count=3
        LOAD space=WEIGHT_SCRATCHPAD     memory=0x1000 index=7 count=3
        LOAD space=BIAS_SCRATCHPAD       memory=0x2000 index=1 count=2
        HALT""",
        {0x1000: bytes([1, 0xFF, 0x80]), 0x2000: int32_bytes(-100000, 7)})
    assert machine.activation_scratchpad[5:8].tolist() == [1, -1, -128]
    assert machine.weight_scratchpad[7:10].tolist() == [1, -1, -128]
    assert machine.bias_scratchpad[1:3].tolist() == [-100000, 7]


def test_store_from_activations_and_accumulators():
    machine = run("""
        LOAD  space=ACTIVATION_SCRATCHPAD memory=0x1000 index=0 count=2
        STORE space=ACTIVATION_SCRATCHPAD memory=0x3000 index=0 count=2
        HALT""",
        {0x1000: bytes([0x7F, 0x81])})
    assert bytes(machine.main_memory[0x3000:0x3002]) == bytes([0x7F, 0x81])

    # A STORE from the accumulators writes four little-endian bytes per element.
    machine.accumulators[0:2] = [-2, 305419896]
    machine.halted = False
    tail = run("STORE space=ACCUMULATORS memory=0x4000 index=0 count=2\nHALT")
    counter = machine.program_counter
    machine.main_memory[counter : counter + 32] = tail.main_memory[0:32]
    machine.run()
    assert bytes(machine.main_memory[0x4000:0x4008]) == int32_bytes(-2, 305419896)


def test_matrix_multiply_small():
    # activations [1, 2, 3]; weight rows [1,0,0], [0,1,0], [-1,-1,-1], [10,10,10]
    weights = np.array([1, 0, 0, 0, 1, 0, -1, -1, -1, 10, 10, 10], np.int8)
    machine = run("""
        LOAD            space=ACTIVATION_SCRATCHPAD memory=0x1000 index=0 count=3
        LOAD            space=WEIGHT_SCRATCHPAD memory=0x1010 index=0 count=12
        MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs=4 inputs=3
        HALT""",
        {0x1000: bytes([1, 2, 3]), 0x1010: weights.tobytes()})
    assert machine.accumulators[0:4].tolist() == [1, 2, -6, 60]


def test_matrix_multiply_accumulate_and_negatives():
    machine = run("""
        LOAD            space=ACTIVATION_SCRATCHPAD memory=0x1000 index=0 count=2
        LOAD            space=WEIGHT_SCRATCHPAD memory=0x1010 index=0 count=4
        MATRIX_MULTIPLY input=0 weights=0 accumulator=3 outputs=2 inputs=2
        MATRIX_MULTIPLY input=0 weights=0 accumulator=3 outputs=2 inputs=2 accumulate=1
        MATRIX_MULTIPLY input=0 weights=0 accumulator=3 outputs=1 inputs=2
        HALT""",
        {0x1000: np.array([-128, 127], np.int8).tobytes(),
         0x1010: np.array([-128, -128, 127, 127], np.int8).tobytes()})
    # row 0: 16384 - 16256 = 128; row 1: -16256 + 16129 = -127
    assert machine.accumulators[3] == 128      # overwritten by the third multiply
    assert machine.accumulators[4] == -254     # accumulated twice


def test_matrix_multiply_wraps_at_32_bits():
    """Two inputs cannot overflow, so use accumulate to push a sum past the 32-bit limit."""
    from python.assembler import assemble_bytes
    from python.simulator import Machine

    machine = Machine(assemble_bytes("""
        LOAD            space=ACTIVATION_SCRATCHPAD memory=0x1000 index=0 count=1
        LOAD            space=WEIGHT_SCRATCHPAD memory=0x1000 index=0 count=1
        MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs=1 inputs=1 accumulate=1
        HALT"""))
    machine.main_memory[0x1000] = 0x7F
    machine.accumulators[0] = 2147483647 - 100
    machine.run()
    assert machine.accumulators[0] == np.int32(-2147483648 + (127 * 127 - 101))


def test_add_bias():
    machine = run("""
        LOAD     space=BIAS_SCRATCHPAD memory=0x2000 index=4 count=3
        ADD_BIAS accumulator=1 bias=4 count=3
        HALT""",
        {0x2000: int32_bytes(5, -5, 2147483647)})
    assert machine.accumulators[1:4].tolist() == [5, -5, 2147483647]

    wrapped = run("""
        LOAD     space=BIAS_SCRATCHPAD memory=0x2000 index=0 count=1
        ADD_BIAS accumulator=0 bias=0 count=1
        ADD_BIAS accumulator=0 bias=0 count=1
        HALT""",
        {0x2000: int32_bytes(2147483647)})
    assert wrapped.accumulators[0] == -2


@pytest.mark.parametrize("shift,rectify,accumulators,expected", [
    (0, 1, [5, -5, 127, 128, 100000], [5, 0, 127, 127, 127]),
    (0, 0, [5, -5, -128, -129, -100000], [5, -5, -128, -128, -128]),
    (3, 1, [8, 7, -8, 1023, 1024], [1, 0, 0, 127, 127]),
    (3, 0, [-1, -8, -9, -1024, -1025], [-1, -1, -2, -128, -128]),
    (12, 1, [503071, 4095, 4096, 520192], [122, 0, 1, 127]),
    (31, 1, [2147483647, -2147483648], [0, 0]),
    (31, 0, [2147483647, -2147483648], [0, -1]),
])
def test_rectified_linear(shift, rectify, accumulators, expected):
    """Zero the negatives, shift right, saturate to 8 bits. Both edges, both signs."""
    from python.assembler import assemble_bytes
    from python.simulator import Machine

    count = len(accumulators)
    machine = Machine(assemble_bytes(
        f"RECTIFIED_LINEAR accumulator=0 destination=100 count={count} "
        f"shift={shift} rectify={rectify}\nHALT"
    ))
    machine.accumulators[:count] = accumulators
    machine.run()
    assert machine.activation_scratchpad[100 : 100 + count].tolist() == expected


def test_range_checks():
    """The simulator is strict on purpose: it exists to catch a bad program."""
    from python.simulator import SimulatorError

    with pytest.raises(SimulatorError):
        run("LOAD space=ACTIVATION_SCRATCHPAD memory=0 index=4000 count=100\nHALT")
    with pytest.raises(SimulatorError):
        run("MATRIX_MULTIPLY input=0 weights=0 accumulator=250 outputs=10 inputs=1\nHALT")
    with pytest.raises(SimulatorError):
        run("STORE space=WEIGHT_SCRATCHPAD memory=0 index=0 count=1\nHALT")


def test_missing_halt():
    from python.simulator import SimulatorError

    with pytest.raises(SimulatorError):
        run("NO_OPERATION")


def test_mnist_end_to_end_on_the_simulator(compiled_program, golden):
    """The compiled program on the simulator must reproduce the NumPy reference exactly."""
    from python.program import Program
    from python.simulator import Machine

    for i in range(20):
        compiled_program.write_input(golden["quantized_pixels"][i])
        machine = Machine(compiled_program.image)
        machine.run()
        out = Program.read_output(machine.main_memory, compiled_program.regions["output"])
        np.testing.assert_array_equal(out, golden["int8_logits"][i], err_msg=f"image {i}")
    assert len(machine.trace) == 12
