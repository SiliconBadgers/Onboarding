"""Stage 4: the ISA simulator, instruction by instruction, then the whole network."""

import numpy as np
import pytest

from cub.stages import skip_unless_started

pytestmark = pytest.mark.stage(4)


@pytest.fixture(autouse=True)
def _started():
    skip_unless_started(4)


def run(asm: str, data: dict[int, bytes] | None = None):
    """Assemble `asm`, place it at 0, drop `data` bytes at the given DRAM addresses, run."""
    from cub.asm import assemble_bytes
    from cub.sim import Machine

    image = bytearray(assemble_bytes(asm))
    m = Machine(image)
    for addr, raw in (data or {}).items():
        m.dram[addr : addr + len(raw)] = raw
    m.run()
    return m


def i32(*vals) -> bytes:
    return np.array(vals, dtype="<i4").tobytes()


def test_load_each_scratchpad():
    m = run("""
        LOAD mem=SPAD_A dram=0x1000 spad=5 count=3
        LOAD mem=SPAD_W dram=0x1000 spad=7 count=3
        LOAD mem=SPAD_B dram=0x2000 spad=1 count=2
        HALT""", {0x1000: bytes([1, 0xFF, 0x80]), 0x2000: i32(-100000, 7)})
    assert m.spad_a[5:8].tolist() == [1, -1, -128]
    assert m.spad_w[7:10].tolist() == [1, -1, -128]
    assert m.spad_b[1:3].tolist() == [-100000, 7]


def test_store_from_spad_a_and_acc():
    m = run("""
        LOAD  mem=SPAD_A dram=0x1000 spad=0 count=2
        STORE mem=SPAD_A dram=0x3000 spad=0 count=2
        HALT""", {0x1000: bytes([0x7F, 0x81])})
    assert bytes(m.dram[0x3000:0x3002]) == bytes([0x7F, 0x81])
    m.acc[0:2] = [-2, 305419896]
    m.halted = False
    m.dram[m.pc : m.pc + 32] = run("STORE mem=ACC dram=0x4000 spad=0 count=2\nHALT").dram[0:32]
    m.run()
    assert bytes(m.dram[0x4000:0x4008]) == i32(-2, 305419896)


def test_matmul_small():
    # A = [1, 2, 3]; W rows: [1,0,0], [0,1,0], [-1,-1,-1], [10,10,10]
    m = run("""
        LOAD   mem=SPAD_A dram=0x1000 spad=0 count=3
        LOAD   mem=SPAD_W dram=0x1010 spad=0 count=12
        MATMUL a=0 w=0 acc=0 n=4 k=3
        HALT""", {0x1000: bytes([1, 2, 3]),
                  0x1010: np.array([1, 0, 0, 0, 1, 0, -1, -1, -1, 10, 10, 10], np.int8).tobytes()})
    assert m.acc[0:4].tolist() == [1, 2, -6, 60]


def test_matmul_accumulate_and_negatives():
    m = run("""
        LOAD   mem=SPAD_A dram=0x1000 spad=0 count=2
        LOAD   mem=SPAD_W dram=0x1010 spad=0 count=4
        MATMUL a=0 w=0 acc=3 n=2 k=2
        MATMUL a=0 w=0 acc=3 n=2 k=2 accumulate=1
        MATMUL a=0 w=0 acc=3 n=1 k=2
        HALT""", {0x1000: np.array([-128, 127], np.int8).tobytes(),
                  0x1010: np.array([-128, -128, 127, 127], np.int8).tobytes()})
    # row0: 16384 - 16256 = 128; row1: -16256 + 16129 = -127
    assert m.acc[3] == 128            # overwritten by the third MATMUL
    assert m.acc[4] == -254           # accumulated twice


def test_matmul_wraps_at_32_bits():
    """K=2 cannot overflow, so use accumulate to push a sum past INT32_MAX."""
    from cub.sim import Machine
    from cub.asm import assemble_bytes

    m = Machine(assemble_bytes("""
        LOAD   mem=SPAD_A dram=0x1000 spad=0 count=1
        LOAD   mem=SPAD_W dram=0x1000 spad=0 count=1
        MATMUL a=0 w=0 acc=0 n=1 k=1 accumulate=1
        HALT"""))
    m.dram[0x1000] = 0x7F
    m.acc[0] = 2147483647 - 100
    m.run()
    assert m.acc[0] == np.int32(-2147483648 + (127 * 127 - 101))


def test_add_bias():
    m = run("""
        LOAD     mem=SPAD_B dram=0x2000 spad=4 count=3
        ADD_BIAS acc=1 bias=4 count=3
        HALT""", {0x2000: i32(5, -5, 2147483647)})
    assert m.acc[1:4].tolist() == [5, -5, 2147483647]
    m2 = run("""
        LOAD     mem=SPAD_B dram=0x2000 spad=0 count=1
        ADD_BIAS acc=0 bias=0 count=1
        ADD_BIAS acc=0 bias=0 count=1
        HALT""", {0x2000: i32(2147483647)})
    assert m2.acc[0] == -2            # wrapped


@pytest.mark.parametrize("shift,relu,acc,expected", [
    (0, 1, [5, -5, 127, 128, 100000], [5, 0, 127, 127, 127]),
    (0, 0, [5, -5, -128, -129, -100000], [5, -5, -128, -128, -128]),
    (3, 1, [8, 7, -8, 1023, 1024], [1, 0, 0, 127, 127]),
    (3, 0, [-1, -8, -9, -1024, -1025], [-1, -1, -2, -128, -128]),
    (12, 1, [503071, 4095, 4096, 520192], [122, 0, 1, 127]),
    (31, 1, [2147483647, -2147483648], [0, 0]),
    (31, 0, [2147483647, -2147483648], [0, -1]),
])
def test_relu(shift, relu, acc, expected):
    from cub.asm import assemble_bytes
    from cub.sim import Machine

    n = len(acc)
    m = Machine(assemble_bytes(f"RELU acc=0 dst=100 count={n} shift={shift} relu={relu}\nHALT"))
    m.acc[:n] = acc
    m.run()
    assert m.spad_a[100 : 100 + n].tolist() == expected


def test_range_checks():
    from cub.sim import SimError

    with pytest.raises(SimError):
        run("LOAD mem=SPAD_A dram=0 spad=4000 count=100\nHALT")
    with pytest.raises(SimError):
        run("MATMUL a=0 w=0 acc=250 n=10 k=1\nHALT")
    with pytest.raises(SimError):
        run("STORE mem=SPAD_W dram=0 spad=0 count=1\nHALT")


def test_missing_halt():
    from cub.sim import SimError

    with pytest.raises(SimError):
        run("NOP")


def test_mnist_end_to_end(compiled_program, golden):
    """The compiled program, on your simulator, must reproduce the NumPy reference exactly."""
    from cub.program import Program
    from cub.sim import Machine

    for i in range(20):
        compiled_program.write_input(golden["x_q"][i])
        m = Machine(compiled_program.image)
        m.run()
        out = Program.read_output(m.dram, compiled_program.regions["output"])
        np.testing.assert_array_equal(out, golden["int8_logits"][i], err_msg=f"image {i}")
    assert len(m.trace) == 12
