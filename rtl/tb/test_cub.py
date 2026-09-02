"""cocotb tests for cub_core (Stage 8).

Every test builds a small DRAM image in Python, runs it on the Python simulator
(cub.sim.Machine) and on the RTL, and compares the results. The simulator is the
golden model; the RTL has to match it bit for bit.

Directed tests use DRAM addresses at or above SCRATCH so they never disturb the MNIST
image that +DRAM_HEX preloads for the end-to-end test.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from cub import isa
from cub.asm import assemble_bytes
from cub.sim import Machine

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = 0x20000               # free DRAM above the MNIST image
MNIST_OUTPUT = 0x19680          # output region of artifacts/mnist.cub
MNIST_INSNS = 0x400             # size of the instruction region

CLK_PERIOD_NS = 10


# --- helpers ------------------------------------------------------------------------


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def run_program(dut, max_cycles: int = 2_000_000) -> int:
    """Pulse start, wait for done, return the number of cycles busy."""
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    cycles = 0
    while True:
        await RisingEdge(dut.clk)
        cycles += 1
        if dut.done.value == 1:
            return cycles
        if cycles > max_cycles:
            raise AssertionError(f"no done after {max_cycles} cycles")


def poke(dut, addr: int, data: bytes) -> None:
    mem = dut.host_mem.mem
    for i, b in enumerate(data):
        mem[addr + i].value = b


def peek(dut, addr: int, n: int) -> bytes:
    mem = dut.host_mem.mem
    return bytes(int(mem[addr + i].value) for i in range(n))


def signed(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def read_array(handle, base: int, n: int, bits: int) -> np.ndarray:
    return np.array([signed(int(handle[base + i].value), bits) for i in range(n)])


class Image:
    """A DRAM image under construction: a program at 0 and data at SCRATCH+."""

    def __init__(self, asm: str):
        self.code = assemble_bytes(asm)
        self.chunks: list[tuple[int, bytes]] = [(0, self.code)]
        self.cursor = SCRATCH

    def place(self, data: bytes | np.ndarray) -> int:
        raw = data.astype(data.dtype.newbyteorder("<")).tobytes() if isinstance(data, np.ndarray) else bytes(data)
        addr = self.cursor
        self.chunks.append((addr, raw))
        self.cursor += (len(raw) + 63) // 64 * 64
        return addr

    def reserve(self, n: int) -> int:
        return self.place(bytes(n))

    def dram(self) -> bytearray:
        end = max(a + len(d) for a, d in self.chunks)
        out = bytearray(end)
        for a, d in self.chunks:
            out[a : a + len(d)] = d
        return out

    def load_into(self, dut) -> None:
        for a, d in self.chunks:
            poke(dut, a, d)


async def run_both(dut, img: Image) -> tuple[Machine, int]:
    """Run the same image on the simulator and the RTL. Returns (machine, rtl_cycles)."""
    m = Machine(img.dram())
    m.run()
    img.load_into(dut)
    cycles = await run_program(dut)
    return m, cycles


def check_spad(dut, name: str, m: Machine, base: int, n: int) -> None:
    """Compare one internal memory range. Only ranges the program wrote are meaningful:
    the RTL's memories start as X and keep values across tests, the simulator's are
    fresh zeros every run."""
    handle = {"spad_a": dut.core.spad_a, "spad_w": dut.core.spad_w, "spad_b": dut.core.spad_b, "acc": dut.core.acc}[name]
    bits = 8 if name in ("spad_a", "spad_w") else 32
    rtl = read_array(handle, base, n, bits)
    ref = getattr(m, name)[base : base + n].astype(np.int64)
    assert (rtl == ref).all(), f"{name}[{base}:{base + n}] RTL {rtl.tolist()} != sim {ref.tolist()}"


rng = np.random.default_rng(1234)


# --- directed tests -----------------------------------------------------------------


@cocotb.test()
async def test_load_and_store_spad_a(dut):
    """LOAD into SPAD_A, STORE back out. Exercises the byte path both ways."""
    await reset(dut)
    data = rng.integers(-128, 128, size=37, dtype=np.int8)
    img = Image("")
    src = img.place(data)
    dst = img.reserve(64)
    img.code = assemble_bytes(f"""
        LOAD  mem=SPAD_A dram={src} spad=5 count=37
        STORE mem=SPAD_A dram={dst} spad=5 count=37
        HALT
    """)
    img.chunks[0] = (0, img.code)
    m, _ = await run_both(dut, img)
    check_spad(dut, "spad_a", m, 5, 37)
    assert peek(dut, dst, 37) == bytes(m.dram[dst : dst + 37]) == data.tobytes()


@cocotb.test()
async def test_load_spad_w_and_spad_b(dut):
    """LOAD into SPAD_W (bytes, at a high address) and SPAD_B (little-endian INT32)."""
    await reset(dut)
    w = rng.integers(-128, 128, size=100, dtype=np.int8)
    b = rng.integers(-(2**31), 2**31, size=9, dtype=np.int64).astype(np.int32)
    img = Image("")
    wa = img.place(w)
    ba = img.place(b)
    img.code = assemble_bytes(f"""
        LOAD mem=SPAD_W dram={wa} spad=130900 count=100
        LOAD mem=SPAD_B dram={ba} spad=200 count=9
        HALT
    """)
    img.chunks[0] = (0, img.code)
    m, _ = await run_both(dut, img)
    check_spad(dut, "spad_w", m, 130900, 100)
    check_spad(dut, "spad_b", m, 200, 9)


@cocotb.test()
async def test_matmul_small(dut):
    """MATMUL with negative values, then STORE from ACC as INT32."""
    await reset(dut)
    n, k = 5, 7
    a = rng.integers(-128, 128, size=k, dtype=np.int8)
    w = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
    img = Image("")
    aa = img.place(a)
    wa = img.place(w)
    out = img.reserve(64)
    img.code = assemble_bytes(f"""
        LOAD   mem=SPAD_A dram={aa} spad=10 count={k}
        LOAD   mem=SPAD_W dram={wa} spad=20 count={n * k}
        MATMUL a=10 w=20 acc=3 n={n} k={k}
        STORE  mem=ACC dram={out} spad=3 count={n}
        HALT
    """)
    img.chunks[0] = (0, img.code)
    m, _ = await run_both(dut, img)
    expect = a.astype(np.int64) @ w.astype(np.int64).T
    check_spad(dut, "acc", m, 3, n)
    got = np.frombuffer(peek(dut, out, 4 * n), dtype="<i4")
    assert (got == expect).all(), f"{got} != {expect}"


@cocotb.test()
async def test_matmul_accumulate_and_extremes(dut):
    """accumulate=1 adds to ACC; all -128 inputs hit the largest product."""
    await reset(dut)
    n, k = 3, 16
    a = np.full(k, -128, dtype=np.int8)
    w = np.full((n, k), -128, dtype=np.int8)
    w[1] = 127
    img = Image("")
    aa = img.place(a)
    wa = img.place(w)
    out = img.reserve(64)
    img.code = assemble_bytes(f"""
        LOAD   mem=SPAD_A dram={aa} spad=0 count={k}
        LOAD   mem=SPAD_W dram={wa} spad=0 count={n * k}
        MATMUL a=0 w=0 acc=0 n={n} k={k}
        MATMUL a=0 w=0 acc=0 n={n} k={k} accumulate=1
        MATMUL a=0 w=0 acc=0 n={n} k={k} accumulate=1
        STORE  mem=ACC dram={out} spad=0 count={n}
        HALT
    """)
    img.chunks[0] = (0, img.code)
    m, _ = await run_both(dut, img)
    expect = 3 * (a.astype(np.int64) @ w.astype(np.int64).T)
    got = np.frombuffer(peek(dut, out, 4 * n), dtype="<i4")
    assert (got == expect).all(), f"{got} != {expect}"
    check_spad(dut, "acc", m, 0, n)


@cocotb.test()
async def test_add_bias_wraps(dut):
    """ADD_BIAS adds INT32 biases, wrapping like hardware."""
    await reset(dut)
    n, k = 4, 2
    a = np.array([1, 1], dtype=np.int8)
    w = np.array([[100, 27], [-100, -28], [0, 0], [50, 50]], dtype=np.int8)
    bias = np.array([2**31 - 100, -(2**31) + 100, -5, 123456], dtype=np.int32)
    img = Image("")
    aa = img.place(a)
    wa = img.place(w)
    ba = img.place(bias)
    out = img.reserve(64)
    img.code = assemble_bytes(f"""
        LOAD     mem=SPAD_A dram={aa} spad=0 count={k}
        LOAD     mem=SPAD_W dram={wa} spad=0 count={n * k}
        LOAD     mem=SPAD_B dram={ba} spad=7 count={n}
        MATMUL   a=0 w=0 acc=2 n={n} k={k}
        ADD_BIAS acc=2 bias=7 count={n}
        STORE    mem=ACC dram={out} spad=2 count={n}
        HALT
    """)
    img.chunks[0] = (0, img.code)
    m, _ = await run_both(dut, img)
    got = np.frombuffer(peek(dut, out, 4 * n), dtype="<i4")
    ref = np.frombuffer(bytes(m.dram[out : out + 4 * n]), dtype="<i4")
    assert (got == ref).all(), f"{got} != {ref}"
    # the first two wrap; check them explicitly so the test documents the intent
    assert got[0] == np.int32(np.int64(127) + (2**31 - 100) - 2**32)
    assert got[1] == np.int32(np.int64(-128) + (-(2**31) + 100) + 2**32)


@cocotb.test()
async def test_relu_shift_saturate(dut):
    """RELU with relu=1 and relu=0, several shifts, saturation on both sides."""
    await reset(dut)
    # Build accumulators by matmul against an identity-ish weight: a=INT8 values
    # times w=1 gives ACC = a; bias pushes them to large magnitudes.
    n, k = 8, 1
    a = np.array([1], dtype=np.int8)
    w = np.ones((n, k), dtype=np.int8)
    bias = np.array([300000, -300000, 127 * 16, -128 * 16 - 1, 5, -5, 2**31 - 2, -(2**31) + 1], dtype=np.int32)
    img = Image("")
    aa = img.place(a)
    wa = img.place(w)
    ba = img.place(bias)
    out = img.reserve(128)
    lines = [
        f"LOAD     mem=SPAD_A dram={aa} spad=0 count={k}",
        f"LOAD     mem=SPAD_W dram={wa} spad=0 count={n * k}",
        f"LOAD     mem=SPAD_B dram={ba} spad=0 count={n}",
        f"MATMUL   a=0 w=0 acc=0 n={n} k={k}",
        f"ADD_BIAS acc=0 bias=0 count={n}",
    ]
    cases = [(1, 0), (1, 4), (1, 12), (0, 0), (0, 4), (0, 12), (1, 31), (0, 31)]
    for i, (relu, shift) in enumerate(cases):
        lines.append(f"RELU acc=0 dst={100 + 16 * i} count={n} shift={shift} relu={relu}")
    for i in range(len(cases)):
        lines.append(f"STORE mem=SPAD_A dram={out + 16 * i} spad={100 + 16 * i} count={n}")
    lines.append("HALT")
    img.code = assemble_bytes("\n".join(lines))
    img.chunks[0] = (0, img.code)
    m, _ = await run_both(dut, img)
    for i, (relu, shift) in enumerate(cases):
        check_spad(dut, "spad_a", m, 100 + 16 * i, n)
        got = np.frombuffer(peek(dut, out + 16 * i, n), dtype=np.int8)
        ref = np.frombuffer(bytes(m.dram[out + 16 * i : out + 16 * i + n]), dtype=np.int8)
        assert (got == ref).all(), f"relu={relu} shift={shift}: {got} != {ref}"


@cocotb.test()
async def test_nop_and_unknown_opcode(dut):
    """NOP and an unknown opcode are skipped; HALT still arrives."""
    await reset(dut)
    img = Image("")
    out = img.reserve(64)
    code = assemble_bytes(f"NOP\nLOAD mem=SPAD_A dram={out} spad=0 count=1\nHALT")
    bogus = bytes([0x77]) + bytes(15)          # not a real opcode
    img.code = code[:16] + bogus + code[16:]
    img.chunks[0] = (0, img.code)
    poke(dut, 0, img.code)
    cycles = await run_program(dut)
    assert cycles < 200


# --- the real thing -----------------------------------------------------------------


@cocotb.test()
async def test_mnist_end_to_end(dut):
    """Run the compiled MNIST program on test image 0 and compare with the golden logits."""
    await reset(dut)
    hex_path = Path(os.environ.get("CUB_DRAM_HEX", ROOT / "rtl" / "build" / "dram.hex"))
    image = bytes(int(line, 16) for line in hex_path.read_text().split())
    # +DRAM_HEX preloaded the bulk of the image at time 0. The directed tests above
    # overwrite the instruction region, so restore that and the input region here.
    poke(dut, 0, image[:MNIST_INSNS])
    golden = np.load(ROOT / "artifacts" / "golden.npz")
    from cub.program import Program

    prog = Program.load(ROOT / "artifacts" / "mnist.cub")
    inp = prog.regions["input"]
    poke(dut, inp.offset, image[inp.offset : inp.end])
    assert prog.regions["output"].offset == MNIST_OUTPUT

    t0 = time.time()
    cycles = await run_program(dut, max_cycles=1_000_000)
    wall = time.time() - t0
    logits = np.frombuffer(peek(dut, MNIST_OUTPUT, 40), dtype="<i4")
    expect = golden["int8_logits"][0]
    dut._log.info(f"MNIST image 0: {cycles} cycles, {wall:.1f}s wall, logits {logits.tolist()}")
    assert (logits == expect).all(), f"RTL {logits.tolist()} != golden {expect.tolist()}"
    assert int(np.argmax(logits)) == int(golden["labels"][0]) == 7
