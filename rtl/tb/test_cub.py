"""cocotb tests for cub_core (Stage 6).

Every test builds a small memory image in Python, runs it on the Python simulator
(cub.simulator.Machine) and on the hardware, and compares the two. The simulator is the
golden model; the hardware has to match it bit for bit. There is no second copy of the
expected answers to get wrong.

Directed tests use main-memory addresses at or above SCRATCH so they never disturb the
MNIST image that +MEMORY_HEX preloads for the end-to-end test.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from cub.assembler import assemble_bytes
from cub.simulator import Machine

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = 0x20000               # free main memory above the MNIST image
MNIST_OUTPUT = 0x19680          # output region of artifacts/mnist.cub
MNIST_INSTRUCTIONS = 0x400      # size of the instruction region

CLOCK_PERIOD_NS = 10


# --- helpers ------------------------------------------------------------------------


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, unit="ns").start())
    dut.reset_n.value = 0
    dut.start.value = 0
    await ClockCycles(dut.clk, 3)
    dut.reset_n.value = 1
    await ClockCycles(dut.clk, 2)


async def run_program(dut, max_cycles: int = 2_000_000) -> int:
    """Set the start bit, poll the done bit, return the number of cycles it took.

    This is exactly what a host driver does over a register interface, written out
    one wire at a time.
    """
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


def write_memory(dut, address: int, data: bytes) -> None:
    storage = dut.main_memory.storage
    for i, byte in enumerate(data):
        storage[address + i].value = byte


def read_memory(dut, address: int, count: int) -> bytes:
    storage = dut.main_memory.storage
    return bytes(int(storage[address + i].value) for i in range(count))


def signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def read_array(handle, base: int, count: int, bits: int) -> np.ndarray:
    return np.array([signed(int(handle[base + i].value), bits) for i in range(count)])


class MemoryImage:
    """A main-memory image under construction: a program at 0 and data at SCRATCH+."""

    def __init__(self, assembly: str):
        self.code = assemble_bytes(assembly)
        self.chunks: list[tuple[int, bytes]] = [(0, self.code)]
        self.cursor = SCRATCH

    def place(self, data: bytes | np.ndarray) -> int:
        if isinstance(data, np.ndarray):
            raw = data.astype(data.dtype.newbyteorder("<")).tobytes()
        else:
            raw = bytes(data)
        address = self.cursor
        self.chunks.append((address, raw))
        self.cursor += (len(raw) + 63) // 64 * 64
        return address

    def reserve(self, count: int) -> int:
        return self.place(bytes(count))

    def set_code(self, assembly: str) -> None:
        self.code = assemble_bytes(assembly)
        self.chunks[0] = (0, self.code)

    def as_bytes(self) -> bytearray:
        end = max(address + len(data) for address, data in self.chunks)
        out = bytearray(end)
        for address, data in self.chunks:
            out[address : address + len(data)] = data
        return out

    def load_into(self, dut) -> None:
        for address, data in self.chunks:
            write_memory(dut, address, data)


async def run_both(dut, image: MemoryImage) -> tuple[Machine, int]:
    """Run the same image on the simulator and the hardware. Returns (machine, cycles)."""
    machine = Machine(image.as_bytes())
    machine.run()
    image.load_into(dut)
    cycles = await run_program(dut)
    return machine, cycles


def check_space(dut, name: str, machine: Machine, base: int, count: int) -> None:
    """Compare one on-chip memory range against the simulator.

    Only ranges the program actually wrote are meaningful: the hardware's memories
    start as X and keep their values across tests, while the simulator's are fresh
    zeros every run.
    """
    handle = {
        "activation_scratchpad": dut.core.activation_scratchpad,
        "weight_scratchpad": dut.core.weight_scratchpad,
        "bias_scratchpad": dut.core.bias_scratchpad,
        "accumulators": dut.core.accumulators,
    }[name]
    bits = 8 if name in ("activation_scratchpad", "weight_scratchpad") else 32
    from_hardware = read_array(handle, base, count, bits)
    from_simulator = getattr(machine, name)[base : base + count].astype(np.int64)
    assert (from_hardware == from_simulator).all(), (
        f"{name}[{base}:{base + count}] hardware {from_hardware.tolist()} "
        f"!= simulator {from_simulator.tolist()}"
    )


rng = np.random.default_rng(1234)


# --- directed tests -----------------------------------------------------------------


@cocotb.test()
async def test_load_and_store_activations(dut):
    """LOAD into the activation scratchpad, STORE it back out. The byte path both ways."""
    await reset(dut)
    data = rng.integers(-128, 128, size=37, dtype=np.int8)
    image = MemoryImage("")
    source = image.place(data)
    destination = image.reserve(64)
    image.set_code(f"""
        LOAD  space=ACTIVATION_SCRATCHPAD memory={source} index=5 count=37
        STORE space=ACTIVATION_SCRATCHPAD memory={destination} index=5 count=37
        HALT
    """)
    machine, _ = await run_both(dut, image)
    check_space(dut, "activation_scratchpad", machine, 5, 37)
    assert (read_memory(dut, destination, 37)
            == bytes(machine.main_memory[destination : destination + 37])
            == data.tobytes())


@cocotb.test()
async def test_load_weights_and_biases(dut):
    """LOAD bytes into the weight scratchpad and little-endian 32-bit values into biases."""
    await reset(dut)
    weights = rng.integers(-128, 128, size=100, dtype=np.int8)
    biases = rng.integers(-(2**31), 2**31, size=9, dtype=np.int64).astype(np.int32)
    image = MemoryImage("")
    weight_address = image.place(weights)
    bias_address = image.place(biases)
    image.set_code(f"""
        LOAD space=WEIGHT_SCRATCHPAD memory={weight_address} index=130900 count=100
        LOAD space=BIAS_SCRATCHPAD   memory={bias_address}   index=200    count=9
        HALT
    """)
    machine, _ = await run_both(dut, image)
    check_space(dut, "weight_scratchpad", machine, 130900, 100)
    check_space(dut, "bias_scratchpad", machine, 200, 9)


@cocotb.test()
async def test_matrix_multiply_small(dut):
    """MATRIX_MULTIPLY with negative values, then STORE from ACCUMULATORS as 32 bits."""
    await reset(dut)
    outputs, inputs = 5, 7
    activations = rng.integers(-128, 128, size=inputs, dtype=np.int8)
    weights = rng.integers(-128, 128, size=(outputs, inputs), dtype=np.int8)
    image = MemoryImage("")
    activation_address = image.place(activations)
    weight_address = image.place(weights)
    out = image.reserve(64)
    image.set_code(f"""
        LOAD            space=ACTIVATION_SCRATCHPAD memory={activation_address} index=10 count={inputs}
        LOAD            space=WEIGHT_SCRATCHPAD memory={weight_address} index=20 count={outputs * inputs}
        MATRIX_MULTIPLY input=10 weights=20 accumulator=3 outputs={outputs} inputs={inputs}
        STORE           space=ACCUMULATORS memory={out} index=3 count={outputs}
        HALT
    """)
    machine, _ = await run_both(dut, image)
    expected = activations.astype(np.int64) @ weights.astype(np.int64).T
    check_space(dut, "accumulators", machine, 3, outputs)
    got = np.frombuffer(read_memory(dut, out, 4 * outputs), dtype="<i4")
    assert (got == expected).all(), f"{got} != {expected}"


@cocotb.test()
async def test_matrix_multiply_accumulate_and_extremes(dut):
    """accumulate=1 adds to what is already there; all -128 inputs hit the largest product."""
    await reset(dut)
    outputs, inputs = 3, 16
    activations = np.full(inputs, -128, dtype=np.int8)
    weights = np.full((outputs, inputs), -128, dtype=np.int8)
    weights[1] = 127
    image = MemoryImage("")
    activation_address = image.place(activations)
    weight_address = image.place(weights)
    out = image.reserve(64)
    image.set_code(f"""
        LOAD            space=ACTIVATION_SCRATCHPAD memory={activation_address} index=0 count={inputs}
        LOAD            space=WEIGHT_SCRATCHPAD memory={weight_address} index=0 count={outputs * inputs}
        MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs={outputs} inputs={inputs}
        MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs={outputs} inputs={inputs} accumulate=1
        MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs={outputs} inputs={inputs} accumulate=1
        STORE           space=ACCUMULATORS memory={out} index=0 count={outputs}
        HALT
    """)
    machine, _ = await run_both(dut, image)
    expected = 3 * (activations.astype(np.int64) @ weights.astype(np.int64).T)
    got = np.frombuffer(read_memory(dut, out, 4 * outputs), dtype="<i4")
    assert (got == expected).all(), f"{got} != {expected}"
    check_space(dut, "accumulators", machine, 0, outputs)


@cocotb.test()
async def test_add_bias_wraps(dut):
    """ADD_BIAS adds 32-bit biases, wrapping the way a 32-bit adder does."""
    await reset(dut)
    outputs, inputs = 4, 2
    activations = np.array([1, 1], dtype=np.int8)
    weights = np.array([[100, 27], [-100, -28], [0, 0], [50, 50]], dtype=np.int8)
    biases = np.array([2**31 - 100, -(2**31) + 100, -5, 123456], dtype=np.int32)
    image = MemoryImage("")
    activation_address = image.place(activations)
    weight_address = image.place(weights)
    bias_address = image.place(biases)
    out = image.reserve(64)
    image.set_code(f"""
        LOAD            space=ACTIVATION_SCRATCHPAD memory={activation_address} index=0 count={inputs}
        LOAD            space=WEIGHT_SCRATCHPAD memory={weight_address} index=0 count={outputs * inputs}
        LOAD            space=BIAS_SCRATCHPAD memory={bias_address} index=7 count={outputs}
        MATRIX_MULTIPLY input=0 weights=0 accumulator=2 outputs={outputs} inputs={inputs}
        ADD_BIAS        accumulator=2 bias=7 count={outputs}
        STORE           space=ACCUMULATORS memory={out} index=2 count={outputs}
        HALT
    """)
    machine, _ = await run_both(dut, image)
    got = np.frombuffer(read_memory(dut, out, 4 * outputs), dtype="<i4")
    reference = np.frombuffer(
        bytes(machine.main_memory[out : out + 4 * outputs]), dtype="<i4"
    )
    assert (got == reference).all(), f"{got} != {reference}"
    # the first two wrap; check them explicitly so the test documents the intent
    assert got[0] == np.int32(np.int64(127) + (2**31 - 100) - 2**32)
    assert got[1] == np.int32(np.int64(-128) + (-(2**31) + 100) + 2**32)


@cocotb.test()
async def test_rectified_linear_shift_and_saturate(dut):
    """RECTIFIED_LINEAR with rectify on and off, several shifts, saturation both ways."""
    await reset(dut)
    # Build accumulators with a multiply by ones (so the accumulator equals the
    # activation) and then a bias big enough to reach interesting magnitudes.
    outputs, inputs = 8, 1
    activations = np.array([1], dtype=np.int8)
    weights = np.ones((outputs, inputs), dtype=np.int8)
    biases = np.array(
        [300000, -300000, 127 * 16, -128 * 16 - 1, 5, -5, 2**31 - 2, -(2**31) + 1],
        dtype=np.int32,
    )
    image = MemoryImage("")
    activation_address = image.place(activations)
    weight_address = image.place(weights)
    bias_address = image.place(biases)
    out = image.reserve(128)
    lines = [
        f"LOAD            space=ACTIVATION_SCRATCHPAD memory={activation_address} index=0 count={inputs}",
        f"LOAD            space=WEIGHT_SCRATCHPAD memory={weight_address} index=0 count={outputs * inputs}",
        f"LOAD            space=BIAS_SCRATCHPAD memory={bias_address} index=0 count={outputs}",
        f"MATRIX_MULTIPLY input=0 weights=0 accumulator=0 outputs={outputs} inputs={inputs}",
        f"ADD_BIAS        accumulator=0 bias=0 count={outputs}",
    ]
    cases = [(1, 0), (1, 4), (1, 12), (0, 0), (0, 4), (0, 12), (1, 31), (0, 31)]
    for i, (rectify, shift) in enumerate(cases):
        lines.append(
            f"RECTIFIED_LINEAR accumulator=0 destination={100 + 16 * i} "
            f"count={outputs} shift={shift} rectify={rectify}"
        )
    for i in range(len(cases)):
        lines.append(
            f"STORE space=ACTIVATION_SCRATCHPAD memory={out + 16 * i} "
            f"index={100 + 16 * i} count={outputs}"
        )
    lines.append("HALT")
    image.set_code("\n".join(lines))
    machine, _ = await run_both(dut, image)
    for i, (rectify, shift) in enumerate(cases):
        check_space(dut, "activation_scratchpad", machine, 100 + 16 * i, outputs)
        got = np.frombuffer(read_memory(dut, out + 16 * i, outputs), dtype=np.int8)
        reference = np.frombuffer(
            bytes(machine.main_memory[out + 16 * i : out + 16 * i + outputs]), dtype=np.int8
        )
        assert (got == reference).all(), (
            f"rectify={rectify} shift={shift}: {got} != {reference}"
        )


@cocotb.test()
async def test_no_operation_and_unknown_opcode(dut):
    """NO_OPERATION and an unknown opcode are skipped; HALT still arrives."""
    await reset(dut)
    image = MemoryImage("")
    out = image.reserve(64)
    code = assemble_bytes(
        f"NO_OPERATION\nLOAD space=ACTIVATION_SCRATCHPAD memory={out} index=0 count=1\nHALT"
    )
    bogus = bytes([0x77]) + bytes(15)          # not a real opcode
    image.code = code[:16] + bogus + code[16:]
    image.chunks[0] = (0, image.code)
    write_memory(dut, 0, image.code)
    cycles = await run_program(dut)
    assert cycles < 200


# --- the real thing -----------------------------------------------------------------


@cocotb.test()
async def test_mnist_end_to_end(dut):
    """Run the compiled MNIST program on test image 0 and compare with the golden answer."""
    await reset(dut)
    hex_path = Path(
        os.environ.get("CUB_MEMORY_HEX", ROOT / "rtl" / "build" / "main_memory.hex")
    )
    image_bytes = bytes(int(line, 16) for line in hex_path.read_text().split())
    # +MEMORY_HEX preloaded the bulk of the image at time 0. The directed tests above
    # overwrote the instruction region, so restore it and the input region here.
    write_memory(dut, 0, image_bytes[:MNIST_INSTRUCTIONS])
    golden = np.load(ROOT / "artifacts" / "golden.npz")
    from cub.program import Program

    program = Program.load(ROOT / "artifacts" / "mnist.cub")
    input_region = program.regions["input"]
    write_memory(dut, input_region.offset, image_bytes[input_region.offset : input_region.end])
    assert program.regions["output"].offset == MNIST_OUTPUT

    started = time.time()
    cycles = await run_program(dut, max_cycles=1_000_000)
    wall_seconds = time.time() - started
    logits = np.frombuffer(read_memory(dut, MNIST_OUTPUT, 40), dtype="<i4")
    expected = golden["int8_logits"][0]
    dut._log.info(
        f"MNIST image 0: {cycles} cycles, {wall_seconds:.1f}s wall, logits {logits.tolist()}"
    )
    assert (logits == expected).all(), (
        f"hardware {logits.tolist()} != golden {expected.tolist()}"
    )
    assert int(np.argmax(logits)) == int(golden["labels"][0]) == 7
