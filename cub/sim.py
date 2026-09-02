"""The Cub ISA simulator: a software model of the core, bit-exact with the RTL.

It is deliberately written the slow, obvious way (loops, not vectorized NumPy) so it
reads like the pseudocode in docs/isa.md and like the RTL you will write in Stage 8.
Speed comes later, if ever.
"""

from __future__ import annotations

import numpy as np

from . import isa
from .isa import Instruction, decode


def wrap_i32(x: int) -> int:
    """Interpret a Python int as a two's-complement INT32, the way a 32-bit adder does."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def clamp_i8(x: int) -> int:
    return max(-128, min(127, x))


class SimError(RuntimeError):
    pass


class Machine:
    """One Cub core plus its DRAM.

    Memories are NumPy arrays of the exact element type in docs/isa.md section 2, so
    an out-of-range store fails instead of silently widening.
    """

    def __init__(self, dram: bytes | bytearray | None = None) -> None:
        self.dram = bytearray(isa.DRAM_BYTES)
        if dram is not None:
            if len(dram) > isa.DRAM_BYTES:
                raise SimError(f"image is {len(dram)} bytes, DRAM is {isa.DRAM_BYTES}")
            self.dram[: len(dram)] = dram
        self.spad_a = np.zeros(isa.SPAD_A_SIZE, dtype=np.int8)
        self.spad_w = np.zeros(isa.SPAD_W_SIZE, dtype=np.int8)
        self.spad_b = np.zeros(isa.SPAD_B_SIZE, dtype=np.int32)
        self.acc = np.zeros(isa.ACC_SIZE, dtype=np.int32)
        self.pc = 0            # byte address of the next instruction
        self.halted = False
        self.trace: list[Instruction] = []   # every instruction executed, in order
        self.cycles = 0        # a rough count: one per MAC, one per element moved

    # --- fetch / execute loop ---------------------------------------------------

    def fetch(self) -> Instruction:
        raw = bytes(self.dram[self.pc : self.pc + isa.INSN_BYTES])
        try:
            insn = decode(raw, strict=True)
        except ValueError as e:
            raise SimError(f"pc=0x{self.pc:X}: {e}") from e
        self.pc += isa.INSN_BYTES
        return insn

    def step(self) -> Instruction:
        if self.halted:
            raise SimError("machine is halted")
        insn = self.fetch()
        self.trace.append(insn)
        handler = {
            isa.OP_NOP: self._exec_nop,
            isa.OP_LOAD: self._exec_load,
            isa.OP_STORE: self._exec_store,
            isa.OP_MATMUL: self._exec_matmul,
            isa.OP_ADD_BIAS: self._exec_add_bias,
            isa.OP_RELU: self._exec_relu,
            isa.OP_HALT: self._exec_halt,
        }[insn.op]
        handler(insn)
        return insn

    def run(self, max_insns: int = 10_000) -> None:
        for _ in range(max_insns):
            self.step()
            if self.halted:
                return
        raise SimError(f"no HALT after {max_insns} instructions")

    # --- helpers ----------------------------------------------------------------

    def _spad(self, mem: int) -> np.ndarray:
        try:
            return {
                isa.MEM_SPAD_A: self.spad_a,
                isa.MEM_SPAD_W: self.spad_w,
                isa.MEM_SPAD_B: self.spad_b,
                isa.MEM_ACC: self.acc,
            }[mem]
        except KeyError:
            raise SimError(f"bad memory code {mem}") from None

    def _check_range(self, what: str, base: int, count: int, size: int) -> None:
        if base + count > size:
            raise SimError(f"{what}: [{base}, {base + count}) exceeds size {size}")

    # --- instruction semantics (docs/isa.md section 4) --------------------------

    def _exec_nop(self, insn: Instruction) -> None:
        pass

    def _exec_halt(self, insn: Instruction) -> None:
        self.halted = True

    def _exec_load(self, insn: Instruction) -> None:
        mem, dram, spad, count = insn["mem"], insn["dram"], insn["spad"], insn["count"]
        if mem == isa.MEM_ACC:
            raise SimError("LOAD into ACC is not allowed")
        dst = self._spad(mem)
        width = isa.MEM_ELEM_BYTES[mem]
        self._check_range("LOAD dram", dram, count * width, isa.DRAM_BYTES)
        self._check_range(f"LOAD {isa.MEM_NAMES[mem]}", spad, count, len(dst))
        raw = self.dram[dram : dram + count * width]
        dst[spad : spad + count] = np.frombuffer(raw, dtype=dst.dtype.newbyteorder("<"))
        self.cycles += count * width

    def _exec_store(self, insn: Instruction) -> None:
        mem, dram, spad, count = insn["mem"], insn["dram"], insn["spad"], insn["count"]
        if mem not in (isa.MEM_SPAD_A, isa.MEM_ACC):
            raise SimError(f"STORE from {isa.MEM_NAMES.get(mem, mem)} is not allowed")
        src = self._spad(mem)
        width = isa.MEM_ELEM_BYTES[mem]
        self._check_range("STORE dram", dram, count * width, isa.DRAM_BYTES)
        self._check_range(f"STORE {isa.MEM_NAMES[mem]}", spad, count, len(src))
        self.dram[dram : dram + count * width] = src[spad : spad + count].astype("<" + src.dtype.str[1:]).tobytes()
        self.cycles += count * width

    def _exec_matmul(self, insn: Instruction) -> None:
        a, w, acc, n, k = insn["a"], insn["w"], insn["acc"], insn["n"], insn["k"]
        accumulate = insn["accumulate"]
        self._check_range("MATMUL SPAD_A", a, k, isa.SPAD_A_SIZE)
        self._check_range("MATMUL SPAD_W", w, n * k, isa.SPAD_W_SIZE)
        self._check_range("MATMUL ACC", acc, n, isa.ACC_SIZE)
        # --- SOLUTION(stage=4): for each output row, dot the input vector with that row of the weight matrix and write (or add to) ACC. Use plain Python ints and wrap_i32 so the result wraps like hardware. ---
        for row in range(n):
            s = 0
            for col in range(k):
                s += int(self.spad_a[a + col]) * int(self.spad_w[w + row * k + col])
            if accumulate:
                s += int(self.acc[acc + row])
            self.acc[acc + row] = wrap_i32(s)
        # --- END SOLUTION ---
        self.cycles += n * k

    def _exec_add_bias(self, insn: Instruction) -> None:
        acc, bias, count = insn["acc"], insn["bias"], insn["count"]
        self._check_range("ADD_BIAS ACC", acc, count, isa.ACC_SIZE)
        self._check_range("ADD_BIAS SPAD_B", bias, count, isa.SPAD_B_SIZE)
        # --- SOLUTION(stage=4): add each bias into its accumulator, wrapping to INT32 ---
        for i in range(count):
            self.acc[acc + i] = wrap_i32(int(self.acc[acc + i]) + int(self.spad_b[bias + i]))
        # --- END SOLUTION ---
        self.cycles += count

    def _exec_relu(self, insn: Instruction) -> None:
        acc, dst, count, shift, relu = insn["acc"], insn["dst"], insn["count"], insn["shift"], insn["relu"]
        self._check_range("RELU ACC", acc, count, isa.ACC_SIZE)
        self._check_range("RELU SPAD_A", dst, count, isa.SPAD_A_SIZE)
        if shift > 31:
            raise SimError(f"RELU shift={shift} is out of range")
        # --- SOLUTION(stage=4): ReLU (if the flag is set), arithmetic shift right, saturate to INT8, write to SPAD_A ---
        for i in range(count):
            v = int(self.acc[acc + i])
            if relu:
                v = max(v, 0)
            v >>= shift              # Python's >> on a negative int is arithmetic
            self.spad_a[dst + i] = clamp_i8(v)
        # --- END SOLUTION ---
        self.cycles += count


def run_image(image: bytes, max_insns: int = 10_000) -> Machine:
    """Convenience: run a DRAM image to HALT and return the finished machine."""
    m = Machine(image)
    m.run(max_insns)
    return m
