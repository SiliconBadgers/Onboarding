"""A software model of the accelerator core, bit-exact with the hardware.

It is deliberately written the slow, obvious way (loops, not vectorized NumPy) so it
reads like the pseudocode in docs/02-instruction-set.md and like the SystemVerilog in
rtl/cub_core.sv. Speed comes later, if ever.

This is the *golden model*: when the hardware disagrees with it, the hardware is wrong.
"""

from __future__ import annotations

import numpy as np

from . import instruction_set
from .instruction_set import Instruction, decode


def wrap_to_int32(value: int) -> int:
    """Read a Python integer as a 32-bit two's-complement value, like a 32-bit adder."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def clamp_to_int8(value: int) -> int:
    return max(-128, min(127, value))


class SimulatorError(RuntimeError):
    pass


class Machine:
    """One accelerator core plus the main memory it reads and writes.

    The five memory spaces are NumPy arrays of exactly the element type listed in
    docs/02-instruction-set.md, so an out-of-range value fails instead of silently
    widening into a bigger type.
    """

    def __init__(self, main_memory: bytes | bytearray | None = None) -> None:
        self.main_memory = bytearray(instruction_set.MAIN_MEMORY_BYTES)
        if main_memory is not None:
            if len(main_memory) > instruction_set.MAIN_MEMORY_BYTES:
                raise SimulatorError(
                    f"image is {len(main_memory)} bytes, main memory is "
                    f"{instruction_set.MAIN_MEMORY_BYTES}"
                )
            self.main_memory[: len(main_memory)] = main_memory
        self.activation_scratchpad = np.zeros(
            instruction_set.ACTIVATION_SCRATCHPAD_SIZE, dtype=np.int8
        )
        self.weight_scratchpad = np.zeros(
            instruction_set.WEIGHT_SCRATCHPAD_SIZE, dtype=np.int8
        )
        self.bias_scratchpad = np.zeros(
            instruction_set.BIAS_SCRATCHPAD_SIZE, dtype=np.int32
        )
        self.accumulators = np.zeros(instruction_set.ACCUMULATOR_COUNT, dtype=np.int32)
        self.program_counter = 0      # byte address of the next instruction
        self.halted = False
        self.trace: list[Instruction] = []   # every instruction executed, in order
        self.cycles = 0               # rough count: one per multiply, one per element moved

    # --- fetch / execute loop ---------------------------------------------------

    def fetch(self) -> Instruction:
        start = self.program_counter
        raw = bytes(self.main_memory[start : start + instruction_set.INSTRUCTION_BYTES])
        try:
            instruction = decode(raw, strict=True)
        except ValueError as error:
            raise SimulatorError(f"program_counter=0x{start:X}: {error}") from error
        self.program_counter += instruction_set.INSTRUCTION_BYTES
        return instruction

    def step(self) -> Instruction:
        if self.halted:
            raise SimulatorError("machine is halted")
        instruction = self.fetch()
        self.trace.append(instruction)
        handler = {
            instruction_set.OPCODE_NO_OPERATION: self._execute_no_operation,
            instruction_set.OPCODE_LOAD: self._execute_load,
            instruction_set.OPCODE_STORE: self._execute_store,
            instruction_set.OPCODE_MATRIX_MULTIPLY: self._execute_matrix_multiply,
            instruction_set.OPCODE_ADD_BIAS: self._execute_add_bias,
            instruction_set.OPCODE_RECTIFIED_LINEAR: self._execute_rectified_linear,
            instruction_set.OPCODE_HALT: self._execute_halt,
        }[instruction.opcode]
        handler(instruction)
        return instruction

    def run(self, max_instructions: int = 10_000) -> None:
        for _ in range(max_instructions):
            self.step()
            if self.halted:
                return
        raise SimulatorError(f"no HALT after {max_instructions} instructions")

    # --- helpers ----------------------------------------------------------------

    def _space(self, space: int) -> np.ndarray:
        try:
            return {
                instruction_set.SPACE_ACTIVATION_SCRATCHPAD: self.activation_scratchpad,
                instruction_set.SPACE_WEIGHT_SCRATCHPAD: self.weight_scratchpad,
                instruction_set.SPACE_BIAS_SCRATCHPAD: self.bias_scratchpad,
                instruction_set.SPACE_ACCUMULATORS: self.accumulators,
            }[space]
        except KeyError:
            raise SimulatorError(f"bad memory space code {space}") from None

    def _check_range(self, what: str, base: int, count: int, size: int) -> None:
        if base + count > size:
            raise SimulatorError(f"{what}: [{base}, {base + count}) exceeds size {size}")

    # --- instruction semantics (docs/02-instruction-set.md) ---------------------

    def _execute_no_operation(self, instruction: Instruction) -> None:
        pass

    def _execute_halt(self, instruction: Instruction) -> None:
        self.halted = True

    def _execute_load(self, instruction: Instruction) -> None:
        space = instruction["space"]
        address = instruction["memory"]
        index = instruction["index"]
        count = instruction["count"]
        if space == instruction_set.SPACE_ACCUMULATORS:
            raise SimulatorError("LOAD into ACCUMULATORS is not allowed")
        destination = self._space(space)
        width = instruction_set.SPACE_ELEMENT_BYTES[space]
        self._check_range(
            "LOAD main memory", address, count * width, instruction_set.MAIN_MEMORY_BYTES
        )
        self._check_range(
            f"LOAD {instruction_set.SPACE_NAMES[space]}", index, count, len(destination)
        )
        raw = self.main_memory[address : address + count * width]
        destination[index : index + count] = np.frombuffer(
            raw, dtype=destination.dtype.newbyteorder("<")
        )
        self.cycles += count * width

    def _execute_store(self, instruction: Instruction) -> None:
        space = instruction["space"]
        address = instruction["memory"]
        index = instruction["index"]
        count = instruction["count"]
        if space not in (
            instruction_set.SPACE_ACTIVATION_SCRATCHPAD,
            instruction_set.SPACE_ACCUMULATORS,
        ):
            name = instruction_set.SPACE_NAMES.get(space, space)
            raise SimulatorError(f"STORE from {name} is not allowed")
        source = self._space(space)
        width = instruction_set.SPACE_ELEMENT_BYTES[space]
        self._check_range(
            "STORE main memory", address, count * width, instruction_set.MAIN_MEMORY_BYTES
        )
        self._check_range(
            f"STORE {instruction_set.SPACE_NAMES[space]}", index, count, len(source)
        )
        little_endian = "<" + source.dtype.str[1:]
        self.main_memory[address : address + count * width] = (
            source[index : index + count].astype(little_endian).tobytes()
        )
        self.cycles += count * width

    def _execute_matrix_multiply(self, instruction: Instruction) -> None:
        input_index = instruction["input"]
        weight_index = instruction["weights"]
        accumulator_index = instruction["accumulator"]
        outputs = instruction["outputs"]
        inputs = instruction["inputs"]
        accumulate = instruction["accumulate"]
        self._check_range(
            "MATRIX_MULTIPLY ACTIVATION_SCRATCHPAD", input_index, inputs,
            instruction_set.ACTIVATION_SCRATCHPAD_SIZE,
        )
        self._check_range(
            "MATRIX_MULTIPLY WEIGHT_SCRATCHPAD", weight_index, outputs * inputs,
            instruction_set.WEIGHT_SCRATCHPAD_SIZE,
        )
        self._check_range(
            "MATRIX_MULTIPLY ACCUMULATORS", accumulator_index, outputs,
            instruction_set.ACCUMULATOR_COUNT,
        )
        for row in range(outputs):
            total = 0
            for column in range(inputs):
                activation = int(self.activation_scratchpad[input_index + column])
                weight = int(self.weight_scratchpad[weight_index + row * inputs + column])
                total += activation * weight
            if accumulate:
                total += int(self.accumulators[accumulator_index + row])
            self.accumulators[accumulator_index + row] = wrap_to_int32(total)
        self.cycles += outputs * inputs

    def _execute_add_bias(self, instruction: Instruction) -> None:
        accumulator_index = instruction["accumulator"]
        bias_index = instruction["bias"]
        count = instruction["count"]
        self._check_range(
            "ADD_BIAS ACCUMULATORS", accumulator_index, count,
            instruction_set.ACCUMULATOR_COUNT,
        )
        self._check_range(
            "ADD_BIAS BIAS_SCRATCHPAD", bias_index, count,
            instruction_set.BIAS_SCRATCHPAD_SIZE,
        )
        for i in range(count):
            total = int(self.accumulators[accumulator_index + i]) + int(
                self.bias_scratchpad[bias_index + i]
            )
            self.accumulators[accumulator_index + i] = wrap_to_int32(total)
        self.cycles += count

    def _execute_rectified_linear(self, instruction: Instruction) -> None:
        accumulator_index = instruction["accumulator"]
        destination_index = instruction["destination"]
        count = instruction["count"]
        shift = instruction["shift"]
        rectify = instruction["rectify"]
        self._check_range(
            "RECTIFIED_LINEAR ACCUMULATORS", accumulator_index, count,
            instruction_set.ACCUMULATOR_COUNT,
        )
        self._check_range(
            "RECTIFIED_LINEAR ACTIVATION_SCRATCHPAD", destination_index, count,
            instruction_set.ACTIVATION_SCRATCHPAD_SIZE,
        )
        if shift > 31:
            raise SimulatorError(f"RECTIFIED_LINEAR shift={shift} is out of range")
        for i in range(count):
            value = int(self.accumulators[accumulator_index + i])
            if rectify:
                value = max(value, 0)
            value >>= shift          # Python's >> on a negative int is already arithmetic
            self.activation_scratchpad[destination_index + i] = clamp_to_int8(value)
        self.cycles += count


def run_memory_image(image: bytes, max_instructions: int = 10_000) -> Machine:
    """Convenience: run a main-memory image to HALT and return the finished machine."""
    machine = Machine(image)
    machine.run(max_instructions)
    return machine
