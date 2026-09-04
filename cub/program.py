"""The main-memory image and the .cub file format.

A compiled program is not just instructions: it is a picture of everything that must
be sitting in main memory before the core starts. Instructions live at address 0, the
weights and biases after them, then a blank space for the input image and a blank
space for the ten results.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import INSTRUCTION_SET_VERSION, instruction_set
from .instruction_set import Instruction, encode

MAGIC = b"CUB1"
HEADER_BYTES = 64
ALIGNMENT = 64


@dataclass
class Region:
    """A named span of the main-memory image."""

    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class Program:
    """A list of instructions plus the main-memory image they expect to run against."""

    instructions: list[Instruction]
    image: bytearray
    regions: dict[str, Region] = field(default_factory=dict)
    output_scale: float = 1.0

    # --- building ---------------------------------------------------------------

    @classmethod
    def new(cls) -> Program:
        return cls([], bytearray(), {})

    def place(self, name: str, data: bytes | np.ndarray | int) -> Region:
        """Append a region to the image at the next 64-byte-aligned offset.

        `data` may be bytes, a NumPy array (its little-endian bytes are appended), or
        an integer meaning "reserve this many zero bytes" (used for the input image
        and the output logits, which the host fills in later).
        """
        if isinstance(data, int):
            raw = bytes(data)
        elif isinstance(data, np.ndarray):
            raw = data.astype(data.dtype.newbyteorder("<")).tobytes()
        else:
            raw = bytes(data)
        padding = (-len(self.image)) % ALIGNMENT
        self.image.extend(bytes(padding))
        region = Region(len(self.image), len(raw))
        self.image.extend(raw)
        self.regions[name] = region
        return region

    def finalize(self) -> None:
        """Write the encoded instructions into the image at offset 0."""
        code = b"".join(encode(i) for i in self.instructions)
        region = self.regions.get("instructions")
        if region is None or region.length < len(code):
            raise ValueError("reserve the 'instructions' region (with place) before finalize()")
        self.image[region.offset : region.offset + len(code)] = code
        if len(self.image) > instruction_set.MAIN_MEMORY_BYTES:
            raise ValueError(
                f"image is {len(self.image)} bytes, main memory is "
                f"{instruction_set.MAIN_MEMORY_BYTES}"
            )

    # --- host-side access -------------------------------------------------------

    def write_input(self, quantized_pixels: np.ndarray) -> None:
        """Drop one quantized image into the input region of the memory picture."""
        region = self.regions["input"]
        raw = quantized_pixels.astype(np.int8).tobytes()
        if len(raw) != region.length:
            raise ValueError(f"input is {len(raw)} bytes, region is {region.length}")
        self.image[region.offset : region.end] = raw

    @staticmethod
    def read_output(image: bytes | bytearray, region: Region) -> np.ndarray:
        """Read the 32-bit results the program's final STORE left in memory."""
        raw = bytes(image[region.offset : region.end])
        return np.frombuffer(raw, dtype="<i4").astype(np.int32)

    # --- file format ------------------------------------------------------------

    def to_bytes(self) -> bytes:
        input_region, output_region = self.regions["input"], self.regions["output"]
        header = struct.pack(
            "<4sIIIIIIIf",
            MAGIC, INSTRUCTION_SET_VERSION, len(self.instructions), len(self.image),
            input_region.offset, input_region.length,
            output_region.offset, output_region.length,
            self.output_scale,
        )
        header += bytes(HEADER_BYTES - len(header))
        return header + bytes(self.image)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: str | Path) -> Program:
        from .instruction_set import decode

        raw = Path(path).read_bytes()
        (magic, version, instruction_count, image_length,
         input_offset, input_length, output_offset, output_length,
         scale) = struct.unpack_from("<4sIIIIIIIf", raw)
        if magic != MAGIC:
            raise ValueError(f"not a .cub file (magic {magic!r})")
        if version != INSTRUCTION_SET_VERSION:
            raise ValueError(f"instruction set version {version}, expected {INSTRUCTION_SET_VERSION}")
        image = bytearray(raw[HEADER_BYTES : HEADER_BYTES + image_length])
        instructions = [
            decode(bytes(image[i * 16 : i * 16 + 16])) for i in range(instruction_count)
        ]
        regions = {
            "instructions": Region(0, instruction_count * 16),
            "input": Region(input_offset, input_length),
            "output": Region(output_offset, output_length),
        }
        return cls(instructions, image, regions, scale)

    def to_hex(self, pad_to: int = instruction_set.MAIN_MEMORY_BYTES) -> str:
        """One byte per line, for SystemVerilog's $readmemh.

        Used by the hardware testbench. Padded with zeros to the full main-memory size
        so the memory is completely initialized and $readmemh has nothing to warn about.
        """
        data = bytes(self.image) + bytes(max(0, pad_to - len(self.image)))
        return "\n".join(f"{byte:02x}" for byte in data) + "\n"
