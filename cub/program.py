"""The DRAM image and the .cub file format (docs/isa.md section 6)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ISA_VERSION, isa
from .isa import Instruction, encode

MAGIC = b"CUB1"
HEADER_BYTES = 64
ALIGN = 64


@dataclass
class Region:
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass
class Program:
    """A list of instructions plus the DRAM image they expect to run against."""

    insns: list[Instruction]
    image: bytearray
    regions: dict[str, Region] = field(default_factory=dict)
    output_scale: float = 1.0

    # --- building ---------------------------------------------------------------

    @classmethod
    def new(cls) -> Program:
        return cls([], bytearray(), {})

    def place(self, name: str, data: bytes | np.ndarray | int) -> Region:
        """Append a region to the image at the next 64-byte-aligned offset.

        `data` may be bytes, a NumPy array (little-endian bytes are appended), or an
        int meaning "reserve this many zero bytes" (used for the input and output).
        """
        if isinstance(data, int):
            raw = bytes(data)
        elif isinstance(data, np.ndarray):
            raw = data.astype(data.dtype.newbyteorder("<")).tobytes()
        else:
            raw = bytes(data)
        pad = (-len(self.image)) % ALIGN
        self.image.extend(bytes(pad))
        r = Region(len(self.image), len(raw))
        self.image.extend(raw)
        self.regions[name] = r
        return r

    def finalize(self) -> None:
        """Write the encoded instructions into the image at offset 0."""
        code = b"".join(encode(i) for i in self.insns)
        r = self.regions.get("insns")
        if r is None or r.length < len(code):
            raise ValueError("reserve the 'insns' region (with place) before finalize()")
        self.image[r.offset : r.offset + len(code)] = code
        if len(self.image) > isa.DRAM_BYTES:
            raise ValueError(f"image is {len(self.image)} bytes, DRAM is {isa.DRAM_BYTES}")

    # --- host-side access -------------------------------------------------------

    def write_input(self, x_q: np.ndarray) -> None:
        r = self.regions["input"]
        raw = x_q.astype(np.int8).tobytes()
        if len(raw) != r.length:
            raise ValueError(f"input is {len(raw)} bytes, region is {r.length}")
        self.image[r.offset : r.end] = raw

    @staticmethod
    def read_output(image: bytes | bytearray, region: Region) -> np.ndarray:
        return np.frombuffer(bytes(image[region.offset : region.end]), dtype="<i4").astype(np.int32)

    # --- file format ------------------------------------------------------------

    def to_bytes(self) -> bytes:
        inp, out = self.regions["input"], self.regions["output"]
        header = struct.pack(
            "<4sIIIIIIIf",
            MAGIC, ISA_VERSION, len(self.insns), len(self.image),
            inp.offset, inp.length, out.offset, out.length, self.output_scale,
        )
        header += bytes(HEADER_BYTES - len(header))
        return header + bytes(self.image)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: str | Path) -> Program:
        from .isa import decode

        raw = Path(path).read_bytes()
        magic, version, n_insns, image_len, in_off, in_len, out_off, out_len, scale = struct.unpack_from("<4sIIIIIIIf", raw)
        if magic != MAGIC:
            raise ValueError(f"not a .cub file (magic {magic!r})")
        if version != ISA_VERSION:
            raise ValueError(f"ISA version {version}, expected {ISA_VERSION}")
        image = bytearray(raw[HEADER_BYTES : HEADER_BYTES + image_len])
        insns = [decode(bytes(image[i * 16 : i * 16 + 16])) for i in range(n_insns)]
        regions = {
            "insns": Region(0, n_insns * 16),
            "input": Region(in_off, in_len),
            "output": Region(out_off, out_len),
        }
        return cls(insns, image, regions, scale)

    def to_hex(self, pad_to: int = isa.DRAM_BYTES) -> str:
        """One byte per line, for Verilog $readmemh. Used by the RTL testbench and the FPGA build.

        Padded with zeros to the full DRAM size so the memory is completely
        initialized and $readmemh has nothing to warn about.
        """
        data = bytes(self.image) + bytes(max(0, pad_to - len(self.image)))
        return "\n".join(f"{b:02x}" for b in data) + "\n"
