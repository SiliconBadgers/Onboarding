"""The onboarding accelerator stack, in Python.

Everything here is written against docs/02-instruction-set.md. Read that first.

    python.instruction_set   what an instruction is, and its 128-bit layout   (Stage 2)
    python.assembler         text <-> instructions                           (Stage 2)
    python.simulator         a software model of the chip                    (Stage 2)
    python.model             the PyTorch network and its training script     (Stage 1)
    python.quantization      decimal numbers -> whole numbers                (Stage 1/3)
    python.compiler          a quantized network -> a runnable program       (Stage 3)
    python.program           the main-memory image and the .cub file format  (Stage 3)
    python.runtime           the host side: quantize, run, read the answer   (Stage 5)
"""

INSTRUCTION_SET_VERSION = 1
