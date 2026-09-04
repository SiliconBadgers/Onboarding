"""The onboarding accelerator stack, in Python.

Everything here is written against docs/02-instruction-set.md. Read that first.

    cub.instruction_set   what an instruction is, and its 128-bit layout   (Stage 2)
    cub.assembler         text <-> instructions                           (Stage 2)
    cub.simulator         a software model of the chip                    (Stage 2)
    cub.model             the PyTorch network and its training script     (Stage 1)
    cub.quantization      decimal numbers -> whole numbers                (Stage 1/3)
    cub.compiler          a quantized network -> a runnable program       (Stage 3)
    cub.program           the main-memory image and the .cub file format  (Stage 3)
    cub.runtime           the host side: quantize, run, read the answer   (Stage 5)
"""

INSTRUCTION_SET_VERSION = 1
