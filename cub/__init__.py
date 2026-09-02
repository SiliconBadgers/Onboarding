"""Cub: the onboarding accelerator stack.

Everything here is written against docs/isa.md. Read that first.

    cub.isa       instruction definitions, encode/decode         (Stage 3)
    cub.asm       assembler and disassembler                     (Stage 3)
    cub.sim       the ISA simulator                              (Stage 4)
    cub.quant     float -> INT8 quantization                     (Stage 2)
    cub.model     the PyTorch MLP and its training script        (Stage 1)
    cub.program   the DRAM image and the .cub file format        (Stage 6)
    cub.compiler  quantized weights -> Cub program               (Stage 6)
    cub.runtime   host side: preprocess, run, argmax             (Stage 7)
"""

ISA_VERSION = 1
