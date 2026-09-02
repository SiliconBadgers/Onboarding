"""Command line: python -m cub <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def cmd_train(args):
    from .model import train

    train(epochs=args.epochs)


def cmd_asm(args):
    from .asm import assemble_bytes

    out = assemble_bytes(Path(args.src).read_text())
    Path(args.out).write_bytes(out)
    print(f"{len(out) // 16} instructions -> {args.out}")


def cmd_disasm(args):
    from .asm import disassemble
    from .program import Program

    p = Path(args.src)
    if p.suffix == ".cub":
        prog = Program.load(p)
        raw = b"".join(prog.image[i * 16 : i * 16 + 16] for i in range(len(prog.insns)))
    else:
        raw = p.read_bytes()
    sys.stdout.write(disassemble(raw))


def cmd_compile(args):
    from .compiler import compile_from_artifacts

    prog, model = compile_from_artifacts()
    prog.save(args.out)
    print(f"{len(prog.insns)} instructions, {len(prog.image)} byte image, shift1={model.layers[0].shift} -> {args.out}")
    for name, r in prog.regions.items():
        print(f"  {name:<8} 0x{r.offset:05X} .. 0x{r.end:05X}  ({r.length} bytes)")


def cmd_run(args):
    from .model import load_test_1k
    from .runtime import ascii_digit, load_program, predict

    prog = load_program(args.program)
    images, labels = load_test_1k()
    i = args.index
    digit, logits = predict(prog, images[i])
    print(ascii_digit(images[i]))
    print(f"label {labels[i]}  predicted {digit}")
    print("logits", np.round(logits, 2))


def cmd_hex(args):
    """Bake one test image into the program and write the DRAM image as hex for the RTL."""
    from .model import load_test_1k
    from .quant import quantize_input
    from .runtime import load_program

    prog = load_program(args.program)
    images, labels = load_test_1k()
    prog.write_input(quantize_input(images[args.index]))
    Path(args.out).write_text(prog.to_hex())
    print(f"image {args.index} (label {labels[args.index]}) baked in -> {args.out}")


def cmd_eval(args):
    from .model import load_test_1k
    from .runtime import load_program, predict

    prog = load_program(args.program)
    images, labels = load_test_1k()
    n = args.count
    correct = sum(predict(prog, images[i])[0] == labels[i] for i in range(n))
    print(f"{correct}/{n} correct ({correct / n:.1%}) on the simulator")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cub")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train the float MLP and write artifacts/")
    p.add_argument("--epochs", type=int, default=3)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("asm", help="assemble a .cubasm file to raw instruction bytes")
    p.add_argument("src")
    p.add_argument("-o", "--out", default="out.bin")
    p.set_defaults(fn=cmd_asm)

    p = sub.add_parser("disasm", help="disassemble raw instruction bytes or a .cub program")
    p.add_argument("src")
    p.set_defaults(fn=cmd_disasm)

    p = sub.add_parser("compile", help="compile the trained MLP to artifacts/mnist.cub")
    p.add_argument("-o", "--out", default="artifacts/mnist.cub")
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("run", help="classify one test image on the simulator")
    p.add_argument("--program", default="artifacts/mnist.cub")
    p.add_argument("--index", type=int, default=0)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("eval", help="accuracy of the compiled program on the simulator")
    p.add_argument("--program", default="artifacts/mnist.cub")
    p.add_argument("--count", type=int, default=200)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("hex", help="write a DRAM image with a test image baked in, for the RTL")
    p.add_argument("--program", default="artifacts/mnist.cub")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("-o", "--out", default="rtl/build/dram.hex")
    p.set_defaults(fn=cmd_hex)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
