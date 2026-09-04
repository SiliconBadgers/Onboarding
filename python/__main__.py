"""Command line: python -m python <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def command_train(args):
    from .model import train

    train(epochs=args.epochs)


def command_assemble(args):
    from .assembler import assemble_bytes

    out = assemble_bytes(Path(args.source).read_text())
    Path(args.out).write_bytes(out)
    print(f"{len(out) // 16} instructions -> {args.out}")


def command_disassemble(args):
    from .assembler import disassemble
    from .program import Program

    path = Path(args.source)
    if path.suffix == ".cub":
        program = Program.load(path)
        raw = b"".join(
            program.image[i * 16 : i * 16 + 16] for i in range(len(program.instructions))
        )
    else:
        raw = path.read_bytes()
    sys.stdout.write(disassemble(raw))


def command_compile(args):
    from .compiler import compile_from_artifacts

    program, model = compile_from_artifacts()
    program.save(args.out)
    print(
        f"{len(program.instructions)} instructions, {len(program.image)} byte image, "
        f"layer-1 shift {model.layers[0].shift} -> {args.out}"
    )
    for name, region in program.regions.items():
        print(f"  {name:<13} 0x{region.offset:05X} .. 0x{region.end:05X}  "
              f"({region.length} bytes)")


def command_run(args):
    from .model import load_test_images
    from .runtime import ascii_digit, load_program, predict

    program = load_program(args.program)
    images, labels = load_test_images()
    index = args.index
    digit, logits = predict(program, images[index])
    print(ascii_digit(images[index]))
    print(f"label {labels[index]}  predicted {digit}")
    print("logits", np.round(logits, 2))


def command_memory_image(args):
    """Bake one test image into the program and write main memory as hex for hardware."""
    from .model import load_test_images
    from .quantization import quantize_input
    from .runtime import load_program

    program = load_program(args.program)
    images, labels = load_test_images()
    program.write_input(quantize_input(images[args.index]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(program.to_hex())
    print(f"image {args.index} (label {labels[args.index]}) baked in -> {args.out}")


def command_accuracy(args):
    from .model import load_test_images
    from .runtime import load_program, predict

    program = load_program(args.program)
    images, labels = load_test_images()
    count = args.count
    correct = sum(predict(program, images[i])[0] == labels[i] for i in range(count))
    print(f"{correct}/{count} correct ({correct / count:.1%}) on the simulator")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python")
    subcommands = parser.add_subparsers(dest="command", required=True)

    sub = subcommands.add_parser("train", help="retrain the network and rewrite artifacts/")
    sub.add_argument("--epochs", type=int, default=3)
    sub.set_defaults(run=command_train)

    sub = subcommands.add_parser("assemble", help="assemble a .cubasm file to instruction bytes")
    sub.add_argument("source")
    sub.add_argument("-o", "--out", default="out.bin")
    sub.set_defaults(run=command_assemble)

    sub = subcommands.add_parser("disassemble", help="turn instruction bytes or a .cub program back into text")
    sub.add_argument("source")
    sub.set_defaults(run=command_disassemble)

    sub = subcommands.add_parser("compile", help="compile the trained network to artifacts/mnist.cub")
    sub.add_argument("-o", "--out", default="artifacts/mnist.cub")
    sub.set_defaults(run=command_compile)

    sub = subcommands.add_parser("run", help="classify one test image on the simulator")
    sub.add_argument("--program", default="artifacts/mnist.cub")
    sub.add_argument("--index", type=int, default=0)
    sub.set_defaults(run=command_run)

    sub = subcommands.add_parser("accuracy", help="accuracy of the compiled program on the simulator")
    sub.add_argument("--program", default="artifacts/mnist.cub")
    sub.add_argument("--count", type=int, default=200)
    sub.set_defaults(run=command_accuracy)

    sub = subcommands.add_parser(
        "memory-image", help="write a main-memory image with a test image baked in, for the hardware"
    )
    sub.add_argument("--program", default="artifacts/mnist.cub")
    sub.add_argument("--index", type=int, default=0)
    sub.add_argument("-o", "--out", default="rtl/build/main_memory.hex")
    sub.set_defaults(run=command_memory_image)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
