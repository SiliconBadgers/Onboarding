"""The onboarding stages, and which files each stage's blanks live in.

Used by the tests (to skip a stage whose blanks are untouched), by tools/progress.py
(to print where an onboardee is), and by tools/make_blanks.py (to know which comment
syntax a file uses).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# stage number -> (title, guide, files that contain that stage's blanks)
STAGES: dict[int, tuple[str, str, tuple[str, ...]]] = {
    0: ("Setup", "docs/00-setup.md", ()),
    1: ("The PyTorch MLP", "docs/01-pytorch-mlp.md", ("cub/model.py",)),
    2: ("Quantization", "docs/02-quantization.md", ("cub/quant.py",)),
    3: ("Encoding and the assembler", "docs/03-encoding.md", ("cub/isa.py", "cub/asm.py")),
    4: ("The simulator", "docs/04-simulator.md", ("cub/sim.py",)),
    5: ("MNIST by hand", "docs/05-mnist-by-hand.md", ("programs/mnist_by_hand.cubasm",)),
    6: ("The compiler", "docs/06-compiler.md", ("cub/compiler.py",)),
    7: ("The runtime and the demo", "docs/07-runtime.md", ("cub/runtime.py",)),
    8: ("RTL", "docs/08-rtl.md", ("rtl/src/cub_core.sv",)),
    9: ("FPGA", "docs/09-fpga.md", ()),
}

# A stage's tests exercise code from earlier stages too. If a prerequisite is unfilled
# the later stage cannot be checked yet, so its tests skip rather than fail.
DEPENDS: dict[int, tuple[int, ...]] = {
    4: (3,),
    5: (3, 4),
    6: (2, 3, 4),
    7: (2, 4),
    8: (3, 4),
}

TODO_RE = re.compile(r"TODO\(onboard, stage (\d+)\)")


def remaining_blanks(stage: int) -> list[tuple[str, int]]:
    """(file, line number) of every unfilled blank for a stage."""
    out = []
    for rel in STAGES[stage][2]:
        path = ROOT / rel
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            m = TODO_RE.search(line)
            if m and int(m.group(1)) == stage:
                out.append((rel, lineno))
    return out


def started(stage: int) -> bool:
    """True once every blank for the stage has been removed."""
    return not remaining_blanks(stage)


def skip_unless_started(stage: int) -> None:
    """Call at the top of a test: skips it while the stage's blanks are still there."""
    import pytest

    for dep in DEPENDS.get(stage, ()):
        if remaining_blanks(dep):
            pytest.skip(f"stage {stage} needs stage {dep} first")
    blanks = remaining_blanks(stage)
    if blanks:
        where = ", ".join(f"{f}:{n}" for f, n in blanks)
        pytest.skip(f"stage {stage} not started: fill in {where}")
