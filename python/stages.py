"""The onboarding stages, and which file each stage's blanks live in.

Used by the tests (to skip a stage whose blanks are untouched), by tools/progress.py
(to print where someone is), and by tools/make_blanks.py (to know which comment syntax
a file uses).

Only two stages have anything to fill in. The other four are reading stages: their
tests check that the repository is intact and that the code does what the guide says
it does, and they always run.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# stage number -> (title, guide, files that contain that stage's blanks)
STAGES: dict[int, tuple[str, str, tuple[str, ...]]] = {
    1: ("PyTorch and whole numbers", "docs/01-pytorch.md", ()),
    2: ("The instruction set", "docs/02-instruction-set.md", ()),
    3: ("The compiler", "docs/03-compiler.md", ()),
    4: ("MNIST by hand", "docs/04-mnist-by-hand.md", ("programs/mnist_by_hand.cubasm",)),
    5: ("Talking to the chip", "docs/05-registers-and-memory.md", ()),
    6: ("The chip", "docs/06-rtl.md", ("rtl/cub_core.sv",)),
}

TODO_PATTERN = re.compile(r"TODO\(onboard, stage (\d+)\)")


def remaining_blanks(stage: int) -> list[tuple[str, int]]:
    """(file, line number) of every unfilled blank for a stage."""
    out = []
    for relative_path in STAGES[stage][2]:
        path = ROOT / relative_path
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            match = TODO_PATTERN.search(line)
            if match and int(match.group(1)) == stage:
                out.append((relative_path, line_number))
    return out


def started(stage: int) -> bool:
    """True once every blank for the stage has been removed."""
    return not remaining_blanks(stage)


def skip_unless_started(stage: int) -> None:
    """Call at the top of a test: skips it while the stage's blanks are still there."""
    import pytest

    blanks = remaining_blanks(stage)
    if blanks:
        where = ", ".join(f"{path}:{line}" for path, line in blanks)
        pytest.skip(f"stage {stage} not started: fill in {where}")
