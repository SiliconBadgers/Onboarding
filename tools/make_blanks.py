#!/usr/bin/env python3
"""Turn solution code into fill-in-the-blank code.

Maintainers write the full solution between marker comments:

    # --- SOLUTION(stage=6): one-line hint shown to the onboardee ---
    ...the real code...
    # --- END SOLUTION ---

This tool replaces every such block with a single line at the same indentation:

    # TODO(onboard, stage 6): one-line hint shown to the onboardee

followed, in Python files, by a `raise NotImplementedError(...)` so the blank
fails loudly rather than silently returning None. SystemVerilog (`//`) and assembly
(`;`) comments get only the TODO line, since neither has anything to raise.

As it stands only Stage 4 (programs/mnist_by_hand.cubasm) and Stage 6
(rtl/src/cub_core.sv) have solution blocks; the Python stages are for reading.

Usage:
    tools/make_blanks.py --check           list every marker block (exit 1 if any exist)
    tools/make_blanks.py --apply           rewrite the files in place (used by tools/sync_main.sh)
    tools/make_blanks.py --apply FILE...   rewrite only these files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cub.stages import STAGES  # noqa: E402

START = re.compile(r"^(?P<indent>\s*)(?P<c>#|//|;)\s*--- SOLUTION\(stage=(?P<stage>\d+)\)(?::\s*(?P<hint>.*?))?\s*---\s*$")
END = re.compile(r"^\s*(#|//|;)\s*--- END SOLUTION ---\s*$")


def blank_text(text: str, path: Path) -> tuple[str, int]:
    out, n, i = [], 0, 0
    lines = text.splitlines(keepends=True)
    while i < len(lines):
        m = START.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not END.match(lines[j]):
            j += 1
        if j == len(lines):
            raise SystemExit(f"{path}:{i + 1}: SOLUTION block without END SOLUTION")
        indent, c, stage, hint = m["indent"], m["c"], int(m["stage"]), (m["hint"] or "").strip()
        out.append(f"{indent}{c} TODO(onboard, stage {stage}): {hint}\n")
        if c == "#":
            out.append(f'{indent}raise NotImplementedError("stage {stage} blank, see {path.name}:{i + 1}")\n')
        n += 1
        i = j + 1
    return "".join(out), n


def all_solution_files() -> list[Path]:
    return [ROOT / f for _, _, files in STAGES.values() for f in files]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    files = [Path(f) for f in args.files] or all_solution_files()

    total = 0
    for path in files:
        if not path.exists():
            continue
        text = path.read_text()
        new, n = blank_text(text, path)
        if n == 0:
            continue
        total += n
        rel = path.relative_to(ROOT) if path.is_absolute() else path
        if args.check:
            print(f"{rel}: {n} solution block(s)")
        else:
            path.write_text(new)
            print(f"{rel}: blanked {n} block(s)")
    if args.check:
        print("no solution blocks found" if total == 0 else f"{total} solution block(s) present")
        return 1 if total else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
