#!/usr/bin/env python3
"""Where is this branch in the onboarding track?

For each stage: are the blanks still there, and do the tests pass? Prints a table and
exits 1 if any *started* stage has failing tests (CI uses this so that an unfilled
stage is not an error but a wrongly filled one is).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from python.stages import STAGES, remaining_blanks  # noqa: E402


def run_stage_tests(stage: int) -> str:
    pattern = f"tests/test_{stage:02d}_*.py"
    files = sorted(ROOT.glob(pattern))
    if not files:
        return "no tests"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, files)],
                       cwd=ROOT, capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-80:]
    if r.returncode == 0:
        return "PASS" if "passed" in last else "SKIPPED"
    return f"FAIL ({last})"


def main() -> int:
    failed = False
    print(f"{'stage':<6}{'title':<30}{'blanks left':<13}{'tests'}")
    for stage, (title, _guide, _files) in STAGES.items():
        blanks = remaining_blanks(stage)
        result = run_stage_tests(stage)
        status = str(len(blanks))
        if result.startswith("FAIL"):
            failed = True
        print(f"{stage:<6}{title:<30}{status:<13}{result}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
