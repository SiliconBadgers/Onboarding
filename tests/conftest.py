import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def golden():
    """artifacts/golden.npz: 1000 test images with the expected result at every layer."""
    with np.load(ROOT / "artifacts" / "golden.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture
def compiled_program():
    """artifacts/mnist.cub, freshly loaded for each test (tests write inputs into it)."""
    from cub.program import Program

    return Program.load(ROOT / "artifacts" / "mnist.cub")


def pytest_configure(config):
    config.addinivalue_line("markers", "stage(n): which onboarding stage a test belongs to")
