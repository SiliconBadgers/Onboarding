"""Setup: the environment works. If this fails, nothing else will pass."""

import numpy as np
import pytest


def test_imports():
    import python
    import python.assembler, python.compiler, python.instruction_set  # noqa: E401,F401
    import python.program, python.quantization, python.runtime, python.simulator  # noqa: E401,F401

    assert python.INSTRUCTION_SET_VERSION == 1


def test_artifacts_present(golden, compiled_program):
    assert golden["images"].shape == (1000, 784)
    assert golden["images"].dtype == np.uint8
    assert len(compiled_program.instructions) == 12


def test_pytorch_optional():
    """PyTorch is only needed for Stage 1. Everything else is NumPy."""
    torch = pytest.importorskip("torch")
    assert torch.__version__
