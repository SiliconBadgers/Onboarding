"""Stage 0: the environment works. If this fails, nothing else will pass."""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(0)


def test_imports():
    import cub
    import cub.asm, cub.compiler, cub.isa, cub.program, cub.quant, cub.runtime, cub.sim  # noqa: E401,F401

    assert cub.ISA_VERSION == 1


def test_artifacts_present(golden, compiled_program):
    assert golden["images"].shape == (1000, 784)
    assert golden["images"].dtype == np.uint8
    assert len(compiled_program.insns) == 12


def test_torch_optional():
    """torch is only needed for Stage 1. Everything else is NumPy."""
    torch = pytest.importorskip("torch")
    assert torch.__version__
