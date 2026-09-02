"""Stage 5: the hand-written MNIST program runs, and matches the compiler."""

import numpy as np
import pytest

from cub.stages import ROOT, skip_unless_started

pytestmark = pytest.mark.stage(5)
SRC = ROOT / "programs" / "mnist_by_hand.cubasm"


@pytest.fixture(scope="module")
def hand_insns():
    skip_unless_started(5)
    skip_unless_started(4)   # the simulator has to work to check this stage
    from cub.asm import assemble

    return assemble(SRC.read_text())


def test_ends_with_halt(hand_insns):
    assert hand_insns[-1].name == "HALT"
    assert len(hand_insns) <= 64


def test_runs_and_matches_golden(hand_insns, compiled_program, golden):
    from cub.isa import encode
    from cub.program import Program
    from cub.sim import Machine

    code = b"".join(encode(i) for i in hand_insns)
    for i in range(10):
        compiled_program.write_input(golden["x_q"][i])
        image = bytearray(compiled_program.image)
        image[0:1024] = bytes(1024)
        image[0 : len(code)] = code
        m = Machine(image)
        m.run()
        out = Program.read_output(m.dram, compiled_program.regions["output"])
        np.testing.assert_array_equal(out, golden["int8_logits"][i], err_msg=f"image {i}")


def test_same_instructions_as_compiler(hand_insns, compiled_program):
    """Not required for correctness, but a nice check: you and the compiler agree."""
    assert [i.name for i in hand_insns] == [i.name for i in compiled_program.insns]
