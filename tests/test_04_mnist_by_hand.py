"""Stage 4: the hand-written MNIST program runs, and agrees with the compiler."""

import numpy as np
import pytest

from cub.stages import ROOT, skip_unless_started

pytestmark = pytest.mark.stage(4)
SOURCE = ROOT / "programs" / "mnist_by_hand.cubasm"


@pytest.fixture(scope="module")
def hand_written():
    skip_unless_started(4)
    from cub.assembler import assemble

    return assemble(SOURCE.read_text())


def test_ends_with_halt(hand_written):
    assert hand_written[-1].name == "HALT"
    assert len(hand_written) <= 64, "the instruction region only has room for 64"


def test_runs_and_matches_golden(hand_written, compiled_program, golden):
    """Your instructions, the compiler's weights, and the same ten answers."""
    from cub.instruction_set import encode
    from cub.program import Program
    from cub.simulator import Machine

    code = b"".join(encode(i) for i in hand_written)
    for i in range(10):
        compiled_program.write_input(golden["quantized_pixels"][i])
        image = bytearray(compiled_program.image)
        image[0:1024] = bytes(1024)          # wipe the compiler's instructions
        image[0 : len(code)] = code          # ... and drop yours in instead
        machine = Machine(image)
        machine.run()
        out = Program.read_output(machine.main_memory, compiled_program.regions["output"])
        np.testing.assert_array_equal(out, golden["int8_logits"][i], err_msg=f"image {i}")


def test_same_instructions_as_the_compiler(hand_written, compiled_program):
    """Not required for correctness, but a good sign: you and the compiler agree."""
    assert [i.name for i in hand_written] == [i.name for i in compiled_program.instructions]
