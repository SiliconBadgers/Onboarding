"""Stage 6: the compiler."""

import numpy as np
import pytest

from cub.stages import skip_unless_started

pytestmark = pytest.mark.stage(6)


@pytest.fixture(scope="module")
def compiled(golden):
    skip_unless_started(6)
    from cub.compiler import compile_from_artifacts

    return compile_from_artifacts()


def test_instruction_sequence(compiled):
    prog, _ = compiled
    assert [i.name for i in prog.insns] == ["LOAD", "LOAD", "LOAD", "MATMUL", "ADD_BIAS", "RELU",
                                            "LOAD", "LOAD", "MATMUL", "ADD_BIAS", "STORE", "HALT"]


def test_matches_committed_program(compiled, compiled_program):
    prog, _ = compiled
    assert prog.insns == compiled_program.insns
    assert bytes(prog.image) == bytes(compiled_program.image)
    assert prog.output_scale == pytest.approx(compiled_program.output_scale)


def test_regions_do_not_overlap(compiled):
    prog, _ = compiled
    spans = sorted((r.offset, r.end, name) for name, r in prog.regions.items())
    for (a0, a1, na), (b0, b1, nb) in zip(spans, spans[1:]):
        assert a1 <= b0, f"{na} overlaps {nb}"
        assert b0 % 64 == 0, f"{nb} is not 64-byte aligned"


def test_end_to_end_on_simulator(compiled, golden):
    skip_unless_started(4)
    from cub.runtime import SimBackend

    prog, _ = compiled
    be = SimBackend()
    for i in range(20):
        out = be.run(prog, golden["x_q"][i])
        np.testing.assert_array_equal(out, golden["int8_logits"][i], err_msg=f"image {i}")


def test_cub_file_roundtrip(compiled, tmp_path):
    from cub.program import Program

    prog, _ = compiled
    prog.save(tmp_path / "x.cub")
    again = Program.load(tmp_path / "x.cub")
    assert again.insns == prog.insns
    assert bytes(again.image) == bytes(prog.image)
    assert again.regions["output"] == prog.regions["output"]
