"""Stage 3: the compiler turns a trained network into a program plus a memory image."""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(3)


@pytest.fixture(scope="module")
def compiled():
    from python.compiler import compile_from_artifacts

    return compile_from_artifacts()


def test_quantization_matches_golden(golden):
    """The whole-number weights and biases the compiler works from."""
    from python.model import load_trained_weights
    from python.quantization import quantize_model

    weights = load_trained_weights()
    model = quantize_model(
        weights["weights1"], weights["biases1"],
        weights["weights2"], weights["biases2"],
        golden["images"],
    )
    assert model.layers[0].shift == int(golden["shift1"])
    np.testing.assert_array_equal(model.layers[0].weights, golden["weights1"])
    np.testing.assert_array_equal(model.layers[0].biases, golden["biases1"])
    np.testing.assert_array_equal(model.layers[1].weights, golden["weights2"])
    np.testing.assert_array_equal(model.layers[1].biases, golden["biases2"])


def test_instruction_sequence(compiled):
    """Five instructions per layer, plus one LOAD for the image and one HALT."""
    program, _ = compiled
    assert [i.name for i in program.instructions] == [
        "LOAD", "LOAD", "LOAD", "MATRIX_MULTIPLY", "ADD_BIAS", "RECTIFIED_LINEAR",
        "LOAD", "LOAD", "MATRIX_MULTIPLY", "ADD_BIAS", "STORE", "HALT",
    ]


def test_matches_committed_program(compiled, compiled_program):
    program, _ = compiled
    assert program.instructions == compiled_program.instructions
    assert bytes(program.image) == bytes(compiled_program.image)
    assert program.output_scale == pytest.approx(compiled_program.output_scale)


def test_regions_do_not_overlap(compiled):
    program, _ = compiled
    spans = sorted((r.offset, r.end, name) for name, r in program.regions.items())
    for (_, first_end, first_name), (second_start, _, second_name) in zip(spans, spans[1:]):
        assert first_end <= second_start, f"{first_name} overlaps {second_name}"
        assert second_start % 64 == 0, f"{second_name} is not 64-byte aligned"


def test_end_to_end_on_the_simulator(compiled, golden):
    from python.runtime import SimulatorBackend

    program, _ = compiled
    backend = SimulatorBackend()
    for i in range(20):
        out = backend.run(program, golden["quantized_pixels"][i])
        np.testing.assert_array_equal(out, golden["int8_logits"][i], err_msg=f"image {i}")


def test_program_file_roundtrip(compiled, tmp_path):
    from python.program import Program

    program, _ = compiled
    program.save(tmp_path / "roundtrip.cub")
    again = Program.load(tmp_path / "roundtrip.cub")
    assert again.instructions == program.instructions
    assert bytes(again.image) == bytes(program.image)
    assert again.regions["output"] == program.regions["output"]
