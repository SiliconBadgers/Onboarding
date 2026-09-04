"""Stage 5: talking to the chip.

The host writes the memory image, writes the input, starts the core, waits for it to
finish, and reads the answer back out of main memory. These tests walk that round trip.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(5)


def test_input_region_round_trip(compiled_program, golden):
    """Writing the input region is how an image gets onto the chip."""
    from cub.quantization import quantize_input

    region = compiled_program.regions["input"]
    assert region.length == 784, "one byte per pixel"

    quantized = quantize_input(golden["images"][0])
    compiled_program.write_input(quantized)
    written = np.frombuffer(
        bytes(compiled_program.image[region.offset : region.end]), dtype=np.int8
    )
    np.testing.assert_array_equal(written, quantized)


def test_output_region_holds_32_bit_logits(compiled_program, golden):
    """The program's last STORE leaves ten 32-bit values for the host to read."""
    from cub.program import Program
    from cub.simulator import Machine

    region = compiled_program.regions["output"]
    assert region.length == 40, "ten values, four bytes each"

    compiled_program.write_input(golden["quantized_pixels"][0])
    machine = Machine(compiled_program.image)
    machine.run()
    assert machine.halted, "the core raises done only after HALT"
    logits = Program.read_output(machine.main_memory, region)
    np.testing.assert_array_equal(logits, golden["int8_logits"][0])


def test_memory_spaces_have_the_documented_sizes():
    """The five memory spaces in docs/05-registers-and-memory.md."""
    from cub import instruction_set
    from cub.simulator import Machine

    machine = Machine()
    assert len(machine.main_memory) == instruction_set.MAIN_MEMORY_BYTES == 256 * 1024
    assert machine.activation_scratchpad.shape == (4096,)
    assert machine.weight_scratchpad.shape == (131072,)
    assert machine.bias_scratchpad.shape == (256,)
    assert machine.accumulators.shape == (256,)
    assert machine.activation_scratchpad.dtype == np.int8
    assert machine.weight_scratchpad.dtype == np.int8
    assert machine.bias_scratchpad.dtype == np.int32
    assert machine.accumulators.dtype == np.int32


def test_scratchpads_only_hold_what_was_loaded():
    """There is no cache. If the program does not LOAD it, it is not there."""
    from cub.simulator import Machine

    machine = Machine()
    assert machine.weight_scratchpad.sum() == 0
    assert machine.accumulators.sum() == 0


def test_predict_matches_golden(compiled_program, golden):
    from cub.runtime import predict

    for i in range(20):
        digit, logits = predict(compiled_program, golden["images"][i])
        assert digit == int(golden["int8_logits"][i].argmax())
        np.testing.assert_allclose(
            logits, golden["int8_logits"][i] / golden["output_scale"], rtol=1e-5
        )


def test_accuracy_on_the_simulator(compiled_program, golden):
    from cub.runtime import predict

    count = 100
    correct = sum(
        predict(compiled_program, golden["images"][i])[0] == golden["labels"][i]
        for i in range(count)
    )
    assert correct / count >= 0.95


def test_command_line_run(capsys):
    from cub.__main__ import main

    main(["run", "--index", "0"])
    assert "predicted 7" in capsys.readouterr().out
