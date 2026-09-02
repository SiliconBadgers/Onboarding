"""Stage 7: image in, digit out."""

import numpy as np
import pytest

from cub.stages import skip_unless_started

pytestmark = pytest.mark.stage(7)


@pytest.fixture(autouse=True)
def _started():
    skip_unless_started(7)
    skip_unless_started(4)


def test_predict_matches_golden(compiled_program, golden):
    from cub.runtime import predict

    for i in range(20):
        digit, logits = predict(compiled_program, golden["images"][i])
        assert digit == int(golden["int8_logits"][i].argmax())
        np.testing.assert_allclose(logits, golden["int8_logits"][i] / golden["output_scale"], rtol=1e-5)


def test_accuracy_on_simulator(compiled_program, golden):
    from cub.runtime import predict

    n = 100
    correct = sum(predict(compiled_program, golden["images"][i])[0] == golden["labels"][i] for i in range(n))
    assert correct / n >= 0.95


def test_cli_run_smoke(capsys):
    from cub.__main__ import main

    main(["run", "--index", "0"])
    out = capsys.readouterr().out
    assert "predicted 7" in out
