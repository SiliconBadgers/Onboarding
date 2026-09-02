"""Stage 2: float -> INT8."""

import numpy as np
import pytest

from cub.stages import skip_unless_started

pytestmark = pytest.mark.stage(2)


@pytest.fixture(autouse=True)
def _started():
    skip_unless_started(2)


def test_quantize_input_extremes():
    from cub.quant import quantize_input

    q = quantize_input(np.array([0, 255], dtype=np.uint8))
    assert q.dtype == np.int8
    assert q[1] == 127, "the brightest pixel must map to exactly 127"
    assert q[0] == -19, "a black pixel is below zero after normalization"


def test_quantize_input_matches_golden(golden):
    from cub.quant import quantize_input

    np.testing.assert_array_equal(quantize_input(golden["images"]), golden["x_q"])


def test_quantize_weights_symmetric():
    from cub.quant import quantize_weights

    w = np.array([[0.5, -1.0], [0.25, 0.0]], dtype=np.float32)
    q, s = quantize_weights(w)
    assert q.dtype == np.int8
    assert s == pytest.approx(127.0)
    np.testing.assert_array_equal(q, [[64, -127], [32, 0]])


def test_choose_shift():
    from cub.quant import choose_shift

    assert choose_shift(127) == 0
    assert choose_shift(128) == 1
    assert choose_shift(255) == 1
    assert choose_shift(256) == 2
    assert choose_shift(503071) == 12


def test_quantized_model_matches_golden(golden):
    from cub.model import load_float_weights
    from cub.quant import quantize_model

    w = load_float_weights()
    qm = quantize_model(w["w1"], w["b1"], w["w2"], w["b2"], golden["images"])
    assert qm.layers[0].shift == int(golden["shift1"])
    np.testing.assert_array_equal(qm.layers[0].w, golden["w1"])
    np.testing.assert_array_equal(qm.layers[0].b, golden["b1"])
    np.testing.assert_array_equal(qm.layers[1].w, golden["w2"])
    np.testing.assert_array_equal(qm.layers[1].b, golden["b2"])


def test_int8_accuracy(golden):
    from cub.model import load_float_weights
    from cub.quant import int8_forward, quantize_input, quantize_model

    w = load_float_weights()
    qm = quantize_model(w["w1"], w["b1"], w["w2"], w["b2"], golden["images"])
    logits = int8_forward(qm, quantize_input(golden["images"]))
    np.testing.assert_array_equal(logits, golden["int8_logits"])
    acc = (logits.argmax(1) == golden["labels"]).mean()
    assert acc >= 0.96, f"INT8 accuracy {acc:.1%}"
