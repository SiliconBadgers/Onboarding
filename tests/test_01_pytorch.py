"""Stage 1: the PyTorch network, and turning its decimal numbers into whole numbers."""

import numpy as np
import pytest

pytestmark = pytest.mark.stage(1)
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def model():
    from cub.model import TRAINED_WEIGHTS_TORCH, MultiLayerPerceptron

    network = MultiLayerPerceptron()
    network.load_state_dict(torch.load(TRAINED_WEIGHTS_TORCH))
    network.eval()
    return network


def test_forward_shape(model):
    """784 numbers in, 10 numbers out, one per digit."""
    images = torch.zeros(5, 1, 28, 28)
    assert model(images).shape == (5, 10)


def test_matches_numpy_reference(model, golden):
    """The two layers and the rectified linear step, in PyTorch and in plain NumPy."""
    from cub.quantization import MNIST_MEAN, MNIST_STANDARD_DEVIATION

    images = (
        torch.tensor(golden["images"][:64]).float() / 255.0 - MNIST_MEAN
    ) / MNIST_STANDARD_DEVIATION
    with torch.no_grad():
        logits = model(images).numpy()
    np.testing.assert_allclose(
        logits, golden["decimal_logits"][:64], rtol=1e-4, atol=1e-4
    )


def test_accuracy(model, golden):
    from cub.quantization import MNIST_MEAN, MNIST_STANDARD_DEVIATION

    images = (
        torch.tensor(golden["images"]).float() / 255.0 - MNIST_MEAN
    ) / MNIST_STANDARD_DEVIATION
    with torch.no_grad():
        predicted = model(images).argmax(1).numpy()
    accuracy = (predicted == golden["labels"]).mean()
    assert accuracy >= 0.96, f"accuracy {accuracy:.1%}"


# --- scaling decimals to whole numbers ---------------------------------------------


def test_brightest_pixel_becomes_127():
    from cub.quantization import quantize_input

    quantized = quantize_input(np.array([0, 255], dtype=np.uint8))
    assert quantized.dtype == np.int8
    assert quantized[1] == 127, "the brightest pixel must map to exactly 127"
    assert quantized[0] == -19, "a black pixel is below zero after normalization"


def test_quantize_weights_is_symmetric():
    """The largest magnitude in the tensor lands on 127, and zero stays zero."""
    from cub.quantization import quantize_weights

    weights = np.array([[0.5, -1.0], [0.25, 0.0]], dtype=np.float32)
    quantized, scale = quantize_weights(weights)
    assert quantized.dtype == np.int8
    assert scale == pytest.approx(127.0)
    np.testing.assert_array_equal(quantized, [[64, -127], [32, 0]])


def test_choose_shift():
    """The smallest right shift that brings an accumulator peak back into 8 bits."""
    from cub.quantization import choose_shift

    assert choose_shift(127) == 0
    assert choose_shift(128) == 1
    assert choose_shift(255) == 1
    assert choose_shift(256) == 2
    assert choose_shift(503071) == 12


def test_whole_number_accuracy(golden):
    """Whole-number inference loses almost nothing against the decimal network."""
    from cub.model import load_trained_weights
    from cub.quantization import int8_forward, quantize_input, quantize_model

    weights = load_trained_weights()
    quantized_model = quantize_model(
        weights["weights1"], weights["biases1"],
        weights["weights2"], weights["biases2"],
        golden["images"],
    )
    logits = int8_forward(quantized_model, quantize_input(golden["images"]))
    np.testing.assert_array_equal(logits, golden["int8_logits"])
    accuracy = (logits.argmax(1) == golden["labels"]).mean()
    assert accuracy >= 0.96, f"whole-number accuracy {accuracy:.1%}"
