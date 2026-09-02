"""Stage 1: the PyTorch MLP forward pass."""

import numpy as np
import pytest

from cub.stages import skip_unless_started

pytestmark = pytest.mark.stage(1)
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def model():
    skip_unless_started(1)
    from cub.model import FLOAT_PT, Mlp

    m = Mlp()
    m.load_state_dict(torch.load(FLOAT_PT))
    m.eval()
    return m


def test_forward_shape(model):
    x = torch.zeros(5, 1, 28, 28)
    assert model(x).shape == (5, 10)


def test_matches_numpy_reference(model, golden):
    """Your forward() must compute what the float reference computes, to float precision."""
    from cub.quant import MNIST_MEAN, MNIST_STD

    x = (torch.tensor(golden["images"][:64]).float() / 255.0 - MNIST_MEAN) / MNIST_STD
    with torch.no_grad():
        out = model(x).numpy()
    np.testing.assert_allclose(out, golden["float_logits"][:64], rtol=1e-4, atol=1e-4)


def test_accuracy(model, golden):
    from cub.quant import MNIST_MEAN, MNIST_STD

    x = (torch.tensor(golden["images"]).float() / 255.0 - MNIST_MEAN) / MNIST_STD
    with torch.no_grad():
        pred = model(x).argmax(1).numpy()
    acc = (pred == golden["labels"]).mean()
    assert acc >= 0.96, f"accuracy {acc:.1%}"
