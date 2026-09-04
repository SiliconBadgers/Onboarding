"""The reference network (Stage 1). A test fixture, not part of the accelerator.

    784 -> 128 -> rectified linear -> 10

Every operation here is one the accelerator can run: two fully connected layers and
one rectified linear step. No batch normalization, no dropout, no softmax. The host
takes the argmax, and it does not need softmax because softmax never changes which
output is largest.

Retrain it with:   python -m python train
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
TRAINED_WEIGHTS_TORCH = ARTIFACTS / "trained_weights.pt"
TRAINED_WEIGHTS_NUMPY = ARTIFACTS / "trained_weights.npz"
TEST_IMAGES = ARTIFACTS / "mnist_test_1000.npz"

INPUT_FEATURES = 28 * 28
HIDDEN_UNITS = 128
OUTPUT_CLASSES = 10

try:
    import torch
    import torch.nn as neural_network
except ImportError:      # the simulator, compiler and hardware never need PyTorch
    torch = None
    neural_network = None


if neural_network is not None:

    class MultiLayerPerceptron(neural_network.Module):
        """Two fully connected layers with a rectified linear step between them."""

        def __init__(self) -> None:
            super().__init__()
            self.hidden_layer = neural_network.Linear(INPUT_FEATURES, HIDDEN_UNITS)
            self.output_layer = neural_network.Linear(HIDDEN_UNITS, OUTPUT_CLASSES)

        def forward(self, images):
            # images arrive as (N, 1, 28, 28) from the data loader, or (N, 784) flat.
            flattened = images.reshape(images.shape[0], -1)
            hidden = torch.relu(self.hidden_layer(flattened))
            return self.output_layer(hidden)


def load_trained_weights(path: Path = TRAINED_WEIGHTS_NUMPY) -> dict[str, np.ndarray]:
    """The trained weights as NumPy arrays.

    weights1 (128, 784), biases1 (128,), weights2 (10, 128), biases2 (10,)
    """
    with np.load(path) as archive:
        return {k: archive[k] for k in ("weights1", "biases1", "weights2", "biases2")}


def load_test_images(path: Path = TEST_IMAGES) -> tuple[np.ndarray, np.ndarray]:
    """1000 MNIST test images as 8-bit pixels (1000, 784) and their labels (1000,)."""
    with np.load(path) as archive:
        return archive["images"], archive["labels"]


def decimal_forward(weights: dict[str, np.ndarray], pixels: np.ndarray) -> np.ndarray:
    """A NumPy forward pass in decimal numbers, for tests that must not need PyTorch."""
    from .quantization import MNIST_MEAN, MNIST_STANDARD_DEVIATION

    normalized = (
        pixels.astype(np.float32) / 255.0 - MNIST_MEAN
    ) / MNIST_STANDARD_DEVIATION
    hidden = np.maximum(normalized @ weights["weights1"].T + weights["biases1"], 0)
    return hidden @ weights["weights2"].T + weights["biases2"]


def train(epochs: int = 3, seed: int = 0, data_dir: str = "data") -> None:
    """Train the network on MNIST and rewrite the files in artifacts/.

    Retraining changes every value downstream, so only do it deliberately.
    """
    import torchvision
    from torchvision import transforms

    from .quantization import MNIST_MEAN, MNIST_STANDARD_DEVIATION

    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STANDARD_DEVIATION,)),
    ])
    train_data = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    train_batches = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_batches = torch.utils.data.DataLoader(test_data, batch_size=1000)

    model = MultiLayerPerceptron()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = neural_network.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for images, labels in train_batches:
            optimizer.zero_grad()
            loss_function(model(images), labels).backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_batches:
                correct += (model(images).argmax(1) == labels).sum().item()
        print(f"epoch {epoch + 1}: test accuracy {correct / len(test_data):.2%}")

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), TRAINED_WEIGHTS_TORCH)
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(
        TRAINED_WEIGHTS_NUMPY,
        weights1=state["hidden_layer.weight"], biases1=state["hidden_layer.bias"],
        weights2=state["output_layer.weight"], biases2=state["output_layer.bias"],
    )

    # A fixed 1000-image slice of the test set, as raw 8-bit pixels, so every later
    # stage can be tested without downloading MNIST.
    raw = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    images = raw.data[:1000].numpy().reshape(1000, -1).astype(np.uint8)
    labels = raw.targets[:1000].numpy().astype(np.int64)
    np.savez_compressed(TEST_IMAGES, images=images, labels=labels)
    print(
        f"wrote {TRAINED_WEIGHTS_TORCH.name}, {TRAINED_WEIGHTS_NUMPY.name}, "
        f"{TEST_IMAGES.name}"
    )


if __name__ == "__main__":
    train()
