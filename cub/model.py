"""The reference MLP (Stage 1). A test fixture, not part of the accelerator.

    784 -> 128 -> ReLU -> 10

Every operation here is one the Cub ISA can run: two nn.Linear layers and one ReLU.
No BatchNorm, no Dropout, no Softmax. Argmax happens on the host, and it does not
need Softmax because Softmax never changes which logit is largest.

Train it with:   python -m cub train
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FLOAT_PT = ARTIFACTS / "mlp_float.pt"
FLOAT_NPZ = ARTIFACTS / "mlp_float.npz"
TEST_1K = ARTIFACTS / "mnist_test_1k.npz"

IN_FEATURES = 28 * 28
HIDDEN = 128
CLASSES = 10

try:
    import torch
    import torch.nn as nn
except ImportError:      # the simulator and compiler never need torch
    torch = None
    nn = None


if nn is not None:

    class Mlp(nn.Module):
        """Two fully connected layers with a ReLU between them."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(IN_FEATURES, HIDDEN)
            self.fc2 = nn.Linear(HIDDEN, CLASSES)

        def forward(self, x):
            # x arrives as (N, 1, 28, 28) from the data loader, or (N, 784) already flat.
            x = x.reshape(x.shape[0], -1)
            # --- SOLUTION(stage=1): fc1, then relu, then fc2. Return the logits. ---
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
            # --- END SOLUTION ---


def load_float_weights(path: Path = FLOAT_NPZ) -> dict[str, np.ndarray]:
    """The trained float weights as NumPy arrays: w1 (128,784), b1 (128,), w2 (10,128), b2 (10,)."""
    with np.load(path) as z:
        return {k: z[k] for k in ("w1", "b1", "w2", "b2")}


def load_test_1k(path: Path = TEST_1K) -> tuple[np.ndarray, np.ndarray]:
    """1000 MNIST test images as uint8 (1000, 784) and their labels (1000,)."""
    with np.load(path) as z:
        return z["images"], z["labels"]


def float_forward(w: dict[str, np.ndarray], pixels: np.ndarray) -> np.ndarray:
    """NumPy float32 forward pass, for tests that must not depend on torch."""
    from .quant import MNIST_MEAN, MNIST_STD

    x = (pixels.astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD
    h = np.maximum(x @ w["w1"].T + w["b1"], 0)
    return h @ w["w2"].T + w["b2"]


def train(epochs: int = 3, seed: int = 0, data_dir: str = "data") -> None:
    """Train the MLP on MNIST and write artifacts/mlp_float.{pt,npz} and mnist_test_1k.npz."""
    import torchvision
    from torchvision import transforms

    from .quant import MNIST_MEAN, MNIST_STD

    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    train_ds = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    model = Mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x).argmax(1) == y).sum().item()
        print(f"epoch {epoch + 1}: test accuracy {correct / len(test_ds):.2%}")

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), FLOAT_PT)
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(FLOAT_NPZ, w1=sd["fc1.weight"], b1=sd["fc1.bias"], w2=sd["fc2.weight"], b2=sd["fc2.bias"])

    # A fixed 1000-image slice of the test set, as raw uint8 pixels, so every later
    # stage can test without downloading MNIST.
    raw = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    images = raw.data[:1000].numpy().reshape(1000, -1).astype(np.uint8)
    labels = raw.targets[:1000].numpy().astype(np.int64)
    np.savez_compressed(TEST_1K, images=images, labels=labels)
    print(f"wrote {FLOAT_PT.name}, {FLOAT_NPZ.name}, {TEST_1K.name}")


if __name__ == "__main__":
    train()
