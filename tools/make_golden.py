#!/usr/bin/env python3
"""Regenerate artifacts/golden.npz from the float weights and the test slice.

Run after retraining (python -m cub train) and recompiling (python -m cub compile).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cub.compiler import compile_from_artifacts  # noqa: E402
from cub.model import float_forward, load_float_weights, load_test_1k  # noqa: E402
from cub.quant import int8_forward, quantize_input  # noqa: E402
from cub.runtime import SimBackend  # noqa: E402


def main() -> None:
    prog, qm = compile_from_artifacts()
    images, labels = load_test_1k()
    w = load_float_weights()
    x_q = quantize_input(images)
    ref = int8_forward(qm, x_q)
    sim = np.stack([SimBackend().run(prog, x_q[i]) for i in range(len(images))])
    assert (sim == ref).all(), "simulator and NumPy reference disagree"
    np.savez_compressed(
        ROOT / "artifacts" / "golden.npz",
        images=images, labels=labels, x_q=x_q,
        float_logits=float_forward(w, images).astype(np.float32),
        int8_logits=ref.astype(np.int32),
        shift1=np.int32(qm.layers[0].shift),
        output_scale=np.float32(qm.output_scale),
        w1=qm.layers[0].w, b1=qm.layers[0].b, w2=qm.layers[1].w, b2=qm.layers[1].b,
    )
    print(f"golden.npz written; INT8 accuracy {(ref.argmax(1) == labels).mean():.1%}")


if __name__ == "__main__":
    main()
