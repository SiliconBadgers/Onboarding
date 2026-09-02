# Stage 1 — The PyTorch MLP

**Goal:** understand exactly what computation the accelerator has to do, by writing
it in PyTorch first.

**File:** `cub/model.py`. **Test:** `pytest tests/test_01_pytorch_mlp.py`.

## The network

```
input   784  (a 28x28 grayscale image, flattened)
fc1     784 -> 128     y1 = W1 x + b1
relu                   h  = max(y1, 0)
fc2     128 -> 10      y2 = W2 h + b2
argmax                 digit = index of the largest y2
```

Two matrix-vector products, one bias add each, one ReLU between them. That is the
whole thing. Every stage after this one is about doing those five operations in
integers, then in hardware.

Things deliberately *not* in the network, and why:

- **No Softmax.** Softmax is monotonic, so it never changes which output is largest.
  The host takes argmax on raw logits and gets the same answer for free.
- **No Dropout or BatchNorm.** They would need to be folded away before compiling.
  Real models have them; Badger's compiler will have to deal with them; Cub does not.
- **No convolutions.** An MLP is the smallest network that is still a real network.
  Convolutions are what Badger adds (as `IM2COL` + `MATMUL`, see `docs/isa.md`
  section 9).

## Your task

Open `cub/model.py` and find `TODO(onboard, stage 1)` inside `Mlp.forward`. The input
has already been flattened to `(N, 784)`. Apply `fc1`, then ReLU, then `fc2`, and
return the logits. Two lines.

`torch.relu` is a function; `self.fc1` is a module you call like a function.

## Check

```bash
pytest tests/test_01_pytorch_mlp.py -v
```

The tests load the committed trained weights into *your* `Mlp` and check that it
reproduces the reference logits and gets at least 96% on a 1000-image slice of the
test set. If the shapes are right but the numbers are wrong, you have the layers in
the wrong order or forgot the ReLU.

## Train it yourself

Not required for the test, but do it once so you have seen it:

```bash
python -m cub train
```

Three epochs on CPU take about a minute and reach roughly 97.3%. This overwrites
`artifacts/mlp_float.*` and `artifacts/mnist_test_1k.npz`. **Do not commit those
files.** Every golden value downstream was generated from the committed weights, and
retraining changes all of them. Run `git checkout artifacts/` when you are done.

## Milestone video

Under two minutes, screen recording:

1. `pytest tests/test_01_pytorch_mlp.py -v` passing.
2. Open `cub/model.py` and explain, in your own words, what each of the two `Linear`
   layers does to the shape of the data.
3. Say why there is no Softmax.

Submit at **[INSERT_GOOGLE_FORM_LINK_HERE]**.

## Read next

`docs/02-quantization.md`. Before you do, look at `artifacts/mlp_float.npz` and answer
for yourself: what is the largest weight magnitude in `w1`? (`np.abs(w["w1"]).max()`)
Stage 2 is about that number.
