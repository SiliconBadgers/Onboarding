# Stage 7 — The runtime and the demo

**Goal:** an image goes in, a digit comes out, and you can explain every step.

**File:** `cub/runtime.py`. **Test:** `pytest tests/test_07_runtime.py`.

## The host boundary

The core does integer arithmetic between the first `LOAD` and the last `STORE`.
Everything else is the host's job, in Python:

1. Normalize and quantize the image (Stage 2's `quantize_input`).
2. Write it into the program's input region.
3. Start the core and wait for `done`.
4. Read the ten INT32 logits.
5. Divide by `output_scale` to get real-valued logits (only for display), and take
   the argmax.

Step 3 is behind a `backend` object with one method, `run(prog, x_q) -> logits`.
`SimBackend` runs the simulator. In Stage 9 the same `predict` function will drive the
FPGA through a different backend, and nothing else changes. That is the point of
drawing the boundary here.

## Your task

`predict`: quantize, run the backend, dequantize with `prog.output_scale`, argmax.
Four lines.

## Check

```bash
pytest tests/test_07_runtime.py -v
python -m cub run --index 12
python -m cub eval --count 500
```

## Milestone video

This is the software-track finale. Under two minutes:

1. `pytest` with every stage 0 to 7 passing (no skips before stage 8).
2. `python -m cub run` on two or three images.
3. In one sentence each, what Stages 2, 4, and 6 did.

Submit at **[INSERT_GOOGLE_FORM_LINK_HERE]**.

## Where you are now

You have a trained model, a quantizer, an ISA, an assembler, a simulator, a compiler,
and a runtime, and they all agree. That is the whole software half of the main
project, at one tenth the size. Read `docs/isa.md` section 9 again; every row should
now make sense.

If you are on the hardware track, continue to `docs/08-rtl.md`. If not, you are done
with the track. Ask your mentor for a first task on
[AI-Inference-Chip](https://github.com/SiliconBadgers/AI-Inference-Chip).
