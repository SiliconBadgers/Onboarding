# Maintaining the track

This file is for people who maintain the onboarding repository, not for onboardees.

## Two branches

- **`solutions`** holds the complete, working code. Every test passes here. This is
  where you edit code, docs, tests, and guides. Keep it private if the organization
  allows it (a private mirror is fine); if it must be public, that is acceptable.
  Onboardees who peek only cheat themselves.
- **`main`** is *generated* from `solutions` by `tools/sync_main.sh`. Never edit it by
  hand. Onboardees branch from it.

The generator strips every solution block into a TODO. A block looks like this in
Python, and the same with `//` in SystemVerilog or `;` in assembly:

```python
# --- SOLUTION(stage=4): one-line hint the onboardee will see ---
for i in range(count):
    ...
# --- END SOLUTION ---
```

and becomes:

```python
# TODO(onboard, stage 4): one-line hint the onboardee will see
raise NotImplementedError("stage 4 blank, see sim.py:141")
```

Rules for a good blank:

- The hint says *what*, never *how*. "Add each bias into its accumulator, wrapping to
  INT32" is right. "Loop over i and call wrap_i32" is the answer.
- The code around the blank must still parse with the block removed. Put setup and
  range checks before the marker and the return after it.
- One idea per blank. If the solution is more than about ten lines, split it.
- The blank's stage must match the file's stage in `cub/stages.py`, or the tests will
  not know to skip it.

## Adding or changing a stage

1. Edit the code on `solutions`, with markers.
2. Write or update the test in `tests/test_NN_*.py`. Call
   `cub.stages.skip_unless_started(N)` first, and add prerequisites to `DEPENDS` in
   `cub/stages.py` if the test exercises earlier stages' code.
3. Write the guide in `docs/NN-*.md` and register the stage in `cub/stages.py`.
4. Run `pytest` on `solutions` (all pass), then `tools/make_blanks.py --apply` in a
   scratch copy and `pytest` again (all skip or pass, none fail).
5. `tools/sync_main.sh` to regenerate `main`.

## Regenerating the artifacts

```bash
python -m cub train      # artifacts/mlp_float.{pt,npz}, mnist_test_1k.npz (needs torch)
python -m cub compile    # artifacts/mnist.cub
python tools/make_golden.py   # artifacts/golden.npz and the encoding table in tests
```

Retraining changes every downstream artifact and every golden value, so do it only
deliberately and commit all of them together.

## Reviewing an onboardee

The CI run on their branch is the primary check. `python tools/progress.py` on a
checkout of their branch shows the same table. The video is for seeing that they can
explain what they built; a passing test with a confused explanation is a signal to
pair with them, not to fail them.
