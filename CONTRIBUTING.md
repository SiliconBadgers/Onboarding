# Maintaining the track

This file is for people who maintain the onboarding repository, not for people going
through it.

## Two branches

- **`solutions`** holds the complete, working code. Every test passes here. This is
  where you edit code, docs, tests and guides.
- **`main`** is *generated* from `solutions` by `tools/sync_main.sh`. Never edit it by
  hand. Onboardees branch from it.

The generator strips every solution block into a TODO. A block looks like this in
Python, and the same with `//` in SystemVerilog or `;` in assembly:

```python
# --- SOLUTION(stage=6): one-line hint the onboardee will see ---
for i in range(count):
    ...
# --- END SOLUTION ---
```

and becomes:

```python
# TODO(onboard, stage 6): one-line hint the onboardee will see
raise NotImplementedError("stage 6 blank, see simulator.py:141")
```

Rules for a good blank:

- The hint says *what* and names *which variables*, never *how*. "Set `running_sum_next`
  to `running_sum` plus the product of `activation_read_data` and `weight_read_data`"
  is right. Pasting the line back is not.
- The code around the blank must still parse with the block removed. Put setup and
  range checks before the marker and the return after it.
- One idea per blank. If the solution is more than about ten lines, split it.
- The blank's stage must match the file's stage in `python/stages.py`, or the tests will
  not know to skip it.

There are deliberately only two stages with blanks — Stage 4 (assembly) and Stage 6
(SystemVerilog). The Python stages are for reading; adding blanks to them would turn a
hardware onboarding track into a Python exercise. If you want to make a Python stage
more active, add a "Things to try" item that breaks something, not a blank.

## Adding or changing a stage

1. Edit the code on `solutions`, with markers if the stage has blanks.
2. Write or update the test in `tests/test_NN_*.py`. If the stage has blanks, call
   `python.stages.skip_unless_started(N)` first.
3. Write the guide in `docs/NN-*.md` and register the stage in `python/stages.py`.
4. Run `pytest` on `solutions` (all pass), then `tools/make_blanks.py --apply` in a
   scratch copy and `pytest` again (all skip or pass, none fail).
5. `tools/sync_main.sh` to regenerate `main`.

## Regenerating the artifacts

```bash
python -m python train      # trained_weights.{pt,npz}, mnist_test_1000.npz (needs PyTorch)
python -m python compile    # artifacts/mnist.cub
python tools/make_golden.py   # artifacts/golden.npz
```

Retraining changes every downstream artifact and every expected value, so do it only
deliberately and commit all of them together. The golden encodings in
`tests/test_02_instruction_set.py` are hand-checked against the specification and do
*not* depend on the weights, so they should never need regenerating — if one changes,
the instruction encoding changed and that is a specification change.

## Access

New onboardees have to be members of the SiliconBadgers GitHub organization before they
can clone or push. They are told to message Bilal Usman on Slack with their GitHub
username or email; whoever handles that sends the organization invitation.

## Reviewing an onboardee

The continuous integration run on their branch is the primary check.
`python tools/progress.py` on a checkout of their branch prints the same table. Pair
with anyone whose tests pass but who cannot explain what they built — a passing test
with a confused explanation is a signal to sit down together, not to fail them.
