# Stage 0 — Setup

**Goal:** a green test run on your own branch.

## 1. Clone and branch

```bash
git clone https://github.com/SiliconBadgers/Onboarding.git
cd Onboarding
git checkout -b FirstnameLastname
```

Add a row for yourself to `ONBOARDEES.md` and commit it. That is your first commit on
the branch, and it is the one your mentor will look for.

## 2. Python environment

You need Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[torch,dev]"
```

Creating the venv normally takes a second or two. If it sits for more than a minute
it is stuck bootstrapping pip, which happens now and then with Homebrew Python.
Press Ctrl-C, run `rm -rf .venv`, and try again. Do not skip the delete: an
interrupted venv has no `activate` script and no pip.

The `torch` extra pulls in PyTorch and torchvision (a few hundred MB). Only Stage 1
needs them. If the install is a problem on your machine, use `pip install -e ".[dev]"`
and the Stage 1 tests will skip; come back to it when you can.

## 3. Run the tests

```bash
pytest
```

You should see something like `3 passed, 60 skipped`. Skipped is correct: every
stage you have not started skips itself. Look at one of the skip reasons to see how
it tells you which file to open:

```bash
pytest tests/test_04_simulator.py -rs
```

Then:

```bash
python tools/progress.py
```

This table is the same thing CI prints on every push.

## 4. Look around

Ten minutes, no more:

- `python -m cub run --index 3` classifies one test image on the simulator. It works
  already because the committed `artifacts/mnist.cub` was compiled by the finished
  pipeline. Your job over the next stages is to rebuild that pipeline yourself.
- `python -m cub disasm artifacts/mnist.cub` shows the twelve instructions that
  classify a digit. By Stage 5 you will be writing these by hand.
- Skim `docs/isa.md`. Do not try to absorb it yet; just see how long it is.

## 5. Push

```bash
git push -u origin FirstnameLastname
```

Open the Actions tab on GitHub and confirm the CI run is green.

## Done when

`pytest` passes (with skips) locally and in CI on your branch. No video for this stage.
