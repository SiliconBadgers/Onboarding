# Setup

**Goal:** access to the organization, the repository on your machine, and a green test
run on your own branch.

## 1. Join the GitHub organization

You cannot clone this repository, push a branch, or open a pull request until you are a
member of the **SiliconBadgers** GitHub organization. This is the first step and
nothing else works before it.

**Message Bilal Usman on Slack.** Include:

- your GitHub username, or the email address your GitHub account uses
- that you are starting the onboarding track

You will get an organization invitation by email. Accept it, then confirm you can see
<https://github.com/SiliconBadgers/Onboarding> while signed in to GitHub.

## 2. Clone and branch

```bash
git clone https://github.com/SiliconBadgers/Onboarding.git
cd Onboarding
git checkout -b FirstnameLastname
```

Work only on that branch. If `git clone` asks for a password, use a personal access
token or set up SSH keys — GitHub stopped accepting account passwords over HTTPS.

## 3. Python environment

You need Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[torch,dev]"
```

On Windows the first two lines depend on your shell. There is usually no `python3`
command, so use `python`. The virtual environment's folder is `Scripts`, not `bin`, and
only Git Bash has `source`.

| Shell | Create | Activate |
|---|---|---|
| Command Prompt | `python -m venv .venv` | `.venv\Scripts\activate` |
| PowerShell | `python -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Git Bash | `python -m venv .venv` | `source .venv/Scripts/activate` |

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

Activation only affects the terminal window you run it in. Your prompt should start
with `(.venv)` afterwards; if it does not, the later `pip` and `pytest` commands will
use the wrong Python.

Creating the environment normally takes a second or two. If it sits for more than a
minute it is stuck bootstrapping pip, which happens now and then with Homebrew Python.
Press Ctrl-C, run `rm -rf .venv`, and try again. Do not skip the delete: an interrupted
environment has no `activate` script and no pip.

The `torch` extra pulls in PyTorch and torchvision (a few hundred megabytes). Only
Stage 1 needs them. If the install is a problem on your machine, use
`pip install -e ".[dev]"` and the Stage 1 tests will skip themselves; come back to it
when you can.

## 4. The hardware simulator (needed for Stage 6)

Stage 6 compiles the SystemVerilog with [Icarus Verilog](https://steveicarus.github.io/iverilog/)
and drives it from Python with cocotb. You can install these now or wait until you get
there.

```bash
brew install icarus-verilog            # macOS
sudo apt-get install iverilog          # Debian/Ubuntu
pip install -e ".[rtl]"
```

Without `iverilog` the Stage 6 test skips with a message telling you so.

## 5. Run the tests

```bash
pytest
```

Everything should pass except the two stages with blanks in them, which skip. Look at a
skip reason to see how it tells you which file to open:

```bash
pytest tests/test_04_mnist_by_hand.py -rs
```

Then:

```bash
python tools/progress.py
```

This table is the same one continuous integration prints on every push.

## 6. Look around

Ten minutes, no more:

- `python -m cub run --index 3` classifies one test image on the simulator. It works
  already, because `artifacts/mnist.cub` was compiled by the finished pipeline. The
  stages ahead are about understanding how each piece of that pipeline works.
- `python -m cub disassemble artifacts/mnist.cub` prints the twelve instructions that
  classify a digit. By Stage 4 you will be writing these yourself.
- Skim [02-instruction-set.md](02-instruction-set.md). Do not try to absorb it yet;
  just see how short it is.

## 7. Push

```bash
git push -u origin FirstnameLastname
```

Open the Actions tab on GitHub and confirm the run on your branch is green.

## Done when

`pytest` passes locally and on your branch in continuous integration.

## Read next

[01-pytorch.md](01-pytorch.md).
