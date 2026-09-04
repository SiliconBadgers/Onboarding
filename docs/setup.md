# Setup

**Goal:** organization access, the repository on your machine, and a green test run on
your own branch.

## 1. Join the GitHub organization

You cannot clone, push, or open a pull request until you are a member of the
**SiliconBadgers** organization. Nothing else works first.

**Message Bilal Usman on Slack** with your GitHub username or the email on your GitHub
account, and say you are starting onboarding. Accept the invitation that arrives by
email, then confirm you can see <https://github.com/SiliconBadgers/Onboarding> while
signed in.

## 2. Clone and branch

```bash
git clone https://github.com/SiliconBadgers/Onboarding.git
cd Onboarding
git checkout -b FirstnameLastname
```

Work only on that branch. If `git clone` asks for a password, use a personal access
token or SSH keys — GitHub no longer accepts account passwords over HTTPS.

## 3. Python environment

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[torch,dev]"
```

On Windows there is usually no `python3`, the folder is `Scripts` rather than `bin`,
and only Git Bash has `source`:

| Shell | Create | Activate |
|---|---|---|
| Command Prompt | `python -m venv .venv` | `.venv\Scripts\activate` |
| PowerShell | `python -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Git Bash | `python -m venv .venv` | `source .venv/Scripts/activate` |

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

Activation affects only the terminal you run it in. Your prompt should start with
`(.venv)`; if it does not, `pip` and `pytest` will use the wrong Python.

If `venv` creation hangs for more than a minute it is stuck bootstrapping pip, which
happens with Homebrew Python. Ctrl-C, `rm -rf .venv`, try again — do not skip the
delete, since an interrupted environment has no `activate` and no pip.

The `torch` extra is a few hundred megabytes and only Stage 1 needs it. If it will not
install, use `pip install -e ".[dev]"`; the Stage 1 tests will skip.

## 4. The hardware simulator (Stage 6)

Install now or when you get there:

```bash
brew install icarus-verilog            # macOS
sudo apt-get install iverilog          # Debian/Ubuntu
pip install -e ".[rtl]"
```

Without `iverilog` the Stage 6 test skips and says so.

## 5. Run the tests

```bash
pytest
```

Everything passes except the two stages with blanks, which skip. `-rs` shows which file
a skip wants you to open:

```bash
pytest tests/test_04_mnist_by_hand.py -rs
python tools/progress.py
```

`progress.py` prints the same table continuous integration does.

## 6. Look around

Ten minutes, no more:

- `python -m cub run --index 3` classifies a test image on the simulator. It already
  works, because `artifacts/mnist.cub` was compiled by the finished pipeline.
- `python -m cub disassemble artifacts/mnist.cub` prints the twelve instructions that
  classify a digit. You will write these yourself in Stage 4.
- Skim [02-instruction-set.md](02-instruction-set.md) — just to see how short it is.

## 7. Push

```bash
git push -u origin FirstnameLastname
```

Check the Actions tab is green.

## Done when

`pytest` passes locally and in continuous integration.

## Read next

[01-pytorch.md](01-pytorch.md).
