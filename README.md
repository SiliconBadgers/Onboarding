# SiliconBadgers Onboarding

This onboarding project will guide you into running a machine learning program on a
chip. By the end you will hand it a 28x28 pixel image of a handwritten digit and get a
prediction of that digit back. Six stages; you write code in two of them.

## Get access first

You must be in the **SiliconBadgers** GitHub organization to clone or push.

**Message Bilal Usman on Slack with your GitHub username or email.** Accept the
invitation, then go to [docs/setup.md](docs/setup.md).

---

## The stack

A neural network is a big pile of multiply-and-add. A chip that runs one is a machine
that does multiply-and-add fast and nothing else.

```
   PyTorch model              2 layers of weights and biases, in decimal numbers
        |
        v
   instruction set            the short list of operations the chip can do
   docs/02-instruction-set.md
        |
        v
   compiler                   scale the decimals to whole numbers, decide where
   python/compiler.py         everything lives in memory, emit 12 instructions
        |
        v
   the chip                   executes those instructions, one at a time
   rtl/cub_core.sv
```

---

## The stages

| Stage | Guide | You will |
|---|---|---|
| — | [docs/setup.md](docs/setup.md) | get access, clone, get a green test run |
| 1 | [PyTorch and whole numbers](docs/01-pytorch.md) | see what operations a network is made of, and how they become integers |
| 2 | [The instruction set](docs/02-instruction-set.md) | learn every instruction and how each one is encoded |
| 3 | [The compiler](docs/03-compiler.md) | follow a trained network turning into a program |
| 4 | [MNIST by hand](docs/04-mnist-by-hand.md) | **write** the whole program yourself, in assembly |
| 5 | [Talking to the chip](docs/05-registers-and-memory.md) | see how data gets in and out |
| 6 | [The chip](docs/06-rtl.md) | **write** the two missing pieces of the SystemVerilog |

Read them in order. Stages 4 and 6 have blanks marked `TODO(onboard, stage N)`; the
rest are reading, and their tests already pass.

## Repository layout

```
docs/          one guide per stage, in reading order
                 02-instruction-set.md is the specification everything else
                 is written against

programs/      the assembly program you write in Stage 4

rtl/           the hardware: the SystemVerilog chip you finish in Stage 6,
               plus the testbench that checks it

python/        the model, quantizer, compiler and simulator, complete and
               working. Read it when a guide points you at a file; you are
               not expected to study it, and you never have to change it

tests/         one file per stage; `pytest` runs them all
artifacts/     the trained weights, the compiled program, the expected answers
tools/         progress.py, and scripts the maintainers use
```

`artifacts/golden.npz` is worth knowing about: it holds 1000 test images with the
correct answer at every step of the stack, so a mistake shows up as a specific wrong
number rather than a vague accuracy drop.

## Commands

```bash
pytest                                            # everything
pytest tests/test_04_mnist_by_hand.py -v          # one stage
python tools/progress.py                          # where am I?

python -m python run --index 3                    # classify a test image on the simulator
python -m python disassemble artifacts/mnist.cub  # the twelve instructions, as text
python -m python compile                          # rebuild artifacts/mnist.cub
python -m python accuracy --count 200             # how often the simulator is right
python -m python memory-image                     # bake a test image for the hardware
make -C rtl                                       # the hardware tests
```
