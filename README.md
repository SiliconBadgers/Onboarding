# SiliconBadgers Onboarding

Follow one digit-recognizing neural network from a PyTorch model, through a compiler,
into a chip design in SystemVerilog. Six stages; you write code in two of them.

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

### 1. The model

The network is a recipe for turning 784 numbers (the pixels of a 28x28 image) into 10
numbers (one score per digit). The largest score wins.

It does that in two steps, and each step is the same two operations:

- **Multiply.** Every output number is built from *all* the input numbers. Multiply
  each input by its own **weight**, add up the results — one number out. Do that once
  per output.
- **Add a bias.** Each output has one more number, its **bias**, added on at the end.

Between the two steps, every negative number is replaced with zero. That is the
**rectified linear** step.

So: 784 pixels -> multiply and add -> 128 numbers -> replace negatives with zero ->
multiply and add -> 10 scores.

**You do not need to know why this recognizes digits.** You need the list of
operations, because that is the list of things the chip must be able to do.
[Stage 1](docs/01-pytorch.md).

### 2. The instruction set

An **instruction set architecture** is the contract between software and hardware: what
operations exist, what they do, and how they are written as bits.

| Instruction | What it does |
|---|---|
| `LOAD` | copy numbers from main memory into the chip |
| `STORE` | copy numbers from the chip back to main memory |
| `MATRIX_MULTIPLY` | multiply each input by a weight and add up the results — a dot product, repeated once per output |
| `ADD_BIAS` | add the biases to the result |
| `RECTIFIED_LINEAR` | replace the negatives with zero, scale back down to 8 bits |
| `HALT` | stop |
| `NO_OPERATION` | nothing |

That is the entire chip — no branches, no loops, no function calls. It reads one
instruction, finishes it, reads the next. The complexity lives in software: the
compiler decides layer 1 is one `MATRIX_MULTIPLY` with 128 outputs and 784 inputs
followed by one `ADD_BIAS` over 128 numbers, and the chip performs 100,352 multiplies
and 128 additions without knowing why. [Stage 2](docs/02-instruction-set.md).

### 3. The compiler

Turns the PyTorch model into a list of instructions, plus a picture of what main memory
must hold before the chip starts — weights, biases, a blank space for the image, a
blank space for the answers. [Stage 3](docs/03-compiler.md); you then write the same
instructions by hand in [Stage 4](docs/04-mnist-by-hand.md).

### 4. Main memory

The chip cannot reach out and take an image from you. Everything arrives through **main
memory**, a large array both the host and the chip can read and write: the host puts
the image there, sets a start bit, waits for a done bit, and reads ten numbers back
out. `LOAD` and `STORE` are how the chip moves data between main memory and its own
small on-chip memories. [Stage 5](docs/05-registers-and-memory.md).

### 5. The chip

The thing that executes the instructions. For `MATRIX_MULTIPLY` it walks two arrays,
multiplying pairs and adding them up. For `RECTIFIED_LINEAR` it walks one array and
replaces every negative number with zero. For `ADD_BIAS` it walks two arrays and adds
them element by element. For `LOAD` it copies bytes.

We describe all of that in **SystemVerilog**, a language for specifying hardware, and
it comes out smaller than you would expect: one state machine, four arrays, one
multiplier. [Stage 6](docs/06-rtl.md).

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

## Working on the track

1. **Branch.** From `main`, create a branch named `FirstnameLastname`.
2. **Fill in the blanks.** Search for `TODO(onboard, stage`. There are four: two in the
   assembly program, two in the SystemVerilog. Each comment names the variables you
   need.
3. **Run the tests.** `pytest` runs everything; unstarted stages skip rather than fail.
   `python tools/progress.py` prints where you are.
4. **Push.** Continuous integration runs the same tests.
5. **Ask questions early.** The point is to learn how the stack fits together, not to
   do it alone.

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
