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
   compiler                   scale the decimals to whole numbers, decide where
   cub/compiler.py            everything lives in memory, emit 12 instructions
        |
        v
   the chip                   executes those instructions, one at a time
   rtl/src/cub_core.sv
        |
        v
      a digit
```

### 1. The model

Two fully connected layers: 784 pixels in, 128 hidden values, 10 outputs. Each layer
multiplies its input by a matrix of **weights**, adds **biases**, and (layer 1 only)
zeroes anything negative — a **rectified linear** step. The largest of the ten outputs
is the answer.

**You do not need to understand why that recognizes digits.** You need the list of
*operations*, because that is the list of things the chip must do.
[Stage 1](docs/01-pytorch.md).

### 2. Whole numbers

Trained weights are decimals like `-0.0428`. Hardware prefers integers: an 8-bit
multiply is one small circuit, and 8-bit weights take a quarter of the memory. So every
tensor is scaled until its largest value lands on 127, then rounded. The chip never
sees a decimal. Google's Tensor Processing Unit does the same thing for the same
reasons. [Stage 1](docs/01-pytorch.md).

### 3. The instruction set

An **instruction set architecture** is the contract between software and hardware: what
operations exist, what they do, and how they are written as bits. Ours has six that do
something, plus `NO_OPERATION`:

| Instruction | What it does |
|---|---|
| `LOAD` | copy numbers from main memory into the chip |
| `STORE` | copy numbers from the chip back to main memory |
| `MATRIX_MULTIPLY` | multiply a vector by a matrix of weights |
| `ADD_BIAS` | add the biases to the result |
| `RECTIFIED_LINEAR` | zero the negatives, scale back down to 8 bits |
| `HALT` | stop |

That is the entire chip — no branches, no loops, no function calls. It reads one
instruction, finishes it, reads the next. The complexity lives in software: the
compiler decides layer 1 is one `MATRIX_MULTIPLY` with 128 outputs and 784 inputs, and
the chip performs 100,352 multiplies without knowing why.
[Stage 2](docs/02-instruction-set.md).

### 4. The compiler

Produces two things: the instructions, and a picture of what main memory must hold
before the chip starts — weights, biases, a blank space for the image, a blank space
for the answers. [Stage 3](docs/03-compiler.md); you then write the same program by
hand in [Stage 4](docs/04-mnist-by-hand.md).

### 5. Talking to the chip

The chip cannot reach out and take an image from you. Everything arrives through
**registers** and **memory**: the host writes the image, sets a start bit, waits for a
done bit, reads ten numbers back. Inside are more registers — program counter,
instruction register, 256 accumulators, four scratchpads — and every instruction is a
rule for moving numbers between them. [Stage 5](docs/05-registers-and-memory.md).

### 6. The hardware

The SystemVerilog that does all this: one state machine, four arrays, one multiplier.
[Stage 6](docs/06-rtl.md).

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
| 6 | [The hardware](docs/06-rtl.md) | **write** the two missing pieces of the core |

Read them in order. Stages 4 and 6 have blanks marked `TODO(onboard, stage N)`; the
rest are reading, and their tests already pass.

## Repository layout

```
docs/                one guide per stage, in reading order
  setup.md             access, clone, install, first test run
  01-pytorch.md        the network and whole numbers
  02-instruction-set.md  every instruction, and its bit layout        <- the specification
  03-compiler.md       network -> instructions
  04-mnist-by-hand.md  your assembly program                          <- you write this
  05-registers-and-memory.md   registers, memory spaces, the host
  06-rtl.md            the SystemVerilog core                         <- you write this

cub/                 the Python half of the stack, complete and readable
  instruction_set.py   what an instruction is; encode and decode 16 bytes
  assembler.py         assembly text <-> instructions
  simulator.py         a software model of the chip, one instruction at a time
  model.py             the PyTorch network and its training script
  quantization.py      decimal numbers -> whole numbers
  compiler.py          a quantized network -> a program
  program.py           the memory image and the .cub file format
  runtime.py           the host side: quantize, run, read the answer
  stages.py            stage bookkeeping used by the tests
  __main__.py          the `python -m cub ...` commands

programs/
  mnist_by_hand.cubasm   Stage 4: the assembly program you write

rtl/                 the hardware half
  src/cub_core.sv      Stage 6: the core, one state machine
  tb/tb_top.sv         simulation top: the core plus a memory
  tb/test_cub.py       the testbench, in Python, using cocotb
  Makefile             run the hardware tests without pytest

tests/               one file per stage; `pytest` runs them all
artifacts/           the trained weights, the compiled program, the expected answers
tools/               progress.py, and scripts the maintainers use
```

Two files to know about:

- **[docs/02-instruction-set.md](docs/02-instruction-set.md)** is the specification.
  The assembler, simulator, compiler and hardware are all written against it. When two
  of them disagree, it decides who is wrong.
- **`artifacts/golden.npz`** holds 1000 test images with the correct answer at every
  step of the stack, so a mistake shows up as a specific wrong number rather than a
  vague accuracy drop.

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
pytest                                        # everything
pytest tests/test_04_mnist_by_hand.py -v      # one stage
python tools/progress.py                      # where am I?

python -m cub run --index 3                   # classify a test image on the simulator
python -m cub disassemble artifacts/mnist.cub # the twelve instructions, as text
python -m cub compile                         # rebuild artifacts/mnist.cub
python -m cub accuracy --count 200            # how often the simulator is right
python -m cub memory-image                    # bake a test image into rtl/build/main_memory.hex
make -C rtl                                   # the hardware tests
```
