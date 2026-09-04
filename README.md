# SiliconBadgers Onboarding

Welcome. Over the next few weeks you will follow one digit-recognizing neural network
all the way down: from a PyTorch model, through a compiler that turns it into machine
instructions, into a chip design written in SystemVerilog that executes those
instructions. Six stages. Two of them ask you to write something; the other four ask
you to read and understand something.

## Before anything else: get access

This repository lives in the **SiliconBadgers** GitHub organization, and you have to be
a member of the organization before you can clone it, push a branch, or open a pull
request. Nothing in this guide works until that is done.

**Message Bilal Usman on Slack with your GitHub username or the email address on your
GitHub account, and ask to be added to the SiliconBadgers organization.** You will get
an invitation by email — accept it, and then continue with [docs/setup.md](docs/setup.md).

---

## The whole stack, in one page

A neural network is a big pile of multiply-and-add. A chip that runs neural networks
is a machine that does multiply-and-add very fast, and nothing else. Everything
between those two sentences is what this repository is about.

```
   PyTorch model            2 layers of weights and biases, in decimal numbers
        |
        |  quantization     scale every decimal number onto a whole number
        v
   whole-number model       the same network, in 8-bit and 32-bit integers
        |
        |  compiler         choose where everything lives in memory,
        v                   then emit instructions that move and multiply it
   a program                12 instructions + one picture of memory
        |
        +----------------------------+
        |                            |
        v                            v
   simulator (Python)          the chip (SystemVerilog)
   cub/simulator.py            rtl/src/cub_core.sv
        |                            |
        v                            v
      a digit  =================  the same digit, bit for bit
```

### 1. The model

The network is two fully connected layers: 784 pixels in, 128 hidden values, 10
outputs, one per digit. Each layer multiplies its input by a matrix of **weights**,
adds a vector of **biases**, and (for the first layer) zeroes anything negative — a
**rectified linear** step. The largest of the ten final numbers is the answer.

**You do not need to understand why any of that recognizes digits.** That is machine
learning, and it is somebody else's problem. What you need is the list of *operations*
a network performs, because that list is exactly the list of things the chip has to be
able to do. [Stage 1](docs/01-pytorch.md) is that list.

### 2. Whole numbers

The trained weights are decimal numbers like `-0.0428`. Hardware would rather have
whole numbers: an 8-bit multiply is one small circuit, a decimal multiply is a large
one, and 8-bit weights take a quarter of the memory 32-bit decimals do. So the whole
network is *scaled* — every tensor gets multiplied by a constant so that its largest
value lands on 127, then rounded to the nearest whole number. The chip only ever sees
integers. Google's Tensor Processing Unit does exactly this, for exactly these
reasons. [Stage 1](docs/01-pytorch.md) covers it.

### 3. The instruction set

An **instruction set architecture** is the contract between software and hardware: the
complete list of operations a chip can perform, what each one does, and how each one
is written down as bits. Ours has six instructions that do something:

| Instruction | What it does |
|---|---|
| `LOAD` | copy numbers from main memory into the chip |
| `STORE` | copy numbers from the chip back to main memory |
| `MATRIX_MULTIPLY` | multiply a vector by a matrix of weights |
| `ADD_BIAS` | add the biases to the result |
| `RECTIFIED_LINEAR` | zero the negatives, scale back down to 8 bits |
| `HALT` | stop |

plus `NO_OPERATION`, which does nothing, for seven opcodes in total.

That is the entire chip. It has no branches, no loops, no function calls. It reads one
instruction, does it completely, and reads the next one — one at a time, in order,
never overlapping. The complicated part of running a neural network is in the
software: the compiler decides that a `MATRIX_MULTIPLY` with 128 outputs and 784
inputs is what layer 1 needs, and the chip obediently performs 100,352 multiplies. All
you have to understand is what the chip does with one instruction.
[Stage 2](docs/02-instruction-set.md).

### 4. The compiler

The compiler takes the whole-number model and produces two things: a list of
instructions, and a picture of what main memory must contain before the chip starts —
the weights, the biases, a blank space for the input image, and a blank space for the
ten answers. It decides every address in that picture, and every operand of every
instruction. [Stage 3](docs/03-compiler.md), and then you write the same program by
hand in [Stage 4](docs/04-mnist-by-hand.md).

### 5. Talking to the chip

The chip cannot reach out and take an image from you. Everything it knows arrives
through **registers** and **memory**: the host writes the memory image, writes the
image to classify, sets a start bit, waits for a done bit, and reads ten numbers back
out. Inside the chip there are more registers — the program counter, the instruction
register, 256 accumulators, four scratchpads. Every instruction is a rule for moving
numbers between them. [Stage 5](docs/05-registers-and-memory.md).

### 6. The hardware

Finally, the SystemVerilog that actually does all this. It is smaller than you expect:
one state machine, four arrays, one multiplier. [Stage 6](docs/06-rtl.md).

---

## The stages

Read the guides in order. Everything is written to be read start to finish.

| Stage | Guide | You will |
|---|---|---|
| — | [docs/setup.md](docs/setup.md) | get access, clone, get a green test run |
| 1 | [PyTorch and whole numbers](docs/01-pytorch.md) | see what operations a network is made of, and how they become integers |
| 2 | [The instruction set](docs/02-instruction-set.md) | learn every instruction and how each one is encoded |
| 3 | [The compiler](docs/03-compiler.md) | follow a trained network turning into a program |
| 4 | [MNIST by hand](docs/04-mnist-by-hand.md) | **write** the whole program yourself, in assembly |
| 5 | [Talking to the chip](docs/05-registers-and-memory.md) | see how data gets in and out through registers and memory |
| 6 | [The hardware](docs/06-rtl.md) | **write** the two missing pieces of the SystemVerilog core |

Stages 4 and 6 have blanks to fill in, marked `TODO(onboard, stage N)`. Every other
stage is reading — its tests already pass, and they are there so you can change
something, watch it break, and change it back.

## How to navigate this repository

```
README.md            you are here: the whole stack in one page
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

Two files are worth knowing about specifically:

- **[docs/02-instruction-set.md](docs/02-instruction-set.md)** is the specification.
  The assembler, the simulator, the compiler and the hardware are all written against
  it. When two of them disagree, that file decides who is wrong.
- **`artifacts/golden.npz`** holds 1000 test images together with the correct answer at
  every step of the stack. Every test compares against it, which is why a mistake
  anywhere shows up as a specific failing number rather than "the accuracy dropped".

## Working on the track

1. **Make your branch.** From `main`, create a branch named `FirstnameLastname` and
   work only on that branch.
2. **Fill in the blanks.** Search for `TODO(onboard, stage`. There are four: two in
   Stage 4, in the assembly program, and two in Stage 6, in the SystemVerilog. The
   comment above each one names every variable you need to touch.
3. **Run the tests.** `pytest` runs everything. A stage you have not started is
   *skipped*, not failed, so the suite is green from day one and only goes red when
   something you wrote is wrong. `python tools/progress.py` prints where you are.
4. **Push.** Continuous integration runs the same tests on every push to your branch.
5. **Ask questions early and often.** The point of the track is to learn how the stack
   fits together, not to prove you can do it alone.

## Commands you will use

```bash
pytest                                        # everything
pytest tests/test_04_mnist_by_hand.py -v      # one stage
python tools/progress.py                      # where am I?

python -m cub run --index 3                   # classify a test image on the simulator
python -m cub disassemble artifacts/mnist.cub # the twelve instructions, as text
python -m cub compile                         # rebuild artifacts/mnist.cub
python -m cub accuracy --count 200            # how often the simulator is right
python -m cub memory-image                    # bake a test image into rtl/build/main_memory.hex
make -C rtl                                   # the hardware tests, with waveforms available
```
