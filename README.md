# SiliconBadgers Onboarding

Welcome. Over the next few weeks you will build, from PyTorch down to a running FPGA,
a small hardware accelerator that recognizes handwritten digits. By the end you will
have written the compiler, the simulator, and the hardware for the same instruction
set, and watched all three agree bit for bit.

Everything you build is a scaled-down version of the real project,
[AI-Inference-Chip](https://github.com/SiliconBadgers/AI-Inference-Chip). The
instruction set here is called **Cub**. When you finish, the real one (**Badger**)
will read as Cub plus a handful of extra features, and you will be ready to work on it.

## What you are building

```
PyTorch MLP  ──►  INT8 weights  ──►  Cub program  ──►  simulator  ──►  digit
 (Stage 1)         (Stage 2)         (Stages 3-6)      (Stage 4)     (Stage 7)
                                          │
                                          └──────────►  RTL core  ──►  FPGA
                                                        (Stage 8)    (Stage 9)
```

The network is a two-layer MLP: 784 inputs, 128 hidden units, 10 outputs. The
instruction set has six instructions. That is enough to teach every idea that matters
and small enough that you can hold all of it in your head.

## The track

Each stage has a guide in `docs/`, one or more blanks to fill in, and a test that
tells you when you are done. Stages 0 to 7 are for everyone. Stages 8 and 9 are the
hardware track; software-focused members can stop after Stage 7 or continue.

| Stage | Title | You will | Guide |
|---|---|---|---|
| 0 | Setup | get the tests running | [docs/00-setup.md](docs/00-setup.md) |
| 1 | The PyTorch MLP | write the forward pass, train it | [docs/01-pytorch-mlp.md](docs/01-pytorch-mlp.md) |
| 2 | Quantization | turn floats into INT8 and pick a shift | [docs/02-quantization.md](docs/02-quantization.md) |
| 3 | Encoding | pack instructions into 128 bits, write the assembler | [docs/03-encoding.md](docs/03-encoding.md) |
| 4 | The simulator | implement MATMUL, ADD_BIAS, RELU | [docs/04-simulator.md](docs/04-simulator.md) |
| 5 | MNIST by hand | write the whole program in assembly | [docs/05-mnist-by-hand.md](docs/05-mnist-by-hand.md) |
| 6 | The compiler | make the computer write Stage 5 for you | [docs/06-compiler.md](docs/06-compiler.md) |
| 7 | Runtime and demo | image in, digit out | [docs/07-runtime.md](docs/07-runtime.md) |
| 8 | RTL | finish the hardware core, pass the cocotb tests | [docs/08-rtl.md](docs/08-rtl.md) |
| 9 | FPGA | build a bitstream, press a button, read a digit | [docs/09-fpga.md](docs/09-fpga.md) |

Before Stage 3, read [docs/isa.md](docs/isa.md). It is the specification every stage
is written against, and it is short.

## How it works

1. **Make your branch.** From `main`, create a branch named `FirstnameLastname`
   and add yourself to [ONBOARDEES.md](ONBOARDEES.md). Work only on that branch.
2. **Fill in the blanks.** Every blank is marked `TODO(onboard, stage N)`. Search for
   that string. Each guide tells you which file and what the code needs to do. There
   are no hidden requirements: if the test passes, the stage is done.
3. **Run the tests.** `pytest` runs everything. A stage you have not started is
   *skipped*, not failed, so the suite is green from day one and only goes red when
   something you wrote is wrong. `python tools/progress.py` prints where you are.
4. **Push.** CI runs the same tests on every push to your branch.
5. **Record the milestone.** Stages 1, 4, 5, 7, 8, and 9 each end with a short
   (under two minutes) screen recording. The guide says what to show. Submit it here:
   **[INSERT_GOOGLE_FORM_LINK_HERE]**, with a link to your branch's green CI run.

Ask questions early and often. The point of the track is to learn how the stack
fits together, not to prove you can do it alone.

## Repository layout

```
cub/            the Python package you complete
  isa.py          instruction definitions, encode/decode      Stage 3
  asm.py          assembler and disassembler                  Stage 3
  sim.py          the ISA simulator                           Stage 4
  quant.py        float -> INT8                               Stage 2
  model.py        the PyTorch MLP                             Stage 1
  program.py      the DRAM image and .cub file format
  compiler.py     quantized weights -> Cub program            Stage 6
  runtime.py      host side: preprocess, run, argmax          Stage 7
  stages.py       stage bookkeeping used by the tests
programs/       mnist_by_hand.cubasm                          Stage 5
rtl/
  src/            the SystemVerilog core                      Stage 8
  tb/             cocotb testbench
  fpga/           board top, constraints, Vivado script       Stage 9
docs/           one guide per stage, plus isa.md
tests/          one test file per stage
artifacts/      trained weights, the compiled program, golden outputs
tools/          progress.py, and maintainer scripts
```

## Commands you will use

```bash
pytest                                 # everything (unstarted stages skip)
pytest tests/test_04_simulator.py -v   # one stage
python tools/progress.py               # where am I?
python -m cub run --index 3            # classify a test image on the simulator
python -m cub disasm artifacts/mnist.cub
python -m cub compile                  # rebuild artifacts/mnist.cub (Stage 6)
python -m cub eval --count 200         # simulator accuracy
```
