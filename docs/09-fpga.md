# Stage 9 — FPGA

**Goal:** the same core, on a board, classifying a digit when you press a button.

**Files:** `rtl/fpga/cub_top.sv`, `rtl/fpga/pynq_z2.xdc`, `rtl/fpga/build.tcl`.
**Board:** PYNQ-Z2 (Zynq-7020). Other boards work with a new constraints file; ask.

## What the top level does

`cub_top` wraps `cub_core` with three things the simulator did not need:

1. **A DRAM.** A 256 KiB block RAM, initialized at synthesis time from `dram.hex`,
   which is the compiled program with one test image already written into the input
   region. On the board there is no Python host, so the image is baked in.
2. **Two buttons.** `BTN0` is reset. `BTN1` is start: synchronized, debounced, and
   edge-detected into a one-cycle `start` pulse.
3. **An argmax.** After `done`, a small state machine reads the ten INT32 logits back
   out of the DRAM and drives the four LEDs with the index of the largest one, in
   binary. That is the digit.

Everything inside `cub_core` is untouched. If Stage 8 passes, this works.

## Build

You need Vivado (the free WebPACK edition covers the 7020). Then:

```bash
python -m cub hex --index 0 -o rtl/build/dram.hex     # image 0 is a 7
cd rtl/fpga
vivado -mode batch -source build.tcl
```

The script runs synthesis, place and route, and writes `rtl/build/cub_top.bit` plus
utilization and timing reports. Read the utilization report: how many block RAMs did
the DRAM and `SPAD_W` take, and how many DSPs did the one multiplier take? Those two
numbers are the beginning of every hardware conversation you will have on the real
project.

Before programming the board, open `rtl/fpga/pynq_z2.xdc` and check every pin against
the board's master constraints file from the vendor. The names in the repository were
taken from public documentation and have not been verified on hardware.

## Run

Program the bitstream (Hardware Manager, or `openFPGALoader`). Press `BTN1`. About
two milliseconds later the LEDs should show `0111`, which is 7. `BTN0` resets the
core and clears the LEDs.

Change the baked-in image (`--index 3` is a zero, `--index 12` is a nine), rebuild,
and try again.

## Milestone video

This is the hardware-track finale. Point a phone at the board:

1. Press the button, show the LEDs.
2. Say which image index is baked in and what the simulator predicts for it
   (`python -m cub run --index N`).
3. From the utilization report, the block RAM and DSP counts.

Submit at **[INSERT_GOOGLE_FORM_LINK_HERE]**.

## What comes next

You have now done the four-way match the main project is built around: PyTorch,
simulator, RTL simulation, and silicon all agree. Badger's roadmap is that same loop
with a bigger network, a real DMA path to the processor, and a MAC array instead of a
single multiplier. Ask your mentor for a first task on
[AI-Inference-Chip](https://github.com/SiliconBadgers/AI-Inference-Chip).
