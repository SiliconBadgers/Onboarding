# Standalone cocotb flow for the accelerator core. From the repository root:
#
#     source .venv/bin/activate
#     python -m cub memory-image -o rtl/build/main_memory.hex   # once, or after recompiling
#     make -C rtl
#
# `pytest tests/test_06_rtl.py` runs the same tests through cocotb's Python runner.

SIM ?= icarus
TOPLEVEL_LANG ?= verilog

RTL_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT    := $(abspath $(RTL_DIR)/..)

VERILOG_SOURCES += $(RTL_DIR)/src/cub_core.sv
VERILOG_SOURCES += $(RTL_DIR)/tb/tb_top.sv

COCOTB_TOPLEVEL     := tb_top
COCOTB_TEST_MODULES := test_cub

COMPILE_ARGS += -g2012
SIM_BUILD    := $(RTL_DIR)/build/sim_build
PLUSARGS     += +MEMORY_HEX=$(RTL_DIR)/build/main_memory.hex
COCOTB_RESULTS_FILE := $(RTL_DIR)/build/results.xml

export PYTHONPATH := $(ROOT):$(RTL_DIR)/tb:$(PYTHONPATH)
export CUB_MEMORY_HEX := $(RTL_DIR)/build/main_memory.hex

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: memory-image
memory-image:
	cd $(ROOT) && python -m cub memory-image -o rtl/build/main_memory.hex

.PHONY: clean-all
clean-all: clean
	rm -rf $(RTL_DIR)/build
