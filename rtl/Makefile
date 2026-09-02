# Standalone cocotb flow for the Cub core. From the repo root:
#
#     source .venv/bin/activate
#     python -m cub hex -o rtl/build/dram.hex     # once, or after recompiling
#     make -C rtl
#
# `pytest tests/test_08_rtl.py` runs the same tests through the cocotb Python runner.

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
PLUSARGS     += +DRAM_HEX=$(RTL_DIR)/build/dram.hex
COCOTB_RESULTS_FILE := $(RTL_DIR)/build/results.xml

export PYTHONPATH := $(ROOT):$(RTL_DIR)/tb:$(PYTHONPATH)
export CUB_DRAM_HEX := $(RTL_DIR)/build/dram.hex

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: hex
hex:
	cd $(ROOT) && python -m cub hex -o rtl/build/dram.hex

.PHONY: clean-all
clean-all: clean
	rm -rf $(RTL_DIR)/build
