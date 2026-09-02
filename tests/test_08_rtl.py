"""Stage 8: the RTL core must match the simulator, instruction by instruction and on MNIST.

This drives the cocotb tests in rtl/tb/test_cub.py under Icarus Verilog. It skips if
the stage's blanks are untouched or if iverilog is not installed.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from cub.stages import skip_unless_started

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl"
BUILD = RTL / "build"


def test_rtl_matches_simulator():
    skip_unless_started(8)
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog is not installed (brew install icarus-verilog / apt install iverilog)")
    from cocotb_tools.runner import get_runner

    BUILD.mkdir(exist_ok=True)
    hex_path = BUILD / "dram.hex"
    if not hex_path.exists():
        from cub.__main__ import main

        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            main(["hex", "--index", "0", "-o", str(hex_path)])
        finally:
            os.chdir(cwd)

    runner = get_runner("icarus")
    runner.build(
        sources=[RTL / "src" / "cub_core.sv", RTL / "tb" / "tb_top.sv"],
        hdl_toplevel="tb_top",
        build_dir=BUILD / "sim_build",
        build_args=["-g2012"],
        timescale=("1ns", "1ps"),
        always=True,
    )
    # The runner builds the simulation's PYTHONPATH from this process's sys.path.
    for p in (ROOT, RTL / "tb"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    runner.test(
        test_module="test_cub",
        hdl_toplevel="tb_top",
        build_dir=BUILD / "sim_build",
        test_dir=BUILD,
        plusargs=[f"+DRAM_HEX={hex_path}"],
        extra_env={"CUB_DRAM_HEX": str(hex_path)},
        results_xml=str(BUILD / "results.xml"),
    )
