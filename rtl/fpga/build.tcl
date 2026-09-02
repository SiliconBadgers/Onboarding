# build.tcl -- Vivado non-project batch flow for cub_top on the PYNQ-Z2.
#
#   python -m cub hex -o rtl/build/dram.hex      # pick the image with --index N
#   vivado -mode batch -source rtl/fpga/build.tcl
#
# Outputs land in rtl/build/vivado/: cub_top.bit, utilization and timing reports.
# This script has not been run against a Vivado installation; it follows the
# standard read/synth/place/route/write_bitstream sequence. If a step fails,
# the log in rtl/build/vivado/vivado.log says which.

set fpga_dir  [file dirname [file normalize [info script]]]
set rtl_dir   [file normalize "$fpga_dir/.."]
set build_dir "$rtl_dir/build"
set out_dir   "$build_dir/vivado"
set part      "xc7z020clg400-1"

file mkdir $out_dir

if {![file exists "$build_dir/dram.hex"]} {
    puts "ERROR: $build_dir/dram.hex not found. Run: python -m cub hex -o rtl/build/dram.hex"
    exit 1
}

# $readmemh("dram.hex") in cub_top.sv is resolved relative to the working directory.
cd $build_dir

read_verilog -sv "$rtl_dir/src/cub_core.sv"
read_verilog -sv "$fpga_dir/cub_top.sv"
read_xdc "$fpga_dir/pynq_z2.xdc"

synth_design -top cub_top -part $part
write_checkpoint -force "$out_dir/post_synth.dcp"
report_utilization -file "$out_dir/utilization_synth.rpt"

opt_design
place_design
route_design
write_checkpoint -force "$out_dir/post_route.dcp"

report_utilization    -file "$out_dir/utilization.rpt"
report_timing_summary -file "$out_dir/timing.rpt"
report_power          -file "$out_dir/power.rpt"

write_bitstream -force "$out_dir/cub_top.bit"
puts "Bitstream written to $out_dir/cub_top.bit"
