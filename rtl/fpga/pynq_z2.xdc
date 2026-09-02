## pynq_z2.xdc -- pin constraints for cub_top on the TUL PYNQ-Z2.
##
## VERIFY BEFORE USE. These pin names were taken from the publicly available
## PYNQ-Z2 master constraints file (sysclk H16, LD0-LD3 R14/P14/N16/M14, BTN0/BTN1
## D19/D20). Check them against the master XDC that ships with your board
## revision before programming; a wrong pin is a silent failure, not a build error.

## 125 MHz system clock
set_property -dict { PACKAGE_PIN H16 IOSTANDARD LVCMOS33 } [get_ports { sysclk }]
create_clock -add -name sys_clk_pin -period 8.00 -waveform {0 4} [get_ports { sysclk }]

## LEDs: led[0] = LD0 ... led[3] = LD3. The predicted digit in binary, LD0 is bit 0.
set_property -dict { PACKAGE_PIN R14 IOSTANDARD LVCMOS33 } [get_ports { led[0] }]
set_property -dict { PACKAGE_PIN P14 IOSTANDARD LVCMOS33 } [get_ports { led[1] }]
set_property -dict { PACKAGE_PIN N16 IOSTANDARD LVCMOS33 } [get_ports { led[2] }]
set_property -dict { PACKAGE_PIN M14 IOSTANDARD LVCMOS33 } [get_ports { led[3] }]

## Buttons: btn[0] = BTN0 (reset), btn[1] = BTN1 (start). Active high when pressed.
set_property -dict { PACKAGE_PIN D19 IOSTANDARD LVCMOS33 } [get_ports { btn[0] }]
set_property -dict { PACKAGE_PIN D20 IOSTANDARD LVCMOS33 } [get_ports { btn[1] }]

## The button inputs are asynchronous by nature and are synchronized in cub_top.
set_false_path -from [get_ports { btn[*] }]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
