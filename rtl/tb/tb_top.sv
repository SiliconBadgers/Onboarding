// tb_top.sv -- simulation top: one cub_core plus a 256 KiB byte-wide host memory.
//
// cocotb drives clk, rst_n and start, and reaches the memory contents through the
// hierarchy as tb_top.host_mem.mem[i]. Pass +DRAM_HEX=<file> to preload the memory
// with a DRAM image (one byte per line, as written by `python -m cub hex`).

`default_nettype none
`timescale 1ns/1ps

module host_mem #(
    parameter int AW = 18
) (
    input  wire logic          clk,
    input  wire logic [AW-1:0] addr,
    input  wire logic          we,
    input  wire logic [7:0]    wdata,
    output      logic [7:0]    rdata
);
    logic [7:0] mem [0:(1 << AW) - 1];

    string hex_file;
    initial begin
        if ($value$plusargs("DRAM_HEX=%s", hex_file)) begin
            $display("host_mem: loading %s", hex_file);
            $readmemh(hex_file, mem);
        end
    end

    always_ff @(posedge clk) begin
        if (we)
            mem[addr] <= wdata;
        rdata <= mem[addr];
    end
endmodule

module tb_top;
    localparam int AW = 18;

    logic          clk;
    logic          rst_n;
    logic          start;
    logic          busy;
    logic          done;
    logic [AW-1:0] dram_addr;
    logic          dram_we;
    logic [7:0]    dram_wdata;
    logic [7:0]    dram_rdata;

    cub_core #(.DRAM_AW(AW)) core (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .busy       (busy),
        .done       (done),
        .dram_addr  (dram_addr),
        .dram_we    (dram_we),
        .dram_wdata (dram_wdata),
        .dram_rdata (dram_rdata)
    );

    host_mem #(.AW(AW)) host_mem (
        .clk   (clk),
        .addr  (dram_addr),
        .we    (dram_we),
        .wdata (dram_wdata),
        .rdata (dram_rdata)
    );
endmodule

`default_nettype wire
