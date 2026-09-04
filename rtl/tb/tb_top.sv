// tb_top.sv -- simulation top: one accelerator core plus a 256 KiB byte-wide memory.
//
// cocotb drives clk, reset_n and start, and reaches the memory contents through the
// hierarchy as tb_top.main_memory.storage[i]. Pass +MEMORY_HEX=<file> to preload the
// memory with an image (one byte per line, as written by
// `python -m cub memory-image`).

`default_nettype none
`timescale 1ns/1ps

module main_memory_model #(
    parameter int ADDRESS_WIDTH = 18
) (
    input  wire logic                     clk,
    input  wire logic [ADDRESS_WIDTH-1:0] address,
    input  wire logic                     write_enable,
    input  wire logic [7:0]               write_data,
    output      logic [7:0]               read_data
);
    logic [7:0] storage [0:(1 << ADDRESS_WIDTH) - 1];

    string hex_file;
    initial begin
        if ($value$plusargs("MEMORY_HEX=%s", hex_file)) begin
            $display("main_memory_model: loading %s", hex_file);
            $readmemh(hex_file, storage);
        end
    end

    always_ff @(posedge clk) begin
        if (write_enable)
            storage[address] <= write_data;
        read_data <= storage[address];
    end
endmodule

module tb_top;
    localparam int ADDRESS_WIDTH = 18;

    logic                     clk;
    logic                     reset_n;
    logic                     start;
    logic                     busy;
    logic                     done;
    logic [ADDRESS_WIDTH-1:0] memory_address;
    logic                     memory_write_enable;
    logic [7:0]               memory_write_data;
    logic [7:0]               memory_read_data;

    cub_core #(.MEMORY_ADDRESS_WIDTH(ADDRESS_WIDTH)) core (
        .clk                 (clk),
        .reset_n             (reset_n),
        .start               (start),
        .busy                (busy),
        .done                (done),
        .memory_address      (memory_address),
        .memory_write_enable (memory_write_enable),
        .memory_write_data   (memory_write_data),
        .memory_read_data    (memory_read_data)
    );

    main_memory_model #(.ADDRESS_WIDTH(ADDRESS_WIDTH)) main_memory (
        .clk          (clk),
        .address      (memory_address),
        .write_enable (memory_write_enable),
        .write_data   (memory_write_data),
        .read_data    (memory_read_data)
    );
endmodule

`default_nettype wire
