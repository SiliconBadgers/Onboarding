// cub_top.sv -- board top for the PYNQ-Z2 (Zynq-7020), Stage 9.
//
// The whole demo lives in the fabric: the DRAM image produced by
// `python -m cub hex` (program + weights + one baked-in test image) initializes a
// block RAM, the core runs it when you press a button, and the four LEDs show the
// predicted digit in binary. No processor, no driver, no host link: the smallest
// possible "the hardware works" demonstration.
//
//   btn[0]  reset (active high while pressed)
//   btn[1]  start: run the program once
//   led[3:0]  the predicted digit, once done (all off while idle or running)
//
// The memory initializer is $readmemh("dram.hex", mem). Vivado resolves that path
// relative to the directory it runs in, so rtl/fpga/build.tcl changes into
// rtl/build (where `python -m cub hex` writes the file) before synthesis. To change
// the digit the board classifies, regenerate the hex with a different --index and
// rebuild the bitstream.

`default_nettype none
`timescale 1ns/1ps

module cub_top (
    input  wire logic       sysclk,     // 125 MHz board oscillator
    input  wire logic [1:0] btn,
    output      logic [3:0] led
);
    localparam int AW = 18;

    // ---------------------------------------------------------------------------
    // Buttons: synchronize, debounce (~8 ms at 125 MHz), detect a rising edge.
    // ---------------------------------------------------------------------------
    logic [1:0] btn_sync0, btn_sync1;
    logic [1:0] btn_stable;
    logic [19:0] debounce_cnt [0:1];

    always_ff @(posedge sysclk) begin
        btn_sync0 <= btn;
        btn_sync1 <= btn_sync0;
    end

    // A button counts as stable once it has held one value for 2**20 cycles.
    // These blocks have no reset on purpose: the reset comes *from* one of them.
    generate
        for (genvar i = 0; i < 2; i++) begin : g_debounce
            always_ff @(posedge sysclk) begin
                if (btn_sync1[i] != btn_stable[i]) begin
                    if (debounce_cnt[i] == 20'hFFFFF) begin
                        btn_stable[i]   <= btn_sync1[i];
                        debounce_cnt[i] <= '0;
                    end else begin
                        debounce_cnt[i] <= debounce_cnt[i] + 20'd1;
                    end
                end else begin
                    debounce_cnt[i] <= '0;
                end
            end
        end
    endgenerate

    logic rst_n;
    logic start_prev, start_pulse;
    assign rst_n = ~btn_stable[0];

    always_ff @(posedge sysclk) begin
        start_prev  <= btn_stable[1];
        start_pulse <= btn_stable[1] & ~start_prev;
    end

    // ---------------------------------------------------------------------------
    // Host memory: 256 KiB block RAM initialized from the DRAM image.
    // ---------------------------------------------------------------------------
    logic [7:0] mem [0:(1 << AW) - 1];
    initial $readmemh("dram.hex", mem);

    logic [AW-1:0] mem_addr;
    logic          mem_we;
    logic [7:0]    mem_wdata;
    logic [7:0]    mem_rdata;

    always_ff @(posedge sysclk) begin
        if (mem_we)
            mem[mem_addr] <= mem_wdata;
        mem_rdata <= mem[mem_addr];
    end

    // ---------------------------------------------------------------------------
    // The core.
    // ---------------------------------------------------------------------------
    logic          busy, done;
    logic [AW-1:0] core_addr;
    logic          core_we;
    logic [7:0]    core_wdata;

    cub_core #(.DRAM_AW(AW)) core (
        .clk        (sysclk),
        .rst_n      (rst_n),
        .start      (start_pulse),
        .busy       (busy),
        .done       (done),
        .dram_addr  (core_addr),
        .dram_we    (core_we),
        .dram_wdata (core_wdata),
        .dram_rdata (mem_rdata)
    );

    // ---------------------------------------------------------------------------
    // Argmax: after done, read the 10 INT32 logits from the output region and
    // find the largest. Two cycles per byte (present address, then capture) keeps
    // the timing trivial; 80 cycles is nothing next to the 205 000 the core takes.
    // ---------------------------------------------------------------------------
    localparam logic [AW-1:0] OUTPUT_ADDR = 18'h19680;   // artifacts/mnist.cub 'output' region

    typedef enum logic [1:0] { A_IDLE, A_ADDR, A_DATA, A_DONE } astate_t;
    astate_t            astate;
    logic [5:0]         abyte;        // 0..39
    logic [23:0]        aword;        // low three bytes of the logit being assembled
    logic signed [31:0] best_val;
    logic [3:0]         best_idx;
    logic               done_prev;
    logic [AW-1:0]      argmax_addr;

    wire logic signed [31:0] logit = {mem_rdata, aword};
    wire logic [3:0]         logit_idx = 4'(abyte[5:2]);

    always_ff @(posedge sysclk) begin
        if (!rst_n) begin
            astate    <= A_IDLE;
            abyte     <= '0;
            aword     <= '0;
            best_val  <= '0;
            best_idx  <= '0;
            done_prev <= 1'b0;
            led       <= 4'b0000;
        end else begin
            done_prev <= done;
            case (astate)
                A_IDLE: begin
                    if (start_pulse)
                        led <= 4'b0000;
                    if (done && !done_prev) begin
                        abyte  <= '0;
                        astate <= A_ADDR;
                    end
                end
                A_ADDR: astate <= A_DATA;          // address presented this cycle
                A_DATA: begin                      // mem_rdata is byte 'abyte'
                    case (abyte[1:0])
                        2'd0: aword[7:0]   <= mem_rdata;
                        2'd1: aword[15:8]  <= mem_rdata;
                        2'd2: aword[23:16] <= mem_rdata;
                        2'd3: begin
                            if (logit_idx == 4'd0 || logit > best_val) begin
                                best_val <= logit;
                                best_idx <= logit_idx;
                            end
                        end
                    endcase
                    if (abyte == 6'd39)
                        astate <= A_DONE;
                    else begin
                        abyte  <= abyte + 6'd1;
                        astate <= A_ADDR;
                    end
                end
                A_DONE: begin
                    led    <= best_idx;
                    astate <= A_IDLE;
                end
            endcase
        end
    end

    assign argmax_addr = OUTPUT_ADDR + AW'(abyte);

    // The core owns the memory port while it runs; the argmax reader afterwards.
    always_comb begin
        if (busy) begin
            mem_addr  = core_addr;
            mem_we    = core_we;
            mem_wdata = core_wdata;
        end else begin
            mem_addr  = argmax_addr;
            mem_we    = 1'b0;
            mem_wdata = 8'h00;
        end
    end

endmodule

`default_nettype wire
