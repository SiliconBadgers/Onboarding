// cub_core.sv -- the Cub v1 accelerator core (docs/isa.md, Stage 8).
//
// One instruction at a time, one multiply-accumulate per cycle. The structure is a
// single state machine: fetch 16 bytes, decode, run one of the execute states until
// its counter runs out, fetch again. Each execute state is a direct transcription of
// the pseudocode in docs/isa.md section 4, and the comments quote it.
//
// Memory interface (docs/isa.md section 7): a byte-wide port to DRAM with a
// one-cycle synchronous read. Present dram_addr in cycle t; dram_rdata holds that
// byte in cycle t+1. Writes take effect at the clock edge of the cycle in which
// dram_we is high.
//
// Internal memories:
//   SPAD_A  4096 x INT8    synchronous read (block RAM)
//   SPAD_W  131072 x INT8  synchronous read (block RAM)
//   SPAD_B  256 x INT32    asynchronous read (distributed RAM)
//   ACC     256 x INT32    asynchronous read (distributed RAM)
//
// The two big scratchpads are synchronous-read so they map to block RAM on the FPGA.
// That is why MATMUL and STORE-from-SPAD_A are written as one-cycle pipelines: the
// address goes out in one cycle and the data comes back in the next.

`default_nettype none
`timescale 1ns/1ps

module cub_core #(
    parameter int DRAM_AW = 18                  // 2**18 = 256 KiB of DRAM
) (
    input  wire logic               clk,
    input  wire logic               rst_n,      // synchronous, active low
    input  wire logic               start,      // pulse: run the program at DRAM address 0
    output      logic               busy,
    output      logic               done,       // high after HALT until the next start
    output      logic [DRAM_AW-1:0] dram_addr,
    output      logic               dram_we,
    output      logic [7:0]         dram_wdata,
    input  wire logic [7:0]         dram_rdata  // valid the cycle after dram_addr
);

    // ---------------------------------------------------------------------------
    // Opcodes and memory codes (docs/isa.md section 4)
    // ---------------------------------------------------------------------------
    localparam logic [7:0] OP_NOP      = 8'h00;
    localparam logic [7:0] OP_LOAD     = 8'h01;
    localparam logic [7:0] OP_STORE    = 8'h02;
    localparam logic [7:0] OP_MATMUL   = 8'h10;
    localparam logic [7:0] OP_ADD_BIAS = 8'h20;
    localparam logic [7:0] OP_RELU     = 8'h30;
    localparam logic [7:0] OP_HALT     = 8'hFF;

    localparam logic [1:0] MEM_SPAD_A = 2'd0;
    localparam logic [1:0] MEM_SPAD_W = 2'd1;
    localparam logic [1:0] MEM_SPAD_B = 2'd2;
    localparam logic [1:0] MEM_ACC    = 2'd3;

    // ---------------------------------------------------------------------------
    // Scratchpads (docs/isa.md section 2)
    // ---------------------------------------------------------------------------
    logic signed [7:0]  spad_a [0:4095];
    logic signed [7:0]  spad_w [0:131071];
    logic signed [31:0] spad_b [0:255];
    logic signed [31:0] acc    [0:255];

    // Synchronous read ports for the two block-RAM scratchpads.
    logic [11:0]        spa_raddr;
    logic [16:0]        spw_raddr;
    logic signed [7:0]  spa_rdata;
    logic signed [7:0]  spw_rdata;

    always_ff @(posedge clk) begin
        spa_rdata <= spad_a[spa_raddr];
        spw_rdata <= spad_w[spw_raddr];
    end

    // ---------------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE,
        S_FETCH,
        S_DECODE,
        S_LOAD,
        S_STORE,
        S_MATMUL,
        S_ADD_BIAS,
        S_RELU,
        S_HALT
    } state_t;

    state_t             state;
    logic [DRAM_AW-1:0] pc;         // byte address of the instruction being fetched
    logic [127:0]       insn;       // the current instruction
    logic [4:0]         fcnt;       // bytes fetched so far (0..16)
    logic [25:0]        idx;        // element / byte counter for LOAD, STORE, ADD_BIAS, RELU
    logic [23:0]        word_buf;   // low three bytes of an INT32 being assembled by LOAD

    // MATMUL pipeline
    logic               mm_issuing; // still presenting addresses
    logic [15:0]        row;        // n, the output being produced
    logic [15:0]        col;        // k, the position within the reduction
    logic [23:0]        w_ptr;      // SPAD_W index of the next weight (avoids row*K+col)
    logic               mac_valid;  // the read issued last cycle has data this cycle
    logic               mac_last;   // ... and it is the last element of its row
    logic [15:0]        mac_row;    // ... for this row
    logic signed [31:0] mac_sum;    // running dot product for the current row
    logic signed [31:0] mac_sum_next;

    // ---------------------------------------------------------------------------
    // Instruction fields (docs/isa.md section 4; bit positions match cub/isa.py)
    // ---------------------------------------------------------------------------
    wire logic [7:0]  opcode      = insn[7:0];
    wire logic        f_flag8     = insn[8];        // MATMUL.accumulate / RELU.relu
    wire logic [1:0]  f_mem       = insn[9:8];      // LOAD/STORE
    wire logic [31:0] f_dram      = insn[47:16];    // LOAD/STORE
    wire logic [23:0] f_spad      = insn[71:48];    // LOAD/STORE
    wire logic [23:0] f_count     = insn[95:72];    // LOAD/STORE
    wire logic [15:0] f_mm_a      = insn[31:16];    // MATMUL
    wire logic [23:0] f_mm_w      = insn[55:32];
    wire logic [15:0] f_mm_acc    = insn[71:56];
    wire logic [15:0] f_mm_n      = insn[87:72];
    wire logic [15:0] f_mm_k      = insn[103:88];
    wire logic [15:0] f_ab_acc    = insn[31:16];    // ADD_BIAS
    wire logic [15:0] f_ab_bias   = insn[47:32];
    wire logic [15:0] f_ab_count  = insn[63:48];
    wire logic [15:0] f_r_acc     = insn[31:16];    // RELU
    wire logic [15:0] f_r_dst     = insn[47:32];
    wire logic [15:0] f_r_count   = insn[63:48];
    wire logic [7:0]  f_r_shift   = insn[71:64];

    // LOAD/STORE geometry. Elements are 4 bytes for SPAD_B and ACC, else 1 byte.
    wire logic        wide       = (f_mem == MEM_SPAD_B) || (f_mem == MEM_ACC);
    wire logic [25:0] total      = wide ? {f_count, 2'b00} : {2'b00, f_count};
    wire logic [25:0] got        = idx - 26'd1;             // byte whose data is valid now
    wire logic [23:0] elem1      = f_spad + got[23:0];      // element index, 1-byte elements
    wire logic [23:0] elem4      = f_spad + got[25:2];      // element index, 4-byte elements
    wire logic [23:0] issue_elem = f_spad + idx[23:0];      // SPAD_A element to read next (STORE)

    // ADD_BIAS / RELU element addresses
    wire logic [15:0] ab_acc_i   = f_ab_acc  + idx[15:0];
    wire logic [15:0] ab_bias_i  = f_ab_bias + idx[15:0];
    wire logic [15:0] r_acc_i    = f_r_acc   + idx[15:0];
    wire logic [15:0] r_dst_i    = f_r_dst   + idx[15:0];
    wire logic [15:0] mm_acc_i   = f_mm_acc  + mac_row;
    wire logic [15:0] mm_a_i     = f_mm_a    + col;

    // ---------------------------------------------------------------------------
    // The MAC. mac_sum_next is what mac_sum becomes after consuming one
    // (activation, weight) pair.
    //
    //     s += SPAD_A[a_addr + k] * SPAD_W[w_addr + n*K + k]
    // ---------------------------------------------------------------------------
    always_comb begin
        mac_sum_next = mac_sum;   // with the blank unfilled nothing accumulates
        // TODO(onboard, stage 8): multiply the activation and weight bytes as signed values and add the product to mac_sum
    end

    // ---------------------------------------------------------------------------
    // The activation function. relu_in is ACC[acc_addr + i]; relu_out is the INT8
    // that goes to SPAD_A[dst_addr + i].
    //
    //     if relu: v = max(v, 0)
    //     v = v >> shift                        (arithmetic)
    //     out = clamp(v, -128, 127)
    // ---------------------------------------------------------------------------
    logic signed [31:0] relu_in;
    logic signed [31:0] relu_tmp;
    logic signed [7:0]  relu_out;

    assign relu_in = acc[r_acc_i[7:0]];

    always_comb begin
        relu_tmp = relu_in;
        relu_out = 8'sd0;         // with the blank unfilled every activation is zero
        // TODO(onboard, stage 8): ReLU when the relu flag is set, arithmetic shift right by the shift field, then saturate to [-128, 127]
    end

    // ---------------------------------------------------------------------------
    // DRAM port and scratchpad read addresses (combinational, by state)
    // ---------------------------------------------------------------------------
    logic signed [31:0] st_acc_word;
    assign st_acc_word = acc[elem4[7:0]];

    always_comb begin
        dram_addr  = '0;
        dram_we    = 1'b0;
        dram_wdata = 8'h00;
        spa_raddr  = '0;
        spw_raddr  = '0;
        case (state)
            S_FETCH: begin
                dram_addr = pc + DRAM_AW'(fcnt);
            end
            S_LOAD: begin
                dram_addr = DRAM_AW'(f_dram + {6'd0, idx});
            end
            S_STORE: begin
                // Read SPAD_A element idx now; write byte 'got' (read last cycle) now.
                spa_raddr  = issue_elem[11:0];
                dram_addr  = DRAM_AW'(f_dram + {6'd0, got});
                dram_we    = (idx != 26'd0);
                dram_wdata = (f_mem == MEM_ACC) ? st_acc_word[8*got[1:0] +: 8] : spa_rdata;
            end
            S_MATMUL: begin
                spa_raddr = mm_a_i[11:0];
                spw_raddr = w_ptr[16:0];
            end
            default: ;
        endcase
    end

    assign busy = (state != S_IDLE);

    // ---------------------------------------------------------------------------
    // The state machine
    // ---------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            pc         <= '0;
            insn       <= '0;
            fcnt       <= '0;
            idx        <= '0;
            word_buf   <= '0;
            mm_issuing <= 1'b0;
            row        <= '0;
            col        <= '0;
            w_ptr      <= '0;
            mac_valid  <= 1'b0;
            mac_last   <= 1'b0;
            mac_row    <= '0;
            mac_sum    <= '0;
        end else begin
            case (state)

            // Wait for start. Programs always begin at DRAM address 0.
            S_IDLE: begin
                if (start) begin
                    done  <= 1'b0;
                    pc    <= '0;
                    fcnt  <= '0;
                    state <= S_FETCH;
                end
            end

            // Fetch 16 bytes. Cycle fcnt presents address pc+fcnt; the byte for
            // address pc+fcnt-1 arrives this cycle and is shifted in from the top,
            // so after 16 shifts byte 0 (the opcode) sits in insn[7:0].
            S_FETCH: begin
                if (fcnt != 5'd0)
                    insn <= {dram_rdata, insn[127:8]};
                if (fcnt == 5'd16) begin
                    fcnt  <= '0;
                    pc    <= pc + DRAM_AW'(16);
                    state <= S_DECODE;
                end else begin
                    fcnt <= fcnt + 5'd1;
                end
            end

            // Dispatch on the opcode. Zero-length operations and illegal memory
            // combinations are skipped, and anything unknown is a NOP.
            S_DECODE: begin
                idx      <= '0;
                word_buf <= '0;
                state    <= S_FETCH;
                case (opcode)
                    OP_LOAD:
                        if (f_mem != MEM_ACC && total != 26'd0)
                            state <= S_LOAD;
                    OP_STORE:
                        if ((f_mem == MEM_SPAD_A || f_mem == MEM_ACC) && total != 26'd0)
                            state <= S_STORE;
                    OP_MATMUL:
                        if (f_mm_n != 16'd0 && f_mm_k != 16'd0) begin
                            mm_issuing <= 1'b1;
                            row        <= '0;
                            col        <= '0;
                            w_ptr      <= f_mm_w;
                            mac_valid  <= 1'b0;
                            mac_sum    <= '0;
                            state      <= S_MATMUL;
                        end
                    OP_ADD_BIAS:
                        if (f_ab_count != 16'd0)
                            state <= S_ADD_BIAS;
                    OP_RELU:
                        if (f_r_count != 16'd0)
                            state <= S_RELU;
                    OP_HALT:
                        state <= S_HALT;
                    default: ;   // NOP and unknown opcodes
                endcase
            end

            // LOAD: copy 'total' bytes from DRAM into the scratchpad, one per cycle.
            // Cycle idx presents address dram_addr+idx; byte 'got' = idx-1 arrives.
            // For SPAD_B the four bytes of each INT32 are collected little-endian.
            //
            //     for i in 0 .. count-1: SPAD[spad_addr + i] = DRAM[dram_addr + i*width ...]
            S_LOAD: begin
                if (idx != 26'd0) begin
                    case (f_mem)
                        MEM_SPAD_A: spad_a[elem1[11:0]] <= dram_rdata;
                        MEM_SPAD_W: spad_w[elem1[16:0]] <= dram_rdata;
                        MEM_SPAD_B: begin
                            case (got[1:0])
                                2'd0: word_buf[7:0]   <= dram_rdata;
                                2'd1: word_buf[15:8]  <= dram_rdata;
                                2'd2: word_buf[23:16] <= dram_rdata;
                                2'd3: spad_b[elem4[7:0]] <= {dram_rdata, word_buf};
                            endcase
                        end
                        default: ;
                    endcase
                end
                if (idx == total)
                    state <= S_FETCH;
                else
                    idx <= idx + 26'd1;
            end

            // STORE: the DRAM write itself is driven combinationally above; this
            // state only runs the counter. Byte 'got' is written while element idx
            // is being read from SPAD_A for the next cycle.
            //
            //     for i in 0 .. count-1: DRAM[dram_addr + i*width ...] = SPAD[spad_addr + i]
            S_STORE: begin
                if (idx == total)
                    state <= S_FETCH;
                else
                    idx <= idx + 26'd1;
            end

            // MATMUL: a one-cycle pipeline over the two block-RAM scratchpads.
            // Issue side: present SPAD_A[a + col] and SPAD_W[w_ptr] for every (row, col).
            // Consume side, one cycle later: add the product into mac_sum, and at the
            // end of a row write it (plus the old ACC value if accumulating) to ACC.
            //
            //     for n: s = 0; for k: s += A[a+k] * W[w+n*K+k]; ACC[acc+n] = (acc? ACC[acc+n] : 0) + s
            S_MATMUL: begin
                if (mm_issuing) begin
                    w_ptr <= w_ptr + 24'd1;
                    if (col == f_mm_k - 16'd1) begin
                        col <= '0;
                        row <= row + 16'd1;
                        if (row == f_mm_n - 16'd1)
                            mm_issuing <= 1'b0;
                    end else begin
                        col <= col + 16'd1;
                    end
                end
                mac_valid <= mm_issuing;
                mac_last  <= (col == f_mm_k - 16'd1);
                mac_row   <= row;
                if (mac_valid) begin
                    if (mac_last) begin
                        acc[mm_acc_i[7:0]] <= (f_flag8 ? acc[mm_acc_i[7:0]] : 32'sd0) + mac_sum_next;
                        mac_sum <= '0;
                    end else begin
                        mac_sum <= mac_sum_next;
                    end
                end
                if (!mm_issuing && !mac_valid)
                    state <= S_FETCH;
            end

            // ADD_BIAS: one element per cycle, INT32 wrapping add.
            //
            //     for i in 0 .. count-1: ACC[acc_addr + i] += SPAD_B[bias_addr + i]
            S_ADD_BIAS: begin
                acc[ab_acc_i[7:0]] <= acc[ab_acc_i[7:0]] + spad_b[ab_bias_i[7:0]];
                if (idx == {10'd0, f_ab_count} - 26'd1)
                    state <= S_FETCH;
                else
                    idx <= idx + 26'd1;
            end

            // RELU: one element per cycle through the activation function above.
            //
            //     for i in 0 .. count-1: SPAD_A[dst_addr + i] = clamp((relu(ACC[acc_addr + i])) >> shift)
            S_RELU: begin
                spad_a[r_dst_i[11:0]] <= relu_out;
                if (idx == {10'd0, f_r_count} - 26'd1)
                    state <= S_FETCH;
                else
                    idx <= idx + 26'd1;
            end

            // HALT: raise done and go back to waiting.
            S_HALT: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
