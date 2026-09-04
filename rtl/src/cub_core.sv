// cub_core.sv -- the accelerator core (docs/02-instruction-set.md, Stage 6).
//
// One instruction at a time, one multiply per cycle. The whole chip is a single state
// machine: fetch 16 bytes, decode them, run one execute state until its counter runs
// out, fetch again. Each execute state is a direct transcription of the pseudocode in
// docs/02-instruction-set.md, and the comments quote it.
//
// How the host drives it (docs/05-registers-and-memory.md):
//   start   pulse high for one cycle to run the program sitting at memory address 0
//   busy    high while the core is working
//   done    high after HALT, until the next start
// On a real chip those three wires are bits in a control register and a status
// register that the host reads and writes over a bus. Here they are just wires,
// because the testbench is the host.
//
// Main memory interface: one byte-wide port with a one-cycle synchronous read.
// Present memory_address in cycle t; memory_read_data holds that byte in cycle t+1.
// Writes take effect at the clock edge of the cycle in which memory_write_enable is high.
//
// The four memory spaces inside the core:
//   ACTIVATION_SCRATCHPAD  4096 x 8 bits     synchronous read (block RAM)
//   WEIGHT_SCRATCHPAD      131072 x 8 bits   synchronous read (block RAM)
//   BIAS_SCRATCHPAD        256 x 32 bits     asynchronous read (distributed RAM)
//   ACCUMULATORS           256 x 32 bits     asynchronous read (distributed RAM)
//
// The two big scratchpads are synchronous-read so a synthesis tool maps them onto
// block RAM. That is why MATRIX_MULTIPLY and STORE are written as one-cycle
// pipelines: the address goes out in one cycle and the data comes back in the next.

`default_nettype none
`timescale 1ns/1ps

module cub_core #(
    parameter int MEMORY_ADDRESS_WIDTH = 18     // 2**18 = 256 KiB of main memory
) (
    input  wire logic                            clk,
    input  wire logic                            reset_n,   // synchronous, active low
    input  wire logic                            start,     // pulse: run the program at address 0
    output      logic                            busy,
    output      logic                            done,      // high after HALT until the next start
    output      logic [MEMORY_ADDRESS_WIDTH-1:0] memory_address,
    output      logic                            memory_write_enable,
    output      logic [7:0]                      memory_write_data,
    input  wire logic [7:0]                      memory_read_data  // valid one cycle later
);

    // ---------------------------------------------------------------------------
    // Opcodes and memory space codes (docs/02-instruction-set.md)
    // ---------------------------------------------------------------------------
    localparam logic [7:0] OPCODE_NO_OPERATION     = 8'h00;
    localparam logic [7:0] OPCODE_LOAD             = 8'h01;
    localparam logic [7:0] OPCODE_STORE            = 8'h02;
    localparam logic [7:0] OPCODE_MATRIX_MULTIPLY  = 8'h10;
    localparam logic [7:0] OPCODE_ADD_BIAS         = 8'h20;
    localparam logic [7:0] OPCODE_RECTIFIED_LINEAR = 8'h30;
    localparam logic [7:0] OPCODE_HALT             = 8'hFF;

    localparam logic [1:0] SPACE_ACTIVATION_SCRATCHPAD = 2'd0;
    localparam logic [1:0] SPACE_WEIGHT_SCRATCHPAD     = 2'd1;
    localparam logic [1:0] SPACE_BIAS_SCRATCHPAD       = 2'd2;
    localparam logic [1:0] SPACE_ACCUMULATORS          = 2'd3;

    // ---------------------------------------------------------------------------
    // The four on-chip memory spaces
    // ---------------------------------------------------------------------------
    logic signed [7:0]  activation_scratchpad [0:4095];
    logic signed [7:0]  weight_scratchpad     [0:131071];
    logic signed [31:0] bias_scratchpad       [0:255];
    logic signed [31:0] accumulators          [0:255];

    // Synchronous read ports for the two block-RAM scratchpads.
    logic [11:0]        activation_read_index;
    logic [16:0]        weight_read_index;
    logic signed [7:0]  activation_read_data;
    logic signed [7:0]  weight_read_data;

    always_ff @(posedge clk) begin
        activation_read_data <= activation_scratchpad[activation_read_index];
        weight_read_data     <= weight_scratchpad[weight_read_index];
    end

    // ---------------------------------------------------------------------------
    // Registers that hold what the core is currently doing
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        STATE_IDLE,
        STATE_FETCH,
        STATE_DECODE,
        STATE_LOAD,
        STATE_STORE,
        STATE_MATRIX_MULTIPLY,
        STATE_ADD_BIAS,
        STATE_RECTIFIED_LINEAR,
        STATE_HALT
    } state_t;

    state_t                          state;
    logic [MEMORY_ADDRESS_WIDTH-1:0] program_counter;  // address of the instruction being fetched
    logic [127:0]                    instruction;      // the instruction register
    logic [4:0]                      fetch_count;      // bytes fetched so far (0..16)
    logic [25:0]                     element_index;    // counter for LOAD/STORE/ADD_BIAS/RECTIFIED_LINEAR
    logic [23:0]                     word_buffer;      // low three bytes of a 32-bit value being assembled

    // MATRIX_MULTIPLY pipeline
    logic               issuing_reads;    // still presenting scratchpad addresses
    logic [15:0]        output_row;       // which output we are producing
    logic [15:0]        input_column;     // where we are within the reduction
    logic [23:0]        weight_pointer;   // weight scratchpad index (avoids recomputing row*inputs+column)
    logic               product_valid;    // the read issued last cycle has data this cycle
    logic               product_last;     // ... and it is the last element of its row
    logic [15:0]        product_row;      // ... belonging to this output
    logic signed [31:0] running_sum;      // dot product so far for the current output
    logic signed [31:0] running_sum_next;

    // ---------------------------------------------------------------------------
    // Instruction operands, sliced straight out of the instruction register.
    // The bit ranges are the same ones listed in docs/02-instruction-set.md and in
    // cub/instruction_set.py. This is the whole of "decode".
    // ---------------------------------------------------------------------------
    wire logic [7:0]  opcode    = instruction[7:0];
    wire logic        flag_bit  = instruction[8];     // MATRIX_MULTIPLY.accumulate / RECTIFIED_LINEAR.rectify

    wire logic [1:0]  field_space  = instruction[9:8];      // LOAD / STORE
    wire logic [31:0] field_memory = instruction[47:16];
    wire logic [23:0] field_index  = instruction[71:48];
    wire logic [23:0] field_count  = instruction[95:72];

    wire logic [15:0] field_multiply_input       = instruction[31:16];   // MATRIX_MULTIPLY
    wire logic [23:0] field_multiply_weights     = instruction[55:32];
    wire logic [15:0] field_multiply_accumulator = instruction[71:56];
    wire logic [15:0] field_multiply_outputs     = instruction[87:72];
    wire logic [15:0] field_multiply_inputs      = instruction[103:88];

    wire logic [15:0] field_bias_accumulator = instruction[31:16];       // ADD_BIAS
    wire logic [15:0] field_bias_index       = instruction[47:32];
    wire logic [15:0] field_bias_count       = instruction[63:48];

    wire logic [15:0] field_rectify_accumulator = instruction[31:16];    // RECTIFIED_LINEAR
    wire logic [15:0] field_rectify_destination = instruction[47:32];
    wire logic [15:0] field_rectify_count       = instruction[63:48];
    wire logic [7:0]  field_rectify_shift       = instruction[71:64];

    // LOAD / STORE geometry. An element is four bytes for the 32-bit spaces, one byte
    // for the 8-bit ones, so `count` elements is `total_bytes` bytes of main memory.
    wire logic        element_is_wide   = (field_space == SPACE_BIAS_SCRATCHPAD)
                                       || (field_space == SPACE_ACCUMULATORS);
    wire logic [25:0] total_bytes       = element_is_wide ? {field_count, 2'b00}
                                                          : {2'b00, field_count};
    wire logic [25:0] byte_ready        = element_index - 26'd1;      // byte whose data is valid now
    wire logic [23:0] narrow_element    = field_index + byte_ready[23:0];
    wire logic [23:0] wide_element      = field_index + byte_ready[25:2];
    wire logic [23:0] store_read_element = field_index + element_index[23:0];

    // ADD_BIAS / RECTIFIED_LINEAR / MATRIX_MULTIPLY element addresses
    wire logic [15:0] bias_accumulator_index = field_bias_accumulator    + element_index[15:0];
    wire logic [15:0] bias_source_index      = field_bias_index          + element_index[15:0];
    wire logic [15:0] rectify_source_index   = field_rectify_accumulator + element_index[15:0];
    wire logic [15:0] rectify_dest_index     = field_rectify_destination + element_index[15:0];
    wire logic [15:0] multiply_output_index  = field_multiply_accumulator + product_row;
    wire logic [15:0] multiply_input_index   = field_multiply_input       + input_column;

    // ---------------------------------------------------------------------------
    // The multiplier. running_sum_next is what running_sum becomes after consuming
    // one (activation, weight) pair.
    //
    //     total += ACTIVATION_SCRATCHPAD[input + k] * WEIGHT_SCRATCHPAD[weights + n*inputs + k]
    // ---------------------------------------------------------------------------
    always_comb begin
        running_sum_next = running_sum;   // with the blank unfilled nothing accumulates
        // TODO(onboard, stage 6): set running_sum_next to running_sum plus the product of the two bytes coming out of the scratchpads this cycle, activation_read_data and weight_read_data. Both are already declared `logic signed`, so a plain `*` between them is a signed multiply -- do not cast either one, and do not introduce an unsigned intermediate.
    end

    // ---------------------------------------------------------------------------
    // The activation function. rectify_input is ACCUMULATORS[accumulator + i];
    // rectify_output is the 8-bit value written to ACTIVATION_SCRATCHPAD[destination + i].
    //
    //     if rectify: value = max(value, 0)
    //     value = value >> shift                    (arithmetic)
    //     out = clamp(value, -128, 127)
    // ---------------------------------------------------------------------------
    logic signed [31:0] rectify_input;
    logic signed [31:0] rectify_value;
    logic signed [7:0]  rectify_output;

    assign rectify_input = accumulators[rectify_source_index[7:0]];

    always_comb begin
        rectify_value  = rectify_input;
        rectify_output = 8'sd0;          // with the blank unfilled every activation is zero
        // TODO(onboard, stage 6): starting from rectify_value = rectify_input, do three things. (1) If flag_bit is set and rectify_input is negative (its bit 31 is 1), set rectify_value to zero. (2) Shift rectify_value right by field_rectify_shift[4:0] using the arithmetic shift `>>>`, not `>>`, so negative values stay negative. (3) Set rectify_output to rectify_value clamped: 127 if it is above 127, -128 if it is below -128, otherwise its low 8 bits.
    end

    // ---------------------------------------------------------------------------
    // Main memory port and scratchpad read addresses (combinational, by state)
    // ---------------------------------------------------------------------------
    logic signed [31:0] store_accumulator_word;
    assign store_accumulator_word = accumulators[wide_element[7:0]];

    always_comb begin
        memory_address        = '0;
        memory_write_enable   = 1'b0;
        memory_write_data     = 8'h00;
        activation_read_index = '0;
        weight_read_index     = '0;
        case (state)
            STATE_FETCH: begin
                memory_address = program_counter + MEMORY_ADDRESS_WIDTH'(fetch_count);
            end
            STATE_LOAD: begin
                memory_address = MEMORY_ADDRESS_WIDTH'(field_memory + {6'd0, element_index});
            end
            STATE_STORE: begin
                // Read element `element_index` from the scratchpad now; write the byte
                // read last cycle (`byte_ready`) to main memory now.
                activation_read_index = store_read_element[11:0];
                memory_address        = MEMORY_ADDRESS_WIDTH'(field_memory + {6'd0, byte_ready});
                memory_write_enable   = (element_index != 26'd0);
                memory_write_data     = (field_space == SPACE_ACCUMULATORS)
                                      ? store_accumulator_word[8*byte_ready[1:0] +: 8]
                                      : activation_read_data;
            end
            STATE_MATRIX_MULTIPLY: begin
                activation_read_index = multiply_input_index[11:0];
                weight_read_index     = weight_pointer[16:0];
            end
            default: ;
        endcase
    end

    assign busy = (state != STATE_IDLE);

    // ---------------------------------------------------------------------------
    // The state machine
    // ---------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!reset_n) begin
            state           <= STATE_IDLE;
            done            <= 1'b0;
            program_counter <= '0;
            instruction     <= '0;
            fetch_count     <= '0;
            element_index   <= '0;
            word_buffer     <= '0;
            issuing_reads   <= 1'b0;
            output_row      <= '0;
            input_column    <= '0;
            weight_pointer  <= '0;
            product_valid   <= 1'b0;
            product_last    <= 1'b0;
            product_row     <= '0;
            running_sum     <= '0;
        end else begin
            case (state)

            // Wait for the host to set the start bit. Programs always begin at
            // main-memory address 0.
            STATE_IDLE: begin
                if (start) begin
                    done            <= 1'b0;
                    program_counter <= '0;
                    fetch_count     <= '0;
                    state           <= STATE_FETCH;
                end
            end

            // Fetch 16 bytes. Cycle `fetch_count` presents address
            // program_counter+fetch_count; the byte for the previous address arrives
            // this cycle and is shifted in from the top, so after 16 shifts byte 0
            // (the opcode) sits in instruction[7:0].
            STATE_FETCH: begin
                if (fetch_count != 5'd0)
                    instruction <= {memory_read_data, instruction[127:8]};
                if (fetch_count == 5'd16) begin
                    fetch_count     <= '0;
                    program_counter <= program_counter + MEMORY_ADDRESS_WIDTH'(16);
                    state           <= STATE_DECODE;
                end else begin
                    fetch_count <= fetch_count + 5'd1;
                end
            end

            // Dispatch on the opcode. Zero-length operations and illegal memory space
            // combinations are skipped, and anything unknown does nothing.
            STATE_DECODE: begin
                element_index <= '0;
                word_buffer   <= '0;
                state         <= STATE_FETCH;
                case (opcode)
                    OPCODE_LOAD:
                        if (field_space != SPACE_ACCUMULATORS && total_bytes != 26'd0)
                            state <= STATE_LOAD;
                    OPCODE_STORE:
                        if ((field_space == SPACE_ACTIVATION_SCRATCHPAD
                             || field_space == SPACE_ACCUMULATORS) && total_bytes != 26'd0)
                            state <= STATE_STORE;
                    OPCODE_MATRIX_MULTIPLY:
                        if (field_multiply_outputs != 16'd0 && field_multiply_inputs != 16'd0) begin
                            issuing_reads  <= 1'b1;
                            output_row     <= '0;
                            input_column   <= '0;
                            weight_pointer <= field_multiply_weights;
                            product_valid  <= 1'b0;
                            running_sum    <= '0;
                            state          <= STATE_MATRIX_MULTIPLY;
                        end
                    OPCODE_ADD_BIAS:
                        if (field_bias_count != 16'd0)
                            state <= STATE_ADD_BIAS;
                    OPCODE_RECTIFIED_LINEAR:
                        if (field_rectify_count != 16'd0)
                            state <= STATE_RECTIFIED_LINEAR;
                    OPCODE_HALT:
                        state <= STATE_HALT;
                    default: ;   // NO_OPERATION and unknown opcodes
                endcase
            end

            // LOAD: copy `total_bytes` bytes from main memory into a scratchpad, one
            // per cycle. Cycle `element_index` presents the next address; the byte for
            // `byte_ready` arrives. For the 32-bit spaces the four bytes of each value
            // are collected little-endian, low byte first.
            //
            //     for i in 0 .. count-1:
            //         SCRATCHPAD[index + i] = MAIN_MEMORY[memory + i*width ...]
            STATE_LOAD: begin
                if (element_index != 26'd0) begin
                    case (field_space)
                        SPACE_ACTIVATION_SCRATCHPAD:
                            activation_scratchpad[narrow_element[11:0]] <= memory_read_data;
                        SPACE_WEIGHT_SCRATCHPAD:
                            weight_scratchpad[narrow_element[16:0]] <= memory_read_data;
                        SPACE_BIAS_SCRATCHPAD: begin
                            case (byte_ready[1:0])
                                2'd0: word_buffer[7:0]   <= memory_read_data;
                                2'd1: word_buffer[15:8]  <= memory_read_data;
                                2'd2: word_buffer[23:16] <= memory_read_data;
                                2'd3: bias_scratchpad[wide_element[7:0]] <= {memory_read_data, word_buffer};
                            endcase
                        end
                        default: ;
                    endcase
                end
                if (element_index == total_bytes)
                    state <= STATE_FETCH;
                else
                    element_index <= element_index + 26'd1;
            end

            // STORE: the memory write itself is driven combinationally above; this
            // state only runs the counter.
            //
            //     for i in 0 .. count-1:
            //         MAIN_MEMORY[memory + i*width ...] = SCRATCHPAD[index + i]
            STATE_STORE: begin
                if (element_index == total_bytes)
                    state <= STATE_FETCH;
                else
                    element_index <= element_index + 26'd1;
            end

            // MATRIX_MULTIPLY: a one-cycle pipeline over the two block-RAM scratchpads.
            // Issue side: present ACTIVATION_SCRATCHPAD[input + column] and
            // WEIGHT_SCRATCHPAD[weight_pointer] for every (row, column) pair.
            // Consume side, one cycle later: add the product into running_sum, and at
            // the end of a row write it (plus the old accumulator if accumulating).
            //
            //     for n: total = 0
            //            for k: total += A[input+k] * W[weights + n*inputs + k]
            //            ACCUMULATORS[accumulator+n] = (accumulate ? old : 0) + total
            STATE_MATRIX_MULTIPLY: begin
                if (issuing_reads) begin
                    weight_pointer <= weight_pointer + 24'd1;
                    if (input_column == field_multiply_inputs - 16'd1) begin
                        input_column <= '0;
                        output_row   <= output_row + 16'd1;
                        if (output_row == field_multiply_outputs - 16'd1)
                            issuing_reads <= 1'b0;
                    end else begin
                        input_column <= input_column + 16'd1;
                    end
                end
                product_valid <= issuing_reads;
                product_last  <= (input_column == field_multiply_inputs - 16'd1);
                product_row   <= output_row;
                if (product_valid) begin
                    if (product_last) begin
                        accumulators[multiply_output_index[7:0]] <=
                            (flag_bit ? accumulators[multiply_output_index[7:0]] : 32'sd0)
                            + running_sum_next;
                        running_sum <= '0;
                    end else begin
                        running_sum <= running_sum_next;
                    end
                end
                if (!issuing_reads && !product_valid)
                    state <= STATE_FETCH;
            end

            // ADD_BIAS: one element per cycle. A 32-bit adder wraps on overflow all by
            // itself, which is exactly the behaviour the simulator models.
            //
            //     for i in 0 .. count-1:
            //         ACCUMULATORS[accumulator + i] += BIAS_SCRATCHPAD[bias + i]
            STATE_ADD_BIAS: begin
                accumulators[bias_accumulator_index[7:0]] <=
                    accumulators[bias_accumulator_index[7:0]] + bias_scratchpad[bias_source_index[7:0]];
                if (element_index == {10'd0, field_bias_count} - 26'd1)
                    state <= STATE_FETCH;
                else
                    element_index <= element_index + 26'd1;
            end

            // RECTIFIED_LINEAR: one element per cycle through the activation function
            // defined above.
            //
            //     for i in 0 .. count-1:
            //         ACTIVATION_SCRATCHPAD[destination + i] =
            //             clamp(rectify(ACCUMULATORS[accumulator + i]) >> shift)
            STATE_RECTIFIED_LINEAR: begin
                activation_scratchpad[rectify_dest_index[11:0]] <= rectify_output;
                if (element_index == {10'd0, field_rectify_count} - 26'd1)
                    state <= STATE_FETCH;
                else
                    element_index <= element_index + 26'd1;
            end

            // HALT: raise done for the host to see, and go back to waiting.
            STATE_HALT: begin
                done  <= 1'b1;
                state <= STATE_IDLE;
            end

            default: state <= STATE_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
