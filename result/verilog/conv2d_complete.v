// ==============================================================================
// FINAL FIXED VERSION - conv2d_complete.v 
// Fixed conv_done reset timing and kernel transition logic
// ==============================================================================

`timescale 1ns/1ps

// ==============================================================================
// sliding_window_3x3 - Two line buffers + 3x3 shift register window
// (No changes needed - this module is working correctly)
// ==============================================================================
module sliding_window_3x3 #(
    parameter DATA_W = 8,
    parameter IMG_W  = 64,
    parameter IMG_H  = 64
)(
    input                  clk,
    input                  rst_n,
    input                  in_valid,
    input      [DATA_W-1:0] pixel_in,

    output reg             window_valid,
    output reg [DATA_W-1:0] w00, w01, w02,
    output reg [DATA_W-1:0] w10, w11, w12,
    output reg [DATA_W-1:0] w20, w21, w22
);
    // Two line buffers hold previous 2 rows
    reg [DATA_W-1:0] linebuf0 [0:IMG_W-1];
    reg [DATA_W-1:0] linebuf1 [0:IMG_W-1];

    // Column and row counters
    reg [$clog2(IMG_W)-1:0] col;
    reg [31:0]              row;

    // Taps from three rows aligned at current column
    wire [DATA_W-1:0] tap0 = linebuf1[col]; // row-2
    wire [DATA_W-1:0] tap1 = linebuf0[col]; // row-1
    wire [DATA_W-1:0] tap2 = pixel_in;      // row

    // 3x3 shift register window
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w00 <= 0; w01 <= 0; w02 <= 0;
            w10 <= 0; w11 <= 0; w12 <= 0;
            w20 <= 0; w21 <= 0; w22 <= 0;
        end else if (in_valid) begin
            // Shift left, insert new taps at right
            w00 <= w01; w01 <= w02; w02 <= tap0;
            w10 <= w11; w11 <= w12; w12 <= tap1;
            w20 <= w21; w21 <= w22; w22 <= tap2;
        end
    end

    // Line buffer management and addressing - RESET BUFFERS ON NEW START
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col <= 0;
            row <= 0;
        end else if (in_valid) begin
            // Write to line buffer and shift rows
            linebuf0[col] <= pixel_in;
            if (row > 0) linebuf1[col] <= linebuf0[col];

            // Advance position
            if (col == IMG_W-1) begin
                col <= 0;
                row <= row + 1;
            end else begin
                col <= col + 1;
            end
        end else begin
            // ADDED: Reset position when not streaming (for next kernel)
            col <= 0;
            row <= 0;
        end
    end

    // Window valid only for interior pixels
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_valid <= 1'b0;
        end else if (in_valid) begin
            window_valid <= (row >= 1) && (row < IMG_H-1) && (col >= 1) && (col < IMG_W-1);
        end else begin
            window_valid <= 1'b0;
        end
    end
endmodule

// ==============================================================================
// conv3x3_mac - 3x3 MAC (No changes needed)
// ==============================================================================
module conv3x3_mac #(
    parameter DATA_W  = 8,
    parameter OUT_W   = 16,
    parameter ACC_W   = 32
)(
    input                      clk,
    input                      rst_n,
    input                      in_valid,
    input      signed [DATA_W:0] k00, k01, k02,
    input      signed [DATA_W:0] k10, k11, k12,
    input      signed [DATA_W:0] k20, k21, k22,
    input      [DATA_W-1:0]     w00, w01, w02,
    input      [DATA_W-1:0]     w10, w11, w12,
    input      [DATA_W-1:0]     w20, w21, w22,
    input      [3:0]            kernel_id,
    input                       relu_en,

    output reg                  out_valid,
    output reg [OUT_W-1:0]      px_out
);
    // Sign-extend pixels for signed multiplication
    wire signed [DATA_W:0] p00 = $signed({1'b0, w00});
    wire signed [DATA_W:0] p01 = $signed({1'b0, w01});
    wire signed [DATA_W:0] p02 = $signed({1'b0, w02});
    wire signed [DATA_W:0] p10 = $signed({1'b0, w10});
    wire signed [DATA_W:0] p11 = $signed({1'b0, w11});
    wire signed [DATA_W:0] p12 = $signed({1'b0, w12});
    wire signed [DATA_W:0] p20 = $signed({1'b0, w20});
    wire signed [DATA_W:0] p21 = $signed({1'b0, w21});
    wire signed [DATA_W:0] p22 = $signed({1'b0, w22});

    // 9 multiply-accumulate operations
    wire signed [ACC_W-1:0] m00 = p00 * k00;
    wire signed [ACC_W-1:0] m01 = p01 * k01;
    wire signed [ACC_W-1:0] m02 = p02 * k02;
    wire signed [ACC_W-1:0] m10 = p10 * k10;
    wire signed [ACC_W-1:0] m11 = p11 * k11;
    wire signed [ACC_W-1:0] m12 = p12 * k12;
    wire signed [ACC_W-1:0] m20 = p20 * k20;
    wire signed [ACC_W-1:0] m21 = p21 * k21;
    wire signed [ACC_W-1:0] m22 = p22 * k22;

    // Three-stage adder tree
    wire signed [ACC_W-1:0] s0 = m00 + m01 + m02;
    wire signed [ACC_W-1:0] s1 = m10 + m11 + m12;
    wire signed [ACC_W-1:0] s2 = m20 + m21 + m22;
    wire signed [ACC_W-1:0] acc = s0 + s1 + s2;

    // Kernel-specific normalization
    reg signed [ACC_W-1:0] norm;
    always @* begin
        case (kernel_id)
            4:  norm = acc / 9;   // Box blur: exact division by 9
            5:  norm = acc >>> 4; // Gaussian: exact right shift by 4 (÷16)
            default: norm = acc;  // Identity/edge filters: no scaling
        endcase
    end

    // ReLU activation and saturation
    wire signed [ACC_W-1:0] relu_applied = (relu_en && norm < 0) ? {ACC_W{1'b0}} : norm;
    wire [OUT_W-1:0] clipped =
        (relu_applied > $signed({1'b0, {OUT_W{1'b1}}})) ? {OUT_W{1'b1}} :
        (relu_applied < 0) ? {OUT_W{1'b0}} :
        relu_applied[OUT_W-1:0];

    // Pipeline output register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            px_out    <= {OUT_W{1'b0}};
        end else begin
            out_valid <= in_valid;
            px_out    <= clipped;
        end
    end
endmodule

// ==============================================================================
// conv2d_engine - FIXED: Proper state reset between kernels
// ==============================================================================
module conv2d_engine #(
    parameter INPUT_WIDTH  = 8,
    parameter OUTPUT_WIDTH = 16,
    parameter IMAGE_WIDTH  = 64,
    parameter IMAGE_HEIGHT = 64
)(
    input  clk,
    input  rst_n,
    input  start_conv,
    input  [3:0] kernel_select,
    input  enable_activation,

    output reg                     valid_out,
    output reg [OUTPUT_WIDTH-1:0]  conv_result,
    output reg                     conv_done,
    output reg [31:0]              pixel_count
);
    // Image memory - loaded from input_image.hex
    reg [INPUT_WIDTH-1:0] image_mem [0:IMAGE_WIDTH*IMAGE_HEIGHT-1];
    initial begin
        $readmemh("input_image.hex", image_mem);
        $display("Loaded input_image.hex into memory");
    end

    // Pixel streaming control
    localparam TOTAL_PIX = IMAGE_WIDTH*IMAGE_HEIGHT;
    reg [31:0] rd_idx;
    reg streaming;
    wire in_valid = streaming;

    // Sliding window instance
    wire win_valid;
    wire [INPUT_WIDTH-1:0] w00,w01,w02,w10,w11,w12,w20,w21,w22;

    sliding_window_3x3 #(
        .DATA_W(INPUT_WIDTH),
        .IMG_W (IMAGE_WIDTH),
        .IMG_H (IMAGE_HEIGHT)
    ) u_window (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid),
        .pixel_in(image_mem[rd_idx]),
        .window_valid(win_valid),
        .w00(w00), .w01(w01), .w02(w02),
        .w10(w10), .w11(w11), .w12(w12),
        .w20(w20), .w21(w21), .w22(w22)
    );

    // Complete kernel bank - All 11 kernels
    reg signed [INPUT_WIDTH:0] k [0:8];
    always @(*) begin
        case (kernel_select)
            4'd0: begin // Identity
                k[0]=0; k[1]=0; k[2]=0; k[3]=0; k[4]=1; k[5]=0; k[6]=0; k[7]=0; k[8]=0;
            end
            4'd1: begin // Prewitt Horizontal
                k[0]=-1; k[1]=0; k[2]=1; k[3]=-1; k[4]=0; k[5]=1; k[6]=-1; k[7]=0; k[8]=1;
            end
            4'd2: begin // Prewitt Vertical
                k[0]=1; k[1]=1; k[2]=1; k[3]=0; k[4]=0; k[5]=0; k[6]=-1; k[7]=-1; k[8]=-1;
            end
            4'd3: begin // Sharpening
                k[0]=0; k[1]=-1; k[2]=0; k[3]=-1; k[4]=5; k[5]=-1; k[6]=0; k[7]=-1; k[8]=0;
            end
            4'd4: begin // Box Blur
                k[0]=1; k[1]=1; k[2]=1; k[3]=1; k[4]=1; k[5]=1; k[6]=1; k[7]=1; k[8]=1;
            end
            4'd5: begin // Gaussian Blur
                k[0]=1; k[1]=2; k[2]=1; k[3]=2; k[4]=4; k[5]=2; k[6]=1; k[7]=2; k[8]=1;
            end
            4'd6: begin // Sobel Horizontal
                k[0]=-1; k[1]=0; k[2]=1; k[3]=-2; k[4]=0; k[5]=2; k[6]=-1; k[7]=0; k[8]=1;
            end
            4'd7: begin // Sobel Vertical
                k[0]=1; k[1]=2; k[2]=1; k[3]=0; k[4]=0; k[5]=0; k[6]=-1; k[7]=-2; k[8]=-1;
            end
            4'd8: begin // Scharr
                k[0]=-3; k[1]=0; k[2]=3; k[3]=-10; k[4]=0; k[5]=10; k[6]=-3; k[7]=0; k[8]=3;
            end
            4'd9: begin // Laplacian
                k[0]=0; k[1]=1; k[2]=0; k[3]=1; k[4]=-4; k[5]=1; k[6]=0; k[7]=1; k[8]=0;
            end
            4'd10: begin // Laplacian Diagonal
                k[0]=1; k[1]=1; k[2]=1; k[3]=1; k[4]=-8; k[5]=1; k[6]=1; k[7]=1; k[8]=1;
            end
            default: begin // Default to Identity
                k[0]=0; k[1]=0; k[2]=0; k[3]=0; k[4]=1; k[5]=0; k[6]=0; k[7]=0; k[8]=0;
            end
        endcase
    end

    // 3x3 MAC instance
    wire mac_valid;
    wire [OUTPUT_WIDTH-1:0] mac_out;
    conv3x3_mac #(
        .DATA_W(INPUT_WIDTH),
        .OUT_W (OUTPUT_WIDTH),
        .ACC_W (32)
    ) u_mac (
        .clk(clk), .rst_n(rst_n),
        .in_valid(win_valid),
        .k00(k[0]), .k01(k[1]), .k02(k[2]),
        .k10(k[3]), .k11(k[4]), .k12(k[5]),
        .k20(k[6]), .k21(k[7]), .k22(k[8]),
        .w00(w00), .w01(w01), .w02(w02),
        .w10(w10), .w11(w11), .w12(w12),
        .w20(w20), .w21(w21), .w22(w22),
        .kernel_id(kernel_select),
        .relu_en(enable_activation),
        .out_valid(mac_valid),
        .px_out(mac_out)
    );

    // COMPLETELY REWRITTEN: Proper state machine for kernel transitions
    reg [31:0] drain_count;
    reg start_conv_prev;
    
    // Detect start_conv rising edge
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_conv_prev <= 1'b0;
        end else begin
            start_conv_prev <= start_conv;
        end
    end
    
    wire start_conv_rising = start_conv && !start_conv_prev;

    // Main state machine - FIXED
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_idx      <= 0;
            streaming   <= 1'b0;
            drain_count <= 0;
            conv_done   <= 1'b0;
            pixel_count <= 0;
        end else begin
            // Start on rising edge of start_conv
            if (start_conv_rising) begin
                rd_idx      <= 0;
                streaming   <= 1'b1;
                drain_count <= 0;
                conv_done   <= 1'b0;
                pixel_count <= 0;
                $display("    Engine: Starting new kernel at time %0t", $time);
            end 
            // Streaming phase
            else if (streaming) begin
                if (rd_idx == TOTAL_PIX-1) begin
                    streaming   <= 1'b0;
                    drain_count <= 0;
                    $display("    Engine: Finished streaming, starting drain at time %0t", $time);
                end else begin
                    rd_idx <= rd_idx + 1;
                end
            end 
            // Drain phase
            else if (!streaming && !conv_done && start_conv) begin
                if (drain_count < 32) begin
                    drain_count <= drain_count + 1;
                end else begin
                    conv_done <= 1'b1;
                    $display("    Engine: Drain complete, asserting conv_done at time %0t", $time);
                end
            end
            // Reset when start_conv goes low
            else if (!start_conv && conv_done) begin
                conv_done <= 1'b0;
                $display("    Engine: Reset conv_done at time %0t", $time);
            end
        end
    end

    // Output valid and result - FIXED: Don't reset pixel_count mid-kernel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out   <= 1'b0;
            conv_result <= {OUTPUT_WIDTH{1'b0}};
        end else begin
            valid_out   <= mac_valid;
            conv_result <= mac_out;
            if (mac_valid && streaming) begin  // Only count during streaming
                pixel_count <= pixel_count + 1;
            end
        end
    end
endmodule

// ==============================================================================
// conv2d_tb - FIXED: Better kernel transition timing
// ==============================================================================
module conv2d_tb;
    // Parameters
    localparam IW = 64;  // Image width
    localparam IH = 64;  // Image height
    localparam EXPECTED_PER_KERNEL = (IW-2)*(IH-2); // 62*62 = 3844
    localparam EXPECTED_TOTAL = 11 * EXPECTED_PER_KERNEL; // 42284

    // Signals
    reg clk, rst_n, start_conv, enable_activation;
    reg [3:0] kernel_select;
    wire valid_out, conv_done;
    wire [15:0] conv_result;
    wire [31:0] pixel_count;

    // Test control
    integer k;
    integer out_fd;
    integer valid_count = 0;
    integer total_outputs = 0;
    integer timeout_count = 0;

    // Kernel names for display
    reg [8*20:1] kernel_names [0:10];
    initial begin
        kernel_names[0]  = "Identity        ";
        kernel_names[1]  = "Prewitt H       ";
        kernel_names[2]  = "Prewitt V       ";
        kernel_names[3]  = "Sharpening      ";
        kernel_names[4]  = "Box Blur        ";
        kernel_names[5]  = "Gaussian Blur   ";
        kernel_names[6]  = "Sobel H         ";
        kernel_names[7]  = "Sobel V         ";
        kernel_names[8]  = "Scharr          ";
        kernel_names[9]  = "Laplacian       ";
        kernel_names[10] = "Laplacian Diag  ";
    end

    // DUT instantiation
    conv2d_engine #(
        .INPUT_WIDTH(8),
        .OUTPUT_WIDTH(16),
        .IMAGE_WIDTH(IW),
        .IMAGE_HEIGHT(IH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start_conv(start_conv),
        .kernel_select(kernel_select),
        .enable_activation(enable_activation),
        .valid_out(valid_out),
        .conv_result(conv_result),
        .conv_done(conv_done),
        .pixel_count(pixel_count)
    );

    // Clock generation - 10ns period (100MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // File initialization
    initial begin
        out_fd = $fopen("output_image.hex", "w");
        if (out_fd == 0) begin
            $display("ERROR: Could not open output_image.hex file!");
            $finish;
        end else begin
            $display("? SUCCESS: output_image.hex opened for writing");
        end

        $display("================================================================================");
        $display("            FINAL FIXED Line-Buffer CNN Convolution System");
        $display("================================================================================");
        $display("Input file:  input_image.hex");
        $display("Output file: output_image.hex");
        $display("Image size:  %0dx%0d pixels", IW, IH);
        $display("Expected outputs per kernel: %0d", EXPECTED_PER_KERNEL);
        $display("Total expected outputs (11 kernels): %0d", EXPECTED_TOTAL);
        $display("FINAL FIXES: Proper state reset, edge detection, debug messages");
        $display("================================================================================");
    end

    // Main test sequence - FIXED kernel transitions
    initial begin
        // Initialize
        rst_n = 0;
        start_conv = 0;
        enable_activation = 1;
        kernel_select = 0;
        valid_count = 0;
        total_outputs = 0;

        // Reset sequence
        #100 rst_n = 1;
        #100; // Let everything settle
        $display("? Reset complete at time %0t", $time);

        // Test each kernel with proper timing
        for (k = 0; k <= 10; k = k + 1) begin
            $display("");
            $display("=== Processing Kernel %0d: %s ===", k, kernel_names[k]);
            kernel_select = k[3:0];
            valid_count = 0;
            timeout_count = 0;

            // Wait for clean state
            #200;
            
            // Start convolution with clean rising edge
            start_conv = 1;
            $display("  Started at time %0t", $time);
            
            // Wait for completion or timeout
            while (!conv_done && timeout_count < 1000000) begin // 10ms timeout
                #10;
                timeout_count = timeout_count + 1;
            end
            
            if (conv_done) begin
                $display("  ? Completed at time %0t", $time);
            end else begin
                $display("  ? Timeout after %0t", $time);
            end
            
            // Clean stop - let conv_done reset
            start_conv = 0;
            #500; // Longer settle time for state reset

            // Report results
            $display("  Testbench valid_count: %0d", valid_count);
            $display("  DUT pixel_count:      %0d", pixel_count);
            
            if (valid_count == EXPECTED_PER_KERNEL) begin
                $display("  ? PASS - Correct outputs");
            end else begin
                $display("  ? FAIL - Expected %0d, got %0d", EXPECTED_PER_KERNEL, valid_count);
            end
            
            total_outputs = total_outputs + valid_count;
        end

        // Final results
        $fclose(out_fd);
        $display("");
        $display("=== FINAL RESULTS ===");
        $display("Total outputs: %0d (expected %0d)", total_outputs, EXPECTED_TOTAL);
        
        if (total_outputs == EXPECTED_TOTAL) begin
            $display("?? SUCCESS: All kernels completed!");
        end else begin
            $display("?? PARTIAL: Missing %0d outputs", EXPECTED_TOTAL - total_outputs);
        end
        
        $finish;
    end

    // Output capture
    always @(posedge clk) begin
        if (valid_out) begin
            valid_count = valid_count + 1;
            $fwrite(out_fd, "%02H\n", conv_result[7:0]);
            
            if (valid_count % 1000 == 0) begin
                $display("    Progress: %0d/3844 (%0d%%)", valid_count, (valid_count * 100) / 3844);
            end
        end
    end

    // Global timeout
    initial begin
        #2000000000; // 2 second total timeout  
        $display("GLOBAL TIMEOUT");
        $fclose(out_fd);
        $finish;
    end

endmodule

