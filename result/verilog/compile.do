# Create work library
vlib work

# Compile all Verilog files
vlog conv2d_engine.v
vlog conv2d_pipelined.v  
vlog mac_unit.v
vlog conv2d_tb.v

# Load testbench
vsim -gui conv2d_tb

# Add waves
add wave -r /*
add wave /conv2d_tb/dut/state
add wave /conv2d_tb/dut/kernel_select
add wave /conv2d_tb/dut/accumulator

# Run simulation
run -all
*/