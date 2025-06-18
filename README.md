# CNVD-VLSI-in-CNN
VLSI Implementation of CNN, done for CNVD, VIT Chennai under the guidance of Dr P Augusta Sophy.

The document uses three images: Baboon, Cameraman and Matlab logo as well, to test the working of the different codes.

## Grayscale Matrix Generation

The Python code [grayscale-mat-gen.py](grayscale-mat-gen.py) converts any given image into a standard size of 25x25 pixels, of grayscale type. The two files are generated, one a row vector of size 625 as input to the 1D pipeline, and two a 25x25 matrix as input to the 2D pipeline.

## 1D CNN Pipeline:

Input → Convolution → ReLU → Max Pooling → Output

Input: From the grayscale matrix generated
Convolution: Kernels are imported from [kernels.py](kernels.py) and used in the pipeline for convolution.
ReLU: Rectified linear unit, introduces non-linearity and thus allowing the network to learn complex patterns in data.
Max Pooling: Custom Max Pooling function is created to generate the output.

Output is saved at each step to be compared with Verilog.