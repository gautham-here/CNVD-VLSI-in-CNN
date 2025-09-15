import numpy as np
from tkinter import Tk, filedialog
import os
from src.python.kernels import KERNELS_1D

def max_pool_1d(input_array, pool_size=2, stride=2):
    pooled = []
    for i in range(0, len(input_array) - pool_size + 1, stride):
        pooled.append(np.max(input_array[i:i + pool_size]))
    return np.array(pooled)

def save_stream(filepath, data_array):
    with open(filepath, "w") as f:
        f.write(" ".join(map(str, data_array)))

def convolve_1d_pipeline():
    # File picker
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select 1D matrix file (e.g., *_matrix_1d.txt)",
        filetypes=[("Text Files", "*.txt")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    # Load matrix
    input_data = np.loadtxt(file_path)
    if input_data.ndim > 1:
        input_data = input_data.flatten()
    print(f"Loaded 1D matrix of shape: {input_data.shape}")

    # Get base name and folder
    base_name = os.path.splitext(os.path.basename(file_path))[0].replace("_matrix_1d", "")
    folder_path = os.path.join(os.getcwd(), base_name)
    os.makedirs(folder_path, exist_ok=True)

    # Show available kernels
    print("\nAvailable kernels:")
    for name in KERNELS_1D.keys():
        print(f" - {name}")

    kernel_name = input("Enter kernel name: ").strip().lower()
    if kernel_name not in KERNELS_1D:
        print("Invalid kernel name.")
        return
    kernel = KERNELS_1D[kernel_name]

    # B2: Convolution
    conv_output = np.convolve(input_data, kernel, mode='valid')
    conv_output_int = np.round(conv_output).astype(np.int16)
    conv_path = os.path.join(folder_path, f"{base_name}_conv1d_{kernel_name}_stream.txt")
    save_stream(conv_path, conv_output_int)
    print(f"Convolution done. Output shape: {conv_output_int.shape}")
    print(f"Saved convolved stream to {conv_path}")

    # B3: ReLU
    relu_output = np.maximum(0, conv_output_int).astype(np.int16)
    relu_path = os.path.join(folder_path, f"{base_name}_relu1d_{kernel_name}_stream.txt")
    save_stream(relu_path, relu_output)
    print(f"ReLU applied. Saved stream to {relu_path}")

    # B4: Max Pooling
    pooled_output = max_pool_1d(relu_output, pool_size=2, stride=2).astype(np.int16)
    pooled_path = os.path.join(folder_path, f"{base_name}_pooled1d_{kernel_name}_stream.txt")
    save_stream(pooled_path, pooled_output)
    print(f"Max pooling done. Output shape: {pooled_output.shape}")
    print(f"Saved pooled stream to {pooled_path}")

if __name__ == "__main__":
    convolve_1d_pipeline()
