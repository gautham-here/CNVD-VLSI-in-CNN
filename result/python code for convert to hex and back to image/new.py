import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def hex_to_images(hex_file_path, output_width=62, output_height=62, num_kernels=11):
    """
    Convert hex file from Verilog CNN output to images
    
    Parameters:
    - hex_file_path: Path to your output_results.hex file
    - output_width: Output image width (62 for 64x64 input with 3x3 kernel)
    - output_height: Output image height (62 for 64x64 input with 3x3 kernel)  
    - num_kernels: Number of kernels (11 in your case)
    """
    
    # Read hex values from file
    with open(hex_file_path, 'r') as f:
        hex_lines = f.read().splitlines()
    
    # Convert hex strings to integers
    pixel_values = []
    for line in hex_lines:
        if line.strip():  # Skip empty lines
            pixel_values.append(int(line.strip(), 16))
    
    print(f"Total pixel values read: {len(pixel_values)}")
    
    # Calculate pixels per kernel
    pixels_per_kernel = output_width * output_height
    expected_total = pixels_per_kernel * num_kernels
    
    print(f"Expected total pixels: {expected_total}")
    print(f"Pixels per kernel: {pixels_per_kernel}")
    
    # Process each kernel's output
    kernel_names = [
        "Identity", "Prewitt_H", "Prewitt_V", "Sharpening", 
        "Box_Blur", "Gaussian_Blur", "Sobel_H", "Sobel_V", 
        "Scharr", "Laplacian", "Laplacian_Diagonal"
    ]
    
    for kernel_idx in range(min(num_kernels, len(kernel_names))):
        # Extract pixel values for this kernel
        start_idx = kernel_idx * pixels_per_kernel
        end_idx = start_idx + pixels_per_kernel
        
        if end_idx <= len(pixel_values):
            kernel_pixels = pixel_values[start_idx:end_idx]
            
            # Convert to numpy array and reshape to 2D
            img_array = np.array(kernel_pixels, dtype=np.uint8)
            img_array = img_array.reshape((output_height, output_width))
            
            # Save as PNG
            img = Image.fromarray(img_array, mode='L')  # 'L' for grayscale
            output_filename = f"kernel_{kernel_idx:02d}_{kernel_names[kernel_idx]}.png"
            img.save(output_filename)
            
            print(f"Saved: {output_filename}")
        else:
            print(f"Warning: Not enough data for kernel {kernel_idx}")
    
    return pixel_values

def display_all_kernels(hex_file_path, output_width=62, output_height=62):
    """
    Display all kernel results in a single plot
    """
    pixel_values = hex_to_images(hex_file_path, output_width, output_height, 11)
    
    # Create subplot for all kernels
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    kernel_names = [
        "Identity", "Prewitt H", "Prewitt V", "Sharpening", 
        "Box Blur", "Gaussian", "Sobel H", "Sobel V", 
        "Scharr", "Laplacian", "Laplacian Diag"
    ]
    
    pixels_per_kernel = output_width * output_height
    
    for i in range(11):
        start_idx = i * pixels_per_kernel
        end_idx = start_idx + pixels_per_kernel
        
        if end_idx <= len(pixel_values):
            kernel_pixels = pixel_values[start_idx:end_idx]
            img_array = np.array(kernel_pixels, dtype=np.uint8)
            img_array = img_array.reshape((output_height, output_width))
            
            axes[i].imshow(img_array, cmap='gray')
            axes[i].set_title(kernel_names[i])
            axes[i].axis('off')
    
    # Hide unused subplot
    axes[11].axis('off')
    
    plt.tight_layout()
    plt.savefig('all_kernels_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison image saved as: all_kernels_comparison.png")

# Usage example:
if __name__ == "__main__":
    # Convert hex file to individual kernel images
    hex_to_images('output_results.hex')
    
    # Display all kernels in one comparison plot
    display_all_kernels('output_results.hex')
