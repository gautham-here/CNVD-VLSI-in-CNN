import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog


def hex_to_images(hex_file_path, output_width=62, output_height=62, num_kernels=11):
    """
    Convert hex file from Verilog CNN output to images
    Handles incomplete data gracefully
    Output images saved in the same folder as input file
    """
    
    if not os.path.exists(hex_file_path):
        print(f"❌ Error: File {hex_file_path} not found!")
        return []
    
    try:
        with open(hex_file_path, 'r') as f:
            hex_lines = f.read().splitlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []
    
    pixel_values = []
    invalid_count = 0
    
    for line_num, line in enumerate(hex_lines):
        if line.strip():
            try:
                pixel_values.append(int(line.strip(), 16))
            except ValueError:
                invalid_count += 1
                if invalid_count <= 5:
                    print(f"⚠ Invalid hex on line {line_num + 1}: '{line.strip()}'")
    
    if invalid_count > 5:
        print(f"⚠ ... and {invalid_count - 5} more invalid lines")
    
    print(f"✅ Total valid pixel values: {len(pixel_values)}")
    
    pixels_per_kernel = output_width * output_height
    expected_total = pixels_per_kernel * num_kernels
    
    print(f"📊 Expected total pixels: {expected_total}")
    print(f"📊 Pixels per kernel: {pixels_per_kernel}")
    
    complete_kernels = len(pixel_values) // pixels_per_kernel
    remaining_pixels = len(pixel_values) % pixels_per_kernel
    
    print(f"🔍 Analysis:")
    print(f"   Complete kernels available: {complete_kernels}")
    print(f"   Remaining pixels: {remaining_pixels}")
    
    if complete_kernels == 0 and remaining_pixels > 0:
        print(f"   ⚠ Partial data: {remaining_pixels}/{pixels_per_kernel} pixels "
              f"({remaining_pixels/pixels_per_kernel*100:.1f}%)")
    
    kernel_names = [
        "Identity", "Prewitt_H", "Prewitt_V", "Sharpening", 
        "Box_Blur", "Gaussian_Blur", "Sobel_H", "Sobel_V", 
        "Scharr", "Laplacian", "Laplacian_Diagonal"
    ]
    
    base_dir = os.path.dirname(hex_file_path)
    base_name = os.path.splitext(os.path.basename(hex_file_path))[0]
    saved_images = 0
    
    for kernel_idx in range(complete_kernels):
        start_idx = kernel_idx * pixels_per_kernel
        end_idx = start_idx + pixels_per_kernel
        
        kernel_pixels = pixel_values[start_idx:end_idx]
        img_array = np.array(kernel_pixels, dtype=np.uint8).reshape((output_height, output_width))
        
        img = Image.fromarray(img_array, mode='L')
        output_filename = os.path.join(
            base_dir, f"{base_name}_kernel_{kernel_idx:02d}_{kernel_names[kernel_idx]}.png"
        )
        img.save(output_filename)
        
        print(f"💾 Saved: {output_filename}")
        saved_images += 1
    
    if remaining_pixels > 0 and complete_kernels < len(kernel_names):
        print(f"\n🔧 Processing partial kernel {complete_kernels}...")
        
        start_idx = complete_kernels * pixels_per_kernel
        partial_pixels = pixel_values[start_idx:]
        
        padded_pixels = partial_pixels + [0] * (pixels_per_kernel - len(partial_pixels))
        img_array = np.array(padded_pixels, dtype=np.uint8).reshape((output_height, output_width))
        
        img = Image.fromarray(img_array, mode='L')
        output_filename = os.path.join(
            base_dir, f"{base_name}_kernel_{complete_kernels:02d}_{kernel_names[complete_kernels]}_PARTIAL.png"
        )
        img.save(output_filename)
        
        print(f"💾 Saved partial: {output_filename}")
        saved_images += 1
    
    print(f"\n✅ Generated {saved_images} image(s)")
    return pixel_values


def display_available_kernels(hex_file_path, output_width=62, output_height=62):
    """
    Display available kernel results (handles incomplete data)
    Saves combined comparison in same folder as input file
    """
    print(f"\n🖼️ Generating display...")
    
    pixel_values = hex_to_images(hex_file_path, output_width, output_height, 11)
    if not pixel_values:
        print("❌ No pixel data available for display")
        return
    
    kernel_names = [
        "Identity", "Prewitt H", "Prewitt V", "Sharpening", 
        "Box Blur", "Gaussian", "Sobel H", "Sobel V", 
        "Scharr", "Laplacian", "Laplacian Diag"
    ]
    
    pixels_per_kernel = output_width * output_height
    complete_kernels = len(pixel_values) // pixels_per_kernel
    remaining_pixels = len(pixel_values) % pixels_per_kernel
    
    images_to_show = complete_kernels + (1 if remaining_pixels > 0 else 0)
    if images_to_show == 0:
        print("❌ No complete or partial kernel data to display")
        return
    
    cols = min(4, images_to_show)
    rows = (images_to_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i in range(complete_kernels):
        start_idx = i * pixels_per_kernel
        end_idx = start_idx + pixels_per_kernel
        img_array = np.array(pixel_values[start_idx:end_idx], dtype=np.uint8).reshape((output_height, output_width))
        axes[i].imshow(img_array, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f"Kernel {i}: {kernel_names[i]}", fontsize=10)
        axes[i].axis('off')
    
    if remaining_pixels > 0 and complete_kernels < len(kernel_names):
        start_idx = complete_kernels * pixels_per_kernel
        padded_pixels = pixel_values[start_idx:] + [0] * (pixels_per_kernel - remaining_pixels)
        img_array = np.array(padded_pixels, dtype=np.uint8).reshape((output_height, output_width))
        axes[complete_kernels].imshow(img_array, cmap='gray', vmin=0, vmax=255)
        axes[complete_kernels].set_title(
            f"Kernel {complete_kernels}: {kernel_names[complete_kernels]} (PARTIAL)",
            fontsize=10, color='orange'
        )
        axes[complete_kernels].axis('off')
    
    for i in range(images_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    base_dir = os.path.dirname(hex_file_path)
    base_name = os.path.splitext(os.path.basename(hex_file_path))[0]
    output_file = os.path.join(base_dir, f"{base_name}_kernels_display_{complete_kernels}complete.png")
    
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison saved: {output_file}")
    except Exception as e:
        print(f"⚠ Could not save comparison: {e}")
    
    try:
        plt.show(block=False)
        print("🖼️ Display window opened")
    except Exception as e:
        print(f"⚠ Could not display plot: {e}")
    
    plt.close()


def analyze_simulation_status(hex_file_path):
    """
    Analyze simulation output and provide recommendations
    """
    print("\n" + "="*60)
    print("🔍 SIMULATION ANALYSIS")
    print("="*60)
    
    if not os.path.exists(hex_file_path):
        print(f"❌ Output file not found: {hex_file_path}")
        return
    
    try:
        with open(hex_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        total_pixels = len(lines)
        expected_per_kernel = 62 * 62
        expected_total = expected_per_kernel * 11
        
        print(f"📊 Pixels found: {total_pixels}")
        print(f"📊 Expected total: {expected_total}")
        print(f"📊 Completion: {total_pixels/expected_total*100:.1f}%")
        
        if total_pixels < expected_per_kernel:
            print("\n❌ INCOMPLETE: Less than 1 complete kernel")
        elif total_pixels < expected_total:
            complete_kernels = total_pixels // expected_per_kernel
            remaining = total_pixels % expected_per_kernel
            print(f"\n⚠ PARTIAL: {complete_kernels} complete kernels, +{remaining} pixels")
        else:
            print("\n✅ COMPLETE: All kernels processed successfully!")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")


if __name__ == "__main__":
    print("🚀 CNN Output Processor")
    print("="*50)
    
    Tk().withdraw()
    hex_file_path = filedialog.askopenfilename(
        title="Select CNN output hex file",
        filetypes=[("Hex files", "*.hex"), ("All files", "*.*")]
    )
    
    if hex_file_path:
        analyze_simulation_status(hex_file_path)
        print(f"\n📸 Processing available images...")
        pixel_values = hex_to_images(hex_file_path)
        if pixel_values:
            display_available_kernels(hex_file_path)
            print(f"\n✅ PROCESSING COMPLETE")
        else:
            print(f"\n❌ No valid data to process")
    else:
        print("❌ No file selected")
