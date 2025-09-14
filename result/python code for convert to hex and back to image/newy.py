import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def hex_to_images(hex_file_path="output_image.hex", output_width=62, output_height=62, num_kernels=11):
    """
    Convert hex file from Verilog CNN output to images
    Handles incomplete data gracefully
    """
    
    # Check if file exists
    if not os.path.exists(hex_file_path):
        print(f"❌ Error: File {hex_file_path} not found!")
        return []
    
    # Read hex values from file
    try:
        with open(hex_file_path, 'r') as f:
            hex_lines = f.read().splitlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []
    
    # Convert hex strings to integers
    pixel_values = []
    invalid_count = 0
    
    for line_num, line in enumerate(hex_lines):
        if line.strip():  # Skip empty lines
            try:
                pixel_values.append(int(line.strip(), 16))
            except ValueError:
                invalid_count += 1
                if invalid_count <= 5:  # Show first 5 errors
                    print(f"⚠ Invalid hex on line {line_num + 1}: '{line.strip()}'")
    
    if invalid_count > 5:
        print(f"⚠ ... and {invalid_count - 5} more invalid lines")
    
    print(f"✅ Total valid pixel values: {len(pixel_values)}")
    
    # Calculate expected values
    pixels_per_kernel = output_width * output_height
    expected_total = pixels_per_kernel * num_kernels
    
    print(f"📊 Expected total pixels: {expected_total}")
    print(f"📊 Pixels per kernel: {pixels_per_kernel}")
    
    # Analyze what we have
    complete_kernels = len(pixel_values) // pixels_per_kernel
    remaining_pixels = len(pixel_values) % pixels_per_kernel
    
    print(f"🔍 Analysis:")
    print(f"   Complete kernels available: {complete_kernels}")
    print(f"   Remaining pixels: {remaining_pixels}")
    
    if complete_kernels == 0 and remaining_pixels > 0:
        print(f"   ⚠ Partial data: {remaining_pixels}/{pixels_per_kernel} pixels ({remaining_pixels/pixels_per_kernel*100:.1f}%)")
    
    # Process available data
    kernel_names = [
        "Identity", "Prewitt_H", "Prewitt_V", "Sharpening", 
        "Box_Blur", "Gaussian_Blur", "Sobel_H", "Sobel_V", 
        "Scharr", "Laplacian", "Laplacian_Diagonal"
    ]
    
    saved_images = 0
    
    # Save complete kernels
    for kernel_idx in range(complete_kernels):
        start_idx = kernel_idx * pixels_per_kernel
        end_idx = start_idx + pixels_per_kernel
        
        kernel_pixels = pixel_values[start_idx:end_idx]
        
        # Convert to numpy array and reshape to 2D
        img_array = np.array(kernel_pixels, dtype=np.uint8)
        img_array = img_array.reshape((output_height, output_width))
        
        # Save as PNG
        img = Image.fromarray(img_array, mode='L')
        output_filename = f"kernel_{kernel_idx:02d}_{kernel_names[kernel_idx]}.png"
        img.save(output_filename)
        
        print(f"💾 Saved: {output_filename}")
        saved_images += 1
    
    # Handle partial kernel data
    if remaining_pixels > 0 and complete_kernels < len(kernel_names):
        print(f"\n🔧 Processing partial kernel {complete_kernels}...")
        
        start_idx = complete_kernels * pixels_per_kernel
        partial_pixels = pixel_values[start_idx:]
        
        # Pad with zeros to make complete image
        padded_pixels = partial_pixels + [0] * (pixels_per_kernel - len(partial_pixels))
        img_array = np.array(padded_pixels, dtype=np.uint8)
        img_array = img_array.reshape((output_height, output_width))
        
        # Save partial image
        img = Image.fromarray(img_array, mode='L')
        output_filename = f"kernel_{complete_kernels:02d}_{kernel_names[complete_kernels]}_PARTIAL.png"
        img.save(output_filename)
        
        print(f"💾 Saved partial: {output_filename}")
        print(f"   📝 Note: Bottom portion padded with zeros")
        saved_images += 1
    
    print(f"\n✅ Generated {saved_images} image(s)")
    return pixel_values


def display_available_kernels(hex_file_path, output_width=62, output_height=62):
    """
    Display available kernel results (handles incomplete data)
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
    
    # Calculate what to display
    images_to_show = complete_kernels
    if remaining_pixels > 0:
        images_to_show += 1
    
    if images_to_show == 0:
        print("❌ No complete or partial kernel data to display")
        return
    
    # Create appropriate subplot layout
    if images_to_show == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        axes = [ax]
    else:
        cols = min(4, images_to_show)
        rows = (images_to_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if images_to_show == 1:
            axes = [axes]
        elif hasattr(axes, 'flatten'):
            axes = axes.flatten()
        else:
            axes = [axes]
    
    # Display complete kernels
    for i in range(complete_kernels):
        start_idx = i * pixels_per_kernel
        end_idx = start_idx + pixels_per_kernel
        
        kernel_pixels = pixel_values[start_idx:end_idx]
        img_array = np.array(kernel_pixels, dtype=np.uint8)
        img_array = img_array.reshape((output_height, output_width))
        
        axes[i].imshow(img_array, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f"Kernel {i}: {kernel_names[i]}", fontsize=10)
        axes[i].axis('off')
    
    # Display partial kernel if exists
    if remaining_pixels > 0 and complete_kernels < len(kernel_names):
        start_idx = complete_kernels * pixels_per_kernel
        partial_pixels = pixel_values[start_idx:]
        
        # Pad with zeros
        padded_pixels = partial_pixels + [0] * (pixels_per_kernel - len(partial_pixels))
        img_array = np.array(padded_pixels, dtype=np.uint8)
        img_array = img_array.reshape((output_height, output_width))
        
        axes[complete_kernels].imshow(img_array, cmap='gray', vmin=0, vmax=255)
        axes[complete_kernels].set_title(f"Kernel {complete_kernels}: {kernel_names[complete_kernels]} (PARTIAL)", 
                                       fontsize=10, color='orange')
        axes[complete_kernels].axis('off')
    
    # Hide unused subplots
    for i in range(images_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save comparison image
    output_file = f'kernels_display_{complete_kernels}complete.png'
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Comparison saved: {output_file}")
    except Exception as e:
        print(f"⚠ Could not save comparison: {e}")
    
    # Show plot (with error handling)
    try:
        plt.show(block=False)  # Non-blocking show
        print("🖼️ Display window opened")
    except Exception as e:
        print(f"⚠ Could not display plot: {e}")
        print("💡 Check the saved PNG files instead")
    
    plt.close()  # Clean up


def analyze_simulation_status(hex_file_path="output_image.hex"):
    """
    Analyze simulation output and provide recommendations
    """
    print("\n" + "="*60)
    print("🔍 SIMULATION ANALYSIS")
    print("="*60)
    
    if not os.path.exists(hex_file_path):
        print(f"❌ Output file not found: {hex_file_path}")
        print("💡 Make sure your ModelSim simulation completed and generated the file")
        return
    
    try:
        with open(hex_file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        total_pixels = len(lines)
        expected_per_kernel = 62 * 62  # 3844
        expected_total = expected_per_kernel * 11  # 42284
        
        print(f"📊 Pixels found: {total_pixels}")
        print(f"📊 Expected total: {expected_total}")
        print(f"📊 Completion: {total_pixels/expected_total*100:.1f}%")
        
        if total_pixels < expected_per_kernel:
            print(f"\n❌ INCOMPLETE: Less than 1 complete kernel")
            print(f"   Got {total_pixels}/{expected_per_kernel} pixels")
            print(f"   Missing: {expected_per_kernel - total_pixels} pixels")
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   1. Check if ModelSim simulation is still running")
            print(f"   2. Verify conv_done signal logic in Verilog code")
            print(f"   3. Increase simulation timeout")
            print(f"   4. Check for infinite loops in FSM")
            
        elif total_pixels < expected_total:
            complete_kernels = total_pixels // expected_per_kernel
            remaining = total_pixels % expected_per_kernel
            print(f"\n⚠ PARTIAL: {complete_kernels} complete kernels")
            if remaining > 0:
                print(f"   Plus {remaining} pixels from kernel {complete_kernels}")
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   1. Let simulation run longer for remaining kernels")
            print(f"   2. Check testbench kernel loop logic")
            
        else:
            print(f"\n✅ COMPLETE: All kernels processed successfully!")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")


# Main execution
if __name__ == "__main__":
    print("🚀 CNN Output Processor")
    print("="*50)
    
    # Analyze simulation first
    analyze_simulation_status('output_image.hex')
    
    # Process available data
    print(f"\n📸 Processing available images...")
    
    try:
        # Convert hex to images (handles partial data)
        pixel_values = hex_to_images('output_image.hex')
        
        if pixel_values:
            # Create display (safe version)
            display_available_kernels('output_image.hex')
            
            print(f"\n✅ PROCESSING COMPLETE")
            print(f"📁 Check your directory for generated PNG files")
            print(f"🖼️ Individual kernel images: kernel_XX_Name.png")
            print(f"🖼️ Comparison image: kernels_display_Xcomplete.png")
        else:
            print(f"\n❌ No valid data to process")
            
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        
    print(f"\n🏁 Done!")
