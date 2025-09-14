import cv2
import numpy as np

def image_to_hex(image_path, output_hex_path, target_width=64, target_height=64):
    """
    Convert image to hex file for Verilog $readmemh
    """
    # Read image
    img = cv2.imread("C:\\Users\\mu2ce\\Documents\\personal\\images.jpg")

    # Convert to grayscale
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # Resize to target dimensions
    resized_img = cv2.resize(gray_img, (target_width, target_height))
    
    # Convert to hex format
    with open(output_hex_path, 'w') as f:
        for row in range(target_height):
            for col in range(target_width):
                pixel_value = resized_img[row, col]
                f.write(f"{pixel_value:02X}\n")  # Write each pixel as 2-digit hex
    
    print(f"Image converted to {output_hex_path}")
    print(f"Image size: {target_width}x{target_height}")
    print(f"Total pixels: {target_width * target_height}")

# Usage example
image_to_hex("your_image.jpg", "input_image.hex", 64, 64)
