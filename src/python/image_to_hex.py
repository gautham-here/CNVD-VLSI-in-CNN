import cv2
import os
from tkinter import Tk, filedialog

def image_to_hex(target_width=64, target_height=64):
    """
    Convert uploaded image to hex file for Verilog $readmemh
    Output hex file will be saved in the same folder as input with '_result.hex' suffix.
    """
    # Hide the root Tk window
    Tk().withdraw()

    # Ask user to select an image
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )

    if not image_path:
        print("No file selected.")
        return

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not read the image.")
        return

    # Convert to grayscale
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Resize to target dimensions
    resized_img = cv2.resize(gray_img, (target_width, target_height))

    # Create output path in the same folder
    base, ext = os.path.splitext(image_path)
    output_hex_path = base + "_result.hex"

    # Convert to hex format
    with open(output_hex_path, 'w') as f:
        for row in range(target_height):
            for col in range(target_width):
                pixel_value = resized_img[row, col]
                f.write(f"{pixel_value:02X}\n")  # Write each pixel as 2-digit hex

    print(f"✅ Image converted to {output_hex_path}")
    print(f"Image size: {target_width}x{target_height}")
    print(f"Total pixels: {target_width * target_height}")


# Usage
image_to_hex(64, 64)
