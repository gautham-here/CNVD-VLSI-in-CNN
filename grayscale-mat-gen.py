import numpy as np
from PIL import Image
from tkinter import Tk, filedialog
import os

def generate_grayscale_matrix(output_size=(25, 25)):
    # Open file picker
    Tk().withdraw()
    image_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not image_path:
        print("No image selected. Exiting.")
        return None, None

    # Extract base name and make output folder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = os.path.join(os.getcwd(), base_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_resized = img.resize(output_size, resample=Image.BICUBIC)

    # Convert to NumPy array and normalize
    matrix = np.array(img_resized, dtype=np.float32) / 255.0

    # Save grayscale matrix
    txt_path = os.path.join(output_folder, f"{base_name}_grayscale.txt")
    np.savetxt(txt_path, matrix, fmt="%.6f", delimiter=",")
    print(f"Grayscale matrix saved to {txt_path}")

    # Save grayscale image
    img_path = os.path.join(output_folder, f"{base_name}_grayscale.png")
    img_resized.save(img_path)
    print(f"Grayscale image saved to {img_path}")

    return matrix, base_name

# Example run
if __name__ == "__main__":
    generate_grayscale_matrix(output_size=(25, 25))
