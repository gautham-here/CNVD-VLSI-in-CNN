import numpy as np
from PIL import Image
from tkinter import Tk, filedialog
import os

def generate_grayscale_matrices(output_size=(25, 25)):
    # File picker
    Tk().withdraw()
    image_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not image_path:
        print("No image selected. Exiting.")
        return None, None

    # Extract base name and create output folder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = os.path.join(os.getcwd(), base_name)
    os.makedirs(output_folder, exist_ok=True)

    # Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_resized = img.resize(output_size, resample=Image.BICUBIC)

    # Convert to NumPy array (uint8)
    matrix = np.array(img_resized, dtype=np.uint8)

    # Save grayscale image
    grayscale_img_path = os.path.join(output_folder, f"{base_name}_grayscale.png")
    img_resized.save(grayscale_img_path)
    print(f"Grayscale image saved to {grayscale_img_path}")

    # Save 2D matrix (space-separated rows)
    txt_2d_path = os.path.join(output_folder, f"{base_name}_matrix_2d.txt")
    with open(txt_2d_path, 'w') as f2d:
        for row in matrix:
            f2d.write(' '.join(map(str, row)) + '\n')
    print(f"2D matrix saved to {txt_2d_path}")

    # Save 1D matrix (flattened, space-separated)
    txt_1d_path = os.path.join(output_folder, f"{base_name}_matrix_1d.txt")
    with open(txt_1d_path, 'w') as f1d:
        f1d.write(' '.join(map(str, matrix.flatten())))
    print(f"1D matrix saved to {txt_1d_path}")

    return matrix, base_name

# Example run
if __name__ == "__main__":
    generate_grayscale_matrices(output_size=(25, 25))
