# CNVD-VLSI-in-CNN: Hardware-Accelerated CNN Implementation

**VLSI Implementation of Convolutional Neural Networks (CNN)**  
Project for **CNVD, VIT Chennai**  
Supervisor: *Dr. P Augusta Sophy*

---

This project demonstrates a robust **Python + Verilog workflow** for hardware-accelerated CNN design and validation.  
- **Python scripts:** Preprocessing, kernel operations, Verilog interfacing, simulation analysis  
- **Verilog:** Hardware implementation of the CNN pipeline

---

## 📂 Python Codes (`src/python/`)

- **grayscale-mat-gen.py:** Convert images to 25×25 grayscale matrices (2D) and 625-element row vectors (1D)
- **kernels.py:** Dictionary of standard 1D & 2D convolution kernels
- **pipeline_1d_cnn.py:** End-to-end 1D CNN pipeline (Conv → ReLU → MaxPool)
- **image_to_hex.py:** Prepare hex files from images for Verilog `$readmemh`
- **img_display.py:** Convert Verilog HEX outputs to images, generate comparison plots
- **simulation_analysis.py:** Advanced analyzer for Verilog CNN outputs: error handling, report generation

---

## 🔄 Workflow Overview



## 🔄 Overall Workflow

```mermaid
flowchart LR
    A[Input Image] --> B[grayscale-mat-gen.py]
    B --> C1[25x25 Matrix (2D Pipeline Input)]
    B --> C2[Row Vector 625 (1D Pipeline Input)]
    C2 --> D[pipeline_1d_cnn.py]
    D --> E[Python CNN Reference Output]

    A --> F[image_to_hex.py]
    F --> G[HEX File for Verilog]
    G --> H[Verilog CNN Simulation]
    H --> I[Output HEX File]
    I --> J[img_display.py / simulation_analysis.py]
    J --> K[Reconstructed Images & Reports]
```


---

## 🖼️ Grayscale Matrix Generation

**File:** `grayscale-mat-gen.py`  
Converts any image into a standardized 25×25 grayscale matrix for CNN input.

- **Outputs:**
    - `<image>_grayscale.png`: resized grayscale image
    - `<image>_matrix_2d.txt`: 25×25 matrix (2D CNN input)
    - `<image>_matrix_1d.txt`: flattened 625-element row vector (1D CNN input)

**Usage:**

```
python grayscale-mat-gen.py
```

---

## ⚙️ Kernels (Filters)

**File:** `kernels.py`  
Predefined 1D & 2D kernels for convolution.

- **1D kernels:** `identity`, `blur`, `gaussian`, `edge`
- **2D kernels:** `identity`, `Prewitt (H/V)`, `Sharpening`, `Box Blur`, `Gaussian`, `Sobel (H/V)`, `Scharr`, `Laplacian`, `Laplacian-Diagonal`

**Access Example:**

```
from kernels import KERNELS_1D, KERNELS_2D
print(KERNELS_1D["gaussian"])
print(KERNELS_2D["sobel_h"])
```

---

## 🔄 1D CNN Pipeline

**File:** `pipeline_1d_cnn.py`  
Simulates a 1D CNN pipeline:

1. **Input:** Load the 625-element row vector (from `grayscale-mat-gen.py`)
2. **Convolution:** Apply a kernel (from `kernels.py`)
3. **ReLU:** Replace negative values with zero
4. **MaxPooling:** Reduce output resolution
5. **Output:** Save results after each stage for validation

**Usage:**

```
python pipeline_1d_cnn.py
```

---

## 📦 Image to HEX for Verilog

**File:** `image_to_hex.py`  
Prepares images as hex files for Verilog hardware simulation:

- Resizes image to 64×64, converts to grayscale
- Each pixel stored as 2-digit hex (`00–FF`) in `<image>_result.hex`

**Usage:**

```
python image_to_hex.py
```

---

## 🖼️ HEX → Images (Postprocessing)

**File:** `img_display.py`  
- Reads CNN output HEX from Verilog
- Splits into 11 kernel outputs (62×62 each)
- Saves each kernel output as PNG, plus a combined comparison plot

**Outputs:**
- `<basename>_kernel_00_Identity.png`
- `<basename>_kernel_01_Prewitt_H.png`
- ...
- `<basename>_all_kernels.png`

**Usage:**

```
python img_display.py
```

---

## 📊 Simulation Analysis

**File:** `simulation_analysis.py`  
Advanced analyzer for Verilog CNN outputs:

- Reads Verilog CNN output HEX robustly
- Detects invalid/incomplete pixel data
- Reconstructs complete/partial kernel outputs (marks incomplete with `_PARTIAL`)
- Generates comparison plots and simulation reports

**Usage:**

```
python simulation_analysis.py
```

---

## 🧩 Full Workflow Summary

- **Preprocessing:**  
    - `grayscale-mat-gen.py` → matrix/vector for Python CNN  
    - `image_to_hex.py` → prepare HEX for Verilog

- **Simulation:**  
    - Verilog CNN → process HEX input for hardware acceleration

- **Postprocessing:**  
    - `pipeline_1d_cnn.py` → software reference CNN for validation  
    - `img_display.py` → reconstruct Verilog outputs  
    - `simulation_analysis.py` → verification, reporting, error handling

---

**Python ensures correctness and flexible debugging; Verilog delivers hardware acceleration for CNNs—enabling both research and scalable deployment.**

---

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.  
