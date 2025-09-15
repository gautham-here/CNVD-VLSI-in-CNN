import numpy as np

# --------- 1D KERNELS (Integer Scaled) ---------

def identity_1d():
    return np.array([0, 1, 0], dtype=np.int8)

def box_blur_1d():
    return np.array([1, 1, 1], dtype=np.int8)

def gaussian_blur_1d():
    return np.array([1, 2, 1], dtype=np.int8)

def sobel_1d():
    return np.array([-1, 0, 1], dtype=np.int8)

# --------- 2D KERNELS (Integer Scaled) ---------

def identity_2d():
    return np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.int8)

def prewitt_h_2d():
    return np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype=np.int8)

def prewitt_v_2d():
    return np.array([[ 1,  1,  1],
                     [ 0,  0,  0],
                     [-1, -1, -1]], dtype=np.int8)

def sharpening_2d():
    return np.array([[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]], dtype=np.int8)

def box_blur_2d():
    return np.ones((3, 3), dtype=np.int8)  # Sum = 9, normalization handled externally

def gaussian_blur_2d():
    return np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]], dtype=np.int8)  # Original /16 externally

def sobel_h_2d():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.int8)

def sobel_v_2d():
    return np.array([[ 1,  2,  1],
                     [ 0,  0,  0],
                     [-1, -2, -1]], dtype=np.int8)

def scharr_2d():
    return np.array([[-3,  0,  3],
                     [-10, 0, 10],
                     [-3,  0,  3]], dtype=np.int8)

def laplacian_2d():
    return np.array([[0,  1, 0],
                     [1, -4, 1],
                     [0,  1, 0]], dtype=np.int8)

def laplacian_diag_2d():
    return np.array([[ 1,  1,  1],
                     [ 1, -8,  1],
                     [ 1,  1,  1]], dtype=np.int8)

# --------- KERNEL DICTIONARIES ---------

KERNELS_1D = {
    "identity": identity_1d(),
    "blur": box_blur_1d(),
    "gaussian": gaussian_blur_1d(),
    "edge": sobel_1d()
}

KERNELS_2D = {
    "identity": identity_2d(),
    "prewitt_h": prewitt_h_2d(),
    "prewitt_v": prewitt_v_2d(),
    "sharpen": sharpening_2d(),
    "box_blur": box_blur_2d(),
    "gaussian": gaussian_blur_2d(),
    "sobel_h": sobel_h_2d(),
    "sobel_v": sobel_v_2d(),
    "scharr": scharr_2d(),
    "laplacian": laplacian_2d(),
    "laplacian_diag": laplacian_diag_2d()
}
