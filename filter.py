import _filter
import numpy as np
from PIL import Image


def bilateral_filter(src, diameter=5, sigma_space=20, sigma_color=20):
    modes = ['L', 'LA', 'YCbCr', 'RGB', 'RGBA']
    assert src.mode in modes
    assert diameter % 2 != 0
    tmp = np.array(src, dtype=np.uint8)
    if tmp.ndim == 2:
        dst = _filter._l_bilateral_solver(
            tmp, diameter, sigma_space, sigma_color)
    elif tmp.ndim == 3:
        if src.mode == 'LA':
            dst = tmp.copy()
            dst[:, :, 1] = _filter._l_bilateral_solver(
                tmp[:, :, 1], diameter, sigma_space, sigma_color)
        if src.mode == 'YCbCr':
            dst = tmp.copy()
            dst[:, :, 0] = _filter._l_bilateral_solver(
                tmp[:, :, 0], diameter, sigma_space, sigma_color)
        elif src.mode == 'RGB':
            dst = _filter._rgb_bilateral_solver(
                tmp, diameter, sigma_space, sigma_color)
        elif src.mode == 'RGBA':
            dst = tmp.copy()
            dst[:, :, 1:] = _filter._rgb_bilateral_solver(
                tmp[:, :, 1:], diameter, sigma_space, sigma_color)
    return Image.fromarray(dst, mode=src.mode)
