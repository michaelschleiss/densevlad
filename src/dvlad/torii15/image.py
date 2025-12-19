from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _rgb_to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    # Match MATLAB rgb2gray for uint8 input: weighted sum + round half up.
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.floor(gray + 0.5).astype(np.uint8)
    return gray


def _vl_imdown_sample(gray: np.ndarray) -> np.ndarray:
    # Mirrors VLFeat vl_imdown(I, 'method', 'sample') with 1-based indexing:
    # I(1:2:floor(end-.5), 1:2:floor(end-.5))
    h, w = gray.shape[:2]
    return gray[: max(h - 1, 0) : 2, : max(w - 1, 0) : 2]


def read_gray_im2single(path: str | Path) -> np.ndarray:
    """
    Loads an image, converts to grayscale (uint8), applies vl_imdown (sample),
    and converts to float32 in [0, 1], matching Torii15's preprocessing:

        img = imread(); rgb2gray(); vl_imdown(); im2single()
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im)
    gray = _rgb_to_gray_uint8(rgb)
    gray = _vl_imdown_sample(gray)
    return gray.astype(np.float32) / 255.0


def read_gray_uint8(path: str | Path) -> np.ndarray:
    """
    Loads an image and returns the grayscale uint8 after vl_imdown.
    Useful for debugging integer-space operations.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im)
    gray = _rgb_to_gray_uint8(rgb)
    return _vl_imdown_sample(gray)


def image_shape_after_imdown(path: str | Path) -> Tuple[int, int]:
    g = read_gray_uint8(path)
    return int(g.shape[0]), int(g.shape[1])

