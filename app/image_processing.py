"""
Image processing utilities for the vessel annotation tool.

Provides fundus detection and image enhancement functions.
"""

import numpy as np
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AppConfig import AppConfig


def detect_fundus_mask(image: np.ndarray) -> np.ndarray:
    """
    Detect fundus region and return binary mask.

    Optimized version with reduced morphological iterations.

    Args:
        image: RGB image as numpy array

    Returns:
        Binary mask of fundus region
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(
        gray, AppConfig.Enhancement.FUNDUS_THRESHOLD, 255, cv2.THRESH_BINARY
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            AppConfig.Enhancement.MORPHOLOGY_KERNEL_SIZE,
            AppConfig.Enhancement.MORPHOLOGY_KERNEL_SIZE,
        ),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=AppConfig.Enhancement.MORPHOLOGY_CLOSE_ITERATIONS,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=AppConfig.Enhancement.MORPHOLOGY_OPEN_ITERATIONS,
    )
    return mask


def enhance_fundus_image(
    image: np.ndarray, sigma: float = None
) -> np.ndarray:
    """
    Apply VascX-style Gaussian contrast enhancement to a fundus image.

    Args:
        image: RGB image as numpy array
        sigma: Gaussian blur sigma. Defaults to AppConfig.Enhancement.GAUSSIAN_SIGMA

    Returns:
        Enhanced image as uint8 numpy array
    """
    if sigma is None:
        sigma = AppConfig.Enhancement.GAUSSIAN_SIGMA

    mask = detect_fundus_mask(image)
    img_float = image.astype(np.float32)
    ksize = int(6 * sigma) | 1
    blurred = cv2.GaussianBlur(
        img_float, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT
    )
    enhanced = cv2.addWeighted(
        img_float,
        AppConfig.Enhancement.WEIGHT_ORIGINAL,
        blurred,
        AppConfig.Enhancement.WEIGHT_BLUR,
        AppConfig.Enhancement.OFFSET,
    )
    np.clip(enhanced, 0, 255, out=enhanced)
    enhanced = enhanced.astype(np.uint8)
    mask_bool = mask == 0
    enhanced[mask_bool] = 0
    return np.ascontiguousarray(enhanced)
