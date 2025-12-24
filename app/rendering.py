"""
Rendering utilities for the vessel annotation tool.

Provides mask-to-RGBA conversion and Qt image conversion functions.
"""

import numpy as np
from PySide6.QtGui import QPixmap, QImage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AppConfig import AppConfig


# LUT accessor functions
def _get_av_lut():
    """Get artery/vein color lookup table."""
    return AppConfig.Color.get_av_lut()


def _get_od_lut():
    """Get optic disc color lookup table."""
    return AppConfig.Color.get_od_lut()


def _get_artery_only_lut():
    """Get artery-only display lookup table."""
    return AppConfig.Color.get_artery_only_lut()


def _get_vein_only_lut():
    """Get vein-only display lookup table."""
    return AppConfig.Color.get_vein_only_lut()


# Mask to RGBA conversion functions
def masks_to_rgba_av(artery_mask, vein_mask, alpha):
    """
    Convert artery and vein masks to RGBA image.

    Args:
        artery_mask: Binary artery mask
        vein_mask: Binary vein mask
        alpha: Opacity (0.0 to 1.0)

    Returns:
        RGBA numpy array
    """
    combined = (artery_mask > 0).astype(np.uint8) | (
        (vein_mask > 0).astype(np.uint8) << 1
    )
    rgba = _get_av_lut()[combined]
    mask = combined > 0
    if mask.any():
        rgba[..., 3] = (mask * int(alpha * 255)).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def mask_to_rgba_od(disc_mask, alpha):
    """
    Convert optic disc mask to RGBA image.

    Args:
        disc_mask: Binary disc mask
        alpha: Opacity (0.0 to 1.0)

    Returns:
        RGBA numpy array
    """
    binary = (disc_mask > 0).astype(np.uint8)
    rgba = _get_od_lut()[binary]
    mask = binary > 0
    if mask.any():
        rgba[..., 3] = (mask * int(alpha * 255)).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def mask_to_rgba_artery_only(artery_mask, alpha):
    """
    Display only artery mask as red (for dedicated artery annotation mode).

    Args:
        artery_mask: Binary artery mask
        alpha: Opacity (0.0 to 1.0)

    Returns:
        RGBA numpy array
    """
    binary = (artery_mask > 0).astype(np.uint8)
    rgba = _get_artery_only_lut()[binary]
    mask = binary > 0
    if mask.any():
        rgba[..., 3] = (mask * int(alpha * 255)).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def mask_to_rgba_vein_only(vein_mask, alpha):
    """
    Display only vein mask as blue (for dedicated vein annotation mode).

    Args:
        vein_mask: Binary vein mask
        alpha: Opacity (0.0 to 1.0)

    Returns:
        RGBA numpy array
    """
    binary = (vein_mask > 0).astype(np.uint8)
    rgba = _get_vein_only_lut()[binary]
    mask = binary > 0
    if mask.any():
        rgba[..., 3] = (mask * int(alpha * 255)).astype(np.uint8)
    return np.ascontiguousarray(rgba)


def masks_to_rgba_region_av(artery_mask, vein_mask, alpha, x0, y0, x1, y1):
    """Convert a region of artery/vein masks to RGBA."""
    return masks_to_rgba_av(artery_mask[y0:y1, x0:x1], vein_mask[y0:y1, x0:x1], alpha)


def mask_to_rgba_region_od(disc_mask, alpha, x0, y0, x1, y1):
    """Convert a region of optic disc mask to RGBA."""
    return mask_to_rgba_od(disc_mask[y0:y1, x0:x1], alpha)


# Qt image conversion functions
def numpy_rgba_to_qimage(rgba):
    """Convert numpy RGBA array to QImage."""
    if not rgba.flags["C_CONTIGUOUS"]:
        rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)


def numpy_rgb_to_qpixmap(rgb):
    """Convert numpy RGB array to QPixmap."""
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)
