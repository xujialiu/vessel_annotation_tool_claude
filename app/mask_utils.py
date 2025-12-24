"""
Mask utilities for the vessel annotation tool.

Provides mask encoding/decoding and validation functions.
"""

import numpy as np


def ensure_mask(mask):
    """Ensure mask is uint8 dtype."""
    if mask is None:
        return None
    return mask if mask.dtype == np.uint8 else mask.astype(np.uint8)


def decode_combined_mask(combined_mask):
    """
    Split combined mask into artery and vein masks.

    The combined mask uses bit encoding:
    - Bit 0: artery
    - Bit 1: vein

    Returns:
        tuple: (artery_mask, vein_mask) as uint8 arrays
    """
    combined_mask = ensure_mask(combined_mask)
    return (combined_mask & 1).astype(np.uint8), ((combined_mask >> 1) & 1).astype(
        np.uint8
    )


def encode_to_combined_mask(artery_mask, vein_mask):
    """
    Combine artery and vein masks into a single mask.

    Uses bit encoding:
    - Bit 0: artery
    - Bit 1: vein

    Returns:
        Combined mask as uint8 array (values 0-3)
    """
    return (artery_mask > 0).view(np.uint8) | ((vein_mask > 0).view(np.uint8) << 1)
