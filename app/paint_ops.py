"""
Paint operations for the vessel annotation tool.

Provides brush mask generation and painting operations.
"""

import numpy as np
from functools import lru_cache


@lru_cache(maxsize=256)
def get_brush_mask(size):
    """
    Get a brush mask for a given diameter size.

    Args:
        size: Brush diameter in pixels

    Returns:
        Circular brush mask as uint8 numpy array
    """
    if size <= 0:
        size = 1

    if size == 1:
        return np.array([[1]], dtype=np.uint8)

    if size == 2:
        return np.ones((2, 2), dtype=np.uint8)

    y, x = np.ogrid[:size, :size]
    center = (size - 1) / 2.0
    dist_sq = (x - center) ** 2 + (y - center) ** 2
    radius = size / 2.0
    return (dist_sq <= radius**2).astype(np.uint8)


class PaintOperations:
    """Static methods for brush painting operations on masks."""

    @staticmethod
    def apply_brush(mask, x, y, size, value):
        """
        Apply brush at position (x, y) with given diameter size.

        Args:
            mask: Target mask array
            x, y: Center position
            size: Brush diameter
            value: 1 to draw, 0 to erase
        """
        h, w = mask.shape

        half = (size - 1) // 2
        extra = size - 1 - half

        x0 = x - half
        y0 = y - half
        x1 = x + extra + 1
        y1 = y + extra + 1

        bx0 = 0
        by0 = 0
        bx1 = size
        by1 = size

        if x0 < 0:
            bx0 = -x0
            x0 = 0
        if y0 < 0:
            by0 = -y0
            y0 = 0
        if x1 > w:
            bx1 = size - (x1 - w)
            x1 = w
        if y1 > h:
            by1 = size - (y1 - h)
            y1 = h

        if x1 <= x0 or y1 <= y0:
            return

        brush = get_brush_mask(size)
        sub_brush = brush[by0:by1, bx0:bx1]
        region = mask[y0:y1, x0:x1]

        if value:
            np.bitwise_or(region, sub_brush, out=region)
        else:
            np.bitwise_and(region, ~sub_brush, out=region)

    @staticmethod
    def modify_artery(artery_mask, vein_mask, x, y, size):
        """
        Modify vein to artery within brush area.

        Converts vein pixels to artery pixels within the brush region.
        """
        h, w = artery_mask.shape

        half = (size - 1) // 2
        extra = size - 1 - half

        x0 = x - half
        y0 = y - half
        x1 = x + extra + 1
        y1 = y + extra + 1

        bx0 = 0
        by0 = 0
        bx1 = size
        by1 = size

        if x0 < 0:
            bx0 = -x0
            x0 = 0
        if y0 < 0:
            by0 = -y0
            y0 = 0
        if x1 > w:
            bx1 = size - (x1 - w)
            x1 = w
        if y1 > h:
            by1 = size - (y1 - h)
            y1 = h

        if x1 <= x0 or y1 <= y0:
            return

        brush = get_brush_mask(size)
        sub_brush = brush[by0:by1, bx0:bx1].astype(bool)

        a_region = artery_mask[y0:y1, x0:x1]
        v_region = vein_mask[y0:y1, x0:x1]
        overlap_mask = sub_brush & (v_region > 0)
        a_region[overlap_mask] = 1
        v_region[overlap_mask] = 0

    @staticmethod
    def modify_vein(artery_mask, vein_mask, x, y, size):
        """
        Modify artery to vein within brush area.

        Converts artery pixels to vein pixels within the brush region.
        """
        h, w = artery_mask.shape

        half = (size - 1) // 2
        extra = size - 1 - half

        x0 = x - half
        y0 = y - half
        x1 = x + extra + 1
        y1 = y + extra + 1

        bx0 = 0
        by0 = 0
        bx1 = size
        by1 = size

        if x0 < 0:
            bx0 = -x0
            x0 = 0
        if y0 < 0:
            by0 = -y0
            y0 = 0
        if x1 > w:
            bx1 = size - (x1 - w)
            x1 = w
        if y1 > h:
            by1 = size - (y1 - h)
            y1 = h

        if x1 <= x0 or y1 <= y0:
            return

        brush = get_brush_mask(size)
        sub_brush = brush[by0:by1, bx0:bx1].astype(bool)

        a_region = artery_mask[y0:y1, x0:x1]
        v_region = vein_mask[y0:y1, x0:x1]
        overlap_mask = sub_brush & (a_region > 0)
        a_region[overlap_mask] = 0
        v_region[overlap_mask] = 1
