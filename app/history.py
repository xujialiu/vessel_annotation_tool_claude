"""
History management for the vessel annotation tool.

Provides undo/redo functionality with zlib compression.
"""

import zlib
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AppConfig import AppConfig


class CompressedHistory:
    """
    Undo/redo history manager with zlib compression.

    Stores mask states efficiently using compression to reduce memory usage.
    """

    def __init__(self, max_size=None):
        """
        Initialize history.

        Args:
            max_size: Maximum number of states to keep. Defaults to AppConfig.History.MAX_SIZE
        """
        if max_size is None:
            max_size = AppConfig.History.MAX_SIZE
        self._history = []
        self._max_size = max_size
        self._current_idx = -1
        self._num_masks = 1

    def clear(self):
        """Clear all history."""
        self._history.clear()
        self._current_idx = -1

    def set_num_masks(self, num):
        """Set the number of masks being tracked."""
        self._num_masks = num

    def push(self, *masks):
        """
        Push a new state onto the history.

        Args:
            *masks: One or more mask arrays to save
        """
        if self._current_idx < len(self._history) - 1:
            self._history = self._history[: self._current_idx + 1]

        combined = np.stack(masks, axis=0)
        compressed = zlib.compress(
            combined.tobytes(), level=AppConfig.History.COMPRESSION_LEVEL
        )
        shape = combined.shape

        self._history.append((compressed, shape, combined.dtype))
        self._current_idx += 1

        if len(self._history) > self._max_size:
            self._history.pop(0)
            self._current_idx -= 1

    def get(self, idx):
        """
        Get state at given index.

        Args:
            idx: History index

        Returns:
            Tuple of mask arrays, or tuple of Nones if index invalid
        """
        if idx < 0 or idx >= len(self._history):
            return tuple([None] * self._num_masks)

        compressed, shape, dtype = self._history[idx]
        combined = np.frombuffer(zlib.decompress(compressed), dtype=dtype).reshape(
            shape
        )
        return tuple(combined[i].copy() for i in range(combined.shape[0]))

    def can_undo(self):
        """Check if undo is available."""
        return self._current_idx > 0

    def can_redo(self):
        """Check if redo is available."""
        return self._current_idx < len(self._history) - 1

    def undo(self):
        """
        Undo to previous state.

        Returns:
            Tuple of mask arrays from previous state
        """
        if self.can_undo():
            self._current_idx -= 1
            return self.get(self._current_idx)
        return tuple([None] * self._num_masks)

    def redo(self):
        """
        Redo to next state.

        Returns:
            Tuple of mask arrays from next state
        """
        if self.can_redo():
            self._current_idx += 1
            return self.get(self._current_idx)
        return tuple([None] * self._num_masks)

    @property
    def current_idx(self):
        """Get current history index."""
        return self._current_idx
