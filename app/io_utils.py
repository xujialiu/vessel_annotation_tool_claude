"""
I/O utilities for the vessel annotation tool.

Provides file I/O, resource path resolution, and logging setup.
"""

import os
import sys
import logging
import numpy as np
from skimage import io


# Module-level logger
logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = None):
    """
    Setup logging configuration.

    Args:
        log_dir: Directory to save log file. If None, logs only to console.
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG level) - only if log_dir is provided
    if log_dir and os.path.isdir(log_dir):
        log_file = os.path.join(log_dir, "app.log")
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Log file: {log_file}")
        except Exception as e:
            logger.warning(f"Could not create log file: {e}")


def resource_path(relative_path):
    """
    Get absolute path to resource, works for development and PyInstaller.

    When running as a PyInstaller bundle, files are extracted to a temp
    folder stored in sys._MEIPASS. This function handles both cases.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Running in normal Python environment
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def imread(filename):
    """Read image using skimage."""
    return io.imread(filename)


def imwrite(filename, array):
    """Write image using skimage with automatic dtype handling."""
    if array.dtype in (np.float32, np.float64):
        array = (array * 255 if array.max() <= 1.0 else array).astype(np.uint8)
    elif array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    io.imsave(filename, array, check_contrast=False)
