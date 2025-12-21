"""
Simple VascX-style Contrast Enhancement for Retinal Fundus Images
"""

import numpy as np
import cv2


def detect_fundus_mask(image: np.ndarray) -> np.ndarray:
    """Detect fundus region and return binary mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def enhance_fundus_image(image: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    Apply VascX-style Gaussian contrast enhancement to a fundus image.

    This enhances vessel visibility by subtracting a Gaussian-blurred version
    of the image, removing low-frequency background variations.

    Args:
        image: RGB fundus image (numpy array, uint8)
        sigma: Gaussian blur sigma (default 50.0, larger = finer detail enhancement)

    Returns:
        Enhanced RGB image (numpy array, uint8)
    """
    # Detect fundus mask
    mask = detect_fundus_mask(image)

    # Convert to float
    img_float = image.astype(np.float32)

    # Mirror image at boundaries to avoid edge artifacts
    border = int(sigma)
    mirrored = cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_REFLECT
    )
    mirrored_float = mirrored.astype(np.float32)

    # Apply Gaussian blur
    ksize = int(6 * sigma) | 1  # Make kernel size odd
    blurred_mirrored = cv2.GaussianBlur(mirrored_float, (ksize, ksize), sigma)

    # Remove the border
    blurred = blurred_mirrored[border:-border, border:-border]

    # Contrast enhancement: enhanced = alpha * original + beta * blurred + gamma
    alpha = 4.0
    beta = -4.0
    gamma = 128.0
    enhanced = alpha * img_float + beta * blurred + gamma

    # Clip to valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    # Apply mask - black out outside fundus region
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    enhanced = np.where(mask_3ch > 0, enhanced, 0)

    return enhanced


# ============== DEMO ==============
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load image
    img_path = "/Users/xujialiu/Works/vessel_annotation_tool/datasets/dataset_exp/images/AITS0013_20231113_144756_Non_myd_L_1855.png"

    # Read as RGB
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Enhance
    enhanced = enhance_fundus_image(image, sigma=50.0)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(enhanced)
    axes[1].set_title("Enhanced", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()