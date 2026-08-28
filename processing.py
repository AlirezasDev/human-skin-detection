import cv2
import numpy as np

from config import MorphologyConfig, DEFAULT_CONFIG


def apply_morphology(
    mask: np.ndarray, config: MorphologyConfig = DEFAULT_CONFIG
) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.kernel_size, config.kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.iterations)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=config.iterations)
    return opened


def keep_largest_components(
    mask: np.ndarray, config: MorphologyConfig = DEFAULT_CONFIG
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(areas)[::-1]
    selected = sorted_indices[: config.max_components]
    result = np.zeros_like(mask)
    for idx in selected:
        if areas[idx] >= config.min_area:
            result[labels == idx + 1] = 255
    return result


def postprocess_mask(
    mask: np.ndarray, config: MorphologyConfig = DEFAULT_CONFIG
) -> np.ndarray:
    morphed = apply_morphology(mask, config)
    filtered = keep_largest_components(morphed, config)
    return filtered


def extract_skin(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)


def apply_black_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image)
    result[mask > 0] = image[mask > 0]
    return result
