"""Image preprocessing: phone photo of a drawing -> clean binary stroke mask."""
import cv2
import numpy as np


def load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def binarize(img: np.ndarray, block: int = 35, c: int = 12) -> np.ndarray:
    """Return uint8 mask: 255 = ink stroke, 0 = paper.

    Adaptive threshold handles uneven lighting / shadows from phone photos;
    a small opening removes paper texture and grid-paper speckle.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def remove_small_blobs(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)
