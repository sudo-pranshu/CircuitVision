"""Image preprocessing: phone photo of a drawing -> clean binary stroke mask."""
import cv2
import numpy as np


def load(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def binarize(img: np.ndarray, kernel: int | None = None, min_contrast: int = 12) -> np.ndarray:
    """Return uint8 mask: 255 = ink stroke, 0 = paper.

    1. Estimate the local paper brightness with a grey-level closing (removes
       every dark feature thinner than the kernel: pen strokes, grid lines,
       text), which also follows shadows and uneven lighting.
    2. Ink contrast = (paper - pixel) / paper, i.e. relative to the local
       paper brightness, so a stroke in shadow scores like one in daylight.
       Grid lines and noise are faint, pen strokes are dark: Otsu on the
       contrast histogram separates the two, with a floor of `min_contrast`
       (on a 0-255 scale).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # larger than any solid ink blob (junction dots, filled symbols), still local for shadows
    k = kernel or max(15, (min(gray.shape[:2]) // 15) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dark = (255.0 * cv2.subtract(paper, gray) / np.maximum(paper, 1)).astype(np.uint8)
    cand = dark[dark > min_contrast]
    thr = min_contrast
    if cand.size > 100:
        otsu, _ = cv2.threshold(cand.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(min_contrast, int(otsu))
    mask = np.where(dark > thr, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def remove_small_blobs(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)
