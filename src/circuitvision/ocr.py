"""Read component values ("10k", "4.7uF") from detected text boxes.

Backend: EasyOCR (pip install easyocr; uses MPS/CUDA if available).
Restricted to digits + unit characters, which cuts misreads a lot.
"""
import numpy as np

from .types import Detection

ALLOW = "0123456789.,kKmMuµnpGRrFfHhVvΩ"

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr  # lazy: heavy import, downloads weights on first use
        _reader = easyocr.Reader(["en"], verbose=False)
    return _reader


def read_values(img: np.ndarray, dets: list[Detection], pad: int = 4) -> None:
    """Fill d.text for every 'text' detection that has none (in place)."""
    todo = [d for d in dets if d.cls == "text" and not d.text]
    if not todo:
        return
    reader = _get_reader()
    h, w = img.shape[:2]
    for d in todo:
        x1, y1, x2, y2 = d.box
        crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            continue
        if (y2 - y1) > 1.5 * (x2 - x1):          # vertical label: rotate upright
            crop = np.rot90(crop, -1).copy()
        res = reader.readtext(crop, allowlist=ALLOW, detail=0, paragraph=True)
        d.text = " ".join(res).strip() or None
