"""Orientation of polarized symbols (diode, voltage source) -> terminal polarity.

Convention inside CircuitVision:
  rotation is in degrees counter-clockwise, one of 0/90/180/270, and at
  rotation 0 the positive terminal (anode, or + of a source) is on PLUS_AT_0.

CGHD stores a rotation per symbol. Check CGHD_CLOCKWISE and PLUS_AT_0 against
2-3 dataset images before trusting polarity results (see README).

Classifier: HOG features + linear SVM (OpenCV), one model per class, trained
on CGHD crops. Each crop is also rotated by 90/180/270 with the label shifted,
so every labelled crop yields 4 training samples.

  python -m circuitvision.orientation --cghd ~/data/cghd --out models
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .types import Detection

SIDES = ("left", "bottom", "right", "top")          # counter-clockwise order
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
POLARIZED = {"diode", "voltage_source"}
ANGLES = (0, 90, 180, 270)

PLUS_AT_0 = "left"
CGHD_CLOCKWISE = False


def from_cghd(rotation) -> int | None:
    if rotation is None:
        return None
    r = int(round(float(rotation) / 90) * 90) % 360
    return (-r) % 360 if CGHD_CLOCKWISE else r


def positive_side(cls: str, rotation: int | None) -> str | None:
    """Side of the bounding box holding the + / anode terminal, or None."""
    if cls not in POLARIZED or rotation is None:
        return None
    return SIDES[(SIDES.index(PLUS_AT_0) + rotation // 90) % 4]


# ---------- features ----------

_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def _square(gray: np.ndarray, size: int = 64) -> np.ndarray:
    h, w = gray.shape
    s = max(h, w)
    canvas = np.full((s, s), 255, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = gray
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def features(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return _HOG.compute(_square(gray)).ravel()


def rotated_samples(crop: np.ndarray, rotation: int):
    """Yield (crop rotated k*90 CCW, new rotation label)."""
    for k in range(4):
        yield np.ascontiguousarray(np.rot90(crop, k)), (rotation + 90 * k) % 360


# ---------- model ----------

class OrientationModel:
    def __init__(self, svms: dict[str, "cv2.ml.SVM"]):
        self.svms = svms

    @staticmethod
    def train_svm(X: np.ndarray, y: np.ndarray):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setC(1.0)
        svm.train(X.astype(np.float32), cv2.ml.ROW_SAMPLE, (y // 90).astype(np.int32))
        return svm

    def predict(self, cls: str, crop: np.ndarray) -> int | None:
        svm = self.svms.get(cls)
        if svm is None or crop.size == 0:
            return None
        _, r = svm.predict(features(crop)[None].astype(np.float32))
        return int(r[0, 0]) * 90

    def annotate(self, img: np.ndarray, dets: list[Detection]) -> None:
        """Set rotation on polarized detections that have none (in place)."""
        for d in dets:
            if d.cls in POLARIZED and d.rotation is None:
                x1, y1, x2, y2 = d.box
                d.rotation = self.predict(d.cls, img[y1:y2, x1:x2])

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        for cls, svm in self.svms.items():
            svm.save(str(out / f"orient_{cls}.xml"))

    @classmethod
    def load(cls, path: Path):
        svms = {p.stem.removeprefix("orient_"): cv2.ml.SVM_load(str(p))
                for p in Path(path).glob("orient_*.xml")}
        return cls(svms)


def accuracy(model: OrientationModel, cls: str, samples) -> float:
    pairs = list(samples)
    return sum(model.predict(cls, c) == r for c, r in pairs) / max(1, len(pairs))


def collect_cghd(root: Path, drafters: set[str] | None = None, exclude=frozenset()):
    """-> {cls: [(crop, rotation)]} from CGHD symbols that carry a rotation."""
    from .cghd_to_yolo import find_image, parse_voc
    data = {c: [] for c in POLARIZED}
    for drafter in sorted(root.glob("drafter_*")):
        if drafter.name in exclude or (drafters and drafter.name not in drafters):
            continue
        for xml in sorted((drafter / "annotations").glob("*.xml")):
            objs = [o for o in parse_voc(xml)[2] if o[0] in POLARIZED and o[6] is not None]
            img_path = find_image(drafter / "images", xml.stem)
            if not objs or img_path is None:
                continue
            img = cv2.imread(str(img_path))
            for cls, x1, y1, x2, y2, _, rot in objs:
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                if crop.size:
                    data[cls].append((crop, from_cghd(rot)))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cghd", required=True, type=Path)
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--test-drafters", default="12")
    a = ap.parse_args()

    test = {f"drafter_{d}" for d in a.test_drafters.split(",")}
    train_data = collect_cghd(a.cghd, exclude=test)
    test_data = collect_cghd(a.cghd, drafters=test)

    svms = {}
    for cls, items in train_data.items():
        samples = [s for crop, r in items for s in rotated_samples(crop, r)]
        if len({r for _, r in samples}) < 2:
            print(f"{cls}: not enough labelled samples, skipped")
            continue
        X = np.stack([features(c) for c, _ in samples])
        y = np.array([r for _, r in samples])
        svms[cls] = OrientationModel.train_svm(X, y)
        print(f"{cls}: trained on {len(samples)} samples")

    model = OrientationModel(svms)
    model.save(a.out)
    for cls, items in test_data.items():
        if cls in svms and items:
            acc = accuracy(model, cls, [s for c, r in items for s in rotated_samples(c, r)])
            print(f"{cls}: test accuracy {acc:.3f} on {len(items) * 4} samples")


if __name__ == "__main__":
    main()
