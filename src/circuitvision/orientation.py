"""Orientation of polarized symbols (diode, voltage source) -> terminal polarity.

Convention inside CircuitVision:
  rotation is in degrees counter-clockwise, one of 0/90/180/270, and at
  rotation 0 the positive terminal (anode, or + of a source) is on PLUS_AT_0.

CGHD stores a rotation per symbol. The defaults below are UNVERIFIED
assumptions. After inspecting the rotation sheets (Gate 1 in EXPERIMENTS.md),
set the verified values without editing code:
  CIRCUITVISION_PLUS_AT_0=left|right|top|bottom
  CIRCUITVISION_CGHD_CLOCKWISE=0|1

Classifier: HOG features + linear SVM (OpenCV), one model per class, trained
on CGHD crops. Each crop is also rotated by 90/180/270 with the label shifted,
so every labelled crop yields 4 training samples.

  python -m circuitvision.orientation --cghd ~/data/cghd --out models
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .types import Detection

SIDES = ("left", "bottom", "right", "top")          # counter-clockwise order
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
POLARIZED = {"diode", "voltage_source"}
ANGLES = (0, 90, 180, 270)

PLUS_AT_0 = os.environ.get("CIRCUITVISION_PLUS_AT_0", "left")
CGHD_CLOCKWISE = os.environ.get("CIRCUITVISION_CGHD_CLOCKWISE", "0") == "1"
CONVENTION_SOURCE = ("environment (set after Gate 1)"
                     if {"CIRCUITVISION_PLUS_AT_0", "CIRCUITVISION_CGHD_CLOCKWISE"} & set(os.environ)
                     else "code default (UNVERIFIED)")
if PLUS_AT_0 not in SIDES:
    raise ValueError(f"CIRCUITVISION_PLUS_AT_0 must be one of {SIDES}, got {PLUS_AT_0!r}")


def from_cghd(rotation) -> int | None:
    """CGHD <rotation> (degrees, 1-degree resolution) -> nearest of 0/90/180/270 in
    CircuitVision's counter-clockwise convention. Unparsable values -> None."""
    try:
        r = int(round(float(rotation) / 90) * 90) % 360
    except (TypeError, ValueError):
        return None
    return (-r) % 360 if CGHD_CLOCKWISE else r


def is_mirrored(value) -> bool | None:
    """CGHD mirror annotation -> True/False, or None when not annotated."""
    if value is None:
        return None
    return str(value).strip().lower() in ("1", "true", "yes", "y")


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


def collect_cghd(root: Path, drafters: set[str] | None = None, exclude=frozenset(), skipped=None):
    """-> {cls: [(crop, rotation)]} from polarized CGHD symbols that carry a rotation.

    Mirrored symbols are skipped (and counted in `skipped`): whether a mirror
    flips polarity depends on the CGHD symbol template, which is not documented.
    """
    from .cghd_to_yolo import find_image, load_for_annotation, parse_voc
    skipped = skipped if skipped is not None else {}
    data = {c: [] for c in POLARIZED}
    for drafter in sorted(root.glob("drafter_*")):
        if drafter.name in exclude or (drafters and drafter.name not in drafters):
            continue
        for xml in sorted((drafter / "annotations").glob("*.xml")):
            w, h, all_objs = parse_voc(xml)
            objs = []
            for o in all_objs:
                if o.cls not in POLARIZED:
                    continue
                if from_cghd(o.rotation) is None:
                    skipped["no_rotation"] = skipped.get("no_rotation", 0) + 1
                elif is_mirrored(o.mirror):
                    skipped["mirrored"] = skipped.get("mirrored", 0) + 1
                else:
                    objs.append(o)
            img_path = find_image(drafter / "images", xml.stem)
            if not objs or img_path is None:
                continue
            img, _ = load_for_annotation(img_path, w, h)
            if img is None:
                continue
            for o in objs:
                crop = img[int(o.y1):int(o.y2), int(o.x1):int(o.x2)]
                if crop.size:
                    data[o.cls].append((crop, from_cghd(o.rotation)))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cghd", required=True, type=Path)
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--test-drafters", default="12")
    ap.add_argument("--exp", type=Path, help="write orientation.json here")
    a = ap.parse_args()

    test = {f"drafter_{d}" for d in a.test_drafters.split(",")}
    skipped_train, skipped_test = {}, {}
    train_data = collect_cghd(a.cghd, exclude=test, skipped=skipped_train)
    test_data = collect_cghd(a.cghd, drafters=test, skipped=skipped_test)
    print("skipped (train):", skipped_train, " skipped (test):", skipped_test)

    svms, report = {}, {"convention": {"PLUS_AT_0": PLUS_AT_0, "CGHD_CLOCKWISE": CGHD_CLOCKWISE,
                                        "source": CONVENTION_SOURCE},
                         "test_drafters": sorted(test), "classes": {},
                         "skipped_train": skipped_train, "skipped_test": skipped_test}
    for cls, items in train_data.items():
        samples = [s for crop, r in items for s in rotated_samples(crop, r)]
        if len({r for _, r in samples}) < 2:
            print(f"{cls}: not enough labelled samples, skipped")
            continue
        X = np.stack([features(c) for c, _ in samples])
        y = np.array([r for _, r in samples])
        svms[cls] = OrientationModel.train_svm(X, y)
        print(f"{cls}: trained on {len(samples)} samples")
        report["classes"][cls] = {"train_crops": len(items), "train_samples": len(samples),
                                  "train_label_counts": {str(k): int(v) for k, v in
                                                         zip(*np.unique(y, return_counts=True))}}

    model = OrientationModel(svms)
    model.save(a.out)
    for cls, items in test_data.items():
        if cls in svms and items:
            raw_acc = accuracy(model, cls, items)
            acc = accuracy(model, cls, [s for c, r in items for s in rotated_samples(c, r)])
            print(f"{cls}: test accuracy {raw_acc:.3f} on {len(items)} real crops, "
                  f"{acc:.3f} on {len(items) * 4} rotated crops")
            report["classes"][cls].update(test_crops=len(items), test_accuracy_real=raw_acc,
                                          test_accuracy_rotated=acc)
    if a.exp:
        a.exp.mkdir(parents=True, exist_ok=True)
        (a.exp / "orientation.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
