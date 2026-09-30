"""Visual checks a human must look at before trusting the data.

  # YOLO labels drawn on the images training will actually see
  python -m circuitvision.visualize labels --yolo data/yolo --out experiments/baseline_v1/label_check -n 24

  # Polarized symbols grouped by their CGHD <rotation> value
  python -m circuitvision.visualize rotation --cghd ~/data/cghd --out experiments/baseline_v1/rotation_check

Images are read with cv2.imread, which applies EXIF orientation, the same
way ultralytics loads them for training. A shifted or rotated box here is a
shifted or rotated label in training.
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .cghd_to_yolo import find_image, parse_voc
from .classes import CLASSES

COLORS = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 140, 255), (160, 0, 160),
          (0, 0, 0), (200, 200, 0), (120, 60, 0), (128, 128, 128)]


def _fit(img, max_side=1280):
    s = max_side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def draw_yolo(img, label_lines):
    h, w = img.shape[:2]
    out = img.copy()
    t = max(2, round(max(h, w) / 800))
    for line in label_lines:
        c, cx, cy, bw, bh = line.split()
        c = int(c)
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1, p2 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(out, p1, p2, COLORS[c % len(COLORS)], t)
        cv2.putText(out, CLASSES[c], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t, COLORS[c % len(COLORS)], max(1, t // 2))
    return out


def pick_stratified(label_files, n, seed=0):
    """Choose n label files so every class shows up, rare classes first."""
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for f in label_files:
        for c in {int(l.split()[0]) for l in f.read_text().split("\n") if l.strip()}:
            by_cls[c].append(f)
    chosen = []
    per_class = max(1, n // max(1, len(by_cls)))
    for c in sorted(by_cls, key=lambda c: len(by_cls[c])):
        pool = [f for f in by_cls[c] if f not in chosen]
        chosen += rng.sample(pool, min(per_class, len(pool)))
    rest = [f for f in label_files if f not in chosen]
    chosen += rng.sample(rest, min(max(0, n - len(chosen)), len(rest)))
    return chosen[:n]


def labels(yolo: Path, out: Path, n: int, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((yolo / "labels").glob("*/*.txt"))
    index = []
    for f in pick_stratified(files, n, seed):
        split = f.parent.name
        img_path = next((yolo / "images" / split).glob(f.stem + ".*"))
        img = cv2.imread(str(img_path))
        lines = [l for l in f.read_text().split("\n") if l.strip()]
        cv2.imwrite(str(out / f"{split}_{f.stem}.jpg"), _fit(draw_yolo(img, lines)),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        counts = defaultdict(int)
        for l in lines:
            counts[CLASSES[int(l.split()[0])]] += 1
        index.append(f"{split}_{f.stem}.jpg  {dict(counts)}")
    (out / "index.txt").write_text("\n".join(index) + "\n")
    print(f"{len(index)} overlays -> {out}")


def _tile(crop, size=96, pad_color=255):
    h, w = crop.shape[:2]
    s = size / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
    tile = np.full((size, size, 3), pad_color, np.uint8)
    y, x = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    tile[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (200, 200, 200), 1)


def rotation(cghd: Path, out: Path, per_group=8, prefixes=("diode", "voltage"), seed=0):
    """One sheet per raw label: a row per rotation value, crops with context."""
    rng = random.Random(seed)
    groups = defaultdict(list)                       # (label, rot) -> [(img_path, box)]
    for xml in sorted(cghd.glob("drafter_*/annotations/*.xml")):
        for name, x1, y1, x2, y2, _, rot in parse_voc(xml, raw=True)[2]:
            if name.lower().startswith(prefixes):
                groups[(name, (rot or "none").strip())].append((xml, (x1, y1, x2, y2)))
    out.mkdir(parents=True, exist_ok=True)
    labels_seen = sorted({k[0] for k in groups})
    for label in labels_seen:
        rows = []
        for rot in sorted({k[1] for k in groups if k[0] == label}, key=lambda r: (r == "none", r)):
            items = groups[(label, rot)]
            tiles = []
            for xml, (x1, y1, x2, y2) in rng.sample(items, min(per_group, len(items))):
                img = cv2.imread(str(find_image(xml.parent.parent / "images", xml.stem)))
                if img is None:
                    continue
                m = int(0.25 * max(x2 - x1, y2 - y1))       # context so wires show
                crop = img[max(0, int(y1) - m):int(y2) + m, max(0, int(x1) - m):int(x2) + m]
                if crop.size:
                    tiles.append(_tile(crop))
            tiles += [np.full((96, 96, 3), 255, np.uint8)] * (per_group - len(tiles))
            head = np.full((96, 150, 3), 255, np.uint8)
            cv2.putText(head, f"rot={rot}", (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(head, f"n={len(items)}", (6, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
            rows.append(np.hstack([head] + tiles))
        sheet = np.vstack(rows)
        cv2.imwrite(str(out / f"rotation_{label}.png"), sheet)
        print(f"{label}: {len(rows)} rotation values -> rotation_{label}.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("labels")
    l.add_argument("--yolo", required=True, type=Path)
    l.add_argument("--out", required=True, type=Path)
    l.add_argument("-n", type=int, default=24)
    r = sub.add_parser("rotation")
    r.add_argument("--cghd", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.cmd == "labels":
        labels(a.yolo, a.out, a.n)
    else:
        rotation(a.cghd, a.out)


if __name__ == "__main__":
    main()
