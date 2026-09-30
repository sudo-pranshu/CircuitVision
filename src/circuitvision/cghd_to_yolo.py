"""Convert CGHD (Pascal VOC XML) to a YOLO dataset restricted to our classes.

CGHD layout:  <root>/drafter_<k>/images/*.jpg  and  .../annotations/*.xml
Split is by drafter, so no drafter appears in two splits (avoids leakage,
since each circuit is photographed 4 times).

Every image with an annotation file is kept, including images with no
in-scope objects (empty label file), so the test split is the whole drafter.
Boxes are clipped to the image; boxes with no area after clipping are dropped
and counted. Conversion statistics go to <out>/conversion_stats.json.

Usage:
  python -m circuitvision.cghd_to_yolo --cghd ~/data/cghd --out data/yolo
"""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .classes import CLASSES, map_cghd

IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_voc(xml_path: Path, raw: bool = False):
    """-> (w, h, objs). objs: (cls, x1, y1, x2, y2, text, rotation).

    raw=True keeps every object with its original CGHD label as cls.
    """
    root = ET.parse(xml_path).getroot()
    w = int(float(root.findtext("size/width") or 0))
    h = int(float(root.findtext("size/height") or 0))
    objs = []
    for o in root.iter("object"):
        name = o.findtext("name", "")
        cls = name if raw else map_cghd(name)
        if cls is None:
            continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((cls, x1, y1, x2, y2, o.findtext("text"), o.findtext("rotation")))
    return w, h, objs


def clip_box(x1, y1, x2, y2, w, h):
    """-> clipped box, or None if it has no area."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    return (x1, y1, x2, y2) if x2 - x1 >= 1 and y2 - y1 >= 1 else None


def to_yolo_lines(w, h, objs, stats: Counter | None = None):
    lines = []
    for cls, x1, y1, x2, y2, *_ in objs:
        box = clip_box(x1, y1, x2, y2, w, h)
        if stats is not None and box != (x1, y1, x2, y2):
            stats["boxes_clipped" if box else "boxes_dropped"] += 1
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXT:
        for cand in (img_dir / (stem + ext), img_dir / (stem + ext.upper())):
            if cand.exists():
                return cand
    return None


def split_of(drafter: str, val: set, test: set) -> str:
    return "val" if drafter in val else "test" if drafter in test else "train"


def convert(cghd: Path, out: Path, val: set, test: set) -> dict:
    stats = Counter()
    per_split = {s: Counter() for s in ("train", "val", "test")}
    ignored = Counter()
    failures = []

    for drafter in sorted(p for p in cghd.glob("drafter_*") if p.is_dir()):
        split = split_of(drafter.name, val, test)
        for xml in sorted((drafter / "annotations").glob("*.xml")):
            img = find_image(drafter / "images", xml.stem)
            if img is None:
                stats["missing_image"] += 1
                failures.append(f"{drafter.name}/{xml.name}: no image")
                continue
            try:
                w, h, raw_objs = parse_voc(xml, raw=True)
            except (ET.ParseError, TypeError, ValueError, AttributeError) as e:
                stats["xml_parse_error"] += 1
                failures.append(f"{drafter.name}/{xml.name}: {e}")
                continue
            if w <= 0 or h <= 0:
                stats["missing_size"] += 1
                failures.append(f"{drafter.name}/{xml.name}: size {w}x{h}")
                continue
            objs = []
            for o in raw_objs:
                cls = map_cghd(o[0])
                if cls is None:
                    ignored[o[0]] += 1
                else:
                    objs.append((cls, *o[1:]))
            lines = to_yolo_lines(w, h, objs, stats)
            name = f"{drafter.name}_{img.name}"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy(img, out / "images" / split / name)
            (out / "labels" / split / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            per_split[split]["images"] += 1
            per_split[split]["objects"] += len(lines)
            per_split[split].update(CLASSES[int(l.split()[0])] for l in lines)

    yaml = [f"path: {out.resolve()}", "train: images/train",
            "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.yaml").write_text("\n".join(yaml) + "\n")

    report = {
        "val_drafters": sorted(val), "test_drafters": sorted(test),
        "splits": {s: dict(c) for s, c in per_split.items()},
        "problems": dict(stats),
        "ignored_classes": dict(ignored.most_common()),
        "failures": failures[:200],
    }
    (out / "conversion_stats.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cghd", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-drafters", default="11", help="comma-separated ids")
    ap.add_argument("--test-drafters", default="12")
    a = ap.parse_args()

    val = {f"drafter_{d}" for d in a.val_drafters.split(",")}
    test = {f"drafter_{d}" for d in a.test_drafters.split(",")}
    r = convert(a.cghd, a.out, val, test)
    for s, c in r["splits"].items():
        print(f"{s:<5} images {c.get('images', 0):>5}  objects {c.get('objects', 0):>6}")
    print("problems:", r["problems"] or "none")
    print("ignored classes (top 10):", dict(list(r["ignored_classes"].items())[:10]))


if __name__ == "__main__":
    main()
