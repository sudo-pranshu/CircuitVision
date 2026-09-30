"""Convert CGHD (Pascal VOC XML) to a YOLO dataset restricted to our classes.

CGHD layout:  <root>/drafter_<k>/images/*.jpg  and  .../annotations/*.xml
Split is by drafter, so no drafter appears in two splits (avoids leakage,
since each circuit is photographed 4 times).

Usage:
  python -m circuitvision.cghd_to_yolo --cghd ~/data/cghd --out data/yolo
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .classes import CLASSES, map_cghd

IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    w = int(root.findtext("size/width"))
    h = int(root.findtext("size/height"))
    objs = []
    for o in root.iter("object"):
        cls = map_cghd(o.findtext("name", ""))
        if cls is None:
            continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        rot = o.findtext("rotation")
        objs.append((cls, x1, y1, x2, y2, o.findtext("text"), rot))
    return w, h, objs


def to_yolo_lines(w, h, objs):
    lines = []
    for cls, x1, y1, x2, y2, *_ in objs:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cghd", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-drafters", default="11", help="comma-separated ids")
    ap.add_argument("--test-drafters", default="12")
    a = ap.parse_args()

    val = {f"drafter_{d}" for d in a.val_drafters.split(",")}
    test = {f"drafter_{d}" for d in a.test_drafters.split(",")}
    counts = {"train": 0, "val": 0, "test": 0}

    for drafter in sorted(p for p in a.cghd.glob("drafter_*") if p.is_dir()):
        split = "val" if drafter.name in val else "test" if drafter.name in test else "train"
        for xml in sorted((drafter / "annotations").glob("*.xml")):
            img = find_image(drafter / "images", xml.stem)
            if img is None:
                continue
            w, h, objs = parse_voc(xml)
            if not objs:
                continue
            name = f"{drafter.name}_{img.name}"
            (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy(img, a.out / "images" / split / name)
            (a.out / "labels" / split / (Path(name).stem + ".txt")).write_text(
                "\n".join(to_yolo_lines(w, h, objs)) + "\n")
            counts[split] += 1

    yaml = [f"path: {a.out.resolve()}", "train: images/train",
            "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (a.out / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print(counts)


if __name__ == "__main__":
    main()
