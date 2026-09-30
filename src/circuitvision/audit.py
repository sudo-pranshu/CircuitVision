"""Audit a raw CGHD download before training. Assumes nothing from the docs.

  python -m circuitvision.audit --cghd ~/data/cghd --out experiments/baseline_v1/audit

Writes audit.json, audit.md and raw_xml_samples.txt (verbatim <object> XML of
polarized symbols, so the rotation/text fields can be read by eye).
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from .cghd_to_yolo import IMG_EXT, find_image, split_of
from .classes import map_cghd

POLAR_RAW = re.compile(r"^(diode|voltage)", re.I)


def drafter_key(p: Path):
    m = re.search(r"-?\d+", p.name)
    return int(m.group()) if m else 10 ** 6


def image_sizes(path: Path):
    """-> (raw w,h ignoring EXIF, displayed w,h with EXIF applied)."""
    raw = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    shown = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if raw is None or shown is None:
        return None, None
    return (raw.shape[1], raw.shape[0]), (shown.shape[1], shown.shape[0])


def audit(cghd: Path, val: set, test: set, read_images: bool = True, n_samples: int = 5):
    drafters = sorted((p for p in cghd.glob("drafter_*") if p.is_dir()), key=drafter_key)
    r = {"root": str(cghd), "drafters": [d.name for d in drafters],
         "val_drafters_present": sorted(val & {d.name for d in drafters}),
         "test_drafters_present": sorted(test & {d.name for d in drafters}),
         "per_drafter": {}, "split_images": Counter(), "raw_classes": Counter(),
         "mapped_by_split": defaultdict(Counter), "boxes": Counter(),
         "size_check": Counter(), "rotation": defaultdict(Counter),
         "rotation_present": Counter(), "text_present": Counter(),
         "text_examples": defaultdict(list), "object_fields": Counter(), "image_dims": Counter()}
    samples, sample_files = [], set()

    for d in drafters:
        split = split_of(d.name, val, test)
        xmls = sorted((d / "annotations").glob("*.xml"))
        imgs = [p for p in (d / "images").glob("*") if p.suffix.lower() in IMG_EXT]
        img_stems = {p.stem for p in imgs}
        paired = [x for x in xmls if x.stem in img_stems]
        r["per_drafter"][d.name] = {
            "split": split, "xml": len(xmls), "images": len(imgs), "paired": len(paired),
            "xml_without_image": len(xmls) - len(paired),
            "images_without_xml": len(img_stems - {x.stem for x in xmls})}
        r["split_images"][split] += len(paired)

        for xml in paired:
            try:
                root = ET.parse(xml).getroot()
            except ET.ParseError:
                r["boxes"]["xml_parse_error"] += 1
                continue
            w = int(float(root.findtext("size/width") or 0))
            h = int(float(root.findtext("size/height") or 0))
            if read_images:
                raw, shown = image_sizes(find_image(d / "images", xml.stem))
                if raw is None:
                    r["size_check"]["unreadable_image"] += 1
                else:
                    r["image_dims"][f"{shown[0]}x{shown[1]}"] += 1
                    key = ("xml_matches_displayed" if (w, h) == shown else
                           "xml_matches_raw_only(EXIF rotated)" if (w, h) == raw else
                           "xml_size_mismatch")
                    r["size_check"][key] += 1
            for o in root.iter("object"):
                name = o.findtext("name", "")
                r["raw_classes"][name] += 1
                r["object_fields"].update(c.tag for c in o)
                mapped = map_cghd(name)
                if mapped:
                    r["mapped_by_split"][split][mapped] += 1
                b = o.find("bndbox")
                try:
                    x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                except (AttributeError, TypeError, ValueError):
                    r["boxes"]["unparsable"] += 1
                    continue
                r["boxes"]["total"] += 1
                if x2 <= x1 or y2 <= y1:
                    r["boxes"]["inverted_or_empty"] += 1
                if x1 < 0 or y1 < 0 or (w and x2 > w) or (h and y2 > h):
                    r["boxes"]["out_of_bounds"] += 1
                rot, text = o.findtext("rotation"), o.findtext("text")
                if rot is not None:
                    r["rotation_present"][name] += 1
                    r["rotation"][name][rot.strip()] += 1
                if text is not None:
                    r["text_present"][name] += 1
                    if len(r["text_examples"][name]) < 8:
                        r["text_examples"][name].append(text)
                if POLAR_RAW.match(name) and len(samples) < n_samples * 3 and \
                        (len(sample_files) < n_samples or str(xml) in sample_files):
                    sample_files.add(str(xml))
                    samples.append(f"--- {d.name}/{xml.name}  (xml size {w}x{h})\n"
                                   + ET.tostring(o, encoding="unicode").strip())

    r["polarized_objects_without_rotation"] = {
        k: v - r["rotation_present"][k] for k, v in r["raw_classes"].items()
        if POLAR_RAW.match(k) and v - r["rotation_present"][k] > 0}
    return json.loads(json.dumps(r)), samples


def to_markdown(r: dict) -> str:
    L = [f"# CGHD audit\n\nRoot: `{r['root']}`\n",
         f"- Drafters found ({len(r['drafters'])}): {', '.join(r['drafters'])}",
         f"- Validation drafter(s) present: {r['val_drafters_present'] or 'NONE: check split'}",
         f"- Test drafter(s) present: {r['test_drafters_present'] or 'NONE: check split'}",
         f"- Paired images per split: {r['split_images']}",
         f"- Boxes: {r['boxes']}",
         f"- XML size vs image: {r['size_check'] or 'not checked (--no-images)'}",
         f"- Fields seen inside <object>: {r['object_fields']}",
         f"- Polarized objects missing <rotation>: {r['polarized_objects_without_rotation'] or 'none'}",
         "\n## Per drafter\n", "| drafter | split | xml | images | paired | xml w/o image | image w/o xml |",
         "|---|---|---|---|---|---|---|"]
    for k, v in r["per_drafter"].items():
        L.append(f"| {k} | {v['split']} | {v['xml']} | {v['images']} | {v['paired']} | "
                 f"{v['xml_without_image']} | {v['images_without_xml']} |")
    L += ["\n## In-scope classes by split\n", "| class | train | val | test |", "|---|---|---|---|"]
    classes = sorted({c for s in r["mapped_by_split"].values() for c in s})
    for c in classes:
        L.append(f"| {c} | " + " | ".join(str(r["mapped_by_split"].get(s, {}).get(c, 0))
                                          for s in ("train", "val", "test")) + " |")
    L += ["\n## All raw CGHD labels\n"]
    L += [f"- {k}: {v}" for k, v in sorted(r["raw_classes"].items(), key=lambda x: -x[1])]
    L += ["\n## Rotation values (raw label -> value: count)\n"]
    L += [f"- {k}: {dict(v)}" for k, v in sorted(r["rotation"].items())]
    L += ["\n## Text field examples\n"]
    L += [f"- {k} ({r['text_present'][k]} with text): {v}" for k, v in sorted(r["text_examples"].items())]
    L += ["\n## Image dimensions (top 10)\n"]
    L += [f"- {k}: {v}" for k, v in sorted(r["image_dims"].items(), key=lambda x: -x[1])[:10]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cghd", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-drafters", default="11")
    ap.add_argument("--test-drafters", default="12")
    ap.add_argument("--no-images", action="store_true", help="skip reading image files")
    a = ap.parse_args()

    val = {f"drafter_{d}" for d in a.val_drafters.split(",")}
    test = {f"drafter_{d}" for d in a.test_drafters.split(",")}
    r, samples = audit(a.cghd, val, test, read_images=not a.no_images)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "audit.json").write_text(json.dumps(r, indent=1))
    md = to_markdown(r)
    (a.out / "audit.md").write_text(md)
    (a.out / "raw_xml_samples.txt").write_text("\n\n".join(samples) + "\n")
    print(md)
    print(f"raw XML samples: {a.out / 'raw_xml_samples.txt'}")


if __name__ == "__main__":
    main()
