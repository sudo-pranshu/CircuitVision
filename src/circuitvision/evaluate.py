"""Batch evaluation over a folder of images with CGHD (VOC XML) annotations.

  # wire tracing alone, on ground-truth boxes
  python -m circuitvision.evaluate --images D/images --annotations D/annotations

  # end to end with the detector, compared against the ground-truth-box run
  python -m circuitvision.evaluate --images D/images --annotations D/annotations \
      --weights runs/detect/train/weights/best.pt

Metrics per image (written to CSV, averaged in the summary):
  terminals_ok   share of elements with exactly 2 terminals
  valid          netlist passes grammar + semantic checks
  simulates      ngspice produced an operating point
  struct_match   (with --weights) same element counts per class and same
                 number of nodes as the ground-truth-box netlist
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from . import grammar, netlist, preprocess, simulate
from .pipeline import detections_from_xml, run


def netlist_for(img, dets):
    comps, _ = run(img, dets)
    return comps, netlist.to_spice(comps)


def signature(comps):
    two = [c for c in comps if len(c.nets) == 2]
    return Counter(c.cls for c in two), len({n for c in two for n in c.nets})


def evaluate_one(img, dets, sim: bool):
    comps, text = netlist_for(img, dets)
    r = {
        "elements": len(comps),
        "terminals_ok": (sum(len(c.nets) == 2 for c in comps) / len(comps)) if comps else 0.0,
        "valid": int(not grammar.validate(text)),
        "simulates": "",
    }
    if sim:
        try:
            simulate.operating_point(text)
            r["simulates"] = 1
        except (RuntimeError, OSError):
            r["simulates"] = 0
    return r, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--weights")
    ap.add_argument("--csv", default="eval.csv")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    sim = simulate.available()
    det = None
    if a.weights:
        from .detect import Detector
        det = Detector(a.weights)

    rows = []
    for xml in sorted(a.annotations.glob("*.xml"))[:a.limit]:
        img_path = next((p for p in a.images.glob(xml.stem + ".*")
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")), None)
        if img_path is None:
            continue
        img = preprocess.load(str(img_path))
        gt_dets = detections_from_xml(xml)
        row, gt_comps = evaluate_one(img, gt_dets, sim)
        row = {"image": img_path.name, **{f"gt_{k}": v for k, v in row.items()}}
        if det:
            pred_dets = det(img)
            for d in gt_dets:                      # reuse GT text for values
                if d.cls == "text":
                    pred_dets.append(d)
            prow, pcomps = evaluate_one(img, pred_dets, sim)
            row.update({f"pred_{k}": v for k, v in prow.items()})
            row["struct_match"] = int(signature(pcomps) == signature(gt_comps))
        rows.append(row)

    if not rows:
        print("no image/annotation pairs found")
        return
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} images -> {a.csv}")
    for k in rows[0]:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        if vals and k != "image":
            print(f"  {k:<20} {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
